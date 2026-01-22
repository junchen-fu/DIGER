import os
import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torch import optim
from tqdm import tqdm
import json
import math
from colorama import init
from utils import ensure_dir, set_color, get_local_time
from accelerate import PartialState
from model import Model
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import get_scheduler
from metrics import *
from utils import *
from vq import AutoSigmaGumbel, AutoSigmaGaussian, AutoSigmaSimple
from collections import defaultdict
from logging import getLogger
init(autoreset=True)
    
    
class Trainer(object):
    def __init__(self, config, model_rec: Model, model_id, accelerator, train_data=None,
                 valid_data=None, test_data=None, eos_token_id=None):
        self.config = config
        self.model_rec = model_rec
        self.model_id = model_id
        self.logger = getLogger()
        
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
        self.code_num = config["code_num"]
        self.code_length = config["code_length"]
        self.learner = config["learner"]
        self.lr_rec = config['lr_rec']
        self.lr_scheduler_type = config["lr_scheduler_type"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.early_stop = config["early_stop"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.gradient_accumulation_steps = config["gradient_accumulation_steps"]
        self.save_path = config["save_path"]
        ensure_dir(self.save_path)

        # Additional configs for loss computation (kept for code compatibility)
        self.sim = config.get('sim', 'cos')  # Similarity metric for contrastive loss
        self.alpha = config.get('alpha', 1)  # Weight for commitment loss
        self.loss_type = config.get('loss_type', 'mse')  # Loss type for VQ
        self.tau = config.get('tau', 0.07)  # Temperature for contrastive loss

        self.accelerator = accelerator

        self.state = PartialState()
        self.world_size = self.state.num_processes
        self.device = self.state.device
        self.all_item_code = None
        self.model_rec.device = self.device

        # Track global training step for tau annealing
        self.global_step = 0

        self.all_metrics = config["metrics"].split(",")
        self.valid_metric = config["valid_metric"]
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.max_steps = self.get_train_steps()
        self.warmup_steps = config["warmup_steps"]
        
        # IMPORTANT: Set parameter trainable status BEFORE creating optimizer
        # Control whether to freeze semantic_embedding via config flag
        freeze_semantic_embedding = bool(config.get('freeze_semantic_embedding', True))

        # Unfreeze all recommender model parameters first
        for param in model_rec.parameters():
            param.requires_grad = True

        # Optionally freeze semantic_embedding (pretrained item embeddings)
        if freeze_semantic_embedding:
            semantic_emb_name = 'semantic_embedding'
            for name, param in model_rec.named_parameters():
                if name.startswith(semantic_emb_name):
                    param.requires_grad = False
        
        self.rec_optimizer = self._build_optimizer(model_rec, self.lr_rec, self.weight_decay)

        if self.lr_scheduler_type == "linear":
            self.rec_lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.rec_optimizer,
                                                                    num_warmup_steps=self.warmup_steps,
                                                                    num_training_steps=self.max_steps)
        elif self.lr_scheduler_type == "constant":
            self.rec_lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.rec_optimizer,
                                                                      num_warmup_steps=self.warmup_steps)
        elif self.lr_scheduler_type == "cosine":
            self.rec_lr_scheduler = get_scheduler(
                            name="cosine",
                            optimizer=self.rec_optimizer,
                            num_warmup_steps=self.warmup_steps,
                            num_training_steps=self.max_steps,
                        )

        self.best_score = 0
        self.best_ckpt = None

        self.model_rec, self.rec_optimizer, self.rec_lr_scheduler, \
        self.model_id, self.train_data, self.valid_data, self.test_data = \
        self.accelerator.prepare(self.model_rec, self.rec_optimizer, self.rec_lr_scheduler,
                                 self.model_id, self.train_data, self.valid_data, self.test_data)

    def _count_parameters(self, model, model_name="Model"):
        """Count total and trainable parameters in a model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        self.log(f"========== {model_name} Parameters ==========")
        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        self.log(f"Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.2f}%)")
        self.log(f"=" * 50)
        
        return total_params, trainable_params

    def _count_module_parameters(self, model, module_name):
        """Count parameters in a specific module"""
        try:
            module = getattr(model, module_name, None)
            if module is not None:
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                self.log(f"  {module_name}: Total={total:,}, Trainable={trainable:,}")
        except:
            pass

    def _build_optimizer(self, model, lr, weight_decay):
        params = model.parameters()
        learner =  self.learner

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=lr, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=lr, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=lr, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=lr)
        return optimizer
    
    def _build_optimizer_from_groups(self, param_groups, weight_decay):
        """Build optimizer from parameter groups with different learning rates."""
        learner = self.learner
        
        # Add weight_decay to all groups if not specified
        for group in param_groups:
            if 'weight_decay' not in group:
                group['weight_decay'] = weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(param_groups)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(param_groups)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(param_groups)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(param_groups)
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(param_groups)
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(param_groups)
        return optimizer

    @staticmethod
    def _gather_tensor(t, local_rank):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[local_rank] = t
        return all_tensors

    @staticmethod
    def gather_tensors(t, local_rank=None):
        if local_rank is None:
            local_rank = dist.get_rank()
        return torch.cat(Trainer._gather_tensor(t, local_rank))

    @staticmethod
    def compute_discrete_contrastive_loss_kl(x_logits, y_logits):
        # kl loss
        code_num = x_logits.size(-1)
        x_logits = F.log_softmax(x_logits.reshape(-1, code_num), dim=-1)
        y_logits = F.log_softmax(y_logits.reshape(-1, code_num), dim=-1)
        loss = F.kl_div(x_logits, y_logits, reduction='batchmean', log_target=True)
        return loss
                                          
    @staticmethod
    def compute_contrastive_loss(query_embeds, semantic_embeds, temperature=0.07, sim="cos", gathered=True):
        if gathered:
            gathered_query_embeds = Trainer.gather_tensors(query_embeds)
            gathered_semantic_embeds = Trainer.gather_tensors(semantic_embeds)
        else:
            gathered_query_embeds = query_embeds
            gathered_semantic_embeds = semantic_embeds

        if sim=="cos":
            gathered_query_embeds = F.normalize(gathered_query_embeds, dim=-1)
            gathered_semantic_embeds = F.normalize(gathered_semantic_embeds, dim=-1)

        effective_bsz = gathered_query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(gathered_query_embeds, gathered_semantic_embeds.transpose(0, 1)) / temperature

        co_loss = F.cross_entropy(similarities, labels)
        return co_loss
    
    @staticmethod
    def get_unique_index(inputs):
        unique_value = torch.unique(inputs).to(inputs.device)
        unique_index = torch.zeros_like(unique_value, device=inputs.device)
        for i, value in enumerate(unique_value):
            unique_index[i] = torch.argwhere(inputs == value).flatten()[0]
        unique_index = unique_index.to(inputs.device)
        return unique_index
        
    def get_train_steps(self, epochs=None):
        len_dataloader = len(self.train_data)
        num_update_steps_per_epoch = len_dataloader // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if epochs is None:
            epochs = self.epochs
        max_steps = math.ceil(epochs * num_update_steps_per_epoch)

        return max_steps

    def _train_epoch_rec(self, epoch_idx, loss_w, freeze_id=False, verbose=True):

        self.model_rec.train()
        # Enable training for model_id (RQ-VAE) for joint optimization training
        self.model_id.train()
        
        # Reset adaptive selection statistics at the start of each epoch
        if dist.is_initialized():
            self.model_id.module.reset_adaptive_selection_stats()
        else:
            self.model_id.reset_adaptive_selection_stats()

        total_num = 0
        total_loss = defaultdict(int)
        iter_data = tqdm(
                    self.train_data,
                    total=len(self.train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    disable=(not verbose) or (not self.accelerator.is_main_process),
                    )

        for batch_idx, batch in enumerate(iter_data):
            with self.accelerator.accumulate(self.model_rec):

                total_num += 1
                
                self.rec_optimizer.zero_grad()
                if hasattr(self, 'id_optimizer'):
                    self.id_optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)

                B = input_ids.size(0)
                input_ids = self.all_item_code[input_ids].clone().detach().reshape(B, -1)
                labels = self.all_item_code[targets].clone().detach().reshape(B, -1)
                attention_mask = (input_ids != -1).bool() 
                
                target_flatten = targets.flatten()
                if dist.is_initialized():
                    target_semantic_embs = self.model_rec.module.semantic_embedding(target_flatten)
                else:
                    target_semantic_embs = self.model_rec.semantic_embedding(target_flatten)
                
                # ============================================================
                # NEW: Latent-space reconstruction (no decoder)
                # ============================================================
                # 1. Encode to latent space z
                if dist.is_initialized():
                    z = self.model_id.module.encoder(target_semantic_embs)  # [B, e_dim]
                else:
                    z = self.model_id.encoder(target_semantic_embs)          # [B, e_dim]
                
                # 2. Quantize in latent space (with Gumbel-Softmax for gradient flow)
                use_gumbel = self.config.get('use_gumbel', not freeze_id)
                
                # Call RQVAE forward (encoder is already called, so we call rq directly)
                # But we need to manually compute parameters for rq()
                if dist.is_initialized():
                    stop_gumbel_epoch = self.model_id.module.stop_gumbel_sampling_epoch
                    use_indicator_ste = self.model_id.module.use_indicator_ste
                    # Get current tau (with annealing if enabled)
                    current_tau = self.model_id.module.get_current_tau(self.global_step)
                else:
                    stop_gumbel_epoch = self.model_id.stop_gumbel_sampling_epoch
                    use_indicator_ste = self.model_id.use_indicator_ste
                    # Get current tau (with annealing if enabled)
                    current_tau = self.model_id.get_current_tau(self.global_step)

                use_gumbel_sampling = (stop_gumbel_epoch == 0) or (epoch_idx < stop_gumbel_epoch)

                if dist.is_initialized():
                    z_hat, vq_loss, target_indices_sampled, target_indices_argmax, _, target_code_logits, balance_loss, gate_reg_loss, sigma = \
                        self.model_id.module.rq(z, use_gumbel=use_gumbel, tau=current_tau,
                                               use_indicator_ste=use_indicator_ste,
                                               use_gumbel_sampling=use_gumbel_sampling,
                                               current_epoch=epoch_idx)
                else:
                    z_hat, vq_loss, target_indices_sampled, target_indices_argmax, _, target_code_logits, balance_loss, gate_reg_loss, sigma = \
                        self.model_id.rq(z, use_gumbel=use_gumbel, tau=current_tau,
                                        use_indicator_ste=use_indicator_ste,
                                        use_gumbel_sampling=use_gumbel_sampling,
                                        current_epoch=epoch_idx)
                
                # 3. Latent space reconstruction loss: ||z_hat - z||^2
                glq_recon_loss = F.mse_loss(z_hat, z)
                
                # 4. QSLoss: Quantization-Semantic Loss (NEW IMPLEMENTATION)

                # Note: get_indices returns [B, num_rq_layers] (e.g., [B, 3])
                # But code_length=4 includes conflict resolution position
                if dist.is_initialized():
                    token_indices = self.model_id.module.get_indices(target_semantic_embs)  # [B, num_rq_layers]
                else:
                    token_indices = self.model_id.get_indices(target_semantic_embs)  # [B, num_rq_layers]

                # Only use the RQ layer indices (first num_rq_layers positions)
                # token_indices shape: [B, num_rq_layers] where num_rq_layers=3
                num_rq_layers = token_indices.shape[1]  # 3
                token_embs_list = []
                for i in range(num_rq_layers):
                    if dist.is_initialized():
                        emb = self.model_rec.module.token_embeddings[i](token_indices[:, i])  # [B, hidden_size]
                    else:
                        emb = self.model_rec.token_embeddings[i](token_indices[:, i])  # [B, hidden_size]
                    token_embs_list.append(emb)

                # Average pooling over RQ positions to get single representation
                token_embs = torch.stack(token_embs_list, dim=1).mean(dim=1)  # [B, hidden_size]

                # Step 4.3: Project tokenizer's latent z to token embedding space
                if dist.is_initialized():
                    z_projected = self.model_rec.module.qs_projector(z)  # [B, e_dim] -> [B, hidden_size]
                else:
                    z_projected = self.model_rec.qs_projector(z)  # [B, e_dim] -> [B, hidden_size]

                # Step 4.4: Bidirectional alignment loss
                qs_beta = self.config.get('qs_beta', 0.25)
                qs_loss = F.mse_loss(z_projected, token_embs.detach()) + \
                          qs_beta * F.mse_loss(z_projected.detach(), token_embs)
                
                # 5. Final GLQ loss = latent_recon + vq_loss
                recon_loss = glq_recon_loss
                
                # Forward pass for recommender model
                outputs = self.model_rec(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
          
                logits = outputs.logits  # (batch, code_len, code_num)
                
                # Code prediction loss
                code_loss = F.cross_entropy(logits.reshape(-1, self.code_num), labels.detach().reshape(-1))

                # NEW: Apply Uncertainty-Weighted Loss if learnable sigma is enabled
                if sigma is not None:
                    # Check for Cosine Annealing Strategy (Epoch-based)
                    use_cosine_annealing = self.config.get('use_cosine_annealing', False)
                    
                    if use_cosine_annealing:
                        # Cosine Annealing: std decays from initial_std to near-0 based on epoch
                        # Formula: std_t = 0.5 * initial_std * (1 + cos(pi * t / T))
                        # where t = current_epoch, T = max_epochs
                        
                        # CRITICAL: Ensure sigma is NOT learnable
                        # Note: sigma here might be a computed tensor from forward pass, so we can't set requires_grad
                        # Instead, we rely on .data.fill_() which modifies the underlying storage
                        # To be extra safe, we can detach it if needed, but .data write is sufficient
                        
                        initial_std = float(self.config.get('initial_std', 1.0))
                        current_epoch = self.global_step / max(1, self.max_steps) * self.epochs
                        T_max = self.epochs
                        
                        # Calculate target std using cosine schedule
                        import math
                        cosine_factor = 0.5 * (1 + math.cos(math.pi * current_epoch / T_max))
                        target_std = initial_std * cosine_factor
                        
                        # Clamp to ensure it doesn't go below a tiny value (e.g., 1e-6)
                        target_std = max(1e-6, target_std)
                        
                        # Convert std to sigma: sigma = log2(std)
                        # This is necessary because the model uses s = 2^sigma internally
                        target_sigma = math.log2(target_std)
                        
                        # Force update the sigma parameter in the model
                        sigma.data.fill_(target_sigma)
                        
                        # Use plain code loss (no uncertainty weighting) as requested
                        if self.global_step % 10 == 0 and self.accelerator.is_main_process:
                            self.log(f"[Cosine Annealing] Epoch={current_epoch:.2f}/{T_max}, sigma={target_sigma:.4f}, std={target_std:.4f} (Fixed)")
                    else:
                        # Existing logic for learnable sigma or plain loss
                        use_plain_code_loss = self.config.get('use_plain_code_loss', False)
                        use_simple_uncertainty_loss = self.config.get('use_simple_uncertainty_loss', False)
                        
                        if use_plain_code_loss:
                            # Use plain code_loss without uncertainty weighting (for ablation)
                            # Still log sigma for monitoring
                            if self.global_step % 10 == 0 and self.accelerator.is_main_process:
                                sigma_val = sigma.item()
                                if use_simple_uncertainty_loss:
                                    self.log(f"[Plain Loss] sigma={sigma_val:.4f} (direct), code_loss={code_loss.item():.4f}")
                                else:
                                    self.log(f"[Plain Loss] sigma={sigma_val:.4f}, std≈{2**sigma_val:.4f}, code_loss={code_loss.item():.4f}")
                        elif use_simple_uncertainty_loss:
                            # NEW: Standard Deviation Uncertainty Decay (SDUD) with optional learnable/adaptive lambda
                            original_code_loss = code_loss.item()
                            sigma_lambda = self.config.get('sigma_lambda', 0.5)
                            
                            # Get the AutoSigmaSimple module instance (assumes all RQ layers use the same module)
                            auto_sigma_module = None
                            if dist.is_initialized():
                                auto_sigma_module = getattr(self.model_id.module.rq.vq_layers[0], 'auto_sigma_module', None)
                            else:
                                auto_sigma_module = getattr(self.model_id.rq.vq_layers[0], 'auto_sigma_module', None)
                            
                            if auto_sigma_module is not None and hasattr(auto_sigma_module, 'compute_uncertainty_loss'):
                                # Use instance method (supports learnable/adaptive lambda)
                                code_loss, actual_lambda = auto_sigma_module.compute_uncertainty_loss(
                                    code_loss, sigma, lambda_bias=sigma_lambda
                                )
                            else:
                                # Fallback to static method (backward compatibility)
                                from vq import AutoSigmaSimple
                                code_loss = AutoSigmaSimple.compute_uncertainty_loss(code_loss, sigma, lambda_bias=sigma_lambda)
                                actual_lambda = sigma_lambda
                            
                            if self.global_step % 10 == 0 and self.accelerator.is_main_process:
                                sigma_val = sigma.item()
                                lambda_val = actual_lambda if isinstance(actual_lambda, float) else actual_lambda.item()
                                import math
                                # Equilibrium: sigma = sqrt(L) - lambda
                                target_sigma = math.sqrt(max(0, original_code_loss)) - lambda_val
                                
                                # Check auto_lambda_mode
                                auto_lambda_mode = self.config.get('auto_lambda_mode', 'fixed')
                                if auto_lambda_mode == 'learnable':
                                    self.log(f"[SDUD] sigma={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (learnable), Target_sigma={target_sigma:.4f}")
                                elif auto_lambda_mode == 'adaptive':
                                    if auto_sigma_module is not None:
                                        loss_ema = auto_sigma_module.loss_ema.item()
                                        self.log(f"[SDUD] sigma={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (adaptive, EMA={loss_ema:.4f}), Target_sigma={target_sigma:.4f}")
                                    else:
                                        self.log(f"[SDUD] sigma={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (adaptive), Target_sigma={target_sigma:.4f}")
                                else:
                                    self.log(f"[SDUD] sigma={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (fixed), Target_sigma={target_sigma:.4f}")
                        else:
                            # Use uncertainty-weighted loss (default)
                            original_code_loss = code_loss.item()  # Save for logging
                            sigma_reg_weight = self.config.get('sigma_reg_weight', 1.0)
                            # NEW: Configurable annealing parameters for different datasets
                            annealing_threshold = self.config.get('annealing_threshold', None)
                            annealing_slow_k = self.config.get('annealing_slow_k', None)
                            annealing_slow_c = self.config.get('annealing_slow_c', None)
                            annealing_fast_k = self.config.get('annealing_fast_k', None)
                            annealing_fast_c = self.config.get('annealing_fast_c', None)
                            code_loss = AutoSigmaGumbel.compute_uncertainty_loss(
                                code_loss, sigma, reg_weight=sigma_reg_weight,
                                annealing_threshold=annealing_threshold,
                                annealing_slow_k=annealing_slow_k,
                                annealing_slow_c=annealing_slow_c,
                                annealing_fast_k=annealing_fast_k,
                                annealing_fast_c=annealing_fast_c
                            )
                            # Log sigma info for debugging
                            if self.global_step % 10 == 0 and self.accelerator.is_main_process:
                                sigma_val = sigma.item()  # Can be negative now
                                # Exponential annealing equilibrium: sigma = -0.589 + 1.298*ln(L_task)
                                import math
                                equilibrium_sigma = -0.589 + 1.298 * math.log(max(0.1, original_code_loss))
                                self.log(f"[Annealing] sigma={sigma_val:.4f}, Loss={original_code_loss:.4f}, Target_sigma={equilibrium_sigma:.4f}")

                # ============================================================
                # ============================================================

                losses = dict(
                    code_loss=code_loss,
                    recon_loss=recon_loss,
                    vq_loss=vq_loss,            # VQ commitment loss
                    qs_loss=qs_loss,
                )

                # Add balance loss if available (only when using Gumbel-Softmax)
                if balance_loss is not None:
                    losses['balance_loss'] = balance_loss
                
                # Add gate regularization loss if available (only when using gate network)
                if gate_reg_loss is not None:
                    losses['gate_loss'] = gate_reg_loss

                loss = sum([v * loss_w.get(k, 0) for k, v in losses.items()])

                self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1)
                if hasattr(self, 'id_optimizer') and not freeze_id:
                    self.accelerator.clip_grad_norm_(self.model_id.parameters(), 1)

                self.rec_optimizer.step()
                self.rec_lr_scheduler.step()
                # CRITICAL: Only update ID optimizer when NOT frozen
                if hasattr(self, 'id_optimizer') and not freeze_id:
                    # NEW: Dynamic sigma learning rate adjustment
                    # When in fast annealing stage (loss < 2.0), increase sigma lr by 10x
                    # Only enabled when use_dynamic_sigma_lr is explicitly set to True
                    use_dynamic_sigma_lr = self.config.get('use_dynamic_sigma_lr', False)
                    if use_dynamic_sigma_lr and hasattr(self, 'lr_sigma') and self.lr_sigma is not None:
                        threshold_loss = 2.0  # Same as in vq.py
                        current_code_loss = code_loss.item() if isinstance(code_loss, torch.Tensor) else code_loss
                        
                        # Check if we're in the fast annealing stage
                        if current_code_loss < threshold_loss:
                            # Fast stage: increase sigma lr by 10x
                            sigma_lr_multiplier = 10.0
                        else:
                            # Slow stage: use base lr
                            sigma_lr_multiplier = 1.0
                        
                        # Update learning rate for sigma parameter group
                        for param_group in self.id_optimizer.param_groups:
                            # Check if this is the sigma parameter group
                            # (it should have the lr_sigma as base lr)
                            if abs(param_group['lr'] - self.lr_sigma) < 1e-8 or \
                               abs(param_group['lr'] - self.lr_sigma * 10.0) < 1e-8:
                                param_group['lr'] = self.lr_sigma * sigma_lr_multiplier
                        
                        # Log lr adjustment (every 50 steps to avoid spam)
                        if self.global_step % 50 == 0 and self.accelerator.is_main_process:
                            stage = "FAST" if current_code_loss < threshold_loss else "SLOW"
                            actual_sigma_lr = self.lr_sigma * sigma_lr_multiplier
                            self.log(f"[Sigma LR] Stage={stage}, Loss={current_code_loss:.4f}, sigma_lr={actual_sigma_lr:.6f} ({sigma_lr_multiplier:.1f}x)")
                    
                    self.id_optimizer.step()
                    self.id_lr_scheduler.step()

                # Increment global step for tau annealing
                self.global_step += 1

                code_loss_mean = self.accelerator.gather(code_loss).mean().item()
                recon_loss_mean = self.accelerator.gather(recon_loss).mean().item()
                vq_loss_mean = self.accelerator.gather(vq_loss).mean().item()
                qs_loss_mean = self.accelerator.gather(qs_loss).mean().item()

                loss_mean = self.accelerator.gather(loss).mean().item()
                loss = dict(
                    loss=loss_mean,
                    code_loss=code_loss_mean,
                    recon_loss=recon_loss_mean,
                    vq_loss=vq_loss_mean,
                    qs_loss=qs_loss_mean,
                )

                # Add balance loss to statistics if available
                if balance_loss is not None:
                    balance_loss_mean = self.accelerator.gather(balance_loss).mean().item()
                    loss['balance_loss'] = balance_loss_mean
                
                # Add gate loss to statistics if available
                if gate_reg_loss is not None:
                    gate_loss_mean = self.accelerator.gather(gate_reg_loss).mean().item()
                    loss['gate_loss'] = gate_loss_mean

                # Add sigma to statistics if available
                if sigma is not None:
                    sigma_mean = self.accelerator.gather(sigma).mean().item()
                    loss['sigma'] = sigma_mean

                for k,v in loss.items():
                    total_loss[k] += v
                iter_data.set_postfix(loss=loss_mean)

        for k in total_loss.keys():
            total_loss[k] = round(total_loss[k]/total_num, 4)
                
        self.accelerator.wait_for_everyone()
        
        return total_loss
    


    def safe_save(self, epoch, code, prefix=''):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrap_model_rec = self.accelerator.unwrap_model(self.model_rec)
            unwrap_model_id = self.accelerator.unwrap_model(self.model_id)
            
            # Add prefix to filename if provided
            filename = f'{prefix}_{epoch}' if prefix else str(epoch)
            
            self.accelerator.save(unwrap_model_rec.state_dict(), f'{self.save_path}/{filename}.pt')
            self.accelerator.save(unwrap_model_id.state_dict(), f'{self.save_path}/{filename}.pt.rqvae')
            json.dump(code.cpu().tolist(), open(f'{self.save_path}/{filename}.code.json', 'w'))
            self.log(f'[Epoch {epoch}] Save model {self.save_path}/{filename}.pt')

        filename = f'{prefix}_{epoch}' if prefix else str(epoch)
        last_checkpoint = f'{self.save_path}/{filename}.pt'
        return last_checkpoint

    def evaluate(self, outputs, labels):
        batch_size, k, _ = outputs.shape  # Assuming outputs is [batch_size, 10, seq_len]
        recall_at_1, recall_at_5, recall_at_10 = [], [], []
        ndcg_at_1, ndcg_at_5, ndcg_at_10 = [], [], []

        for i in range(batch_size):
            label = labels[i].unsqueeze(0)  # [1, seq_len]
            out = outputs[i]
                
            matches = torch.all(torch.eq(out.unsqueeze(1), label.unsqueeze(0)), dim=2)  # [10, 1, seq_len] -> [10, 1]
            matches = matches.any(dim=1).cpu().numpy()  # [10]

            # Recall
            recall_at_1.append(matches[:1].sum() / 1.0)
            recall_at_5.append(matches[:5].sum() / 1.0)  # Assuming each label has only 1 correct match.
            recall_at_10.append(matches.sum() / 1.0)

            # NDCG (binary relevance)
            ndcg_at_1.append(ndcg_at_k(matches, 1))
            ndcg_at_5.append(ndcg_at_k(matches, 5))
            ndcg_at_10.append(ndcg_at_k(matches, 10))

        # Calculate mean metrics
        metrics = {
            "recall@1": np.sum(recall_at_1),
            "recall@5": np.sum(recall_at_5),
            "recall@10": np.sum(recall_at_10),
            "ndcg@1": np.sum(ndcg_at_1),
            "ndcg@5": np.sum(ndcg_at_5),
            "ndcg@10": np.sum(ndcg_at_10),
        }

        return metrics

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss_dict):
        train_loss_output = (
            "[Epoch %d] [time: %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        if isinstance(loss_dict, dict):
            train_loss_output += "train loss" + str(list(loss_dict.items()))
        else:
            train_loss_output += "train loss" + ": %.4f" % loss_dict
        return train_loss_output + "]"

    def train(self, verbose=True):
        """
        Joint optimization training: Train both recommender and RQ-VAE together.
        Uses Gumbel-Softmax for differentiable code selection.
        """
        stop = False
        cur_eval_step = 0
        self.best_score = 0
        self.best_result = {}
        self.best_ckpt = None
        loss_w = defaultdict(int)

        # Initialize item codes from RQ-VAE
        all_item_code = self.get_code(epoch_idx=-1, verbose=verbose)
        self.all_item_code = torch.tensor(all_item_code).to(self.device)

        # Enable joint optimization training mode
        # Backward compatibility: also check for old 'end_to_end' key
        joint_optimization = self.config.get('joint_optimization', self.config.get('end_to_end', False))
        
        if joint_optimization:
            # Unfreeze all recommender model parameters first
            for param in self.model_rec.parameters():
                param.requires_grad = True
            
            # Optionally freeze semantic_embedding
            freeze_semantic_embedding = bool(self.config.get('freeze_semantic_embedding', True))
            if freeze_semantic_embedding:
                semantic_emb_name = 'module.semantic_embedding' if dist.is_initialized() else 'semantic_embedding'
                for name, param in self.model_rec.named_parameters():
                    if name.startswith(semantic_emb_name):
                        param.requires_grad = False
            
            self.log(f'[Training Mode] Recommender model unfrozen (semantic_embedding frozen)')
            
            # Unfreeze RQ-VAE for joint optimization training
            for param in self.model_id.parameters():
                param.requires_grad = True
            
            # Freeze encoder/RQ if specified
            freeze_id_encoder = self.config.get('freeze_id_encoder', False)
            freeze_id_encoder_layers = self.config.get('freeze_id_encoder_layers', 0)  # Number of layers to freeze from bottom
            freeze_rq = self.config.get('freeze_rq', False)
            
            if freeze_id_encoder:
                # If freeze_id_encoder_layers is set, freeze only the bottom N layers
                # Otherwise, freeze all encoder layers (backward compatibility)
                if freeze_id_encoder_layers > 0:
                    # Freeze bottom N layers (closest to RQ-VAE input)
                    # MLPLayers structure: Dropout -> Linear -> (BN) -> (Activation) -> ...
                    # We need to find Linear layers and freeze the first N layers
                    encoder_modules = list(self.model_id.encoder.mlp_layers.children())
                    linear_layer_idx = 0
                    frozen_count = 0
                    
                    for module in encoder_modules:
                        if isinstance(module, nn.Linear):
                            if linear_layer_idx < freeze_id_encoder_layers:
                                for param in module.parameters():
                                    param.requires_grad = False
                                frozen_count += 1
                            linear_layer_idx += 1
                    
                    total_linear_layers = sum(1 for m in encoder_modules if isinstance(m, nn.Linear))
                    self.log(f'[Training Mode] ID tokenizer encoder: {frozen_count}/{total_linear_layers} layers FROZEN (bottom {freeze_id_encoder_layers} layers)')
                else:
                    # Freeze all encoder layers (original behavior)
                    for param in self.model_id.encoder.parameters():
                        param.requires_grad = False
                    self.log(f'[Training Mode] ID tokenizer encoder FROZEN (all layers)')
            if freeze_rq:
                for param in self.model_id.rq.parameters():
                    param.requires_grad = False
                self.log(f'[Training Mode] ID tokenizer RQ quantizer FROZEN')
            
            # Build optimizer for RQ-VAE
            self.lr_id = self.config.get('lr_id', self.lr_rec * 0.1)  # Use smaller lr for RQ-VAE
            
            # NEW: Separate learning rate for sigma and lambda parameters
            self.lr_sigma = self.config.get('lr_sigma', None)
            self.lr_lambda = self.config.get('lr_lambda', None)  # NEW: separate lr for lambda
            use_separate_sigma_lr = self.lr_sigma is not None and self.config.get('use_learnable_sigma_gumbel', False)
            use_learnable_lambda = self.config.get('auto_lambda_mode', 'fixed') == 'learnable'
            
            if use_separate_sigma_lr or (use_learnable_lambda and self.lr_lambda is not None):
                # Separate sigma/lambda parameters from other RQ-VAE parameters
                sigma_params = []
                lambda_params = []
                other_params = []
                
                for name, param in self.model_id.named_parameters():
                    if param.requires_grad:
                        if 'lambda_param' in name.lower():
                            lambda_params.append(param)
                        elif 'sigma' in name.lower():
                            sigma_params.append(param)
                        else:
                            other_params.append(param)
                
                # Create optimizer with parameter groups
                param_groups = []
                if len(other_params) > 0:
                    param_groups.append({'params': other_params, 'lr': self.lr_id})
                if len(sigma_params) > 0 and use_separate_sigma_lr:
                    param_groups.append({'params': sigma_params, 'lr': self.lr_sigma})
                if len(lambda_params) > 0 and self.lr_lambda is not None:
                    param_groups.append({'params': lambda_params, 'lr': self.lr_lambda})
                elif len(lambda_params) > 0:
                    # If no separate lr_lambda specified, use lr_sigma or lr_id
                    lambda_lr = self.lr_sigma if self.lr_sigma is not None else self.lr_id
                    param_groups.append({'params': lambda_params, 'lr': lambda_lr})
                
                self.id_optimizer = self._build_optimizer_from_groups(param_groups, self.weight_decay)
                
                self.log(f'[Training Mode] Using SEPARATE learning rates:')
                self.log(f'  - RQ-VAE parameters: lr={self.lr_id}')
                if len(sigma_params) > 0:
                    self.log(f'  - Sigma parameters: lr={self.lr_sigma} ({len(sigma_params)} params)')
                if len(lambda_params) > 0:
                    lambda_lr = self.lr_lambda if self.lr_lambda is not None else (self.lr_sigma if self.lr_sigma is not None else self.lr_id)
                    self.log(f'  - Lambda parameters: lr={lambda_lr} ({len(lambda_params)} params)')
            else:
                # Standard: all RQ-VAE parameters use the same learning rate
                self.id_optimizer = self._build_optimizer(self.model_id, self.lr_id, self.weight_decay)
            
            # Build scheduler for RQ-VAE
            if self.lr_scheduler_type == "linear":
                self.id_lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.id_optimizer,
                                                                        num_warmup_steps=self.warmup_steps,
                                                                        num_training_steps=self.max_steps)
            elif self.lr_scheduler_type == "constant":
                self.id_lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.id_optimizer,
                                                                          num_warmup_steps=self.warmup_steps)
            elif self.lr_scheduler_type == "cosine":
                self.id_lr_scheduler = get_scheduler(
                                name="cosine",
                                optimizer=self.id_optimizer,
                                num_warmup_steps=self.warmup_steps,
                                num_training_steps=self.max_steps,
                            )
            
            # Prepare RQ-VAE optimizer and scheduler with accelerator
            self.id_optimizer, self.id_lr_scheduler = \
                self.accelerator.prepare(self.id_optimizer, self.id_lr_scheduler)
            
            # Set loss weights for joint optimization training
            loss_w['code_loss'] = self.config.get('code_loss_weight', 1.0)
            loss_w['recon_loss'] = self.config.get('recon_loss_weight', 1.0)
            loss_w['vq_loss'] = self.config.get('vq_loss_weight', 0.25)
            loss_w['qs_loss'] = self.config.get('qs_loss_weight', 0.1)
            loss_w['balance_loss'] = self.config.get('balance_loss_weight', 0.1)
            loss_w['gate_loss'] = self.config.get('gate_loss_weight', 0.1)
            loss_w['kl_loss'] = self.config.get('kl_loss_weight', 0.0)
            loss_w['dec_cl_loss'] = self.config.get('dec_cl_loss_weight', 0.0)

            self.log(f'[Training Mode] Joint optimization training enabled')
            self.log(f'[Training Mode] Loss weights: {dict(loss_w)}')
        else:
            # Unfreeze all recommender model parameters first
            for param in self.model_rec.parameters():
                param.requires_grad = True
            
            # Optionally freeze semantic_embedding
            freeze_semantic_embedding = bool(self.config.get('freeze_semantic_embedding', True))
            if freeze_semantic_embedding:
                semantic_emb_name = 'module.semantic_embedding' if dist.is_initialized() else 'semantic_embedding'
                for name, param in self.model_rec.named_parameters():
                    if name.startswith(semantic_emb_name):
                        param.requires_grad = False
            
            self.log(f'[Training Mode] Recommender model unfrozen (semantic_embedding frozen)')

            # Freeze RQ-VAE completely
            for param in self.model_id.parameters():
                param.requires_grad = False

            # Set loss weights: only code_loss
            loss_w['code_loss'] = 1
            loss_w['vq_loss'] = 0
            loss_w['qs_loss'] = 0
            loss_w['balance_loss'] = 0
            loss_w['gate_loss'] = 0
            loss_w['kl_loss'] = 0
            loss_w['dec_cl_loss'] = 0
            loss_w['recon_loss'] = 0

            self.log(f'[Training Mode] Frozen RQ-VAE mode (original)')

        # Print parameter statistics
        self.log("")
        self._count_parameters(self.model_rec, "Recommender Model")
        self._count_parameters(self.model_id, "ID Tokenizer (RQ-VAE)")
        
        # Print detailed breakdown for ID tokenizer
        self.log("ID Tokenizer Module Breakdown:")
        self._count_module_parameters(self.model_id, "encoder")
        self._count_module_parameters(self.model_id, "rq")
        
        # Check for learnable sigma parameters
        if dist.is_initialized():
            model_id_unwrapped = self.model_id.module
        else:
            model_id_unwrapped = self.model_id
        
        sigma_params = [name for name, p in model_id_unwrapped.named_parameters() if 'sigma' in name.lower()]
        if sigma_params:
            self.log("")
            self.log(f"========== Learnable Sigma Parameters (Base-2 Exponential) ==========")
            for name in sigma_params:
                param = dict(model_id_unwrapped.named_parameters())[name]
                sigma_val = param.data.item()
                s_val = 2 ** sigma_val
                self.log(f"  {name}: sigma={sigma_val:.6f}, requires_grad={param.requires_grad}")
                self.log(f"    -> Noise scale: s = 2^sigma = 2^{sigma_val:.3f} = {s_val:.6f}")
            self.log(f"=" * 50)
        
        # Calculate total parameters
        total_rec_params = sum(p.numel() for p in self.model_rec.parameters())
        total_id_params = sum(p.numel() for p in self.model_id.parameters())
        trainable_rec_params = sum(p.numel() for p in self.model_rec.parameters() if p.requires_grad)
        trainable_id_params = sum(p.numel() for p in self.model_id.parameters() if p.requires_grad)
        
        self.log("")
        self.log(f"========== Overall Statistics ==========")
        self.log(f"Total parameters (all models): {total_rec_params + total_id_params:,}")
        self.log(f"Trainable parameters (all models): {trainable_rec_params + trainable_id_params:,}")
        self.log(f"Frozen parameters (all models): {(total_rec_params + total_id_params) - (trainable_rec_params + trainable_id_params):,}")
        self.log(f"=" * 50)
        self.log("")

        for epoch_idx in range(self.epochs):
            self.accelerator.wait_for_everyone()

            # Staged unfreezing: freeze RQ-VAE for first N epochs
            freeze_id_epochs = self.config.get('freeze_id_epochs', 0)
            if joint_optimization and epoch_idx < freeze_id_epochs:
                # Freeze RQ-VAE
                for param in self.model_id.parameters():
                    param.requires_grad = False
                if epoch_idx == 0:
                    self.log(f'[Training Mode] RQ-VAE FROZEN for first {freeze_id_epochs} epochs')
                    self.log(f'[Training Mode] Will unfreeze at epoch {freeze_id_epochs}')
            elif joint_optimization and epoch_idx == freeze_id_epochs:
                # Unfreeze RQ-VAE at specified epoch
                for param in self.model_id.parameters():
                    param.requires_grad = True

                # Re-apply encoder/RQ freeze if specified
                freeze_id_encoder = self.config.get('freeze_id_encoder', False)
                freeze_id_encoder_layers = self.config.get('freeze_id_encoder_layers', 0)
                freeze_rq = self.config.get('freeze_rq', False)

                if freeze_id_encoder:
                    # If freeze_id_encoder_layers is set, freeze only the bottom N layers
                    if freeze_id_encoder_layers > 0:
                        encoder_modules = list(self.model_id.encoder.mlp_layers.children())
                        linear_layer_idx = 0
                        
                        for module in encoder_modules:
                            if isinstance(module, nn.Linear):
                                if linear_layer_idx < freeze_id_encoder_layers:
                                    for param in module.parameters():
                                        param.requires_grad = False
                                linear_layer_idx += 1
                    else:
                        # Freeze all encoder layers
                        for param in self.model_id.encoder.parameters():
                            param.requires_grad = False
                if freeze_rq:
                    for param in self.model_id.rq.parameters():
                        param.requires_grad = False

                self.log(f'[Training Mode] RQ-VAE UNFROZEN at epoch {epoch_idx}!')
                if freeze_id_encoder:
                    if freeze_id_encoder_layers > 0:
                        encoder_modules = list(self.model_id.encoder.mlp_layers.children())
                        total_linear_layers = sum(1 for m in encoder_modules if isinstance(m, nn.Linear))
                        self.log(f'[Training Mode] (encoder: {freeze_id_encoder_layers}/{total_linear_layers} bottom layers still frozen)')
                    else:
                        self.log(f'[Training Mode] (encoder still frozen)')
                if freeze_rq:
                    self.log(f'[Training Mode] (RQ quantizer still frozen)')

            # Adjust loss weights based on freeze status
            is_id_frozen = joint_optimization and epoch_idx < freeze_id_epochs
            if is_id_frozen:
                # During freeze: only train recommender, disable RQ-VAE losses
                current_loss_w = {
                    'code_loss': loss_w['code_loss'],
                    'recon_loss': 0.0,  # Disable
                    'vq_loss': 0.0,     # Disable
                    'qs_loss': 0.0,     # Disable
                    'balance_loss': 0.0,  # Disable
                    'gate_loss': 0.0,  # Disable
                }
            else:
                # After unfreeze: use full loss weights
                current_loss_w = loss_w

            # Train
            training_start_time = time()
            train_loss = self._train_epoch_rec(epoch_idx, loss_w=current_loss_w, freeze_id=is_id_frozen, verbose=verbose)
            training_end_time = time()
            
            # Print adaptive selection statistics at the end of each epoch
            if self.config.get('use_adaptive_selection', False) and not is_id_frozen:
                if dist.is_initialized():
                    stats = self.model_id.module.get_adaptive_selection_stats()
                else:
                    stats = self.model_id.get_adaptive_selection_stats()
                
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )

            self.log(train_loss_output)
            self.log(f'[Epoch {epoch_idx}] REC lr: {self.rec_lr_scheduler.get_lr()}')
            if hasattr(self, 'id_lr_scheduler'):
                self.log(f'[Epoch {epoch_idx}] ID lr: {self.id_lr_scheduler.get_lr()}')

            # Regenerate codes every epoch for joint optimization training
            if joint_optimization:
                all_item_code = self.get_code(epoch_idx=epoch_idx, verbose=verbose)
                self.all_item_code = torch.tensor(all_item_code).to(self.device)

            # Evaluate every epoch (as requested)
            metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)
            total_metrics = metrics

            if total_metrics[self.valid_metric] > self.best_score:
                self.best_score = total_metrics[self.valid_metric]
                self.best_result = total_metrics
                cur_eval_step = 0
                self.best_ckpt = self.safe_save(epoch_idx, self.all_item_code)
            else:
                cur_eval_step += 1

            if cur_eval_step >= self.early_stop:
                stop = True

            self.log(f'[Epoch {epoch_idx}] Val Results: {total_metrics}')

            self.accelerator.wait_for_everyone()

            if stop:
                break

        # ============================================
        # Load Stage 1 checkpoint if provided (before testing)
        # ============================================
        stage2_epochs = self.config.get('stage2_epochs', 0)
        stage1_ckpt_path = self.config.get('stage1_checkpoint', None)
        
        # If user provided a Stage 1 checkpoint, load it first
        if stage1_ckpt_path is not None:
            self.log("")
            self.log("="*60)
            self.log("Loading provided Stage 1 checkpoint")
            self.log("="*60)
            self.log(f"Checkpoint: {stage1_ckpt_path}")
            self.log("")
            
            # Load checkpoint
            if dist.is_initialized():
                safe_load(self.model_rec.module, stage1_ckpt_path, verbose=verbose)
                safe_load(self.model_id.module, stage1_ckpt_path+'.rqvae', verbose=verbose)
            else:
                safe_load(self.model_rec, stage1_ckpt_path, verbose=verbose)
                safe_load(self.model_id, stage1_ckpt_path+'.rqvae', verbose=verbose)
            
            # Load codes
            best_code = json.load(open(stage1_ckpt_path[:-3]+'.code.json'))
            self.all_item_code = torch.tensor(best_code).to(self.device)
            
            # Evaluate the loaded checkpoint to establish baseline
            self.log("Evaluating loaded checkpoint on validation set...")
            initial_metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)
            self.best_score = initial_metrics[self.valid_metric]
            self.best_result = initial_metrics
            self.best_ckpt = stage1_ckpt_path
            self.log(f"Initial {self.valid_metric}: {self.best_score:.6f}")
            self.log(f"Initial Validation Results: {initial_metrics}")
            self.log("")
            
            self.stage1_test_results = None
        
        # Only run Stage 1 test if we actually ran Stage 1 training
        if stage1_ckpt_path is None:
            # ============================================
            # Stage 1: Test on best model
            # ============================================
            self.log("")
            self.log("="*60)
            self.log("Stage 1 Training Complete!")
            self.log(f"Best Stage 1 {self.valid_metric}: {self.best_score:.6f}")
            self.log(f"Best Stage 1 Validation Results: {self.best_result}")
            self.log("="*60)
            self.log("")
            
            # Test Stage 1 best model
            self.log("Testing Stage 1 best model on test set...")
            stage1_test_results = self.test(verbose=verbose, model_file=self.best_ckpt)
            self.log("")
            self.log("="*60)
            self.log(f"Stage 1 Test Results: {stage1_test_results}")
            self.log("="*60)
            self.log("")
            
            # Store stage 1 results for comparison later
            self.stage1_test_results = stage1_test_results

        if stage2_epochs > 0 and joint_optimization:
            self.log("")
            self.log("="*60)
            self.log("Starting Stage 2: Training Recommender with Frozen ID Tokenizer")
            self.log("="*60)
            self.log(f"Stage 2 epochs: {stage2_epochs}")
            
            # Checkpoint already loaded above if stage1_ckpt_path was provided
            if stage1_ckpt_path is not None:
                stage1_ckpt = stage1_ckpt_path
                self.log(f"Using already-loaded Stage 1 checkpoint: {stage1_ckpt}")
            else:
                # Load best checkpoint from Stage 1 training
                stage1_ckpt = self.best_ckpt
                self.log(f"Loading best checkpoint from Stage 1: {stage1_ckpt}")
                
                # Load checkpoint
                if dist.is_initialized():
                    safe_load(self.model_rec.module, stage1_ckpt, verbose=verbose)
                    safe_load(self.model_id.module, stage1_ckpt+'.rqvae', verbose=verbose)
                else:
                    safe_load(self.model_rec, stage1_ckpt, verbose=verbose)
                    safe_load(self.model_id, stage1_ckpt+'.rqvae', verbose=verbose)
                
                # Load codes
                best_code = json.load(open(stage1_ckpt[:-3]+'.code.json'))
                self.all_item_code = torch.tensor(best_code).to(self.device)
            
            # Freeze entire ID tokenizer
            for param in self.model_id.parameters():
                param.requires_grad = False
            self.log('[Stage 2] ID Tokenizer COMPLETELY FROZEN')
            
            # ====== CRITICAL: Force deterministic quantization in Stage 2 ======
            # Override stop_gumbel_sampling_epoch to force pure deterministic
            if dist.is_initialized():
                self.model_id.module.stop_gumbel_sampling_epoch = -1
                self.log('[Stage 2] Force deterministic: stop_gumbel_sampling_epoch = -1')
            else:
                self.model_id.stop_gumbel_sampling_epoch = -1
                self.log('[Stage 2] Force deterministic: stop_gumbel_sampling_epoch = -1')
            
            # Adjust loss weights: only use code_loss for recommender
            stage2_loss_w = {
                'code_loss': loss_w.get('code_loss', 1.0),
                'recon_loss': 0.0,
                'vq_loss': 0.0,
                'qs_loss': 0.0,
                'balance_loss': 0.0,
                'gate_loss': 0.0,
                'kl_loss': 0.0,
                'dec_cl_loss': 0.0,
            }
            self.log(f'[Stage 2] Loss weights: {stage2_loss_w}')
            
            # Update learning rate for stage 2 if specified
            stage2_lr_rec = self.config.get('stage2_lr_rec', self.lr_rec)
            if stage2_lr_rec != self.lr_rec:
                self.log(f'[Stage 2] Updating learning rate: {self.lr_rec} -> {stage2_lr_rec}')
                for param_group in self.rec_optimizer.param_groups:
                    param_group['lr'] = stage2_lr_rec
            
            # Update early_stop for stage 2 if specified
            stage2_early_stop = self.config.get('stage2_early_stop', self.early_stop)
            
            # Reset training state for stage 2
            # If Stage 1 was skipped (using external checkpoint), initialize with defaults
            stage2_best_score = self.best_score if self.best_score > 0 else 0.0
            stage2_best_result = self.best_result if self.best_result else {}
            stage2_best_ckpt = self.best_ckpt
            cur_eval_step = 0
            stop = False
            
            self.log("")
            self.log("[Stage 2] Starting training loop...")
            self.log("")
            
            # Stage 2 training loop
            for epoch_idx in range(stage2_epochs):
                self.accelerator.wait_for_everyone()
                
                # Train (freeze_id=True to skip ID optimizer)
                training_start_time = time()
                train_loss = self._train_epoch_rec(epoch_idx, loss_w=stage2_loss_w, freeze_id=True, verbose=verbose)
                training_end_time = time()
                
                train_loss_output = self._generate_train_loss_output(
                    epoch_idx, training_start_time, training_end_time, train_loss
                )
                
                self.log(f"[Stage 2 Epoch {epoch_idx}] {train_loss_output}")
                self.log(f'[Stage 2 Epoch {epoch_idx}] REC lr: {self.rec_lr_scheduler.get_lr()}')
                
                # Evaluate
                metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)
                
                if metrics[self.valid_metric] > stage2_best_score:
                    stage2_best_score = metrics[self.valid_metric]
                    stage2_best_result = metrics
                    cur_eval_step = 0
                    stage2_best_ckpt = self.safe_save(epoch_idx, self.all_item_code, prefix='stage2')
                    self.log(f'[Stage 2 Epoch {epoch_idx}] New best model saved!')
                else:
                    cur_eval_step += 1
                
                self.log(f'[Stage 2 Epoch {epoch_idx}] Val Results: {metrics}')
                self.log(f'[Stage 2 Epoch {epoch_idx}] Best {self.valid_metric}: {stage2_best_score:.6f}')
                
                self.accelerator.wait_for_everyone()
                
                if cur_eval_step >= stage2_early_stop:
                    self.log(f"[Stage 2] Early stopping triggered at epoch {epoch_idx}")
                    stop = True
                    break
            
            # Update best results with stage 2 results
            self.best_score = stage2_best_score
            self.best_result = stage2_best_result
            self.best_ckpt = stage2_best_ckpt
            
            self.log("")
            self.log("="*60)
            self.log(f"Stage 2 Training Complete!")
            self.log(f"Best Stage 2 {self.valid_metric}: {stage2_best_score:.6f}")
            self.log(f"Best Stage 2 Validation Results: {stage2_best_result}")
            self.log("="*60)
            self.log("")
            
            # Test Stage 2 best model
            self.log("Testing Stage 2 best model on test set...")
            stage2_test_results = self.test(verbose=verbose, model_file=stage2_best_ckpt)
            self.log("")
            self.log("="*60)
            self.log(f"Stage 2 Test Results: {stage2_test_results}")
            self.log("="*60)
            self.log("")
            
            # Compare Stage 1 vs Stage 2 (only if Stage 1 was run)
            if self.stage1_test_results is not None:
                self.log("="*60)
                self.log("Stage 1 vs Stage 2 Comparison:")
                self.log("="*60)
                self.log(f"Stage 1 Test Results: {self.stage1_test_results}")
                self.log(f"Stage 2 Test Results: {stage2_test_results}")
                
                # Calculate improvements
                for metric_name in self.stage1_test_results.keys():
                    stage1_val = self.stage1_test_results[metric_name]
                    stage2_val = stage2_test_results[metric_name]
                    improvement = ((stage2_val - stage1_val) / stage1_val * 100) if stage1_val > 0 else 0
                    self.log(f"{metric_name}: {stage1_val:.6f} -> {stage2_val:.6f} ({improvement:+.2f}%)")
                self.log("="*60)
                self.log("")
            else:
                self.log("="*60)
                self.log("Stage 1 was skipped (used provided checkpoint)")
                self.log(f"Stage 2 Test Results: {stage2_test_results}")
                self.log("="*60)
                self.log("")

        return self.best_score
    
    @torch.no_grad()
    def test(self, verbose=True, model_file=None, prefix_allowed_tokens_fn=None):
        test_results=None
        if self.test_data is not None:
            metrics = self._test_epoch(load_best_model=True, model_file=model_file,
                                       prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, verbose=verbose)

            test_results = metrics
        return test_results

    @torch.no_grad()
    def _test_epoch(self, code=None, test_data=None, load_best_model=False, model_file=None,
                    prefix_allowed_tokens_fn=None, verbose=True):
        
        if test_data is None:
            test_data = self.test_data

        if load_best_model:
            ckpt_file = model_file or self.best_ckpt
            if dist.is_initialized():
                safe_load(self.model_rec.module, ckpt_file, verbose=verbose)
                safe_load(self.model_id.module, ckpt_file+'.rqvae', verbose=verbose)
            else:
                safe_load(self.model_rec, ckpt_file, verbose=verbose)
                safe_load(self.model_id, ckpt_file+'.rqvae', verbose=verbose)

            code = json.load(open(ckpt_file[:-3]+'.code.json'))

            message_output = "Loading model parameters from {}".format(
                ckpt_file
            )
            self.log(message_output)

        self.model_rec.eval()
        self.model_id.eval()

        iter_data = tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )

        if isinstance(code, torch.Tensor):
            code = code.cpu().tolist()

        total = 0
        metrics = {m: 0 for m in self.all_metrics}

        code2item = defaultdict(list)
        for i, c in enumerate(code[1:]):
            code2item[str(c)].append(i+1)

        item_code = torch.tensor(code).to(self.device)

        for batch_idx, data in enumerate(iter_data):
            input_ids, attention_mask, labels \
                = data["input_ids"].to(self.device), data["attention_mask"].to(self.device), data["targets"].to(self.device)

            B = input_ids.size(0)
            input_ids = item_code[input_ids].clone().detach().reshape(B, -1)
            labels = item_code[labels].clone().detach().reshape(B, -1)
            attention_mask = (input_ids != -1).bool() 

            if dist.is_initialized():
                preds = self.model_rec.module.generate(input_ids=input_ids, attention_mask=attention_mask, n_return_sequences=10)
                all_preds, all_labels = self.accelerator.gather_for_metrics((preds, labels))
                _metrics = self.evaluate(all_preds, all_labels)
                total += len(all_labels)
            else:
                preds = self.model_rec.generate(input_ids=input_ids, attention_mask=attention_mask, n_return_sequences=10)
                _metrics = self.evaluate(preds, labels)
                total += len(labels)

            for m in metrics.keys():
                metrics[m] += _metrics[m]

        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        return metrics
    
    @torch.no_grad()
    def get_code(self, epoch_idx, verbose=True):
        self.model_rec.eval()
        self.model_id.eval()
        if dist.is_initialized():
            all_item_embs = self.model_rec.module.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.module.get_indices(all_item_embs).detach().cpu().numpy()
        else:
            all_item_embs = self.model_rec.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.get_indices(all_item_embs).detach().cpu().numpy()
        

        if verbose:
            for i in range(self.code_length-1):
                self.log(f'[Epoch {epoch_idx}] Evaluation {self.save_path}/{epoch_idx}.pt Code balance {balance(all_item_prefix[:, i].tolist(), ncentroids=self.code_num)} Used code num of level {i+1}: {len(set(all_item_prefix[:, i].tolist()))}')

            self.log(f'[Epoch {epoch_idx}] Evaluation {self.save_path}/{epoch_idx}.pt Code confilct {conflict(all_item_prefix.tolist())}')
        
        all_item_prefix = all_item_prefix.tolist()

        tokens2item = defaultdict(list)
        all_item_tokens = [[-1] * self.code_length]  # Dynamic based on code_length
        max_conflict = 0
        for i in range(len(all_item_prefix)):
            str_id = ' '.join(map(str, all_item_prefix[i]))
            tokens2item[str_id].append(i+1)
            all_item_tokens.append(all_item_prefix[i]+[len(tokens2item[str_id])-1])
            max_conflict = max(max_conflict, len(tokens2item[str_id]))
        self.log(f'[Epoch {epoch_idx}] [TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
        if max_conflict > self.code_num:
            raise ValueError(
                f'[TOKENIZER] RQ-VAE semantic IDs conflict with codebook size: '
                f'{max_conflict} > {self.code_num}. Please increase the codebook size.'
            )

        return all_item_tokens

    def log(self, message, level='info'):
        return log(message, self.accelerator, self.logger, level=level)

