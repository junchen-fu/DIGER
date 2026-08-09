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
import contextlib
from itertools import islice
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


def accumulation_windows(iterable, window_size):
    iterator = iter(iterable)
    while True:
        window = list(islice(iterator, window_size))
        if not window:
            return
        yield window


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

                                                                               
        self.sim = config.get('sim', 'cos')                                          
        self.alpha = config.get('alpha', 1)                              
        self.loss_type = config.get('loss_type', 'mse')                    
        self.tau = config.get('tau', 0.07)                                    

        self.accelerator = accelerator

        self.state = PartialState()
        self.world_size = self.state.num_processes
        self.device = self.state.device
        self.all_item_code = None
        self.model_rec.device = self.device

                                                      
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

        self._configure_trainable_parameters(model_rec, model_id)

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
        self.process_seed = init_device_seed(config['seed'], self.accelerator.process_index)

    def _configure_trainable_parameters(self, model_rec, model_id):
        for param in model_rec.parameters():
            param.requires_grad = True
        if self.config.get('freeze_semantic_embedding', True):
            model_rec.semantic_embedding.requires_grad_(False)

        if not self.config.get('end_to_end', False):
            model_id.requires_grad_(False)
            return

        model_id.requires_grad_(True)
        if self.config.get('freeze_id_encoder', False):
            freeze_layers = int(self.config.get('freeze_id_encoder_layers', 0))
            if freeze_layers <= 0:
                model_id.encoder.requires_grad_(False)
            else:
                linear_layer_idx = 0
                for module in model_id.encoder.mlp_layers.children():
                    if isinstance(module, nn.Linear):
                        if linear_layer_idx < freeze_layers:
                            module.requires_grad_(False)
                        linear_layer_idx += 1
        if self.config.get('freeze_rq', False):
            model_id.rq.requires_grad_(False)

    def _count_parameters(self, model, model_name="Model"):
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
        learner = self.learner

                                                         
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
        len_dataloader = math.ceil(len(self.train_data) / self.world_size)
        num_update_steps_per_epoch = math.ceil(len_dataloader / self.gradient_accumulation_steps)
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if epochs is None:
            epochs = self.epochs
        max_steps = math.ceil(epochs * num_update_steps_per_epoch)

        return max_steps

    def _apply_code_loss_uncertainty(self, code_loss, sigma, model_id):
        if (
            sigma is None
            or self.config.get('use_plain_code_loss', False)
            or self.config.get('use_cosine_annealing', False)
        ):
            return code_loss, None

        if self.config.get('use_simple_uncertainty_loss', False):
            auto_sigma_module = getattr(model_id.rq.vq_layers[0], 'auto_sigma_module', None)
            sigma_lambda = self.config.get('sigma_lambda', 0.5)
            if auto_sigma_module is not None and hasattr(auto_sigma_module, 'compute_uncertainty_loss'):
                return auto_sigma_module.compute_uncertainty_loss(
                    code_loss, sigma, lambda_bias=sigma_lambda
                )
            effective_sigma = (sigma.abs() + sigma_lambda).clamp(min=1e-6)
            return (
                code_loss / (2 * effective_sigma ** 2) + torch.log(effective_sigma),
                sigma_lambda,
            )

        transformed = AutoSigmaGumbel.compute_uncertainty_loss(
            code_loss,
            sigma,
            reg_weight=self.config.get('sigma_reg_weight', 1.0),
            annealing_threshold=self.config.get('annealing_threshold'),
            annealing_slow_k=self.config.get('annealing_slow_k'),
            annealing_slow_c=self.config.get('annealing_slow_c'),
            annealing_fast_k=self.config.get('annealing_fast_k'),
            annealing_fast_c=self.config.get('annealing_fast_c'),
        )
        return transformed, None

    def _validate_preserved_forward_batch_config(self, loss_w):
        if loss_w.get('qs_loss', 0) != 0:
            raise ValueError(
                'preserve_reference_forward_batch currently requires qs_loss_weight=0'
            )
        if self.config.get('auto_lambda_mode', 'fixed') == 'adaptive':
            raise ValueError(
                'preserve_reference_forward_batch does not support adaptive lambda updates'
            )
    def _train_epoch_rec_preserving_forward_batch(self, epoch_idx, loss_w, verbose=True):
        self._validate_preserved_forward_batch_config(loss_w)
        self.model_rec.train()
        self.model_id.train()

        model_rec = self.accelerator.unwrap_model(self.model_rec)
        model_id = self.accelerator.unwrap_model(self.model_id)
        model_id.reset_adaptive_selection_stats()

        accumulation_steps = self.gradient_accumulation_steps
        total_num = 0
        total_loss = defaultdict(int)
        iter_data = tqdm(
            self.train_data,
            total=len(self.train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )
        if epoch_idx == 0:
            global_forward_batch = (
                self.config['batch_size'] * self.world_size * accumulation_steps
            )
            self.log(
                '[Batch] RQ-VAE forward preserves the full accumulation window: '
                f'global_forward_batch={global_forward_batch}'
            )

        for batch_window in accumulation_windows(iter_data, accumulation_steps):
            actual_steps = len(batch_window)
            total_num += 1
            self.rec_optimizer.zero_grad()
            if hasattr(self, 'id_optimizer'):
                self.id_optimizer.zero_grad()

            prepared_batches = []
            target_parts = []
            target_sizes = []
            for batch in batch_window:
                raw_input_ids = batch['input_ids'].to(self.device)
                raw_attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)
                batch_size = raw_input_ids.size(0)
                input_ids = self.all_item_code[raw_input_ids].clone().detach().reshape(batch_size, -1)
                labels = self.all_item_code[targets].clone().detach().reshape(batch_size, -1)
                attention_mask = (input_ids != -1).bool()
                flat_targets = targets.flatten()

                prepared_batches.append((input_ids, attention_mask, labels))
                target_parts.append(flat_targets)
                target_sizes.append(flat_targets.numel())

            target_flatten = torch.cat(target_parts, dim=0)
            target_semantic_embs = model_rec.semantic_embedding(target_flatten)
            use_gumbel = self.config.get('use_gumbel', True)
            z_hat, vq_loss, _, _, target_code_logits, balance_loss, gate_reg_loss, sigma, z = \
                self.model_id(
                    target_semantic_embs,
                    use_gumbel=use_gumbel,
                    current_epoch=epoch_idx,
                    global_step=self.global_step,
                    return_latent=True,
                )

            recon_loss = F.mse_loss(z_hat, z)
            token_indices = model_id.get_indices(target_semantic_embs)
            z_parts = z.split(target_sizes, dim=0)
            token_index_parts = token_indices.split(target_sizes, dim=0)

            if sigma is not None and self.config.get('use_cosine_annealing', False):
                current_epoch = self.global_step / max(1, self.max_steps) * self.epochs
                cosine_factor = 0.5 * (1 + math.cos(math.pi * current_epoch / self.epochs))
                target_std = max(1e-6, float(self.config.get('initial_std', 1.0)) * cosine_factor)
                sigma.data.fill_(math.log2(target_std))

            quantizer_losses = {
                'recon_loss': recon_loss,
                'vq_loss': vq_loss,
            }
            if balance_loss is not None:
                quantizer_losses['balance_loss'] = balance_loss
            if gate_reg_loss is not None:
                quantizer_losses['gate_loss'] = gate_reg_loss
            quantizer_loss = sum(
                value * loss_w.get(name, 0)
                for name, value in quantizer_losses.items()
            )
            if self.world_size > 1:
                quantizer_loss = quantizer_loss + 0.0 * target_code_logits.float().sum()

            code_losses = []
            raw_code_losses = []
            qs_losses = []
            actual_lambda = None
            for micro_idx, ((input_ids, attention_mask, labels), z_part, token_index_part) in enumerate(
                zip(prepared_batches, z_parts, token_index_parts)
            ):
                is_last = micro_idx == actual_steps - 1
                sync_context = (
                    contextlib.nullcontext()
                    if is_last
                    else self.accelerator.no_sync(self.model_rec)
                )
                with sync_context:
                    outputs = self.model_rec(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        quantizer_latent=z_part.detach(),
                        token_indices=token_index_part,
                        qs_beta=self.config.get('qs_beta', 0.25),
                    )
                    raw_code_loss = F.cross_entropy(
                        outputs.logits.reshape(-1, self.code_num),
                        labels.detach().reshape(-1),
                    )
                    code_loss, current_lambda = self._apply_code_loss_uncertainty(
                        raw_code_loss,
                        sigma.detach() if sigma is not None else None,
                        model_id,
                    )
                    if current_lambda is not None:
                        actual_lambda = current_lambda

                    raw_code_losses.append(raw_code_loss.detach())
                    code_losses.append(code_loss.detach())
                    qs_losses.append(outputs.qs_loss.detach())

                    micro_scale = accumulation_steps / actual_steps
                    backward_loss = (
                        code_loss * loss_w.get('code_loss', 0)
                        + outputs.qs_loss * loss_w.get('qs_loss', 0)
                    ) * micro_scale
                    if self.world_size > 1:
                        backward_loss = backward_loss + 0.0 * (
                            outputs.seq_project_latents.sum()
                            + outputs.dec_latents.sum()
                        )

                    if is_last:
                        backward_loss = backward_loss + accumulation_steps * quantizer_loss
                        if sigma is not None and not self.config.get('use_plain_code_loss', False):
                            mean_raw_code_loss = torch.stack(raw_code_losses).mean()
                            sigma_objective, actual_lambda = self._apply_code_loss_uncertainty(
                                mean_raw_code_loss, sigma, model_id
                            )
                            sigma_proxy = sigma_objective - sigma_objective.detach()
                            backward_loss = backward_loss + (
                                accumulation_steps
                                * loss_w.get('code_loss', 0)
                                * sigma_proxy
                            )

                    self.accelerator.backward(backward_loss)

            self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1)
            if hasattr(self, 'id_optimizer'):
                self.accelerator.clip_grad_norm_(self.model_id.parameters(), 1)

            mean_code_loss = torch.stack(code_losses).mean()
            mean_raw_code_loss = torch.stack(raw_code_losses).mean()
            mean_qs_loss = torch.stack(qs_losses).mean()

            self.rec_optimizer.step()
            self.rec_lr_scheduler.step()
            if hasattr(self, 'id_optimizer'):
                if self.config.get('use_dynamic_sigma_lr', False) and hasattr(self, 'lr_sigma'):
                    sigma_lr_multiplier = 10.0 if mean_code_loss.item() < 2.0 else 1.0
                    for param_group in self.id_optimizer.param_groups:
                        if abs(param_group['lr'] - self.lr_sigma) < 1e-8 or \
                                abs(param_group['lr'] - self.lr_sigma * 10.0) < 1e-8:
                            param_group['lr'] = self.lr_sigma * sigma_lr_multiplier
                self.id_optimizer.step()
                self.id_lr_scheduler.step()

            if (
                sigma is not None
                and self.global_step % 10 == 0
                and self.accelerator.is_main_process
                and self.config.get('use_simple_uncertainty_loss', False)
                and not self.config.get('use_plain_code_loss', False)
                and actual_lambda is not None
            ):
                lambda_value = (
                    actual_lambda
                    if isinstance(actual_lambda, float)
                    else actual_lambda.item()
                )
                target_sigma = math.sqrt(max(0, mean_raw_code_loss.item())) - lambda_value
                self.log(
                    f'[Simple Uncertainty] sigma={sigma.item():.4f}, '
                    f'Loss={mean_raw_code_loss.item():.4f}, lambda={lambda_value:.4f} '
                    f'(fixed), Target_sigma={target_sigma:.4f}'
                )

            self.global_step += 1

            loss_for_logging = mean_code_loss * loss_w.get('code_loss', 0) + quantizer_loss.detach()
            loss_values = {
                'loss': self.accelerator.gather(loss_for_logging).mean().item(),
                'code_loss': self.accelerator.gather(mean_code_loss).mean().item(),
                'recon_loss': self.accelerator.gather(recon_loss.detach()).mean().item(),
                'vq_loss': self.accelerator.gather(vq_loss.detach()).mean().item(),
                'qs_loss': self.accelerator.gather(mean_qs_loss).mean().item(),
            }
            if balance_loss is not None:
                loss_values['balance_loss'] = self.accelerator.gather(
                    balance_loss.detach()
                ).mean().item()
            if gate_reg_loss is not None:
                loss_values['gate_loss'] = self.accelerator.gather(
                    gate_reg_loss.detach()
                ).mean().item()
            if sigma is not None:
                loss_values['sigma'] = self.accelerator.gather(sigma.detach()).mean().item()

            for name, value in loss_values.items():
                total_loss[name] += value
            iter_data.set_postfix(loss=loss_values['loss'])

        for name in total_loss:
            total_loss[name] = round(total_loss[name] / total_num, 4)

        self.accelerator.wait_for_everyone()
        return total_loss

    def _train_epoch_rec(self, epoch_idx, loss_w, freeze_id=False, verbose=True):

        preserve_forward_batch = self.config.get('preserve_reference_forward_batch', True)
        if (
            preserve_forward_batch
            and self.gradient_accumulation_steps > 1
            and self.config.get('end_to_end', False)
            and not freeze_id
        ):
            return self._train_epoch_rec_preserving_forward_batch(
                epoch_idx, loss_w=loss_w, verbose=verbose
            )

        self.model_rec.train()
                                                                       
        self.model_id.train()

        model_rec = self.accelerator.unwrap_model(self.model_rec)
        model_id = self.accelerator.unwrap_model(self.model_id)
        model_id.reset_adaptive_selection_stats()

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
            with self.accelerator.accumulate(self.model_rec, self.model_id):

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
                target_semantic_embs = model_rec.semantic_embedding(target_flatten)
                use_gumbel = self.config.get('use_gumbel', not freeze_id)
                z_hat, vq_loss, _, _, target_code_logits, balance_loss, gate_reg_loss, sigma, z = \
                    self.model_id(
                        target_semantic_embs,
                        use_gumbel=use_gumbel,
                        current_epoch=epoch_idx,
                        global_step=self.global_step,
                        return_latent=True,
                    )

                                                                      
                                                            
                glq_recon_loss = F.mse_loss(z_hat, z)

                                                                            
                                                                                
                                                           

                                                                                  
                                                                             
                                                                         
                token_indices = model_id.get_indices(target_semantic_embs)
                qs_beta = self.config.get('qs_beta', 0.25)

                                                            
                                                              
                recon_loss = glq_recon_loss                 

                                                    
                outputs = self.model_rec(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels,
                                         quantizer_latent=z,
                                         token_indices=token_indices,
                                         qs_beta=qs_beta)

                logits = outputs.logits                               
                qs_loss = outputs.qs_loss

                                      
                code_loss = F.cross_entropy(logits.reshape(-1, self.code_num), labels.detach().reshape(-1))

                                                                                    
                if sigma is not None:
                                                                       
                    use_cosine_annealing = self.config.get('use_cosine_annealing', False)

                    if use_cosine_annealing:
                                                                                                
                                                                                    
                                                                 

                                                                 
                                                                                                                      
                                                                                                 
                                                                                                     

                        initial_std = float(self.config.get('initial_std', 1.0))
                        current_epoch = self.global_step / max(1, self.max_steps) * self.epochs
                        T_max = self.epochs

                                                                    
                        import math
                        cosine_factor = 0.5 * (1 + math.cos(math.pi * current_epoch / T_max))
                        target_std = initial_std * cosine_factor

                                                                                       
                        target_std = max(1e-6, target_std)

                                                                 
                                                                                         
                        target_sigma = math.log2(target_std)

                                                                       
                        sigma.data.fill_(target_sigma)

                                                                                     
                        if self.accelerator.sync_gradients and self.global_step % 10 == 0 and self.accelerator.is_main_process:
                            self.log(f"[Cosine Annealing] Epoch={current_epoch:.2f}/{T_max}, σ={target_sigma:.4f}, std={target_std:.4f} (Fixed)")
                    else:
                                                                          
                        use_plain_code_loss = self.config.get('use_plain_code_loss', False)
                        use_simple_uncertainty_loss = self.config.get('use_simple_uncertainty_loss', False)

                        if use_plain_code_loss:
                                                                                              
                                                            
                            if self.accelerator.sync_gradients and self.global_step % 10 == 0 and self.accelerator.is_main_process:
                                sigma_val = sigma.item()
                                if use_simple_uncertainty_loss:
                                    self.log(f"[Plain Loss] σ={sigma_val:.4f} (direct), code_loss={code_loss.item():.4f}")
                                else:
                                    self.log(f"[Plain Loss] σ={sigma_val:.4f}, std≈{2**sigma_val:.4f}, code_loss={code_loss.item():.4f}")
                        elif use_simple_uncertainty_loss:
                                                                                                  
                            original_code_loss = code_loss.item()
                            sigma_lambda = self.config.get('sigma_lambda', 0.5)

                            auto_sigma_module = getattr(model_id.rq.vq_layers[0], 'auto_sigma_module', None)

                            if auto_sigma_module is not None and hasattr(auto_sigma_module, 'compute_uncertainty_loss'):
                                                                                          
                                code_loss, actual_lambda = auto_sigma_module.compute_uncertainty_loss(
                                    code_loss, sigma, lambda_bias=sigma_lambda
                                )
                            else:
                                                                                    
                                from vq import AutoSigmaSimple
                                code_loss = AutoSigmaSimple.compute_uncertainty_loss(code_loss, sigma, lambda_bias=sigma_lambda)
                                actual_lambda = sigma_lambda

                            if self.accelerator.sync_gradients and self.global_step % 10 == 0 and self.accelerator.is_main_process:
                                sigma_val = sigma.item()
                                lambda_val = actual_lambda if isinstance(actual_lambda, float) else actual_lambda.item()
                                import math
                                                                       
                                target_sigma = math.sqrt(max(0, original_code_loss)) - lambda_val

                                                        
                                auto_lambda_mode = self.config.get('auto_lambda_mode', 'fixed')
                                if auto_lambda_mode == 'learnable':
                                    self.log(f"[Simple Uncertainty] σ={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (learnable), Target_σ={target_sigma:.4f}")
                                elif auto_lambda_mode == 'adaptive':
                                    if auto_sigma_module is not None:
                                        loss_ema = auto_sigma_module.loss_ema.item()
                                        self.log(f"[Simple Uncertainty] σ={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (adaptive, EMA={loss_ema:.4f}), Target_σ={target_sigma:.4f}")
                                    else:
                                        self.log(f"[Simple Uncertainty] σ={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (adaptive), Target_σ={target_sigma:.4f}")
                                else:
                                    self.log(f"[Simple Uncertainty] σ={sigma_val:.4f}, Loss={original_code_loss:.4f}, λ={lambda_val:.4f} (fixed), Target_σ={target_sigma:.4f}")
                        else:
                                                                     
                            original_code_loss = code_loss.item()                    
                            sigma_reg_weight = self.config.get('sigma_reg_weight', 1.0)
                                                                                           
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
                                                          
                            if self.accelerator.sync_gradients and self.global_step % 10 == 0 and self.accelerator.is_main_process:
                                sigma_val = sigma.item()                       
                                                                                                      
                                import math
                                equilibrium_sigma = -0.589 + 1.298 * math.log(max(0.1, original_code_loss))
                                self.log(f"[Annealing] σ={sigma_val:.4f}, Loss={original_code_loss:.4f}, Target_σ={equilibrium_sigma:.4f}")

                                                                              
                                                   
                                                     
                                                                              

                losses = dict(
                    code_loss=code_loss,                         
                    recon_loss=recon_loss,                       
                    vq_loss=vq_loss,                                
                    qs_loss=qs_loss,                                      
                )

                                                                                
                if balance_loss is not None:
                    losses['balance_loss'] = balance_loss

                                                                                          
                if gate_reg_loss is not None:
                    losses['gate_loss'] = gate_reg_loss

                loss = sum([v * loss_w.get(k, 0) for k, v in losses.items()])
                if self.world_size > 1:
                    loss = loss + 0.0 * (
                        outputs.seq_project_latents.sum()
                        + outputs.dec_latents.sum()
                        + target_code_logits.float().sum()
                    )

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1)
                    if hasattr(self, 'id_optimizer') and not freeze_id:
                        self.accelerator.clip_grad_norm_(self.model_id.parameters(), 1)

                self.rec_optimizer.step()
                if self.accelerator.sync_gradients:
                    self.rec_lr_scheduler.step()
                                                                    
                if hasattr(self, 'id_optimizer') and not freeze_id:
                                                                 
                                                                                         
                                                                                      
                    use_dynamic_sigma_lr = self.config.get('use_dynamic_sigma_lr', False)
                    if self.accelerator.sync_gradients and use_dynamic_sigma_lr and hasattr(self, 'lr_sigma') and self.lr_sigma is not None:
                        threshold_loss = 2.0                    
                        current_code_loss = code_loss.item() if isinstance(code_loss, torch.Tensor) else code_loss

                                                                    
                        if current_code_loss < threshold_loss:
                                                                  
                            sigma_lr_multiplier = 10.0
                        else:
                                                     
                            sigma_lr_multiplier = 1.0

                                                                        
                        for param_group in self.id_optimizer.param_groups:
                                                                        
                                                                      
                            if abs(param_group['lr'] - self.lr_sigma) < 1e-8 or \
                               abs(param_group['lr'] - self.lr_sigma * 10.0) < 1e-8:
                                param_group['lr'] = self.lr_sigma * sigma_lr_multiplier

                                                                          
                        if self.global_step % 50 == 0 and self.accelerator.is_main_process:
                            stage = "FAST" if current_code_loss < threshold_loss else "SLOW"
                            actual_sigma_lr = self.lr_sigma * sigma_lr_multiplier
                            self.log(f"[Sigma LR] Stage={stage}, Loss={current_code_loss:.4f}, σ_lr={actual_sigma_lr:.6f} ({sigma_lr_multiplier:.1f}x)")

                    self.id_optimizer.step()
                    if self.accelerator.sync_gradients:
                        self.id_lr_scheduler.step()

                if self.accelerator.sync_gradients:
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

                                                             
                if balance_loss is not None:
                    balance_loss_mean = self.accelerator.gather(balance_loss).mean().item()
                    loss['balance_loss'] = balance_loss_mean

                                                          
                if gate_reg_loss is not None:
                    gate_loss_mean = self.accelerator.gather(gate_reg_loss).mean().item()
                    loss['gate_loss'] = gate_loss_mean

                                                      
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

                                                
            filename = f'{prefix}_{epoch}' if prefix else str(epoch)

            self.accelerator.save(unwrap_model_rec.state_dict(), f'{self.save_path}/{filename}.pt')
            self.accelerator.save(unwrap_model_id.state_dict(), f'{self.save_path}/{filename}.pt.rqvae')
            json.dump(code.cpu().tolist(), open(f'{self.save_path}/{filename}.code.json', 'w'))
            self.log(f'[Epoch {epoch}] Save model {self.save_path}/{filename}.pt')

        filename = f'{prefix}_{epoch}' if prefix else str(epoch)
        last_checkpoint = f'{self.save_path}/{filename}.pt'
        return last_checkpoint

    def evaluate(self, outputs, labels):
        batch_size, k, _ = outputs.shape                                                 
        recall_at_1, recall_at_5, recall_at_10 = [], [], []
        ndcg_at_1, ndcg_at_5, ndcg_at_10 = [], [], []

        for i in range(batch_size):
            label = labels[i].unsqueeze(0)                
            out = outputs[i]

            matches = torch.all(torch.eq(out.unsqueeze(1), label.unsqueeze(0)), dim=2)                               
            matches = matches.any(dim=1).cpu().numpy()        

                    
            recall_at_1.append(matches[:1].sum() / 1.0)
            recall_at_5.append(matches[:5].sum() / 1.0)                                                 
            recall_at_10.append(matches.sum() / 1.0)

                                     
            ndcg_at_1.append(ndcg_at_k(matches, 1))
            ndcg_at_5.append(ndcg_at_k(matches, 5))
            ndcg_at_10.append(ndcg_at_k(matches, 10))

                                
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
        stop = False
        cur_eval_step = 0
        self.best_score = 0
        self.best_result = {}
        self.best_ckpt = None
        loss_w = defaultdict(int)

                                           
        all_item_code = self.get_code(epoch_idx=-1, verbose=verbose)
        self.all_item_code = torch.tensor(all_item_code).to(self.device)

                                         
        end_to_end = self.config.get('end_to_end', False)
        model_rec_unwrapped = self.accelerator.unwrap_model(self.model_rec)
        model_id_unwrapped = self.accelerator.unwrap_model(self.model_id)

        if end_to_end:
            self.log(f'[Training Mode] Recommender model unfrozen (semantic_embedding frozen)')

            freeze_id_encoder = self.config.get('freeze_id_encoder', False)
            freeze_id_encoder_layers = self.config.get('freeze_id_encoder_layers', 0)
            freeze_rq = self.config.get('freeze_rq', False)

            if freeze_id_encoder:
                if freeze_id_encoder_layers > 0:
                    encoder_modules = list(model_id_unwrapped.encoder.mlp_layers.children())
                    total_linear_layers = sum(1 for m in encoder_modules if isinstance(m, nn.Linear))
                    self.log(f'[Training Mode] ID tokenizer encoder: {freeze_id_encoder_layers}/{total_linear_layers} layers FROZEN (bottom {freeze_id_encoder_layers} layers)')
                else:
                    self.log(f'[Training Mode] ID tokenizer encoder FROZEN (all layers)')
            if freeze_rq:
                self.log(f'[Training Mode] ID tokenizer RQ quantizer FROZEN')

                                        
            self.lr_id = self.config.get('lr_id', self.lr_rec * 0.1)                             

                                                                         
            self.lr_sigma = self.config.get('lr_sigma', None)
            self.lr_lambda = self.config.get('lr_lambda', None)                               
            use_separate_sigma_lr = self.lr_sigma is not None and self.config.get('use_learnable_sigma_gumbel', False)
            use_learnable_lambda = self.config.get('auto_lambda_mode', 'fixed') == 'learnable'

            if use_separate_sigma_lr or (use_learnable_lambda and self.lr_lambda is not None):
                                                                               
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

                                                        
                param_groups = []
                if len(other_params) > 0:
                    param_groups.append({'params': other_params, 'lr': self.lr_id})
                if len(sigma_params) > 0 and use_separate_sigma_lr:
                    param_groups.append({'params': sigma_params, 'lr': self.lr_sigma})
                if len(lambda_params) > 0 and self.lr_lambda is not None:
                    param_groups.append({'params': lambda_params, 'lr': self.lr_lambda})
                elif len(lambda_params) > 0:
                                                                               
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
                                                                            
                self.id_optimizer = self._build_optimizer(self.model_id, self.lr_id, self.weight_decay)

                                        
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

                                                                     
            self.id_optimizer, self.id_lr_scheduler = \
                self.accelerator.prepare(self.id_optimizer, self.id_lr_scheduler)

                                                      
            loss_w['code_loss'] = self.config.get('code_loss_weight', 1.0)
            loss_w['recon_loss'] = self.config.get('recon_loss_weight', 1.0)
            loss_w['vq_loss'] = self.config.get('vq_loss_weight', 0.25)
            loss_w['qs_loss'] = self.config.get('qs_loss_weight', 0.1)             
            loss_w['balance_loss'] = self.config.get('balance_loss_weight', 0.1)                            
            loss_w['gate_loss'] = self.config.get('gate_loss_weight', 0.1)                        
            loss_w['kl_loss'] = self.config.get('kl_loss_weight', 0.0)
            loss_w['dec_cl_loss'] = self.config.get('dec_cl_loss_weight', 0.0)

            self.log(f'[Training Mode] End-to-end training enabled')
            self.log(f'[Training Mode] Loss weights: {dict(loss_w)}')
        else:
            self.log(f'[Training Mode] Recommender model unfrozen (semantic_embedding frozen)')

                                              
            loss_w['code_loss'] = 1
            loss_w['vq_loss'] = 0
            loss_w['qs_loss'] = 0                      
            loss_w['balance_loss'] = 0                            
            loss_w['gate_loss'] = 0                         
            loss_w['kl_loss'] = 0
            loss_w['dec_cl_loss'] = 0
            loss_w['recon_loss'] = 0

            self.log(f'[Training Mode] Frozen RQ-VAE mode (original)')

                                    
        self.log("")
        self._count_parameters(self.model_rec, "Recommender Model")
        self._count_parameters(self.model_id, "ID Tokenizer (RQ-VAE)")

                                                   
        self.log("ID Tokenizer Module Breakdown:")
        self._count_module_parameters(model_id_unwrapped, "encoder")
        self._count_module_parameters(model_id_unwrapped, "rq")

        sigma_params = [name for name, p in model_id_unwrapped.named_parameters() if 'sigma' in name.lower()]
        if sigma_params:
            self.log("")
            self.log(f"========== Learnable Sigma Parameters (Base-2 Exponential) ==========")
            for name in sigma_params:
                param = dict(model_id_unwrapped.named_parameters())[name]
                sigma_val = param.data.item()
                s_val = 2 ** sigma_val
                self.log(f"  {name}: σ={sigma_val:.6f}, requires_grad={param.requires_grad}")
                self.log(f"    -> Noise scale: s = 2^σ = 2^{sigma_val:.3f} = {s_val:.6f}")
            self.log(f"=" * 50)

                                    
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

                                                                 
            freeze_id_epochs = self.config.get('freeze_id_epochs', 0)
            if end_to_end and epoch_idx < freeze_id_epochs:
                if epoch_idx == 0:
                    self.log(f'[Training Mode] RQ-VAE FROZEN for first {freeze_id_epochs} epochs')
                    self.log(f'[Training Mode] Will unfreeze at epoch {freeze_id_epochs}')
            elif end_to_end and freeze_id_epochs > 0 and epoch_idx == freeze_id_epochs:
                freeze_id_encoder = self.config.get('freeze_id_encoder', False)
                freeze_id_encoder_layers = self.config.get('freeze_id_encoder_layers', 0)
                freeze_rq = self.config.get('freeze_rq', False)

                self.log(f'[Training Mode] RQ-VAE UNFROZEN at epoch {epoch_idx}!')
                if freeze_id_encoder:
                    if freeze_id_encoder_layers > 0:
                        encoder_modules = list(model_id_unwrapped.encoder.mlp_layers.children())
                        total_linear_layers = sum(1 for m in encoder_modules if isinstance(m, nn.Linear))
                        self.log(f'[Training Mode] (encoder: {freeze_id_encoder_layers}/{total_linear_layers} bottom layers still frozen)')
                    else:
                        self.log(f'[Training Mode] (encoder still frozen)')
                if freeze_rq:
                    self.log(f'[Training Mode] (RQ quantizer still frozen)')

                                                        
            is_id_frozen = end_to_end and epoch_idx < freeze_id_epochs
            if is_id_frozen:
                                                                              
                current_loss_w = {
                    'code_loss': loss_w['code_loss'],
                    'recon_loss': 0.0,           
                    'vq_loss': 0.0,              
                    'qs_loss': 0.0,              
                    'balance_loss': 0.0,           
                    'gate_loss': 0.0,           
                }
            else:
                                                       
                current_loss_w = loss_w

                   
            training_start_time = time()
            train_loss = self._train_epoch_rec(epoch_idx, loss_w=current_loss_w, freeze_id=is_id_frozen, verbose=verbose)
            training_end_time = time()

                                                                          
            if self.config.get('use_adaptive_selection', False) and not is_id_frozen:
                stats = model_id_unwrapped.get_adaptive_selection_stats()

                if stats['total_count'] > 0:
                    self.log(f"\n{'='*60}")
                    self.log(f"Epoch {epoch_idx} - 自适应选择统计:")
                    self.log(f"{'='*60}")

                                                                          
                    if stats.get('use_gate_network', False):
                        self.log(f"模式: Gate Network (基于embedding的可学习门控)")
                        self.log(f"门控策略: gate <= 0.5 使用Gumbel, gate > 0.5 使用确定性")
                        if stats.get('avg_gate_reg_loss') is not None:
                            self.log(f"平均Gate正则化Loss: {stats['avg_gate_reg_loss']:.6f}")
                    elif stats.get('use_soft_frequency', False):
                        self.log(f"模式: Soft Frequency Threshold (可学习)")
                        self.log(f"学习到的阈值: {stats['learned_threshold']:.6f} (平均频率: {1.0/256:.6f})")
                        self.log(f"阈值logit: {stats['threshold_logit']:.4f}")
                    else:
                        self.log(f"模式: Hard Threshold Ratio (固定)")
                        self.log(f"阈值比例: {self.config.get('hot_threshold_ratio', 1.5)}")

                    self.log(f"总样本数: {stats['total_count']}")
                    self.log(f"使用 Gumbel 采样 (热门code): {stats['gumbel_count']} ({stats['gumbel_ratio']*100:.2f}%)")
                    self.log(f"使用确定性索引 (冷门code): {stats['deterministic_count']} ({stats['deterministic_ratio']*100:.2f}%)")

                                                             
                    if 'per_layer' in stats and len(stats['per_layer']) > 0:
                        self.log(f"\n各量化层详细统计:")
                        for i, layer_stats in enumerate(stats['per_layer']):
                            if layer_stats['total_count'] > 0:
                                layer_info = f"  Layer {i}: Gumbel={layer_stats['gumbel_ratio']*100:>5.2f}%, " \
                                           f"Det={layer_stats['deterministic_ratio']*100:>5.2f}%, " \
                                           f"Total={layer_stats['total_count']}"
                                if layer_stats.get('use_soft_frequency', False):
                                    layer_info += f", Learned_α={layer_stats['learned_threshold']:.6f}"
                                self.log(layer_info)
                    self.log(f"{'='*60}\n")

            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )

            self.log(train_loss_output)
            self.log(f'[Epoch {epoch_idx}] REC lr: {self.rec_lr_scheduler.get_lr()}')
            if hasattr(self, 'id_lr_scheduler'):
                self.log(f'[Epoch {epoch_idx}] ID lr: {self.id_lr_scheduler.get_lr()}')

            self.accelerator.wait_for_everyone()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

                                                                  
            if end_to_end:
                all_item_code = self.get_code(epoch_idx=epoch_idx, verbose=verbose)
                self.all_item_code = torch.tensor(all_item_code).to(self.device)

                                                 
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if stop:
                break

                                                      
                                                              
                                                      
        legacy_frozen_key = 'stage' + '2'
        frozen_phase_epochs = self.config.get(f'{legacy_frozen_key}_epochs', 0)
        stage1_ckpt_path = self.config.get('stage1_checkpoint', None)

                                                              
        if stage1_ckpt_path is not None:
            self.log("")
            self.log("="*60)
            self.log("Loading provided Stage 1 checkpoint")
            self.log("="*60)
            self.log(f"Checkpoint: {stage1_ckpt_path}")
            self.log("")

                             
            safe_load(self.accelerator.unwrap_model(self.model_rec), stage1_ckpt_path, verbose=verbose)
            safe_load(self.accelerator.unwrap_model(self.model_id), stage1_ckpt_path+'.rqvae', verbose=verbose)

                        
            best_code = json.load(open(stage1_ckpt_path[:-3]+'.code.json'))
            self.all_item_code = torch.tensor(best_code).to(self.device)

                                                                  
            self.log("Evaluating loaded checkpoint on validation set...")
            initial_metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)
            self.best_score = initial_metrics[self.valid_metric]
            self.best_result = initial_metrics
            self.best_ckpt = stage1_ckpt_path
            self.log(f"Initial {self.valid_metric}: {self.best_score:.6f}")
            self.log(f"Initial Validation Results: {initial_metrics}")
            self.log("")

            self.stage1_test_results = None

                                                                   
        if stage1_ckpt_path is None:
                                                          
                                         
                                                          
            self.log("")
            self.log("="*60)
            self.log("Stage 1 Training Complete!")
            self.log(f"Best Stage 1 {self.valid_metric}: {self.best_score:.6f}")
            self.log(f"Best Stage 1 Validation Results: {self.best_result}")
            self.log("="*60)
            self.log("")

                                     
            self.log("Testing Stage 1 best model on test set...")
            stage1_test_results = self.test(verbose=verbose, model_file=self.best_ckpt)
            self.log("")
            self.log("="*60)
            self.log(f"Stage 1 Test Results: {stage1_test_results}")
            self.log("="*60)
            self.log("")

                                                        
            self.stage1_test_results = stage1_test_results

        if frozen_phase_epochs > 0 and end_to_end:
            self.log("")
            self.log("="*60)
            self.log("Starting frozen-tokenizer recommender training")
            self.log("="*60)
            self.log(f"Frozen-tokenizer training epochs: {frozen_phase_epochs}")

                                                                              
            if stage1_ckpt_path is not None:
                stage1_ckpt = stage1_ckpt_path
                self.log(f"Using already-loaded Stage 1 checkpoint: {stage1_ckpt}")
            else:
                                                            
                stage1_ckpt = self.best_ckpt
                self.log(f"Loading best checkpoint from Stage 1: {stage1_ckpt}")

                                 
                safe_load(self.accelerator.unwrap_model(self.model_rec), stage1_ckpt, verbose=verbose)
                safe_load(self.accelerator.unwrap_model(self.model_id), stage1_ckpt+'.rqvae', verbose=verbose)

                            
                best_code = json.load(open(stage1_ckpt[:-3]+'.code.json'))
                self.all_item_code = torch.tensor(best_code).to(self.device)

                                        
            self.log('[Frozen Tokenizer] ID Tokenizer COMPLETELY FROZEN')

            self.accelerator.unwrap_model(self.model_id).stop_gumbel_sampling_epoch = -1
            self.log('[Frozen Tokenizer] Force deterministic: stop_gumbel_sampling_epoch = -1')

                                                                     
            frozen_phase_loss_w = {
                'code_loss': loss_w.get('code_loss', 1.0),
                'recon_loss': 0.0,
                'vq_loss': 0.0,
                'qs_loss': 0.0,
                'balance_loss': 0.0,
                'gate_loss': 0.0,
                'kl_loss': 0.0,
                'dec_cl_loss': 0.0,
            }
            self.log(f'[Frozen Tokenizer] Loss weights: {frozen_phase_loss_w}')

                                                           
            frozen_phase_lr_rec = self.config.get(f'{legacy_frozen_key}_lr_rec', self.lr_rec)
            if frozen_phase_lr_rec != self.lr_rec:
                self.log(f'[Frozen Tokenizer] Updating learning rate: {self.lr_rec} -> {frozen_phase_lr_rec}')
                for param_group in self.rec_optimizer.param_groups:
                    param_group['lr'] = frozen_phase_lr_rec

                                                        
            frozen_phase_early_stop = self.config.get(f'{legacy_frozen_key}_early_stop', self.early_stop)

                                              
                                                                                          
            frozen_phase_best_score = self.best_score if self.best_score > 0 else 0.0
            frozen_phase_best_result = self.best_result if self.best_result else {}
            frozen_phase_best_ckpt = self.best_ckpt
            cur_eval_step = 0
            stop = False

            self.log("")
            self.log("[Frozen Tokenizer] Starting training loop...")
            self.log("")

                                   
            for epoch_idx in range(frozen_phase_epochs):
                self.accelerator.wait_for_everyone()

                                                             
                training_start_time = time()
                train_loss = self._train_epoch_rec(epoch_idx, loss_w=frozen_phase_loss_w, freeze_id=True, verbose=verbose)
                training_end_time = time()

                train_loss_output = self._generate_train_loss_output(
                    epoch_idx, training_start_time, training_end_time, train_loss
                )

                self.log(f"[Frozen Tokenizer Epoch {epoch_idx}] {train_loss_output}")
                self.log(f'[Frozen Tokenizer Epoch {epoch_idx}] REC lr: {self.rec_lr_scheduler.get_lr()}')

                          
                metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)

                if metrics[self.valid_metric] > frozen_phase_best_score:
                    frozen_phase_best_score = metrics[self.valid_metric]
                    frozen_phase_best_result = metrics
                    cur_eval_step = 0
                    frozen_phase_best_ckpt = self.safe_save(epoch_idx, self.all_item_code, prefix='frozen_tokenizer')
                    self.log(f'[Frozen Tokenizer Epoch {epoch_idx}] New best model saved!')
                else:
                    cur_eval_step += 1

                self.log(f'[Frozen Tokenizer Epoch {epoch_idx}] Val Results: {metrics}')
                self.log(f'[Frozen Tokenizer Epoch {epoch_idx}] Best {self.valid_metric}: {frozen_phase_best_score:.6f}')

                self.accelerator.wait_for_everyone()

                if cur_eval_step >= frozen_phase_early_stop:
                    self.log(f"[Frozen Tokenizer] Early stopping triggered at epoch {epoch_idx}")
                    stop = True
                    break

                                                      
            self.best_score = frozen_phase_best_score
            self.best_result = frozen_phase_best_result
            self.best_ckpt = frozen_phase_best_ckpt

            self.log("")
            self.log("="*60)
            self.log(f"Frozen-tokenizer training complete!")
            self.log(f"Best frozen-tokenizer {self.valid_metric}: {frozen_phase_best_score:.6f}")
            self.log(f"Best frozen-tokenizer validation results: {frozen_phase_best_result}")
            self.log("="*60)
            self.log("")

                                     
            self.log("Testing frozen-tokenizer best model on test set...")
            frozen_phase_test_results = self.test(verbose=verbose, model_file=frozen_phase_best_ckpt)
            self.log("")
            self.log("="*60)
            self.log(f"Frozen-tokenizer test results: {frozen_phase_test_results}")
            self.log("="*60)
            self.log("")

                                                                  
            if self.stage1_test_results is not None:
                self.log("="*60)
                self.log("Stage 1 vs frozen-tokenizer comparison:")
                self.log("="*60)
                self.log(f"Stage 1 Test Results: {self.stage1_test_results}")
                self.log(f"Frozen-tokenizer test results: {frozen_phase_test_results}")

                                        
                for metric_name in self.stage1_test_results.keys():
                    stage1_val = self.stage1_test_results[metric_name]
                    frozen_phase_val = frozen_phase_test_results[metric_name]
                    improvement = ((frozen_phase_val - stage1_val) / stage1_val * 100) if stage1_val > 0 else 0
                    self.log(f"{metric_name}: {stage1_val:.6f} -> {frozen_phase_val:.6f} ({improvement:+.2f}%)")
                self.log("="*60)
                self.log("")
            else:
                self.log("="*60)
                self.log("Stage 1 was skipped (used provided checkpoint)")
                self.log(f"Frozen-tokenizer test results: {frozen_phase_test_results}")
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
            safe_load(self.accelerator.unwrap_model(self.model_rec), ckpt_file, verbose=verbose)
            safe_load(self.accelerator.unwrap_model(self.model_id), ckpt_file+'.rqvae', verbose=verbose)

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

            if self.accelerator.num_processes > 1:
                preds = self.accelerator.unwrap_model(self.model_rec).generate(input_ids=input_ids, attention_mask=attention_mask, n_return_sequences=10)
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
        model_rec = self.accelerator.unwrap_model(self.model_rec)
        model_id = self.accelerator.unwrap_model(self.model_id)
        all_item_embs = model_rec.semantic_embedding.weight.data[1:]
        all_item_prefix = model_id.get_indices(all_item_embs).detach().cpu().numpy()


        if verbose:
            for i in range(self.code_length-1):
                self.log(f'[Epoch {epoch_idx}] Evaluation {self.save_path}/{epoch_idx}.pt Code balance {balance(all_item_prefix[:, i].tolist(), ncentroids=self.code_num)} Used code num of level {i+1}: {len(set(all_item_prefix[:, i].tolist()))}')

            self.log(f'[Epoch {epoch_idx}] Evaluation {self.save_path}/{epoch_idx}.pt Code confilct {conflict(all_item_prefix.tolist())}')

        all_item_prefix = all_item_prefix.tolist()

        tokens2item = defaultdict(list)
        all_item_tokens = [[-1] * self.code_length]                                
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
