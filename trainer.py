import os
import hashlib
import random
import signal
import subprocess
import sys
import tempfile
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
from dataclasses import dataclass
from colorama import init
from utils import ensure_dir, set_color, get_local_time
from accelerate import PartialState
from model import Model
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import get_scheduler
from metrics import *
from utils import *
from vq import (
    AutoSigmaGumbel,
    AutoSigmaGaussian,
    AutoSigmaSimple,
    TRUE_E2E_ASSIGNMENT_MODES,
)
from collections import defaultdict
from logging import getLogger
init(autoreset=True)


TRAINING_CHECKPOINT_VERSION = 1
_RESUME_RUNTIME_CONFIG_KEYS = {
    'accelerator',
    'config_path',
    'device',
    'evaluate_test_at_end',
    'allow_resume_code_mismatch',
    'allow_resume_config_mismatch',
    'resume_from',
    'run_local_time',
    'save_path',
    'stop_after_epoch',
    'use_ddp',
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def resumable_config_snapshot(config):
    return _json_safe({
        key: value
        for key, value in config.items()
        if key not in _RESUME_RUNTIME_CONFIG_KEYS
    })


def resumable_config_fingerprint(config):
    payload = json.dumps(
        resumable_config_snapshot(config), sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def semantic_id_map_sha256(item_codes):
    """Return a stable hash for a complete integer SID map."""
    array = np.asarray(item_codes, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def semantic_id_change_stats(previous_codes, current_codes, prefix_length):
    """Measure item-level SID changes, excluding the padding row."""
    previous = np.asarray(previous_codes, dtype=np.int64)
    current = np.asarray(current_codes, dtype=np.int64)
    if previous.shape != current.shape:
        raise ValueError(
            f'SID map shapes differ: {previous.shape} vs {current.shape}'
        )
    if previous.ndim != 2:
        raise ValueError('SID maps must be rank-2')
    if not 0 < int(prefix_length) < previous.shape[1]:
        raise ValueError('prefix_length must leave at least one suffix column')

    previous = previous[1:]
    current = current[1:]
    item_count = int(current.shape[0])

    def _rate(mask):
        changed = int(np.count_nonzero(mask))
        return {
            'changed_items': changed,
            'change_rate': changed / max(item_count, 1),
        }

    prefix_changed = np.any(
        previous[:, :prefix_length] != current[:, :prefix_length], axis=1
    )
    full_changed = np.any(previous != current, axis=1)
    suffix_changed = previous[:, prefix_length] != current[:, prefix_length]
    per_level = []
    for level_index in range(prefix_length):
        level = _rate(previous[:, level_index] != current[:, level_index])
        level['level'] = level_index + 1
        per_level.append(level)
    return {
        'item_count': item_count,
        'full': _rate(full_changed),
        'prefix': _rate(prefix_changed),
        'suffix': _rate(suffix_changed),
        'per_level': per_level,
    }


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    required = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    missing = required.difference(state)
    if missing:
        raise ValueError(f'RNG checkpoint is missing fields: {sorted(missing)}')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    cuda_states = state['torch_cuda']
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA RNG state cannot be restored without CUDA')
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                'CUDA RNG device count mismatch: '
                f'{len(cuda_states)} in checkpoint vs {torch.cuda.device_count()} visible'
            )
        torch.cuda.set_rng_state_all(cuda_states)


def atomic_torch_save(payload, path):
    path = os.path.abspath(os.fspath(path))
    ensure_dir(os.path.dirname(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=os.path.basename(path) + '.', suffix='.tmp',
            dir=os.path.dirname(path), delete=False,
        ) as file_obj:
            temporary = file_obj.name
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_save(payload, path):
    path = os.path.abspath(os.fspath(path))
    ensure_dir(os.path.dirname(path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8',
            prefix=os.path.basename(path) + '.', suffix='.tmp',
            dir=os.path.dirname(path), delete=False,
        ) as file_obj:
            temporary = file_obj.name
            json.dump(_json_safe(payload), file_obj, indent=2, sort_keys=True)
            file_obj.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class BatchItemIndex:
    unique_item_ids: torch.Tensor
    history_inverse: torch.Tensor
    target_inverse: torch.Tensor


def deduplicate_batch_items(input_ids, attention_mask, targets, padding_id=0):
    """Deduplicate valid history and target items for one local forward batch."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    valid_history_mask = attention_mask.bool() & input_ids.ne(padding_id)
    valid_history = input_ids[valid_history_mask]
    flat_targets = targets.reshape(-1)
    if flat_targets.eq(padding_id).any():
        raise ValueError("padding items cannot be recommendation targets")

    combined = torch.cat([valid_history, flat_targets], dim=0)
    unique_item_ids, inverse = torch.unique(
        combined, sorted=True, return_inverse=True
    )
    history_inverse = torch.full_like(input_ids, -1)
    history_inverse[valid_history_mask] = inverse[:valid_history.numel()]
    target_inverse = inverse[valid_history.numel():].view_as(targets)
    return BatchItemIndex(
        unique_item_ids=unique_item_ids,
        history_inverse=history_inverse,
        target_inverse=target_inverse,
    )


def cached_hard_straight_through(current_soft, cached_hard_ids):
    """Use cached hard semantic IDs in forward and current assignments in backward."""
    if current_soft.dim() != cached_hard_ids.dim() + 1:
        raise ValueError(
            "current_soft must have exactly one code dimension beyond cached_hard_ids"
        )
    if current_soft.shape[:-1] != cached_hard_ids.shape:
        raise ValueError("cached hard IDs must match the item and RQ-layer dimensions")
    if not current_soft.is_floating_point():
        raise TypeError("current_soft must be a floating-point probability tensor")
    code_number = current_soft.shape[-1]
    cached_hard_ids = cached_hard_ids.detach().long()
    if cached_hard_ids.numel() and (
        cached_hard_ids.min().item() < 0
        or cached_hard_ids.max().item() >= code_number
    ):
        raise ValueError("cached semantic IDs must be valid non-padding code indices")
    cached_one_hot = F.one_hot(
        cached_hard_ids, num_classes=code_number
    ).to(dtype=current_soft.dtype, device=current_soft.device)
    return cached_one_hot - current_soft.detach() + current_soft


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
        self.training_mode = config.get('training_mode', 'alternating_baseline')
        supported_training_modes = {'alternating_baseline'} | TRUE_E2E_ASSIGNMENT_MODES
        if self.training_mode not in supported_training_modes:
            raise ValueError(f'Unsupported training_mode: {self.training_mode}')
        self.is_true_e2e = self.training_mode in TRUE_E2E_ASSIGNMENT_MODES
        if self.is_true_e2e:
            if self.world_size != 1:
                raise ValueError('True-E2E modes are intentionally single-process only')
            if self.gradient_accumulation_steps != 1:
                raise ValueError(
                    'True-E2E modes currently require gradient_accumulation_steps=1'
                )
            if self.code_length != len(config['num_emb_list']) + 1:
                raise ValueError(
                    'True-E2E modes require one collision suffix after the RQ levels'
                )
            if self.training_mode != 'true_e2e_plain':
                if float(config.get('gumbel_noise_scale', 1.0)) < 0:
                    raise ValueError('gumbel_noise_scale must be non-negative')
            if 'sdud' in self.training_mode:
                if not config.get('use_learnable_sigma_gumbel', False):
                    raise ValueError('SDUD modes require use_learnable_sigma_gumbel=true')
                if not config.get('use_simple_uncertainty_loss', False):
                    raise ValueError('SDUD modes require direct learnable noise scale')
                if float(config.get('sigma_lambda', 0.0)) <= 0:
                    raise ValueError('SDUD modes require positive sigma_lambda')

        self.all_item_code = None
        self.model_rec.device = self.device

                                                      
        self.global_step = 0
        self.stop_requested = False
        self.stop_request_reason = None
        self.last_validation_metrics = None
        self.last_codebook_stats = None
        self.last_assignment_stats = None
        self.last_rec_gradient_report = None
        self.manifest_path = os.path.join(self.save_path, 'manifest.json')
        if self.is_true_e2e and hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._request_graceful_stop)

        self.all_metrics = config["metrics"].split(",")
        self.valid_metric = config["valid_metric"]
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())
        if int(config['num_beams']) < self.max_topk:
            raise ValueError(
                f'num_beams={config["num_beams"]} is smaller than the largest '
                f'requested metric cutoff {self.max_topk}'
            )

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
        self.best_epoch = None

        self.model_rec, self.rec_optimizer, self.rec_lr_scheduler, \
        self.model_id, self.train_data, self.valid_data, self.test_data = \
        self.accelerator.prepare(self.model_rec, self.rec_optimizer, self.rec_lr_scheduler,
                                 self.model_id, self.train_data, self.valid_data, self.test_data)
        self.process_seed = init_device_seed(config['seed'], self.accelerator.process_index)

    def _request_graceful_stop(self, signum, frame):
        del frame
        self.stop_requested = True
        self.stop_request_reason = f'signal_{signal.Signals(signum).name.lower()}'

    def _configure_trainable_parameters(self, model_rec, model_id):
        for param in model_rec.parameters():
            param.requires_grad = True
        if self.config.get('freeze_semantic_embedding', True):
            model_rec.semantic_embedding.requires_grad_(False)

        trains_tokenizer = (
            self.config.get('end_to_end', False)
            or self.config.get('training_mode') in TRUE_E2E_ASSIGNMENT_MODES
        )
        if not trains_tokenizer:
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
        module = getattr(model, module_name, None)
        if module is not None:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            self.log(f"  {module_name}: Total={total:,}, Trainable={trainable:,}")

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

    @staticmethod
    def _combined_gradient_norm(gradients):
        squared_norm = 0.0
        for gradient in gradients:
            if gradient is not None:
                squared_norm += gradient.detach().float().pow(2).sum().item()
        return math.sqrt(squared_norm)

    def _recommendation_gradient_report(self, semantic_loss, tokenizer_output):
        model_id = self.accelerator.unwrap_model(self.model_id)
        encoder_parameters = [
            parameter for parameter in model_id.encoder.parameters()
            if parameter.requires_grad
        ]
        codebook_parameters = [
            quantizer.embedding.weight for quantizer in model_id.rq.vq_layers
            if quantizer.embedding.weight.requires_grad
        ]
        assignment_logits = [
            level.assignment_logits for level in tokenizer_output.levels
        ]
        requested = encoder_parameters + codebook_parameters + assignment_logits
        gradients = torch.autograd.grad(
            semantic_loss,
            requested,
            retain_graph=True,
            allow_unused=True,
        )

        encoder_end = len(encoder_parameters)
        codebook_end = encoder_end + len(codebook_parameters)
        report = {
            'encoder': self._combined_gradient_norm(gradients[:encoder_end]),
            'codebooks': [
                self._combined_gradient_norm([gradient])
                for gradient in gradients[encoder_end:codebook_end]
            ],
            'assignments': [
                self._combined_gradient_norm([gradient])
                for gradient in gradients[codebook_end:]
            ],
        }
        required = report['codebooks'] + report['assignments']
        if encoder_parameters:
            required = [report['encoder']] + required
        if any(not math.isfinite(value) or value <= 1e-10 for value in required):
            raise RuntimeError(
                f'recommendation-only gradient contract failed: {report}'
            )
        self.last_rec_gradient_report = report
        return report

    @staticmethod
    def _true_e2e_semantic_loss(raw_semantic_loss):
        """UD changes assignment noise, never recommendation-loss scale."""
        return raw_semantic_loss

    def _train_epoch_true_e2e(self, epoch_idx, verbose=True):
        """Minimal single-process True-E2E assignment training."""
        if not hasattr(self, 'id_optimizer'):
            raise RuntimeError('True-E2E training requires a tokenizer optimizer')
        self.model_rec.train()
        self.model_id.train()
        model_rec = self.accelerator.unwrap_model(self.model_rec)
        model_id = self.accelerator.unwrap_model(self.model_id)
        model_id.reset_adaptive_selection_stats()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        total_num = 0
        total_loss = defaultdict(float)
        iterator = tqdm(
            self.train_data,
            total=len(self.train_data),
            ncols=100,
            desc=set_color(f'True-E2E {epoch_idx}', 'pink'),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )
        assignment_forward = self.config.get('assignment_forward', 'hard_st')
        if assignment_forward not in {'hard_st', 'soft'}:
            raise ValueError(
                f'Unsupported assignment_forward: {assignment_forward}'
            )
        for batch_idx, batch in enumerate(iterator):
            self.rec_optimizer.zero_grad()
            self.id_optimizer.zero_grad()

            raw_input_ids = batch['input_ids'].to(self.device)
            raw_attention_mask = batch['attention_mask'].to(self.device).bool()
            targets = batch['targets'].to(self.device)
            batch_index = deduplicate_batch_items(
                raw_input_ids,
                raw_attention_mask,
                targets,
                padding_id=self.pad_token_id,
            )

            unique_embeddings = model_rec.semantic_embedding(
                batch_index.unique_item_ids
            )
            if self.config.get('use_tau_annealing', False):
                assignment_temperature = model_id.get_current_tau(
                    self.global_step
                )
            else:
                assignment_temperature = float(
                    self.config.get('assignment_temperature', 2.0)
                )
            tokenizer_output = self.model_id(
                unique_embeddings,
                current_epoch=epoch_idx,
                global_step=self.global_step,
                return_structured=True,
                deterministic_st=self.training_mode == 'true_e2e_plain',
                assignment_temperature=assignment_temperature,
                assignment_mode=self.training_mode,
                assignment_forward=assignment_forward,
            )
            current_soft_probabilities = torch.stack(
                [level.soft_probabilities for level in tokenizer_output.levels],
                dim=1,
            )
            sampled_forward_probabilities = torch.stack(
                [
                    level.forward_probabilities
                    for level in tokenizer_output.levels
                ],
                dim=1,
            )
            cached_semantic_ids = self.all_item_code[
                batch_index.unique_item_ids, :current_soft_probabilities.shape[1]
            ].detach()
            use_cached_sinkhorn_forward = bool(
                self.config.get('use_cached_sinkhorn_forward', True)
            )
            if use_cached_sinkhorn_forward:
                unique_probabilities = cached_hard_straight_through(
                    current_soft_probabilities, cached_semantic_ids
                )
            else:
                unique_probabilities = sampled_forward_probabilities
            history_probabilities = unique_probabilities[
                batch_index.history_inverse.clamp_min(0)
            ]
            target_probabilities = unique_probabilities[
                batch_index.target_inverse
            ].squeeze(1)

            history_suffix = self.all_item_code[raw_input_ids, -1]
            target_suffix = self.all_item_code[targets.reshape(-1), -1]
            inputs_embeds, expanded_attention_mask = (
                model_rec.get_mixture_input_embeddings(
                    history_probabilities,
                    history_suffix,
                    raw_attention_mask,
                )
            )
            decoder_inputs_embeds = model_rec.get_differentiable_decoder_inputs(
                target_probabilities
            )
            outputs = self.model_rec(
                inputs_embeds=inputs_embeds,
                attention_mask=expanded_attention_mask,
                decoder_inputs_embeds=decoder_inputs_embeds,
            )
            raw_semantic_loss, suffix_loss = (
                model_rec.compute_differentiable_code_losses(
                    outputs.logits,
                    target_probabilities,
                    target_suffix,
                    suffix_logits=outputs.suffix_logits,
                )
            )
            semantic_loss = self._true_e2e_semantic_loss(raw_semantic_loss)
            target_positions = batch_index.target_inverse.reshape(-1)
            recon_loss_per_item = (
                tokenizer_output.quantized - tokenizer_output.latent
            ).pow(2).mean(dim=-1)
            recon_loss = recon_loss_per_item[target_positions].mean()
            vq_loss = tokenizer_output.vq_loss_per_item[
                target_positions
            ].mean()
            code_loss_weight = float(self.config.get('code_loss_weight', 1.0))
            recon_loss_weight = float(self.config.get('recon_loss_weight', 1.0))
            vq_loss_weight = float(self.config.get('vq_loss_weight', 1.0))
            loss = (
                code_loss_weight * (semantic_loss + suffix_loss)
                + recon_loss_weight * recon_loss
                + vq_loss_weight * vq_loss
            )

            if batch_idx == 0:
                self.log(
                    '[Recommendation Forward] '
                    f'assignment_forward={assignment_forward}, '
                    f'cached_sinkhorn_forward={use_cached_sinkhorn_forward}, '
                    f'temperature={assignment_temperature:.6f}'
                )
                gradient_report = self._recommendation_gradient_report(
                    semantic_loss, tokenizer_output
                )
                self.log(
                    '[Gradient Attribution] recommendation-only '
                    f'encoder={gradient_report["encoder"]:.8e}, '
                    f'codebooks={gradient_report["codebooks"]}, '
                    f'assignments={gradient_report["assignments"]}'
                )
                sampled_hard_indices = torch.stack(
                    [level.hard_indices.detach() for level in tokenizer_output.levels],
                    dim=1,
                )
                raw_argmax_indices = torch.stack(
                    [
                        level.assignment_logits.detach().argmax(dim=-1)
                        for level in tokenizer_output.levels
                    ],
                    dim=1,
                )
                train_forward_indices = unique_probabilities.detach().argmax(dim=-1)
                sample_prefix_vs_cached = sampled_hard_indices.eq(
                    cached_semantic_ids
                ).all(dim=-1).float().mean().item()
                train_prefix_vs_cached = train_forward_indices.eq(
                    cached_semantic_ids
                ).all(dim=-1).float().mean().item()
                assignment_stats = []
                for level_index, level in enumerate(tokenizer_output.levels):
                    probabilities = level.soft_probabilities.detach().float()
                    entropy = -(
                        probabilities
                        * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1).mean().item()
                    max_probability = probabilities.max(dim=-1).values.mean().item()
                    usage = torch.bincount(
                        level.hard_indices.detach().reshape(-1),
                        minlength=probabilities.shape[-1],
                    )
                    dead_codes = usage.eq(0).sum().item()
                    sample_vs_raw_argmax = sampled_hard_indices[:, level_index].eq(
                        raw_argmax_indices[:, level_index]
                    ).float().mean().item()
                    sample_vs_cached_sinkhorn = sampled_hard_indices[:, level_index].eq(
                        cached_semantic_ids[:, level_index]
                    ).float().mean().item()
                    train_hard_vs_cached_sinkhorn = train_forward_indices[:, level_index].eq(
                        cached_semantic_ids[:, level_index]
                    ).float().mean().item()
                    level_stats = {
                        'level': level_index + 1,
                        'entropy': entropy,
                        'max_probability': max_probability,
                        'used': int((usage > 0).sum().item()),
                        'dead': int(dead_codes),
                        'sample_vs_raw_argmax': sample_vs_raw_argmax,
                        'sample_vs_cached_sinkhorn': sample_vs_cached_sinkhorn,
                        'train_hard_vs_cached_sinkhorn': train_hard_vs_cached_sinkhorn,
                        'sample_prefix_vs_cached_sinkhorn': sample_prefix_vs_cached,
                        'train_prefix_vs_cached_sinkhorn': train_prefix_vs_cached,
                        'temperature': float(assignment_temperature),
                    }
                    uncertainty_parts = []
                    if level.noise_scale is not None:
                        level_stats['noise_scale'] = (
                            level.noise_scale.detach().float().mean().item()
                        )
                        uncertainty_parts.append(
                            f'noise_scale={level_stats["noise_scale"]:.6f}'
                        )
                    if level.effective_noise_scale is not None:
                        effective_scale = level.effective_noise_scale.detach().float()
                        level_stats['effective_noise_mean'] = effective_scale.mean().item()
                        uncertainty_parts.append(
                            f'effective_noise_mean={level_stats["effective_noise_mean"]:.6f}'
                        )
                    if level.frequency_scores is not None:
                        frequency = level.frequency_scores.detach().float()
                        level_stats['frequency_mean'] = frequency.mean().item()
                        uncertainty_parts.append(
                            f'frequency_mean={level_stats["frequency_mean"]:.6f}'
                        )
                    if level.stochastic_mask is not None:
                        level_stats['stochastic_ratio'] = (
                            level.stochastic_mask.detach().float().mean().item()
                        )
                        uncertainty_parts.append(
                            'stochastic_ratio='
                            f'{level_stats["stochastic_ratio"]:.6f}'
                        )
                    assignment_stats.append(level_stats)
                    self.log(
                        f'[Assignment L{level_index}] entropy={entropy:.6f}, '
                        f'max_probability={max_probability:.6f}, '
                        f'used={(usage > 0).sum().item()}, dead={dead_codes}, '
                        f'sample_vs_raw_argmax={sample_vs_raw_argmax:.6f}, '
                        f'sample_vs_cached_sinkhorn={sample_vs_cached_sinkhorn:.6f}, '
                        f'train_hard_vs_cached_sinkhorn={train_hard_vs_cached_sinkhorn:.6f}, '
                        f'temperature={assignment_temperature:.6f}'
                        + (
                            ', ' + ', '.join(uncertainty_parts)
                            if uncertainty_parts else ''
                        )
                    )
                self.log(
                    '[Assignment Prefix] '
                    f'sample_vs_cached_sinkhorn={sample_prefix_vs_cached:.6f}, '
                    f'train_hard_vs_cached_sinkhorn={train_prefix_vs_cached:.6f}'
                )
                self.last_assignment_stats = assignment_stats

            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1.0)
            self.accelerator.clip_grad_norm_(self.model_id.parameters(), 1.0)
            self.rec_optimizer.step()
            self.id_optimizer.step()
            self.rec_lr_scheduler.step()
            self.id_lr_scheduler.step()
            self.global_step += 1

            values = {
                'loss': loss.detach(),
                'semantic_loss': semantic_loss.detach(),
                'raw_semantic_loss': raw_semantic_loss.detach(),
                'suffix_loss': suffix_loss.detach(),
                'recon_loss': recon_loss.detach(),
                'vq_loss': vq_loss.detach(),
                'unique_items': torch.tensor(
                    float(batch_index.unique_item_ids.numel()),
                    device=self.device,
                ),
            }
            for name, value in values.items():
                total_loss[name] += value.float().item()
            total_num += 1
            iterator.set_postfix(loss=values['loss'].item())

        if total_num == 0:
            raise RuntimeError('True-E2E training received an empty training loader')
        for name in total_loss:
            total_loss[name] = round(total_loss[name] / total_num, 6)
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            total_loss['peak_memory_mb'] = round(peak_memory_mb, 2)
            self.log(f'[Memory] peak allocated={peak_memory_mb:.2f} MiB')

        self.accelerator.wait_for_everyone()
        return dict(total_loss)

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

    def _repo_metadata(self):
        repo_root = os.path.dirname(os.path.abspath(__file__))

        def run_git(*arguments):
            completed = subprocess.run(
                ['git', *arguments], cwd=repo_root,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
            return completed.stdout if completed.returncode == 0 else b''

        head = run_git('rev-parse', 'HEAD').decode('utf-8').strip() or None
        status = run_git('status', '--short').decode('utf-8').splitlines()
        diff = run_git('diff', '--binary', '--no-ext-diff')
        return {
            'repo_root': repo_root,
            'git_head': head,
            'git_status': status,
            'git_diff_sha256': hashlib.sha256(diff).hexdigest(),
        }

    def _initialize_manifest(self, resume_from=None):
        if not self.accelerator.is_main_process:
            return
        repo = self._repo_metadata()
        manifest = {
            'schema_version': 1,
            'status': 'running',
            'created_at': get_local_time(),
            'updated_at': get_local_time(),
            'dataset': self.config['dataset'],
            'seed': int(self.config['seed']),
            'save_path': os.path.abspath(self.save_path),
            'config_path': self.config.get('config_path'),
            'resolved_config': resumable_config_snapshot(self.config),
            'config_fingerprint': resumable_config_fingerprint(self.config),
            'command': [sys.executable, *sys.argv],
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'data_path': os.path.abspath(self.config['data_path']),
            'initial_rqvae_checkpoint': self.config.get('rqvae_path'),
            'resume_from': os.path.abspath(resume_from) if resume_from else None,
            'best_epoch': self.best_epoch,
            'best_validation': _json_safe(self.best_result),
            'best_checkpoint': self.best_ckpt,
            **repo,
        }
        atomic_json_save(manifest, self.manifest_path)

    def _update_manifest(self, **updates):
        if not self.accelerator.is_main_process:
            return
        manifest = {}
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, 'r', encoding='utf-8') as file_obj:
                manifest = json.load(file_obj)
        safe_updates = _json_safe(updates)
        for key, value in safe_updates.items():
            if isinstance(value, dict) and isinstance(manifest.get(key), dict):
                manifest[key].update(value)
            else:
                manifest[key] = value
        manifest['updated_at'] = get_local_time()
        manifest['best_epoch'] = self.best_epoch
        manifest['best_validation'] = _json_safe(self.best_result)
        manifest['best_checkpoint'] = self.best_ckpt
        atomic_json_save(manifest, self.manifest_path)

    def _training_checkpoint_payload(self, epoch, cur_eval_step, legacy_checkpoint):
        if self.all_item_code is None:
            raise RuntimeError('Cannot save resumable checkpoint without all_item_code')
        model_rec = self.accelerator.unwrap_model(self.model_rec)
        model_id = self.accelerator.unwrap_model(self.model_id)
        payload = {
            'format': 'diger_true_e2e_training_state',
            'version': TRAINING_CHECKPOINT_VERSION,
            'epoch': int(epoch),
            'next_epoch': int(epoch) + 1,
            'global_step': int(self.global_step),
            'cur_eval_step': int(cur_eval_step),
            'best_score': float(self.best_score),
            'best_result': _json_safe(self.best_result),
            'best_epoch': self.best_epoch,
            'best_checkpoint': (
                os.path.abspath(self.best_ckpt) if self.best_ckpt else None
            ),
            'legacy_checkpoint': os.path.abspath(legacy_checkpoint),
            'last_validation_metrics': _json_safe(self.last_validation_metrics),
            'last_codebook_stats': _json_safe(self.last_codebook_stats),
            'last_assignment_stats': _json_safe(self.last_assignment_stats),
            'last_rec_gradient_report': _json_safe(self.last_rec_gradient_report),
            'all_item_code': self.all_item_code.detach().cpu(),
            'model_rec': model_rec.state_dict(),
            'model_id': model_id.state_dict(),
            'rec_optimizer': self.rec_optimizer.state_dict(),
            'rec_scheduler': self.rec_lr_scheduler.state_dict(),
            'id_optimizer': (
                self.id_optimizer.state_dict() if hasattr(self, 'id_optimizer') else None
            ),
            'id_scheduler': (
                self.id_lr_scheduler.state_dict()
                if hasattr(self, 'id_lr_scheduler') else None
            ),
            'rng_state': capture_rng_state(),
            'config': resumable_config_snapshot(self.config),
            'config_fingerprint': resumable_config_fingerprint(self.config),
            'code_state': self._repo_metadata(),
        }
        return payload

    def _save_resume_checkpoint(
        self, epoch, cur_eval_step, legacy_checkpoint, resume_path=None,
    ):
        self.accelerator.wait_for_everyone()
        if resume_path is None:
            resume_path = f'{legacy_checkpoint}.resume'
        resume_path = os.path.abspath(resume_path)
        if self.accelerator.is_main_process:
            payload = self._training_checkpoint_payload(
                epoch, cur_eval_step, legacy_checkpoint
            )
            atomic_torch_save(payload, resume_path)
            self.log(f'[Epoch {epoch}] Save resumable state {resume_path}')
            self._update_manifest(
                latest_epoch=int(epoch),
                latest_validation=self.last_validation_metrics,
                latest_codebook_stats=self.last_codebook_stats,
                latest_resume_checkpoint=resume_path,
            )
        self.accelerator.wait_for_everyone()
        return resume_path

    @staticmethod
    def _config_mismatches(saved, current):
        keys = sorted(set(saved) | set(current))
        return {
            key: {'saved': saved.get(key), 'current': current.get(key)}
            for key in keys
            if saved.get(key) != current.get(key)
        }

    def _load_resume_checkpoint(self, resume_path):
        resume_path = os.path.abspath(os.fspath(resume_path))
        checkpoint = load_torch_checkpoint(resume_path, map_location='cpu')
        if checkpoint.get('format') != 'diger_true_e2e_training_state':
            raise ValueError(f'Not a DIGER resumable checkpoint: {resume_path}')
        if checkpoint.get('version') != TRAINING_CHECKPOINT_VERSION:
            raise ValueError(
                f'Unsupported training checkpoint version: {checkpoint.get("version")}'
            )

        current_config = resumable_config_snapshot(self.config)
        saved_config = checkpoint.get('config', {})
        if checkpoint.get('config_fingerprint') != resumable_config_fingerprint(
            self.config
        ):
            mismatches = self._config_mismatches(saved_config, current_config)
            if not self.config.get('allow_resume_config_mismatch', False):
                raise ValueError(
                    'Resume configuration mismatch; exact continuation refused: '
                    f'{mismatches}'
                )
            self.log(f'[Resume] WARNING configuration mismatch allowed: {mismatches}')

        saved_code = checkpoint.get('code_state', {})
        current_code = self._repo_metadata()
        code_keys = ('git_head', 'git_diff_sha256')
        code_mismatches = {
            key: {'saved': saved_code.get(key), 'current': current_code.get(key)}
            for key in code_keys
            if saved_code.get(key) != current_code.get(key)
        }
        if code_mismatches and not self.config.get(
            'allow_resume_code_mismatch', False
        ):
            raise ValueError(
                'Resume code mismatch; exact continuation refused: '
                f'{code_mismatches}'
            )
        if code_mismatches:
            self.log(f'[Resume] WARNING code mismatch allowed: {code_mismatches}')

        model_rec = self.accelerator.unwrap_model(self.model_rec)
        model_id = self.accelerator.unwrap_model(self.model_id)
        model_rec.load_state_dict(checkpoint['model_rec'], strict=True)
        model_id.load_state_dict(checkpoint['model_id'], strict=True)
        self.rec_optimizer.load_state_dict(checkpoint['rec_optimizer'])
        self.rec_lr_scheduler.load_state_dict(checkpoint['rec_scheduler'])
        if checkpoint.get('id_optimizer') is not None:
            if not hasattr(self, 'id_optimizer'):
                raise RuntimeError('Checkpoint contains ID optimizer but trainer does not')
            self.id_optimizer.load_state_dict(checkpoint['id_optimizer'])
        if checkpoint.get('id_scheduler') is not None:
            if not hasattr(self, 'id_lr_scheduler'):
                raise RuntimeError('Checkpoint contains ID scheduler but trainer does not')
            self.id_lr_scheduler.load_state_dict(checkpoint['id_scheduler'])

        self.global_step = int(checkpoint['global_step'])
        self.best_score = float(checkpoint['best_score'])
        self.best_result = checkpoint.get('best_result', {})
        self.best_epoch = checkpoint.get('best_epoch')
        self.best_ckpt = checkpoint.get('best_checkpoint')
        self.last_validation_metrics = checkpoint.get('last_validation_metrics')
        self.last_codebook_stats = checkpoint.get('last_codebook_stats')
        self.last_assignment_stats = checkpoint.get('last_assignment_stats')
        self.last_rec_gradient_report = checkpoint.get('last_rec_gradient_report')
        self.all_item_code = checkpoint['all_item_code'].to(self.device)
        restore_rng_state(checkpoint['rng_state'])
        next_epoch = int(checkpoint['next_epoch'])
        cur_eval_step = int(checkpoint['cur_eval_step'])
        self.log(
            f'[Resume] Strictly restored {resume_path}: next_epoch={next_epoch}, '
            f'global_step={self.global_step}, best_epoch={self.best_epoch}, '
            f'best_{self.valid_metric}={self.best_score:.6f}'
        )
        return next_epoch, cur_eval_step

    def _save_latest_resumable_bundle(self, epoch, cur_eval_step, reason):
        legacy_checkpoint = self.safe_save(
            epoch, self.all_item_code, prefix='latest'
        )
        resume_path = self._save_resume_checkpoint(
            epoch, cur_eval_step, legacy_checkpoint
        )
        self._update_manifest(
            status='completed' if reason == 'max_epochs' else 'stopped',
            stop_reason=reason,
            stopped_epoch=int(epoch),
            latest_checkpoint=os.path.abspath(legacy_checkpoint),
            latest_resume_checkpoint=resume_path,
        )
        return legacy_checkpoint, resume_path

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

    def _is_best_checkpoint_candidate(self, metrics):
        return (
            self.best_ckpt is None
            or metrics[self.valid_metric] > self.best_score
        )

    def train(self, verbose=True):
        stop = False
        cur_eval_step = 0
        self.best_score = 0
        self.best_result = {}
        self.best_ckpt = None
        self.best_epoch = None
        loss_w = defaultdict(int)

        resume_from = self.config.get('resume_from')

        if resume_from is None:
            all_item_code = self.get_code(epoch_idx=-1, verbose=verbose)
            self.all_item_code = torch.tensor(all_item_code).to(self.device)

                                         
        true_e2e = self.is_true_e2e
        end_to_end = self.config.get('end_to_end', False) or true_e2e
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

            if true_e2e:
                self.log(
                    '[Training Mode] Single-GPU True-E2E enabled; '
                    f'assignment_mode={self.training_mode}'
                )
                self.log(
                    '[Training Mode] Objective: semantic recommendation + hard suffix '
                    '+ latent reconstruction + VQ'
                )
            else:
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
            direct_sigma = self.config.get('use_simple_uncertainty_loss', False)
            sigma_parameterization = (
                'Direct Non-negative Scale'
                if direct_sigma else 'Base-2 Exponential'
            )
            self.log(
                f"========== Learnable Sigma Parameters ({sigma_parameterization}) =========="
            )
            for name in sigma_params:
                param = dict(model_id_unwrapped.named_parameters())[name]
                sigma_val = param.data.item()
                self.log(f"  {name}: σ={sigma_val:.6f}, requires_grad={param.requires_grad}")
                if direct_sigma:
                    self.log(f"    -> Noise scale: s = |σ| = {abs(sigma_val):.6f}")
                else:
                    s_val = 2 ** sigma_val
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

        start_epoch = 0
        if resume_from is not None:
            start_epoch, cur_eval_step = self._load_resume_checkpoint(resume_from)
            if start_epoch >= self.epochs:
                raise ValueError(
                    f'Resume next_epoch {start_epoch} is not below epochs={self.epochs}'
                )
        self._initialize_manifest(resume_from=resume_from)

        for epoch_idx in range(start_epoch, self.epochs):
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
            if true_e2e:
                train_loss = self._train_epoch_true_e2e(
                    epoch_idx, verbose=verbose
                )
            else:
                train_loss = self._train_epoch_rec(
                    epoch_idx,
                    loss_w=current_loss_w,
                    freeze_id=is_id_frozen,
                    verbose=verbose,
                )
            training_end_time = time()

                                                                          
            if (
                self.config.get('use_adaptive_selection', False)
                or 'frqud' in self.training_mode
            ) and not is_id_frozen:
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
            self.last_validation_metrics = total_metrics

            is_best = self._is_best_checkpoint_candidate(total_metrics)
            if is_best:
                self.best_score = total_metrics[self.valid_metric]
                self.best_result = total_metrics
                self.best_epoch = epoch_idx
                cur_eval_step = 0
                self.best_ckpt = self.safe_save(epoch_idx, self.all_item_code)
            else:
                cur_eval_step += 1

            if cur_eval_step >= self.early_stop:
                stop = True

            self.log(f'[Epoch {epoch_idx}] Val Results: {total_metrics}')

            if is_best:
                self._save_resume_checkpoint(
                    epoch_idx, cur_eval_step, self.best_ckpt
                )

            self.accelerator.wait_for_everyone()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            stop_after_epoch = int(self.config.get('stop_after_epoch', -1))
            stop_reason = None
            if stop:
                stop_reason = 'early_stop'
            elif self.stop_requested:
                stop_reason = self.stop_request_reason or 'external_request'
            elif stop_after_epoch >= 0 and epoch_idx >= stop_after_epoch:
                stop_reason = 'requested_epoch_limit'
                self.log(
                    f'[Training] Requested stop after epoch {epoch_idx}; '
                    'the scheduler horizon remains configured by epochs.'
                )
            elif epoch_idx == self.epochs - 1:
                stop_reason = 'max_epochs'

            if stop_reason is not None:
                self.log(
                    f'[Training] Saving latest resumable checkpoint before '
                    f'stop: reason={stop_reason}'
                )
                self._save_latest_resumable_bundle(
                    epoch_idx, cur_eval_step, stop_reason
                )
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

                                     
            if self.config.get('evaluate_test_at_end', True):
                self.log("Testing Stage 1 best model on test set...")
                stage1_test_results = self.test(
                    verbose=verbose, model_file=self.best_ckpt
                )
                self.log("")
                self.log("="*60)
                self.log(f"Stage 1 Test Results: {stage1_test_results}")
                self.log("="*60)
                self.log("")
                self.stage1_test_results = stage1_test_results
            else:
                self.stage1_test_results = None
                self.log(
                    '[Screening] Test evaluation disabled; checkpoint '
                    'selection used validation only.'
                )

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

                                     
            if not self.config.get('evaluate_test_at_end', True):
                self.log(
                    '[Screening] Frozen-tokenizer test evaluation disabled.'
                )
                return self.best_score

            self.log("Testing frozen-tokenizer best model on test set...")
            frozen_phase_test_results = self.test(
                verbose=verbose, model_file=frozen_phase_best_ckpt
            )
            self.log("")
            self.log("="*60)
            self.log(f"Frozen-tokenizer test results: {frozen_phase_test_results}")
            self.log("="*60)
            self.log("")
            self.final_test_results = frozen_phase_test_results

                                                                  
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
        max_eval_batches = int(self.config.get('max_eval_batches', 0))

        for batch_idx, data in enumerate(iter_data):
            if max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
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
        all_item_prefix = model_id.get_indices(
            all_item_embs, use_sinkhorn=True
        ).detach().cpu().numpy()

        sinkhorn_stats = []
        for level_index in range(self.code_length - 1):
            level_codes = all_item_prefix[:, level_index].tolist()
            used = len(set(level_codes))
            sinkhorn_stats.append(
                {
                    'level': level_index + 1,
                    'used': used,
                    'dead': self.code_num - used,
                    'balance': balance(level_codes, ncentroids=self.code_num),
                }
            )

        nearest_stats = None
        if self.is_true_e2e:
            nearest_prefix = model_id.get_indices(
                all_item_embs, use_sinkhorn=False
            ).detach().cpu().numpy()
            nearest_stats = []
            for level_index in range(self.code_length - 1):
                level_codes = nearest_prefix[:, level_index].tolist()
                nearest_stats.append(
                    {
                        'level': level_index + 1,
                        'used': len(set(level_codes)),
                        'dead': self.code_num - len(set(level_codes)),
                        'balance': balance(
                            level_codes, ncentroids=self.code_num
                        ),
                    }
                )
            self.log(
                f'[Epoch {epoch_idx}] [TOKENIZER] nearest-code diagnostic '
                f'(not used for evaluation): {nearest_stats}'
            )


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
        collision_prefixes = sum(
            len(item_ids) > 1 for item_ids in tokens2item.values()
        )
        collision_items = sum(
            len(item_ids) for item_ids in tokens2item.values()
            if len(item_ids) > 1
        )
        item_count = max(len(all_item_prefix), 1)
        previous_change = None
        if self.all_item_code is not None:
            previous_codes = self.all_item_code
            if torch.is_tensor(previous_codes):
                previous_codes = previous_codes.detach().cpu().numpy()
            previous_change = semantic_id_change_stats(
                previous_codes,
                all_item_tokens,
                prefix_length=self.code_length - 1,
            )

        map_hash = semantic_id_map_sha256(all_item_tokens)
        self.last_codebook_stats = {
            'epoch': int(epoch_idx),
            'sinkhorn': sinkhorn_stats,
            'nearest': nearest_stats,
            'prefix_unique': len(tokens2item),
            'collision_prefixes': collision_prefixes,
            'collision_items': collision_items,
            'collision_item_rate': collision_items / item_count,
            'maximum_conflict': max_conflict,
            'suffix_used': max_conflict,
            'map_sha256': map_hash,
            'previous_epoch_change': previous_change,
        }
        self.log(
            f'[Epoch {epoch_idx}] [SID MAP] map_sha256={map_hash}, '
            f'previous_change={previous_change}'
        )
        self.log(
            f'[Epoch {epoch_idx}] [TOKENIZER] prefix_unique={len(tokens2item)}, '
            f'collision_prefixes={collision_prefixes}, '
            f'collision_items={collision_items}, '
            f'collision_item_rate={collision_items / item_count:.6f}, '
            f'suffix_used={max_conflict}, suffix_exceeds_256={max_conflict > 256}'
        )
        suffix_capacity = model_rec.token_embeddings[-1].num_embeddings
        if max_conflict > suffix_capacity:
            raise RuntimeError(
                f'[TOKENIZER] RQ-VAE semantic IDs conflict with suffix capacity: '
                f'{max_conflict} > {suffix_capacity}. The run is invalid and was stopped.'
            )

        return all_item_tokens

    def log(self, message, level='info'):
        return log(message, self.accelerator, self.logger, level=level)
