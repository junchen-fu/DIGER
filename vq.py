import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional, Tuple
from layers import *


TRUE_E2E_ASSIGNMENT_MODES = {
    "true_e2e_plain",
    "true_e2e_gumbel_fixed",
    "true_e2e_frqud",
    "true_e2e_sdud",
    "true_e2e_sdud_frqud",
}
TRUE_E2E_GUMBEL_MODES = TRUE_E2E_ASSIGNMENT_MODES - {"true_e2e_plain"}


@dataclass
class GumbelStraightThroughOutput:
    soft_probabilities: torch.Tensor
    hard_one_hot: torch.Tensor
    straight_through_probabilities: torch.Tensor
    hard_indices: torch.Tensor
    noisy_logits: torch.Tensor
    gumbel_noise: torch.Tensor


def deterministic_straight_through(logits, temperature=1.0):
    """Return a hard-forward, soft-backward categorical assignment."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    soft_probabilities = F.softmax(logits / temperature, dim=-1)
    hard_indices = soft_probabilities.argmax(dim=-1)
    hard_one_hot = F.one_hot(
        hard_indices, num_classes=logits.shape[-1]
    ).to(dtype=soft_probabilities.dtype)
    straight_through_probabilities = (
        hard_one_hot - soft_probabilities.detach() + soft_probabilities
    )
    return (
        soft_probabilities,
        hard_one_hot,
        straight_through_probabilities,
        hard_indices,
    )


def sample_gumbel_like(logits):
    uniform = torch.rand_like(logits)
    epsilon = torch.finfo(uniform.dtype).eps
    uniform = uniform.clamp(min=epsilon, max=1.0 - epsilon)
    return -torch.log(-torch.log(uniform))


def gumbel_straight_through(
    logits,
    temperature=1.0,
    noise_scale=1.0,
    add_noise=True,
    gumbel_noise=None,
):
    """Return hard-forward, soft-backward assignments with explicit noise scale."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scale = torch.as_tensor(noise_scale, dtype=logits.dtype, device=logits.device)
    if torch.any(scale < 0):
        raise ValueError("noise_scale must be non-negative")
    while scale.dim() < logits.dim():
        scale = scale.unsqueeze(-1)

    if add_noise:
        if gumbel_noise is None:
            gumbel_noise = sample_gumbel_like(logits)
        elif gumbel_noise.shape != logits.shape:
            raise ValueError("gumbel_noise must match logits")
        noisy_logits = (logits + scale * gumbel_noise) / temperature
    else:
        gumbel_noise = torch.zeros_like(logits)
        noisy_logits = logits / temperature

    soft_probabilities = F.softmax(noisy_logits, dim=-1)
    hard_indices = soft_probabilities.argmax(dim=-1)
    hard_one_hot = F.one_hot(
        hard_indices, num_classes=logits.shape[-1]
    ).to(dtype=soft_probabilities.dtype)
    straight_through_probabilities = (
        hard_one_hot - soft_probabilities.detach() + soft_probabilities
    )
    return GumbelStraightThroughOutput(
        soft_probabilities=soft_probabilities,
        hard_one_hot=hard_one_hot,
        straight_through_probabilities=straight_through_probabilities,
        hard_indices=hard_indices,
        noisy_logits=noisy_logits,
        gumbel_noise=gumbel_noise,
    )


@dataclass
class QuantizerLevelOutput:
    assignment_logits: torch.Tensor
    soft_probabilities: torch.Tensor
    hard_one_hot: torch.Tensor
    straight_through_probabilities: torch.Tensor
    forward_probabilities: torch.Tensor
    hard_indices: torch.Tensor
    residual_input: torch.Tensor
    quantized_vector: torch.Tensor
    vq_loss: torch.Tensor
    vq_loss_per_item: torch.Tensor
    balance_loss: Optional[torch.Tensor] = None
    gate_reg_loss: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None
    noise_scale: Optional[torch.Tensor] = None
    effective_noise_scale: Optional[torch.Tensor] = None
    frequency_scores: Optional[torch.Tensor] = None
    stochastic_mask: Optional[torch.Tensor] = None
    gumbel_noise: Optional[torch.Tensor] = None


@dataclass
class ResidualQuantizerOutput:
    quantized: torch.Tensor
    levels: Tuple[QuantizerLevelOutput, ...]
    vq_loss: torch.Tensor
    vq_loss_per_item: torch.Tensor
    indices: torch.Tensor
    balance_loss: Optional[torch.Tensor] = None
    gate_reg_loss: Optional[torch.Tensor] = None
    mean_sigma: Optional[torch.Tensor] = None


@dataclass
class RQVAEOutput:
    quantized: torch.Tensor
    latent: torch.Tensor
    levels: Tuple[QuantizerLevelOutput, ...]
    vq_loss: torch.Tensor
    vq_loss_per_item: torch.Tensor
    indices: torch.Tensor
    balance_loss: Optional[torch.Tensor] = None
    gate_reg_loss: Optional[torch.Tensor] = None
    mean_sigma: Optional[torch.Tensor] = None


class AutoSigmaGaussian(nn.Module):
    def __init__(self, config, temperature=1.0):
        super().__init__()
                                                          
                                                          
                                           
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
            if initial_std <= 1e-5:
                initial_sigma = -20.0                    
            else:
                import math
                initial_sigma = math.log2(initial_std)
        else:
            initial_sigma = config.get('initial_sigma', 1.0)

        self.sigma = nn.Parameter(torch.tensor(initial_sigma))

                                              
        self.temperature = temperature

    def forward(self, logits, tau=1.0, hard=False, dim=-1):
                                            
        sigma_value = self.sigma

                                                            
        s = torch.pow(2.0, sigma_value)
                                                               
        s = s.clamp(min=1e-5, max=100.0)

        if self.training:
                                                       
            gaussian = torch.randn_like(logits)

                                                
                                                                            
            if s.item() < 1e-4:
                noise = torch.zeros_like(logits)
            else:
                noise = gaussian * s

                                                       
            noisy_logits = (logits + noise) / tau

            y_soft = noisy_logits.softmax(dim)

            if hard:
                                            
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
                                               
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
        else:
                                           
            if hard:
                index = logits.max(dim, keepdim=True)[1]
                ret = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            else:
                                                              
                ret = (logits / tau).softmax(dim)

                                
        return ret, sigma_value

    @staticmethod
    def compute_uncertainty_loss(task_loss, sigma, reg_weight=1.0):
        k = 0.77
        c = 1.448783 * reg_weight

                                                  
        exp_term = task_loss * torch.exp(-k * sigma)
        reg_term = c * sigma

        return exp_term + reg_term


class AutoSigmaGumbel(nn.Module):
    def __init__(self, config, temperature=1.0):
        super().__init__()
                                                                                
                                                            
                                           
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
            if initial_std <= 1e-5:
                initial_sigma = -20.0                    
            else:
                import math
                initial_sigma = math.log2(initial_std)
        else:
            initial_sigma = config.get('initial_sigma', 1.0)

        self.sigma = nn.Parameter(torch.tensor(initial_sigma))

                                              
        self.temperature = temperature

    def forward(self, logits, tau=1.0, hard=False, dim=-1):
                                                                                     
                                                        
        sigma_value = self.sigma

                                                            
        s = torch.pow(2.0, sigma_value)
                                                               
        s = s.clamp(min=1e-5, max=100.0)                                                   

        if self.training:
                                   
            gumbels = -torch.empty_like(logits).exponential_().log()

                                                        
                                                                            
            if s.item() < 1e-4:
                noise = torch.zeros_like(logits)
            else:
                noise = gumbels * s

                                                       
            gumbel_logits = (logits + noise) / tau

            y_soft = gumbel_logits.softmax(dim)

            if hard:
                                            
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
                                               
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
        else:
                                
            if hard:
                index = logits.max(dim, keepdim=True)[1]
                ret = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            else:
                                                              
                ret = (logits / tau).softmax(dim)

                                                                
        return ret, sigma_value

    @staticmethod
    def compute_uncertainty_loss(task_loss, sigma, reg_weight=1.0,
                                 annealing_threshold=None,
                                 annealing_slow_k=None, annealing_slow_c=None,
                                 annealing_fast_k=None, annealing_fast_c=None):
                                                                     
                          

                                                                  
        threshold_loss = annealing_threshold if annealing_threshold is not None else 2.0

        if task_loss >= threshold_loss:
                              
            k = annealing_slow_k if annealing_slow_k is not None else 0.458145
            c = (annealing_slow_c if annealing_slow_c is not None else 1.361442) * reg_weight
        else:
                              
            k = annealing_fast_k if annealing_fast_k is not None else 0.018127
            c = (annealing_fast_c if annealing_fast_c is not None else 0.036916) * reg_weight

                                    
                                                  
        exp_term = task_loss * torch.exp(-k * sigma)
        reg_term = c * sigma

        return exp_term + reg_term


class AutoSigmaSimple(nn.Module):
    def __init__(self, config, temperature=1.0):
        super().__init__()
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
        else:
            initial_std = config.get('initial_sigma', 1.0)

        self.sigma = nn.Parameter(torch.tensor(initial_std))
        self.temperature = temperature

                          
        self.auto_lambda_mode = config.get('auto_lambda_mode', 'fixed')

        if self.auto_lambda_mode == 'learnable':
                                               
            initial_lambda = config.get('sigma_lambda', 1.8)
            self.lambda_param = nn.Parameter(torch.tensor(initial_lambda))
        elif self.auto_lambda_mode == 'adaptive':
                                                    
                                      
            self.register_buffer('loss_ema', torch.tensor(5.0))                              
            self.ema_momentum = config.get('lambda_ema_momentum', 0.99)
            self.lambda_param = None                                
        else:                
            self.lambda_param = None

    def forward(self, logits, tau=1.0, hard=False, dim=-1):
                                           
        sigma_value = self.sigma
                                                                    
        s = sigma_value.abs().clamp(min=1e-5, max=100.0)

        if self.training:
            gumbels = -torch.empty_like(logits).exponential_().log()

            if s.item() < 1e-4:
                noise = torch.zeros_like(logits)
            else:
                noise = gumbels * s

            gumbel_logits = (logits + noise) / tau
            y_soft = gumbel_logits.softmax(dim)

            if hard:
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
        else:
            if hard:
                index = logits.max(dim, keepdim=True)[1]
                ret = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            else:
                ret = (logits / tau).softmax(dim)

        return ret, sigma_value

    def update_lambda_ema(self, task_loss_value):
        if self.auto_lambda_mode == 'adaptive':
            with torch.no_grad():
                self.loss_ema = self.ema_momentum * self.loss_ema + (1 - self.ema_momentum) * task_loss_value

    def get_lambda(self):
        if self.auto_lambda_mode == 'learnable':
                                                                       
            return self.lambda_param.abs().clamp(min=0.5, max=3.0)
        elif self.auto_lambda_mode == 'adaptive':
                                                                                 
                                                                           
            loss_ema_clamped = self.loss_ema.clamp(min=0.25, max=25.0)
            return torch.sqrt(loss_ema_clamped)
        else:
                                                                 
            return None

    def compute_uncertainty_loss(self, task_loss, sigma, lambda_bias=0.5):
                                  
        lambda_value = self.get_lambda()
        if lambda_value is None:
            lambda_value = lambda_bias

                                        
        if self.auto_lambda_mode == 'adaptive' and self.training:
            self.update_lambda_ema(task_loss.detach())

                      
                                                                              
        effective_sigma = sigma.abs() + lambda_value
        effective_sigma = effective_sigma.clamp(min=1e-6)

        denom = 2 * (effective_sigma ** 2)
        mse_term = task_loss / denom
        log_term = torch.log(effective_sigma)

        return mse_term + log_term, lambda_value


@torch.no_grad()
def sinkhorn_algorithm(
    distances,
    epsilon,
    sinkhorn_iterations,
):
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]                              
    K = Q.shape[1]                                                    

                               
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
                    
    for it in range(sinkhorn_iterations):

                                                                    
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

                                                                    
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B                                                       
    return Q


class RQVAE(nn.Module):
    def __init__(self, config, in_dim=768,):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.e_dim = config['e_dim']

        self.layers = config['layers']
        self.dropout_prob = config['dropout_prob']
        self.bn = config['bn']
        self.quant_loss_weight = config['alpha']
        self.beta = config['beta']
        self.vq_type = config['vq_type']
        self.gumbel_tau = config.get('gumbel_tau', 1.0)
                                                                
        self.use_indicator_ste = config.get('use_indicator_ste', True)
                                                                         
                                                                                     
        self.stop_gumbel_sampling_epoch = config.get('stop_gumbel_sampling_epoch', 0)

                                   
                                                                                               
        self.use_tau_annealing = config.get('use_tau_annealing', False)
        self.tau_anneal_init = config.get('tau_anneal_init', 2.0)                   
        self.tau_anneal_min = config.get('tau_anneal_min', 0.5)                     
        self.tau_anneal_rate = config.get('tau_anneal_rate', 0.0003)               
        self.configured_tau_decay = config.get('tau_anneal_decay')
        if (
            self.configured_tau_decay is not None
            and not 0.0 < float(self.configured_tau_decay) <= 1.0
        ):
            raise ValueError('tau_anneal_decay must be in (0, 1]')
                                                                                 
        self.warmup_gumbel_epochs = config.get('warmup_gumbel_epochs', 0)
                                                                      
                                                        
                                                                                                                                  
        self.gumbel_hard_switch_epoch = config.get('gumbel_hard_switch_epoch', 50)

        if self.vq_type in ["vq"]:
            self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        else:
            raise NotImplementedError


        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.rq = ResidualVectorQuantizer(config=config)

    def get_current_tau(self, global_step):
        if not self.use_tau_annealing:
            return self.gumbel_tau

        decay = self.configured_tau_decay
        if decay is not None:
            tau = self.tau_anneal_init * (decay ** global_step)
        else:
            import math
            tau = self.tau_anneal_init * math.exp(-self.tau_anneal_rate * global_step)
        tau = max(self.tau_anneal_min, tau)
        return tau

    def forward(self, x, use_gumbel=False, current_epoch=0, global_step=0,
                return_latent=False, return_structured=False,
                deterministic_st=False, assignment_temperature=None,
                assignment_mode=None, assignment_forward="hard_st"):
        if assignment_mode is not None and assignment_mode not in TRUE_E2E_ASSIGNMENT_MODES:
            raise ValueError(f"Unsupported assignment_mode: {assignment_mode}")
        if assignment_forward not in {"hard_st", "soft"}:
            raise ValueError(
                f"Unsupported assignment_forward: {assignment_forward}"
            )
                                                                         
                                                                                 
                                                                                        
        use_gumbel_sampling = (self.stop_gumbel_sampling_epoch == 0) or (current_epoch < self.stop_gumbel_sampling_epoch)

                                                           
        current_tau = self.get_current_tau(global_step)
        if assignment_temperature is not None:
            current_tau = float(assignment_temperature)

        latent = self.encoder(x)
        if return_structured:
            rq_output = self.rq(
                latent, use_gumbel=use_gumbel, tau=current_tau,
                use_indicator_ste=self.use_indicator_ste,
                use_gumbel_sampling=use_gumbel_sampling,
                current_epoch=current_epoch,
                return_structured=True,
                deterministic_st=deterministic_st,
                assignment_mode=assignment_mode,
                assignment_forward=assignment_forward,
            )
            return RQVAEOutput(
                quantized=rq_output.quantized,
                latent=latent,
                levels=rq_output.levels,
                vq_loss=rq_output.vq_loss,
                vq_loss_per_item=rq_output.vq_loss_per_item,
                indices=rq_output.indices,
                balance_loss=rq_output.balance_loss,
                gate_reg_loss=rq_output.gate_reg_loss,
                mean_sigma=rq_output.mean_sigma,
            )

        x_q, rq_loss, indices, code_one_hot, logit, balance_loss, gate_reg_loss, mean_sigma = self.rq(
            latent, use_gumbel=use_gumbel, tau=current_tau,
            use_indicator_ste=self.use_indicator_ste,
            use_gumbel_sampling=use_gumbel_sampling,
            current_epoch=current_epoch
        )

        outputs = (x_q, rq_loss, indices, code_one_hot, logit, balance_loss, gate_reg_loss, mean_sigma)
        if return_latent:
            return outputs + (latent,)
        return outputs

    @torch.no_grad()
    def get_indices(self, xs, use_sinkhorn=True):
        x_e = self.encoder(xs)
        indices = self.rq.get_indices(x_e, use_sinkhorn=use_sinkhorn)
        return indices

    @torch.no_grad()
    def get_maxk_indices(self, xs, maxk=1, used=False):

        x_e = self.encoder(xs)
        all_indices, fix = self.rq.get_maxk_indices(x_e, maxk=maxk, used=used)
        return all_indices, fix

    def get_codebook(self):
        return self.rq.get_codebook()

    def get_adaptive_selection_stats(self):
        return self.rq.get_adaptive_selection_stats()

    def reset_adaptive_selection_stats(self):
        self.rq.reset_adaptive_selection_stats()


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_e_list = config['num_emb_list']
        self.num_quantizers = len(self.n_e_list)
        self.vq_type = config['vq_type']
        self.dist = config['dist']
        self.warmup_gumbel_epochs = config.get('warmup_gumbel_epochs', 0)
                      
        sk_epsilons = config.get('sk_epsilons', [0.0] * len(self.n_e_list))
        if len(sk_epsilons) != len(self.n_e_list):
            sk_epsilons = [sk_epsilons[0] if sk_epsilons else 0.0] * len(self.n_e_list)

        if self.vq_type == "vq":
            self.vq_layers = nn.ModuleList([VectorQuantizer(config=config, n_e=n_e, dist=self.dist, sk_epsilon=sk_eps)
                                            for n_e, sk_eps in zip(self.n_e_list, sk_epsilons)])
        else:
            raise NotImplementedError

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook.detach().cpu())
        return torch.stack(all_codebook)

    def get_adaptive_selection_stats(self):
        total_gumbel = 0
        total_deterministic = 0
        stats_per_layer = []

                                           
        use_soft_frequency = False
        avg_learned_threshold = None

        for i, quantizer in enumerate(self.vq_layers):
            stats = quantizer.get_adaptive_selection_stats()
            stats_per_layer.append(stats)
            total_gumbel += stats['gumbel_count']
            total_deterministic += stats['deterministic_count']

                                    
            if i == 0:
                use_soft_frequency = stats.get('use_soft_frequency', False)
                if use_soft_frequency and stats.get('learned_threshold') is not None:
                    avg_learned_threshold = stats['learned_threshold']

        total = total_gumbel + total_deterministic
        result = {
            'gumbel_count': total_gumbel,
            'deterministic_count': total_deterministic,
            'total_count': total,
            'gumbel_ratio': total_gumbel / total if total > 0 else 0.0,
            'deterministic_ratio': total_deterministic / total if total > 0 else 0.0,
            'per_layer': stats_per_layer
        }

                            
        if use_soft_frequency:
            result['use_soft_frequency'] = True
            result['learned_threshold'] = avg_learned_threshold
                                   
            thresholds = [s.get('learned_threshold') for s in stats_per_layer
                         if s.get('learned_threshold') is not None]
            if thresholds:
                result['learned_threshold'] = sum(thresholds) / len(thresholds)
                result['threshold_logit'] = stats_per_layer[0].get('threshold_logit', 0.0)
        else:
            result['use_soft_frequency'] = False
            result['learned_threshold'] = None

                          
        use_gate_network = False
        avg_gate_reg_loss = None
        if len(stats_per_layer) > 0:
            use_gate_network = stats_per_layer[0].get('use_gate_network', False)
            if use_gate_network:
                                       
                gate_losses = [s.get('avg_gate_reg_loss', 0.0) for s in stats_per_layer
                              if s.get('avg_gate_reg_loss') is not None]
                if gate_losses:
                    avg_gate_reg_loss = sum(gate_losses) / len(gate_losses)
        result['use_gate_network'] = use_gate_network
        result['avg_gate_reg_loss'] = avg_gate_reg_loss

        return result

    def reset_adaptive_selection_stats(self):
        for quantizer in self.vq_layers:
            quantizer.reset_adaptive_selection_stats()

    @torch.no_grad()
    def get_indices(self, x, use_sinkhorn=True):
        all_indices = []
        residual = x
        for i in range(len(self.vq_layers)):
            x_res, _, indices, _, _, _, _, _ = self.vq_layers[i](
                residual,
                use_sinkhorn=use_sinkhorn,
            )
            residual = residual - x_res

            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=-1)

        return all_indices

    def forward(self, x, use_gumbel=False, tau=1.0, use_indicator_ste=True,
                use_gumbel_sampling=True, current_epoch=None,
                return_structured=False, deterministic_st=False,
                assignment_mode=None, assignment_forward="hard_st"):
        if return_structured:
            levels = []
            residual = x
            quantized = torch.zeros_like(x)
            for quantizer in self.vq_layers:
                level = quantizer(
                    residual,
                    use_sinkhorn=(
                        False
                        if deterministic_st or assignment_mode is not None
                        else None
                    ),
                    use_gumbel=use_gumbel,
                    tau=tau,
                    use_indicator_ste=use_indicator_ste,
                    use_gumbel_sampling=use_gumbel_sampling,
                    current_epoch=current_epoch,
                    return_structured=True,
                    deterministic_st=deterministic_st,
                    assignment_mode=assignment_mode,
                    assignment_forward=assignment_forward,
                )
                levels.append(level)
                residual = residual - level.quantized_vector
                quantized = quantized + level.quantized_vector

            vq_loss_per_item = torch.stack(
                [level.vq_loss_per_item for level in levels], dim=0
            ).mean(dim=0)
            vq_loss = vq_loss_per_item.mean()
            balance_losses = [
                level.balance_loss for level in levels
                if level.balance_loss is not None
            ]
            gate_losses = [
                level.gate_reg_loss for level in levels
                if level.gate_reg_loss is not None
            ]
            sigmas = [level.sigma for level in levels if level.sigma is not None]
            return ResidualQuantizerOutput(
                quantized=quantized,
                levels=tuple(levels),
                vq_loss=vq_loss,
                vq_loss_per_item=vq_loss_per_item,
                indices=torch.stack(
                    [level.hard_indices for level in levels], dim=-1
                ),
                balance_loss=(
                    torch.stack(balance_losses).mean() if balance_losses else None
                ),
                gate_reg_loss=(
                    torch.stack(gate_losses).mean() if gate_losses else None
                ),
                mean_sigma=torch.stack(sigmas).mean() if sigmas else None,
            )

        all_losses = []
        all_indices = []
        all_one_hots = []
        all_logits = []
        all_balance_losses = []
        all_gate_reg_losses = []
        all_sigmas = []

                                      
        in_warmup = False
        if self.warmup_gumbel_epochs > 0 and current_epoch is not None:
            in_warmup = current_epoch < self.warmup_gumbel_epochs

                                                                           
                                      
                                        
        sample_use_gumbel_mask = None                                      
        if use_gumbel and use_gumbel_sampling and self.vq_layers[0].use_adaptive_selection:
            if in_warmup:
                batch_size = x.shape[0]
                sample_use_gumbel_mask = torch.ones(batch_size, dtype=torch.bool, device=x.device)
            else:
                                          
                sample_use_gumbel_mask = self.vq_layers[0].get_sequence_level_decision(x)

        x_q = 0
        x_q_detach = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, one_hot, logit, balance_loss, gate_reg_loss, sigma = quantizer(
                residual, use_gumbel=use_gumbel, tau=tau,
                use_indicator_ste=use_indicator_ste,
                use_gumbel_sampling=use_gumbel_sampling,
                sample_use_gumbel_mask=sample_use_gumbel_mask,           
                current_epoch=current_epoch                          
            )
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_one_hots.append(one_hot)
            all_logits.append(logit)
            if balance_loss is not None:
                all_balance_losses.append(balance_loss)
            if gate_reg_loss is not None:
                all_gate_reg_losses.append(gate_reg_loss)
            if sigma is not None:
                all_sigmas.append(sigma)


        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_one_hots = torch.stack(all_one_hots, dim=1)                              
        all_logits = torch.stack(all_logits, dim=1)

                                                    
        mean_balance_loss = None
        if len(all_balance_losses) > 0:
            mean_balance_loss = torch.stack(all_balance_losses).mean()

                                                                
        mean_gate_reg_loss = None
        if len(all_gate_reg_losses) > 0:
            mean_gate_reg_loss = torch.stack(all_gate_reg_losses).mean()

                                             
        mean_sigma = None
        if len(all_sigmas) > 0:
            mean_sigma = torch.stack(all_sigmas).mean()

        return x_q, mean_losses, all_indices, all_one_hots, all_logits, mean_balance_loss, mean_gate_reg_loss, mean_sigma


class VectorQuantizer(nn.Module):
    def __init__(self, config, n_e, dist, sk_epsilon=0.0):
        super().__init__()
        self.n_e = n_e
        self.dist = dist
        self.e_dim = config['e_dim']
        self.beta = config['beta']

        self.kmeans_init = config['kmeans_init']
        self.kmeans_iters = config['kmeans_iters']
        self.sk_epsilon = sk_epsilon
        self.sk_iters = config.get('sk_iters', 50)           
        self.sync_quantizer_stats = config.get('sync_quantizer_stats', True)
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
                                                                      
                                                        
                                                                                                                                  
        self.gumbel_hard_switch_epoch = config.get('gumbel_hard_switch_epoch', 50)

                                                                                   
                                                                            
        self.force_deterministic = config.get('force_deterministic', False)

                                                                        
                                                                                             
                                                                                
        self.use_pure_ste = config.get('use_pure_ste', False)

        self.initted = False if self.kmeans_init else True
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

                                                        
        self.use_gaq = config.get('use_gaq', False)             
        self.gaq_gamma = config.get('gaq_gamma', 0.6)            
        self.gaq_eps = config.get('gaq_eps', 1e-7)                  
        self.register_buffer('N_ema', torch.zeros(self.n_e))                        

                                                        
        self.use_adaptive_selection = config.get('use_adaptive_selection', False)
        self.register_buffer('code_usage_ema', torch.ones(self.n_e) / self.n_e)
        self.usage_momentum = config.get('usage_momentum', 0.99)
        self.hot_threshold_ratio = config.get('hot_threshold_ratio', 1.5)
        self.gumbel_noise_scale = float(config.get('gumbel_noise_scale', 1.0))

                                                    
        self.use_soft_frequency = config.get('use_soft_frequency', False)
        if self.use_soft_frequency:
                                                
            self.dead_code_threshold_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.dead_code_threshold_logit = None

                                                    
        self.use_gate_network = config.get('use_gate_network', False)
        if self.use_gate_network:
                                               
            gate_hidden = config.get('gate_hidden_dim', self.e_dim // 2)
            self.gate_network = nn.Sequential(
                nn.Linear(self.e_dim, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
                nn.Sigmoid()                            
            )

                                                
                                                        
            with torch.no_grad():
                for module in self.gate_network:
                    if isinstance(module, nn.Linear):
                                      
                        module.weight.data.normal_(0, 0.001)         
                        if module.bias is not None:
                            module.bias.data.zero_()             
        else:
            self.gate_network = None

                                                                       
        self.use_learnable_sigma = config.get('use_learnable_sigma_gumbel', False)
        self.noise_type = config.get('noise_type', 'gumbel')                          

        if self.use_learnable_sigma:
                                                          
            if self.noise_type == 'gaussian':
                self.auto_sigma_module = AutoSigmaGaussian(config)
            else:                   
                if config.get('use_simple_uncertainty_loss', False):
                    self.auto_sigma_module = AutoSigmaSimple(config)
                else:
                    self.auto_sigma_module = AutoSigmaGumbel(config)

    def get_codebook(self):
        return self.embedding.weight

    def _distributed_training_enabled(self):
        return (
            self.training
            and self.sync_quantizer_stats
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _gather_detached_rows(self, tensor):
        local_size = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
        all_sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(all_sizes, local_size)
        sizes = [int(size.item()) for size in all_sizes]
        max_size = max(sizes)

        padded = tensor.detach()
        if tensor.shape[0] < max_size:
            padding = torch.zeros(
                max_size - tensor.shape[0], *tensor.shape[1:],
                device=tensor.device, dtype=tensor.dtype
            )
            padded = torch.cat([padded, padding], dim=0)

        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
        rows = torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0)
        offset = sum(sizes[:dist.get_rank()])
        return rows, offset

    def _distributed_batch_mean(self, probabilities):
        if not self._distributed_training_enabled():
            return probabilities.mean(dim=0)

        local_sum = probabilities.sum(dim=0)
        global_sum = local_sum.detach().clone()
        dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)

        total_count = torch.tensor(
            float(probabilities.shape[0]), device=probabilities.device,
            dtype=probabilities.dtype
        )
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

        global_mean = global_sum / total_count.clamp_min(1)
        gradient_proxy = local_sum * (dist.get_world_size() / total_count.clamp_min(1))
        return global_mean + gradient_proxy - gradient_proxy.detach()

    def get_codebook_entry(self, indices, shape=None):
                                      
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def get_adaptive_selection_stats(self):
        if not hasattr(self, '_gumbel_count'):
            stats = {'gumbel_count': 0, 'deterministic_count': 0, 'total_count': 0,
                    'gumbel_ratio': 0.0, 'deterministic_ratio': 0.0}
        else:
            total = self._gumbel_count + self._deterministic_count
            if total == 0:
                stats = {'gumbel_count': 0, 'deterministic_count': 0, 'total_count': 0,
                        'gumbel_ratio': 0.0, 'deterministic_ratio': 0.0}
            else:
                stats = {
                    'gumbel_count': self._gumbel_count,
                    'deterministic_count': self._deterministic_count,
                    'total_count': total,
                    'gumbel_ratio': self._gumbel_count / total,
                    'deterministic_ratio': self._deterministic_count / total
                }

                               
        if self.use_soft_frequency and self.dead_code_threshold_logit is not None:
            learned_threshold = torch.sigmoid(self.dead_code_threshold_logit) / self.n_e
            stats['use_soft_frequency'] = True
            stats['learned_threshold'] = learned_threshold.item()
            stats['threshold_logit'] = self.dead_code_threshold_logit.item()
        else:
            stats['use_soft_frequency'] = False
            stats['learned_threshold'] = None

                             
        if self.use_gate_network and self.gate_network is not None:
            stats['use_gate_network'] = True
                                 
            if hasattr(self, '_gate_reg_loss_count') and self._gate_reg_loss_count > 0:
                stats['avg_gate_reg_loss'] = self._gate_reg_loss_sum / self._gate_reg_loss_count
            else:
                stats['avg_gate_reg_loss'] = 0.0
        else:
            stats['use_gate_network'] = False
            stats['avg_gate_reg_loss'] = None

        return stats

    def reset_adaptive_selection_stats(self):
        self._gumbel_count = 0
        self._deterministic_count = 0
        self._gate_reg_loss_sum = 0.0
        self._gate_reg_loss_count = 0

    def get_sequence_level_decision(self, x):
        if not self.use_adaptive_selection:
            return None

                       
        latent = x.view(-1, self.e_dim)

                          
        if self.dist.lower() == 'l2':
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(latent, self.embedding.weight.t())
        elif self.dist.lower() == 'dot':
            d = -torch.matmul(latent, self.embedding.weight.t())
        elif self.dist.lower() == 'cos':
            d = -torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t())
        else:
            raise NotImplementedError

        indices_deterministic = torch.argmin(d, dim=-1)

                  
        if self.use_gate_network:
                                                            
            assigned_embeddings = self.embedding(indices_deterministic)
            gate_logits = self.gate_network[:-1](assigned_embeddings).squeeze(-1)       

                                        
                                          
            gate_binary = (gate_logits > 0.0).float()       

                                     
                                                   
            gate_values = gate_binary - gate_logits.detach() + gate_logits       

                                                  
            is_hot = (gate_values < 0.5)                    

        elif self.use_soft_frequency:
                                       
            activity_scores, _ = self.soft_threshold_operation(self.code_usage_ema)
            assigned_scores = activity_scores[indices_deterministic.view(-1)]
            is_hot = assigned_scores > 0

        else:
                                    
            avg_freq = 1.0 / self.n_e
            hot_threshold = self.hot_threshold_ratio * avg_freq
            assigned_freqs = self.code_usage_ema[indices_deterministic.view(-1)]
            is_hot = assigned_freqs > hot_threshold

        return is_hot                      

    def soft_threshold_operation(self, frequencies):
        if not self.use_soft_frequency or self.dead_code_threshold_logit is None:
                               
            avg_freq = 1.0 / self.n_e
            return frequencies, avg_freq

                                           
                            
        threshold = torch.sigmoid(self.dead_code_threshold_logit) / self.n_e

                                                     
                              
        activity_scores = F.relu(frequencies - threshold)

        return activity_scores, threshold

    def init_emb(self, data):

        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
                         
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, detach=True, use_sinkhorn=None, use_gumbel=False,
                tau=1.0, use_indicator_ste=True, use_gumbel_sampling=True,
                sample_use_gumbel_mask=None, current_epoch=None,
                return_structured=False, deterministic_st=False,
                assignment_mode=None, assignment_forward="hard_st"):
        if assignment_forward not in {"hard_st", "soft"}:
            raise ValueError(
                f"Unsupported assignment_forward: {assignment_forward}"
            )
                       
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.dist.lower() == 'l2':
                                                                       
            d = torch.sum(latent**2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t()- \
                2 * torch.matmul(latent, self.embedding.weight.t())

        elif self.dist.lower() == 'dot':
            d = torch.matmul(latent, self.embedding.weight.t())
            d = -d
        elif self.dist.lower() == 'cos':
            d = torch.matmul(F.normalize(latent, dim=-1), F.normalize(self.embedding.weight, dim=-1).t())
            d = -d
        else:
            raise NotImplementedError

                          
                                                       
        if use_sinkhorn is None:
            should_use_sinkhorn = self.sk_epsilon > 0
        else:
            should_use_sinkhorn = use_sinkhorn and self.sk_epsilon > 0

        if should_use_sinkhorn:
            if self._distributed_training_enabled():
                all_distances, local_offset = self._gather_detached_rows(d)
                global_max = all_distances.max()
                global_min = all_distances.min()
                middle = (global_max + global_min) / 2
                amplitude = (global_max - middle).clamp_min(1e-5)

                d = ((d - middle) / amplitude).double()
                sinkhorn_distances = (all_distances - middle) / amplitude
                Q = sinkhorn_algorithm(
                    sinkhorn_distances.double(), self.sk_epsilon, self.sk_iters
                )
                Q = Q[local_offset:local_offset + latent.shape[0]]
            else:
                d = self.center_distance_for_constraint(d).double()
                Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise RuntimeError("Sinkhorn produced non-finite assignments")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)

                                                        
        code_one_hot = F.one_hot(indices, self.n_e).float()

                                                                       
        if self.training and self.use_gaq:
            with torch.no_grad():
                                          
                B = indices.numel()
                Ind_hard = F.one_hot(indices.view(-1), num_classes=self.n_e).float()         
                n_k = Ind_hard.sum(dim=0)                                                  

                                                                     
                self.N_ema.mul_(self.gaq_gamma).add_((1.0 - self.gaq_gamma) * (n_k / max(B, 1)))

                                                                 
                latent_flat = x.view(-1, self.e_dim).detach()                               
                      
                denom = n_k.clamp_min(1.0).unsqueeze(-1)                                     
                anchors = (Ind_hard.t() @ latent_flat) / denom                               

                                                       
                                                       
                alpha = torch.exp(- self.N_ema * self.n_e / (1.0 - self.gaq_gamma) - self.gaq_eps)
                alpha = alpha.clamp(max=1.0).unsqueeze(-1)                                   

                                                       
                update_mask = (n_k > 0).float().unsqueeze(-1)                                
                alpha = alpha * update_mask

                                                              
                self.embedding.weight.mul_(1.0 - alpha).add_(alpha * anchors)
                                                      

                                                                                     
        balance_loss = None
        threshold_reg_loss = None                
        gate_reg_loss = None                                      
        sigma = None                         
        assignment_logits = None
        soft_probabilities = None
        hard_one_hot = None
        straight_through_probabilities = None
        forward_probabilities = None
        noise_scale = None
        effective_noise_scale = None
        frequency_scores = None
        stochastic_mask = None
        gumbel_noise = None

        if deterministic_st:
            probability_shape = x.shape[:-1] + (self.n_e,)
            assignment_logits = (-d).view(probability_shape)
            if assignment_logits.dtype != self.embedding.weight.dtype:
                assignment_logits = assignment_logits.float()
            (
                soft_probabilities,
                hard_one_hot,
                straight_through_probabilities,
                indices,
            ) = deterministic_straight_through(
                assignment_logits, temperature=tau
            )
            forward_probabilities = straight_through_probabilities
            x_q = torch.matmul(
                forward_probabilities, self.embedding.weight
            ).view(x.shape)
        elif assignment_mode in TRUE_E2E_GUMBEL_MODES:
            assignment_logits = -d
            if assignment_logits.dtype != self.embedding.weight.dtype:
                assignment_logits = assignment_logits.float()

            deterministic_indices = assignment_logits.argmax(dim=-1)
            uses_frequency_decay = "frqud" in assignment_mode
            uses_standard_deviation_decay = "sdud" in assignment_mode

            if uses_frequency_decay:
                if self.training:
                    with torch.no_grad():
                        counts = torch.bincount(
                            deterministic_indices.reshape(-1), minlength=self.n_e
                        ).to(dtype=self.code_usage_ema.dtype)
                        batch_frequency = counts / counts.sum().clamp_min(1)
                        self.code_usage_ema.mul_(self.usage_momentum).add_(
                            batch_frequency, alpha=1.0 - self.usage_momentum
                        )
                frequency_scores = self.code_usage_ema[
                    deterministic_indices.reshape(-1)
                ]
                threshold = self.hot_threshold_ratio / self.n_e
                stochastic_mask = frequency_scores > threshold
            else:
                stochastic_mask = torch.ones_like(
                    deterministic_indices, dtype=torch.bool
                )

            if uses_standard_deviation_decay:
                if not hasattr(self, "auto_sigma_module") or not isinstance(
                    self.auto_sigma_module, AutoSigmaSimple
                ):
                    raise RuntimeError(
                        "True-E2E SDUD requires use_learnable_sigma_gumbel=true "
                        "and use_simple_uncertainty_loss=true"
                    )
                sigma = self.auto_sigma_module.sigma
                noise_scale = sigma.abs().clamp(min=0.0, max=100.0)
            else:
                noise_scale = torch.as_tensor(
                    self.gumbel_noise_scale,
                    dtype=assignment_logits.dtype,
                    device=assignment_logits.device,
                )

            if not self.training:
                stochastic_mask = torch.zeros_like(stochastic_mask)
            effective_noise_scale = (
                stochastic_mask.to(dtype=assignment_logits.dtype) * noise_scale
            )
            assignment = gumbel_straight_through(
                assignment_logits,
                temperature=tau,
                noise_scale=effective_noise_scale,
                add_noise=self.training,
            )
            soft_probabilities = assignment.soft_probabilities
            hard_one_hot = assignment.hard_one_hot
            straight_through_probabilities = (
                assignment.straight_through_probabilities
            )
            indices = assignment.hard_indices
            gumbel_noise = assignment.gumbel_noise
            forward_probabilities = (
                straight_through_probabilities
                if assignment_forward == "hard_st"
                else soft_probabilities
            )
            x_q = torch.matmul(
                forward_probabilities, self.embedding.weight
            ).view(x.shape)

            if self.training:
                if not hasattr(self, '_gumbel_count'):
                    self.reset_adaptive_selection_stats()
                self._gumbel_count += int(stochastic_mask.sum().item())
                self._deterministic_count += int((~stochastic_mask).sum().item())
        elif use_gumbel and self.training:
                                                                            
            logits = -d
                                                            
            if logits.dtype != self.embedding.weight.dtype:
                logits = logits.float()

                                                              
                                         
                                                      
                                                                   
                                                                                                                    
            use_hard = False
            if current_epoch is not None and self.gumbel_hard_switch_epoch > 0:
                use_hard = current_epoch >= self.gumbel_hard_switch_epoch

                                                                      
            if hasattr(self, 'auto_sigma_module'):
                Ind_soft_gumbel, sigma = self.auto_sigma_module(logits, tau=tau, hard=use_hard, dim=-1)
            else:
                Ind_soft_gumbel = F.gumbel_softmax(logits, tau=tau, hard=use_hard, dim=-1)
                                                  
                sigma = None

                                                   
                                                                                      

                                               
            if self.force_deterministic:
                use_gumbel_sampling = False
                                                                        
                Ind_soft_gumbel = F.softmax(logits / tau, dim=-1)

                                                             
            if self.use_pure_ste:
                                                                     
                                                         
                use_gumbel_sampling = False
                                                                      

                                                     
            if use_gumbel_sampling and self.use_adaptive_selection:
                              
                indices_deterministic = indices

                                  
                with torch.no_grad():
                    counts = torch.bincount(indices_deterministic.view(-1), minlength=self.n_e).float()
                    if self._distributed_training_enabled():
                        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    current_freq = counts / max(counts.sum(), 1)
                    self.code_usage_ema = self.usage_momentum * self.code_usage_ema + \
                                         (1 - self.usage_momentum) * current_freq

                                         
                indices_gumbel = torch.argmax(Ind_soft_gumbel, dim=-1)

                                        
                                                         
                if sample_use_gumbel_mask is not None:
                                            
                    is_hot = sample_use_gumbel_mask
                else:
                                            
                                 
                    if self.use_gate_network:
                                                                      
                                               
                        assigned_embeddings = self.embedding(indices_deterministic)              

                                            
                        gate_logits = self.gate_network[:-1](assigned_embeddings).squeeze(-1)       

                                                    
                                                      
                                                                  
                        gate_binary = (gate_logits > 0.0).float()       

                                                 
                                                               
                        gate_values = gate_binary - gate_logits.detach() + gate_logits       

                                                                
                                           
                        is_hot = (gate_values < 0.5)                             

                                                                       
                        gate_reg_loss = None

                    elif self.use_soft_frequency:
                                                     
                        activity_scores, learned_threshold = self.soft_threshold_operation(self.code_usage_ema)
                                            
                        assigned_scores = activity_scores[indices_deterministic.view(-1)]
                                                 
                        is_hot = assigned_scores > 0

                                            
                                      
                        avg_freq = 1.0 / self.n_e
                        target_threshold = self.hot_threshold_ratio * avg_freq
                        threshold_reg_loss = F.mse_loss(
                            torch.sigmoid(self.dead_code_threshold_logit) / self.n_e,
                            torch.tensor(target_threshold, device=self.dead_code_threshold_logit.device)
                        )
                    else:
                                                    
                        avg_freq = 1.0 / self.n_e
                        hot_threshold = self.hot_threshold_ratio * avg_freq
                        assigned_freqs = self.code_usage_ema[indices_deterministic.view(-1)]
                        is_hot = assigned_freqs > hot_threshold

                                    
                num_gumbel = is_hot.sum().item()
                num_deterministic = (~is_hot).sum().item()

                                 
                if not hasattr(self, '_gumbel_count'):
                    self._gumbel_count = 0
                    self._deterministic_count = 0
                    self._gate_reg_loss_sum = 0.0
                    self._gate_reg_loss_count = 0
                self._gumbel_count += num_gumbel
                self._deterministic_count += num_deterministic

                                      
                if gate_reg_loss is not None:
                    self._gate_reg_loss_sum += gate_reg_loss.item()
                    self._gate_reg_loss_count += 1

                                                        
                indices_selected = torch.where(
                    is_hot.view_as(indices_deterministic),
                    indices_gumbel.view_as(indices_deterministic),
                    indices_deterministic
                )

                                                               
                                               
                Ind_soft_det = F.softmax(logits / tau, dim=-1)

                                                      
                                            
                is_hot_expanded = is_hot.view(-1, 1).expand_as(Ind_soft_gumbel)
                Ind_soft = torch.where(
                    is_hot_expanded,
                    Ind_soft_gumbel,                                
                    Ind_soft_det                              
                )
            elif use_gumbel_sampling:
                                                                           
                                                                     
                indices_selected = torch.argmax(Ind_soft_gumbel, dim=-1)
                Ind_soft = Ind_soft_gumbel
            else:
                                                              
                                                                 
                indices_selected = indices
                Ind_soft = Ind_soft_gumbel

            Ind_hard = F.one_hot(indices_selected, self.n_e).float()

                                                                        
            indices = indices_selected
            assignment_logits = logits
            soft_probabilities = Ind_soft
            hard_one_hot = Ind_hard
            straight_through_probabilities = (
                Ind_hard - Ind_soft.detach() + Ind_soft
            )
            forward_probabilities = straight_through_probabilities

            if use_indicator_ste:
                                                        
                                                                                               
                Ind = Ind_hard - Ind_soft.detach() + Ind_soft
                                                                         
                x_q = torch.matmul(Ind, self.embedding.weight)
                x_q = x_q.view(x.shape)
            else:
                                                        
                                                                         
                x_q_soft = torch.matmul(Ind_soft, self.embedding.weight)
                x_q_soft = x_q_soft.view(x.shape)

                                                                              
                x_q_hard = self.embedding(indices_selected).view(x.shape)

                                                                                   
                x_q = x_q_hard + (x_q_soft - x_q_soft.detach())
                x_q = x_q_hard + (x_q_soft - x_q_soft.detach())

                                                 
                                                                                
                                                                                
            avg_probs = self._distributed_batch_mean(Ind_soft)

                                                                
            uniform_dist = torch.ones_like(avg_probs) / self.n_e

                                                                             
                                                             
                                                               
            balance_loss = torch.abs(avg_probs - uniform_dist).sum()

                                                                  
                                                                

                                                                                
                                      
                                            
                               
                                       
               

                                                                 
                                                                      
                                                                    
                                                  

                                                                                
            if threshold_reg_loss is not None:
                if balance_loss is None:
                    balance_loss = threshold_reg_loss * 0.01       
                else:
                    balance_loss = balance_loss + threshold_reg_loss * 0.01

                                                        
                                                         

                                                   
                                                                               
            if self.use_pure_ste:
                                                           
                x_q = self.embedding(indices).view(x.shape)

                                                                         
                                                                             
                Ind_soft_for_balance = F.softmax(logits / tau, dim=-1)
                avg_probs = self._distributed_batch_mean(Ind_soft_for_balance)
                uniform_dist = torch.ones_like(avg_probs) / self.n_e
                balance_loss = torch.abs(avg_probs - uniform_dist).sum()
        else:
                                                
            x_q = self.embedding(indices).view(x.shape)
            assignment_logits = -d
            if assignment_logits.dtype != self.embedding.weight.dtype:
                assignment_logits = assignment_logits.float()
            soft_probabilities = F.softmax(assignment_logits / tau, dim=-1)
            hard_one_hot = F.one_hot(indices, self.n_e).to(
                dtype=soft_probabilities.dtype
            )
            straight_through_probabilities = hard_one_hot
            forward_probabilities = hard_one_hot

                                    
                                                                
                                                                     
        if self.dist.lower() == 'l2':
            codebook_loss_per_item = (x_q - x.detach()).pow(2).mean(dim=-1)
            commitment_loss_per_item = (x_q.detach() - x).pow(2).mean(dim=-1)
            vq_loss_per_item = (
                codebook_loss_per_item
                + self.beta * commitment_loss_per_item
            )
            loss = vq_loss_per_item.mean()
        elif self.dist.lower() in ['dot', 'cos']:
            vq_loss_per_item = self.beta * F.cross_entropy(
                -d, indices.detach(), reduction='none'
            ).view(x.shape[:-1])
            loss = vq_loss_per_item.mean()
        else:
            raise NotImplementedError

                                                                    
                                                                              
                                                                    
                                                             
        if (
            assignment_mode is None
            and not deterministic_st
            and ((use_gumbel and self.use_pure_ste) or not use_gumbel)
        ):
            x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        logit = d

        if return_structured:
            probability_shape = x.shape[:-1] + (self.n_e,)
            if assignment_logits.shape != probability_shape:
                assignment_logits = assignment_logits.view(probability_shape)
            if soft_probabilities.shape != probability_shape:
                soft_probabilities = soft_probabilities.view(probability_shape)
            if hard_one_hot.shape != probability_shape:
                hard_one_hot = hard_one_hot.view(probability_shape)
            if straight_through_probabilities.shape != probability_shape:
                straight_through_probabilities = (
                    straight_through_probabilities.view(probability_shape)
                )
            if forward_probabilities.shape != probability_shape:
                forward_probabilities = forward_probabilities.view(
                    probability_shape
                )
            return QuantizerLevelOutput(
                assignment_logits=assignment_logits,
                soft_probabilities=soft_probabilities,
                hard_one_hot=hard_one_hot,
                straight_through_probabilities=straight_through_probabilities,
                forward_probabilities=forward_probabilities,
                hard_indices=indices,
                residual_input=x,
                quantized_vector=x_q,
                vq_loss=loss,
                vq_loss_per_item=vq_loss_per_item,
                balance_loss=balance_loss,
                gate_reg_loss=gate_reg_loss,
                sigma=sigma,
                noise_scale=noise_scale,
                effective_noise_scale=effective_noise_scale,
                frequency_scores=frequency_scores,
                stochastic_mask=stochastic_mask,
                gumbel_noise=gumbel_noise,
            )

        return x_q, loss, indices, code_one_hot, logit, balance_loss, gate_reg_loss, sigma
