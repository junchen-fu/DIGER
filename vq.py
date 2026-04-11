import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class AutoSigmaGaussian(nn.Module):
    """
    Learnable Sigma Gaussian-Softmax module with Exponential Annealing.
    
    Uses Gaussian (Normal) noise instead of Gumbel noise for randomization.
    
    Key differences from Gumbel:
    - Gaussian noise: N(0, s^2) with s = 2^sigma
    - Standard Gaussian std = 1.0 (vs Gumbel std = pi/sqrt6 ≈ 1.28)
    - More symmetric noise distribution
    
    Annealing: Loss 5→1.7 drives Sigma 1.5→0.1 automatically
    """
    def __init__(self, config, temperature=1.0):
        super().__init__()
        # Parameter: sigma (can be negative for annealing)
        # Parameter: sigma (can be negative for annealing)
        # NEW: Support initial_std directly
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
            if initial_std <= 1e-5:
                initial_sigma = -20.0 # Hard zero trigger
            else:
                import math
                initial_sigma = math.log2(initial_std)
        else:
            initial_sigma = config.get('initial_sigma', 1.0)
            
        self.sigma = nn.Parameter(torch.tensor(initial_sigma))
        
        # Parameter: temperature (default 1.0)
        self.temperature = temperature

    def forward(self, logits, tau=1.0, hard=False, dim=-1):
        """
        Forward pass with Gaussian noise.
        
        Args:
            logits: Unnormalized log probabilities [Batch, ..., NumClasses]
            tau: Temperature for scaling (from scheduler)
            hard: If True, return one-hot approximation (straight-through)
            dim: Softmax dimension
            
        Returns:
            y: Softmax output (or hard one-hot)
            sigma: The current sigma value
        """
        # Allow negative sigma for annealing
        sigma_value = self.sigma
        
        # Transform sigma to actual noise scale: s = 2^sigma
        s = torch.pow(2.0, sigma_value)
        # Clamp s to reasonable range (allow s<1 for annealing)
        s = s.clamp(min=1e-5, max=100.0)
        
        if self.training:
            # Generate Gaussian (Normal) noise: N(0, 1)
            gaussian = torch.randn_like(logits)
            
            # Scale noise by s: noise ~ N(0, s^2)
            # NEW: Hard zero for very small sigma to ensure true determinism
            if s.item() < 1e-4:
                noise = torch.zeros_like(logits)
            else:
                noise = gaussian * s
            
            # Apply: y = (logits + noise) / temperature
            noisy_logits = (logits + noise) / tau
            
            y_soft = noisy_logits.softmax(dim)
            
            if hard:
                # Straight-through estimator
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
                # Forward: hard, Backward: soft
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
        else:
            # Inference behavior (no noise)
            if hard:
                index = logits.max(dim, keepdim=True)[1]
                ret = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            else:
                # Standard softmax without noise for inference
                ret = (logits / tau).softmax(dim)
        
        # Return the sigma value
        return ret, sigma_value

    @staticmethod
    def compute_uncertainty_loss(task_loss, sigma, reg_weight=1.0):
        """
        Exponential annealing loss for Gaussian noise scheduling.
        
        Same as Gumbel version:
            L = L_task * exp(-k * sigma) + c * sigma
        
        At equilibrium: sigma = -0.589 + 1.298 * ln(L_task)
        - L=5.0  → sigma=1.5 (strong exploration)
        - L=1.7  → sigma=0.1 (near-deterministic)
        
        Args:
            task_loss: The original task loss
            sigma: The learnable parameter
            reg_weight: Scaling factor for c
                       
        Returns:
            weighted_loss: The combined loss
        """
        k = 0.77
        c = 1.448783 * reg_weight
        
        # L = L_task * exp(-k * sigma) + c * sigma
        exp_term = task_loss * torch.exp(-k * sigma)
        reg_term = c * sigma
        
        return exp_term + reg_term


class AutoSigmaGumbel(nn.Module):
    """
    Learnable Sigma Gumbel-Softmax module with Uncertainty-Weighted Loss.
    
    Key Design: 
    - Learnable parameter 'sigma' is clamped to [0, +∞) via ReLU-like operation
    - Actual noise scale: s = 2^sigma (minimum s=1 when sigma=0)
    - Cannot go below standard Gumbel noise level (s=1)
    
    Equilibrium behavior:
    - When L_task > 1: sigma = log2(sqrt(L_task)) > 0, s > 1 (exploration)
    - When L_task ≤ 1: sigma tries to go negative but is clamped to 0
    - At equilibrium with L_task ≤ 1: sigma = 0, s = 1 (standard Gumbel)
    
    Loss formulation: L = L_task / (2 * s^2) + sigma * ln(2)
    where s = 2^max(0, sigma)
    
    Note: The ReLU constraint prevents sigma from becoming negative, providing
    a natural floor on the noise level at s=1.
    """
    def __init__(self, config, temperature=1.0):
        super().__init__()
        # Parameter: sigma (can be negative for annealing to near-deterministic)
        # Initialize to 1.0 for moderate initial exploration
        # NEW: Support initial_std directly
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
            if initial_std <= 1e-5:
                initial_sigma = -20.0 # Hard zero trigger
            else:
                import math
                initial_sigma = math.log2(initial_std)
        else:
            initial_sigma = config.get('initial_sigma', 1.0)
            
        self.sigma = nn.Parameter(torch.tensor(initial_sigma))
        
        # Parameter: temperature (default 1.0)
        self.temperature = temperature

    def forward(self, logits, tau=1.0, hard=False, dim=-1):
        """
        Args:
            logits: Unnormalized log probabilities [Batch, ..., NumClasses]
            tau: Temperature for scaling (from scheduler)
            hard: If True, return one-hot approximation (straight-through)
            dim: Softmax dimension
            
        Returns:
            y: Softmax output (or hard one-hot)
            sigma: The current sigma value (network output parameter, clamped to ≥0)
        """
        # Allow negative sigma for annealing (to achieve near-deterministic behavior)
        # s = 2^sigma can now be < 1.0 for reduced noise
        sigma_value = self.sigma
        
        # Transform sigma to actual noise scale: s = 2^sigma
        s = torch.pow(2.0, sigma_value)
        # Clamp s to reasonable range (allow s<1 for annealing)
        s = s.clamp(min=1e-5, max=100.0)  # min=1e-5 allows sigma down to very small values
        
        if self.training:
            # Generate Gumbel noise
            gumbels = -torch.empty_like(logits).exponential_().log()
            
            # Scale noise by s: noise = gumbel_noise * s
            # NEW: Hard zero for very small sigma to ensure true determinism
            if s.item() < 1e-4:
                noise = torch.zeros_like(logits)
            else:
                noise = gumbels * s
            
            # Apply: y = (logits + noise) / temperature
            gumbel_logits = (logits + noise) / tau
            
            y_soft = gumbel_logits.softmax(dim)
            
            if hard:
                # Straight-through estimator
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
                # Forward: hard, Backward: soft
                ret = y_hard - y_soft.detach() + y_soft
            else:
                ret = y_soft
        else:
            # Inference behavior
            if hard:
                index = logits.max(dim, keepdim=True)[1]
                ret = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            else:
                # Standard softmax without noise for inference
                ret = (logits / tau).softmax(dim)
        
        # Return the sigma value (can be negative for annealing)
        return ret, sigma_value

    @staticmethod
    def compute_uncertainty_loss(task_loss, sigma, reg_weight=1.0, 
                                 annealing_threshold=None, 
                                 annealing_slow_k=None, annealing_slow_c=None,
                                 annealing_fast_k=None, annealing_fast_c=None):
        """
        Exponential annealing loss for adaptive noise scheduling.
        
        Formula:
            L = L_task * exp(-k * sigma) + c * sigma
        
        Mathematical Derivation:
            At equilibrium (dL/dsigma = 0):
                -k * L_task * exp(-k * sigma) + c = 0
                => sigma = ln(k * L_task / c) / k
                => sigma = (1/k) * ln(L_task) + (ln(k) - ln(c)) / k
                => sigma = a * ln(L_task) + b
        
        SLOW-THEN-FAST Piecewise Annealing (configurable):
            Piecewise schedule: slow phase first, then fast phase.

            Default (Beauty dataset):
            Phase 1 (slow): L ∈ [2.2, 5.0] → sigma ∈ [-1, 1]
            - L=5.0  → sigma=1 (std=2, mild initial exploration)
            - L=2.2  → sigma=-1 (std=0.5, handoff point)
            - k=0.458, c=1.361 (slow annealing)

            Phase 2 (fast): L ∈ [1.6, 2.2] → sigma ∈ [-20, -1]
            - L=2.2  → sigma=-1 (std=0.5, handoff point)
            - L=1.6  → sigma=-20 (std≈0, faster convergence)
            - k=0.018, c=0.037 (fast annealing)
            
            Instruments dataset (configurable):
            - Threshold: 2.5 (vs 2.2 for beauty)
            - Slow phase: k=0.385, c=1.2 (gentler than beauty)
            - Fast phase: k=0.020, c=0.040 (similar to beauty)
        
        Args:
            task_loss: The original task loss (e.g., CrossEntropy)
            sigma: The learnable parameter (can be negative for annealing)
            reg_weight: Scaling factor for c (default=1.0)
            annealing_threshold: Loss threshold for switching phases (default=2.0)
            annealing_slow_k: k parameter for slow phase (default=0.458145)
            annealing_slow_c: c parameter for slow phase (default=1.361442)
            annealing_fast_k: k parameter for fast phase (default=0.018127)
            annealing_fast_c: c parameter for fast phase (default=0.036916)
                       
        Returns:
            weighted_loss: The combined loss
        """
        # SLOW-THEN-FAST annealing (piecewise) - Configurable version
        
        # Use provided parameters or defaults (for beauty dataset)
        threshold_loss = annealing_threshold if annealing_threshold is not None else 2.0
        
        if task_loss >= threshold_loss:
            k = annealing_slow_k if annealing_slow_k is not None else 0.458145
            c = (annealing_slow_c if annealing_slow_c is not None else 1.361442) * reg_weight
        else:
            k = annealing_fast_k if annealing_fast_k is not None else 0.018127
            c = (annealing_fast_c if annealing_fast_c is not None else 0.036916) * reg_weight
        
        # Exponential annealing loss
        # L = L_task * exp(-k * sigma) + c * sigma
        exp_term = task_loss * torch.exp(-k * sigma)
        reg_term = c * sigma
        
        return exp_term + reg_term


class AutoSigmaSimple(nn.Module):
    """
    Standard Deviation Uncertainty Decay (SDUD) Module.
    
    Key Differences from AutoSigmaGumbel:
    1. Sigma is directly the std (or noise scale), no 2^x conversion.
    2. Uses SDUD loss: L = L / (2(s+lambda)^2) + log(s+lambda).
    
    Learnable Lambda Options:
    - auto_lambda_mode='fixed': Use fixed sigma_lambda (default, backward compatible)
    - auto_lambda_mode='learnable': Lambda is a learnable parameter
    - auto_lambda_mode='adaptive': Lambda = sqrt(EMA(loss)), auto-computed
    """
    def __init__(self, config, temperature=1.0):
        super().__init__()
        if 'initial_std' in config:
            initial_std = float(config['initial_std'])
        else:
            initial_std = config.get('initial_sigma', 1.0)
            
        self.sigma = nn.Parameter(torch.tensor(initial_std))
        self.temperature = temperature
        
        # Auto-lambda mode
        self.auto_lambda_mode = config.get('auto_lambda_mode', 'fixed')
        
        if self.auto_lambda_mode == 'learnable':
            # Make lambda a learnable parameter
            initial_lambda = config.get('sigma_lambda', 1.8)
            self.lambda_param = nn.Parameter(torch.tensor(initial_lambda))
        elif self.auto_lambda_mode == 'adaptive':
            # Use EMA of loss to auto-compute lambda
            # Lambda = sqrt(EMA(loss))
            self.register_buffer('loss_ema', torch.tensor(5.0))  # Initialize with high value
            self.ema_momentum = config.get('lambda_ema_momentum', 0.99)
            self.lambda_param = None  # Will be computed dynamically
        else:  # 'fixed' mode
            self.lambda_param = None

    def forward(self, logits, tau=1.0, hard=False, dim=-1):
        # Sigma is directly the noise scale
        sigma_value = self.sigma
        # Ensure noise scale is positive and reasonable for sampling
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
        """
        Update the EMA of loss for adaptive lambda computation.
        Should be called during training before compute_uncertainty_loss.
        """
        if self.auto_lambda_mode == 'adaptive':
            with torch.no_grad():
                self.loss_ema = self.ema_momentum * self.loss_ema + (1 - self.ema_momentum) * task_loss_value
    
    def get_lambda(self):
        """
        Get the current lambda value based on the mode.
        """
        if self.auto_lambda_mode == 'learnable':
            # Lambda is learnable, clamp to reasonable range [0.5, 3.0]
            return self.lambda_param.abs().clamp(min=0.5, max=3.0)
        elif self.auto_lambda_mode == 'adaptive':
            # Lambda = sqrt(EMA(loss)), representing expected loss at convergence
            # Clamp loss_ema to avoid sqrt of negative or very small values
            loss_ema_clamped = self.loss_ema.clamp(min=0.25, max=25.0)
            return torch.sqrt(loss_ema_clamped)
        else:
            # Fixed mode, return None to use external lambda_bias
            return None
    
    def compute_uncertainty_loss(self, task_loss, sigma, lambda_bias=0.5):
        """
        Compute uncertainty loss with optional learnable/adaptive lambda.
        
        Formula: L = L_task / (2 * (sigma + lambda)^2) + log(sigma + lambda)
        
        At equilibrium: sigma = sqrt(L_task) - lambda
        Therefore, lambda should be approximately sqrt(L_converge) for sigma -> 0 at convergence.
        
        Args:
            task_loss: The original task loss (scalar tensor)
            sigma: The learnable sigma parameter
            lambda_bias: Fixed lambda value (used only in 'fixed' mode)
        
        Returns:
            weighted_loss: The combined loss
            actual_lambda: The lambda value used (for logging)
        """
        # Get lambda based on mode
        lambda_value = self.get_lambda()
        if lambda_value is None:
            lambda_value = lambda_bias
        
        # Update EMA if in adaptive mode
        if self.auto_lambda_mode == 'adaptive' and self.training:
            self.update_lambda_ema(task_loss.detach())
        
        # Compute loss
        # Formula: L = L_task / (2 * (sigma + lambda)^2) + log(sigma + lambda)
        effective_sigma = sigma.abs() + lambda_value
        effective_sigma = effective_sigma.clamp(min=1e-6)
        
        denom = 2 * (effective_sigma ** 2)
        mse_term = task_loss / denom
        log_term = torch.log(effective_sigma)
        
        return mse_term + log_term, lambda_value


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(- distances / epsilon)

    B = Q.shape[0] # number of samples to assign
    K = Q.shape[1] # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K


    Q *= B # the colomns must sum to 1 so that Q is an assignment
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
        # New: Control gradient flow level (indicator vs vector)
        self.use_indicator_ste = config.get('use_indicator_ste', True)
        # New: Control when to stop using Gumbel sampling in forward pass
        # After this many epochs, use deterministic argmin instead of Gumbel sampling
        self.stop_gumbel_sampling_epoch = config.get('stop_gumbel_sampling_epoch', 0)

        # New: Gumbel tau annealing
        # If enabled, tau will decay from initial_tau to min_tau following exponential schedule
        self.use_tau_annealing = config.get('use_tau_annealing', False)
        self.tau_anneal_init = config.get('tau_anneal_init', 2.0)      # Initial tau
        self.tau_anneal_min = config.get('tau_anneal_min', 0.5)        # Minimum tau
        self.tau_anneal_rate = config.get('tau_anneal_rate', 0.0003)   # Decay rate
        # Warmup: force pure Gumbel (without gate network) for the first N epochs
        self.warmup_gumbel_epochs = config.get('warmup_gumbel_epochs', 0)
        # New: Control when to switch from soft to hard Gumbel-Softmax
        # If set to 0: always use soft mode (hard=False)
        # If set to >0: use soft mode (hard=False) before this epoch, then switch to hard mode (hard=True) from this epoch onwards
        self.gumbel_hard_switch_epoch = config.get('gumbel_hard_switch_epoch', 50)

        if self.vq_type in ["vq"]:
            self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        else:
            raise NotImplementedError


        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.rq = ResidualVectorQuantizer(config=config)

    def get_current_tau(self, global_step):
        """
        Calculate current tau based on annealing schedule.

        Args:
            global_step: Current global training step

        Returns:
            current_tau: Annealed tau value
        """
        if not self.use_tau_annealing:
            return self.gumbel_tau

        # Exponential decay: tau = max(min_tau, init_tau * exp(-rate * step))
        import math
        tau = self.tau_anneal_init * math.exp(-self.tau_anneal_rate * global_step)
        tau = max(self.tau_anneal_min, tau)
        return tau

    def forward(self, x, use_gumbel=False, current_epoch=0, global_step=0):
        """
        Forward pass of RQVAE (encoder + quantizer only, no decoder).

        Args:
            x: Input tensor
            use_gumbel: If True, use Gumbel-Softmax for differentiable quantization
            current_epoch: Current training epoch (used to determine if Gumbel sampling should be used)
            global_step: Current global training step (used for tau annealing)

        Returns:
            x_q: Quantized latent (no reconstruction to original space)
            rq_loss: Quantization loss
            indices: Hard code indices (for recommender model)
            code_one_hot: One-hot encoded codes
            logit: Logits from quantization
            balance_loss: Codebook balance loss (or None if not using Gumbel)
            gate_reg_loss: Gate network regularization loss (or None if not using gate network)
            mean_sigma: Mean learnable sigma (or None if not using learnable sigma)
        """
        # Determine whether to use Gumbel sampling based on current epoch
        # If stop_gumbel_sampling_epoch = 0 (default), always use Gumbel sampling
        # If stop_gumbel_sampling_epoch > 0, stop using Gumbel sampling after that epoch
        use_gumbel_sampling = (self.stop_gumbel_sampling_epoch == 0) or (current_epoch < self.stop_gumbel_sampling_epoch)

        # Calculate current tau (with annealing if enabled)
        current_tau = self.get_current_tau(global_step)

        latent = self.encoder(x)
        x_q, rq_loss, indices, code_one_hot, logit, balance_loss, gate_reg_loss, mean_sigma = self.rq(
            latent, use_gumbel=use_gumbel, tau=current_tau,
            use_indicator_ste=self.use_indicator_ste,
            use_gumbel_sampling=use_gumbel_sampling,
            current_epoch=current_epoch
        )

        return x_q, rq_loss, indices, code_one_hot, logit, balance_loss, gate_reg_loss, mean_sigma

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
        """Return statistics for adaptive Gumbel vs. deterministic selection."""
        return self.rq.get_adaptive_selection_stats()
    
    def reset_adaptive_selection_stats(self):
        """Reset adaptive selection counters."""
        self.rq.reset_adaptive_selection_stats()
    

class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantizer for multi-scale quantization.
    """

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
        """Aggregate adaptive selection stats across all quantizer layers."""
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
        """Reset adaptive selection counters on every quantizer layer."""
        for quantizer in self.vq_layers:
            quantizer.reset_adaptive_selection_stats()

    @torch.no_grad()
    def get_indices(self, x, use_sinkhorn=True):
        all_indices = []
        residual = x
        for i in range(len(self.vq_layers)):
            x_res, _, indices, _, _, _, _, _ = self.vq_layers[i](residual, use_sinkhorn=use_sinkhorn)
            residual = residual - x_res

            all_indices.append(indices)

        all_indices = torch.stack(all_indices, dim=-1)

        return all_indices

    def forward(self, x, use_gumbel=False, tau=1.0, use_indicator_ste=True, use_gumbel_sampling=True, current_epoch=None):
        """
        Forward pass of Residual Vector Quantizer.

        Args:
            x: Input tensor
            use_gumbel: If True, use Gumbel-Softmax for differentiable quantization
            tau: Temperature for Gumbel-Softmax
            use_indicator_ste: If True, use indicator-level STE; else use vector-level STE
            use_gumbel_sampling: If True, sample indices from Gumbel-Softmax (stochastic);
                                If False, use deterministic argmin
            current_epoch: Current training epoch (optional, used for warmup gating control)

        Returns:
            x_q: Quantized tensor (differentiable if use_gumbel=True)
            mean_losses: Average quantization loss
            all_indices: Hard code indices (always discrete, for recommender)
            all_one_hots: One-hot codes
            all_logits: Logits from each quantizer
            mean_balance_loss: Average balance loss across all quantizers (or None)
            mean_gate_reg_loss: Average gate regularization loss across all quantizers (or None)
            mean_sigma: Mean learnable sigma (or None)
        """
        all_losses = []
        all_indices = []
        all_one_hots = []
        all_logits = []
        all_balance_losses = []
        all_gate_reg_losses = []
        all_sigmas = []

        # ====== Warmup control ======
        in_warmup = False
        if self.warmup_gumbel_epochs > 0 and current_epoch is not None:
            in_warmup = current_epoch < self.warmup_gumbel_epochs

        # ====== NEW: Sequence-level decision for adaptive selection ======
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
        all_one_hots = torch.stack(all_one_hots, dim=1) # (batch, code_len, code_num)
        all_logits = torch.stack(all_logits, dim=1)

        # Average balance loss across all quantizers
        mean_balance_loss = None
        if len(all_balance_losses) > 0:
            mean_balance_loss = torch.stack(all_balance_losses).mean()

        # Average gate regularization loss across all quantizers
        mean_gate_reg_loss = None
        if len(all_gate_reg_losses) > 0:
            mean_gate_reg_loss = torch.stack(all_gate_reg_losses).mean()
            
        # Average sigma across all quantizers
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
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # New: Control when to switch from soft to hard Gumbel-Softmax
        # If set to 0: always use soft mode (hard=False)
        # If set to >0: use soft mode (hard=False) before this epoch, then switch to hard mode (hard=True) from this epoch onwards
        self.gumbel_hard_switch_epoch = config.get('gumbel_hard_switch_epoch', 50)
        
        # NEW: Force pure deterministic (no Gumbel noise) even when use_gumbel=True
        # This is for ablation study: train RQ but without exploration noise
        self.force_deterministic = config.get('force_deterministic', False)
        
        # NEW: Pure STE (Straight-Through Estimator) - identity gradient
        # When True: forward uses hard indices, backward uses identity gradient (not softmax)
        # This is for ablation study comparing STE vs softmax-weighted gradients
        self.use_pure_ste = config.get('use_pure_ste', False)
        
        # When True: forward uses Gumbel sampling, backward only updates selected embedding
        # When False: forward uses Gumbel sampling, backward updates all embeddings (soft, default)
        # This is for ablation study: Soft Update vs Hard Update of codebook
        self.use_hard_codebook_update = config.get('use_hard_codebook_update', False)
        
        self.enable_grad_stats = config.get('enable_grad_stats', False)

        self.initted = False if self.kmeans_init else True
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        # GAQ (Global Anchor Quantization) configuration
        self.use_gaq = config.get('use_gaq', False)
        self.gaq_gamma = config.get('gaq_gamma', 0.6)
        self.gaq_eps = config.get('gaq_eps', 1e-7)
        self.register_buffer('N_ema', torch.zeros(self.n_e))
        
        self.use_adaptive_selection = config.get('use_adaptive_selection', False)
        self.register_buffer('code_usage_ema', torch.ones(self.n_e) / self.n_e)
        self.usage_momentum = config.get('usage_momentum', 0.99)
        self.hot_threshold_ratio = config.get('hot_threshold_ratio', 1.5)
        
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
            
        # NEW: Learnable Sigma with adaptive noise (Gumbel or Gaussian)
        self.use_learnable_sigma = config.get('use_learnable_sigma_gumbel', False)
        self.noise_type = config.get('noise_type', 'gumbel')  # 'gumbel' or 'gaussian'
        
        if self.use_learnable_sigma:
            # Instantiate the appropriate AutoSigma module
            if self.noise_type == 'gaussian':
                self.auto_sigma_module = AutoSigmaGaussian(config)
            else:  # default: gumbel
                if config.get('use_simple_uncertainty_loss', False):
                    self.auto_sigma_module = AutoSigmaSimple(config)
                else:
                    self.auto_sigma_module = AutoSigmaGumbel(config)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q
    
    def get_adaptive_selection_stats(self):
        """Return statistics for adaptive Gumbel vs. deterministic selection."""
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
        """Reset adaptive selection counters."""
        self._gumbel_count = 0
        self._deterministic_count = 0
        self._gate_reg_loss_sum = 0.0
        self._gate_reg_loss_count = 0
    
    def get_sequence_level_decision(self, x):
        """
        Decide per sample whether to use Gumbel or deterministic quantization at the
        full RQ stack level. Call only on the first quantizer; the same decision
        applies to all subsequent layers.

        Args:
            x: Input tensor [B, D]

        Returns:
            sample_use_gumbel_mask: [B] bool tensor; True means use Gumbel for that sample.
        """
        if not self.use_adaptive_selection:
            return None
        
        # Flatten input
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
            gate_logits = self.gate_network[:-1](assigned_embeddings).squeeze(-1)  # [B]
            
            # ====== Identity STE ======
            gate_binary = (gate_logits > 0.0).float()  # [B]
            
            gate_values = gate_binary - gate_logits.detach() + gate_logits  # [B]
            
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
        
        return is_hot  # [B] boolean tensor
    
    def soft_threshold_operation(self, frequencies):
        """
        Soft-Threshold Operation: S(freq, α) = ReLU(freq - α)
        
        Args:
            frequencies: Per-code usage frequencies, shape [K].

        Returns:
            activity_scores: Activity scores [K]; > 0 = active code, 0 = dead code.
            threshold: Scalar threshold in use.
        """
        if not self.use_soft_frequency or self.dead_code_threshold_logit is None:
            avg_freq = 1.0 / self.n_e
            return frequencies, avg_freq
        
        threshold = torch.sigmoid(self.dead_code_threshold_logit) / self.n_e
        
        # Soft-Threshold: S(freq, α) = ReLU(freq - α)
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
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, detach=True, use_sinkhorn=None, use_gumbel=False, tau=1.0, use_indicator_ste=True, use_gumbel_sampling=True, sample_use_gumbel_mask=None, current_epoch=None):
        """
        Forward pass of Vector Quantizer with Gumbel-Softmax support.

        Args:
            x: Input tensor
            detach: Whether to detach (legacy parameter, kept for compatibility)
            use_sinkhorn: Whether to use Sinkhorn algorithm
            use_gumbel: If True, use Gumbel-Softmax for differentiable quantization
            tau: Temperature for Gumbel-Softmax
            use_indicator_ste: If True, use indicator-level STE (default);
                              If False, use vector-level STE (legacy)
            use_gumbel_sampling: If True, sample indices from Gumbel-Softmax (stochastic);
                                If False, use deterministic argmin (default behavior)
            sample_use_gumbel_mask: Optional [B] bool mask for sequence-level choice.
                If set, reuse this mask instead of recomputing (keeps RQ layers consistent).
            current_epoch: Current training epoch (optional, used for hard/soft Gumbel switch)

        Returns:
            x_q: Quantized tensor (differentiable if use_gumbel=True)
            loss: Quantization loss
            indices: Hard code indices (always discrete)
            code_one_hot: One-hot codes
            logit: Logits/distances
            balance_loss: Codebook balance loss (only when use_gumbel=True and training)
            gate_reg_loss: Gate network regularization loss (only when use_gate_network=True)
            sigma: Learnable sigma (or None)
        """
        # Flatten input
        latent = x.view(-1, self.e_dim)
        
        if not self.initted and self.training:
            self.init_emb(latent)
        
        if self.dist.lower() == 'l2':
            # Calculate the L2 Norm between latent and Embedded weights
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
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print(f"Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)
        
        # Always keep hard indices for recommender model
        code_one_hot = F.one_hot(indices, self.n_e).float()
        
        # Use Gumbel-Softmax for differentiable quantization when training with joint optimization
        balance_loss = None
        threshold_reg_loss = None
        gate_reg_loss = None
        sigma = None  # NEW: Initialize sigma
        
        if use_gumbel and self.training:
            # Convert distances to logits (negative distance for similarity)
            logits = -d
            # Ensure logits are float32 (same as embeddings)
            if logits.dtype != self.embedding.weight.dtype:
                logits = logits.float()

            # Apply Gumbel-Softmax to get soft one-hot vectors
            use_hard = False
            if current_epoch is not None and self.gumbel_hard_switch_epoch > 0:
                use_hard = current_epoch >= self.gumbel_hard_switch_epoch
            
            # NEW: Use Learnable Sigma (Gumbel or Gaussian) if enabled
            if hasattr(self, 'auto_sigma_module'):
                Ind_soft_gumbel, sigma = self.auto_sigma_module(logits, tau=tau, hard=use_hard, dim=-1)
            else:
                Ind_soft_gumbel = F.gumbel_softmax(logits, tau=tau, hard=use_hard, dim=-1)
                # No learnable sigma, keep as None
                sigma = None

            # ====== Gumbel Sampling Control ======
            # Decide whether to use stochastic Gumbel sampling or deterministic argmin
            
            # NEW: Force deterministic override
            if self.force_deterministic:
                use_gumbel_sampling = False
                # Ensure gradients are also deterministic (pure softmax)
                Ind_soft_gumbel = F.softmax(logits / tau, dim=-1)
            
            # NEW: Pure STE override - skip to classic VQ STE
            if self.use_pure_ste:
                # Use classic VQ STE: forward hard, backward identity
                # Skip all the Gumbel/softmax logic below
                use_gumbel_sampling = False
                # We'll handle this after the adaptive selection block
                
            if use_gumbel_sampling and self.use_adaptive_selection:
                indices_deterministic = indices
                
                with torch.no_grad():
                    counts = torch.bincount(indices_deterministic.view(-1), minlength=self.n_e).float()
                    current_freq = counts / max(counts.sum(), 1)
                    self.code_usage_ema = self.usage_momentum * self.code_usage_ema + \
                                         (1 - self.usage_momentum) * current_freq
                
                indices_gumbel = torch.argmax(Ind_soft_gumbel, dim=-1)
                
                if sample_use_gumbel_mask is not None:
                    is_hot = sample_use_gumbel_mask
                else:
                    if self.use_gate_network:
                        assigned_embeddings = self.embedding(indices_deterministic)  # [B, e_dim]
                        
                        gate_logits = self.gate_network[:-1](assigned_embeddings).squeeze(-1)  # [B]
                        
                        # ====== Identity STE ======
                        gate_binary = (gate_logits > 0.0).float()  # [B]
                        
                        gate_values = gate_binary - gate_logits.detach() + gate_logits  # [B]
                        
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
                # Sample hard indices from Gumbel distribution (stochastic)
                # This makes the forward pass stochastic based on tau
                indices_selected = torch.argmax(Ind_soft_gumbel, dim=-1)
                Ind_soft = Ind_soft_gumbel
            else:
                # Use deterministic argmin (original behavior)
                # This is more stable but doesn't explore as much
                indices_selected = indices
                Ind_soft = Ind_soft_gumbel

            Ind_hard = F.one_hot(indices_selected, self.n_e).float()

            # Update indices to use selected ones (affects forward pass)
            indices = indices_selected

            if use_indicator_ste:
                if self.use_hard_codebook_update:
                    x_q = self.embedding(indices_selected).view(x.shape)
                else:
                    # ====== NEW: Indicator-level STE ======
                    # Gradient flows through Ind_soft, forward uses Ind_hard (from Gumbel sampling)
                    Ind = Ind_hard - Ind_soft.detach() + Ind_soft
                    # Combine indicator with codebook to get quantized vector
                    x_q = torch.matmul(Ind, self.embedding.weight)
                    x_q = x_q.view(x.shape)
            else:
                # ====== LEGACY: Vector-level STE ======
                # Use soft one-hot to get differentiable quantized vector
                x_q_soft = torch.matmul(Ind_soft, self.embedding.weight)
                x_q_soft = x_q_soft.view(x.shape)

                # Also compute hard quantization for the straight-through part
                x_q_hard = self.embedding(indices_selected).view(x.shape)

                # Straight-through estimator: forward uses hard, backward uses soft
                x_q = x_q_hard + (x_q_soft - x_q_soft.detach())


            # ====== Codebook Balance Loss ======
            # Calculate average probability across batch for each codebook entry
            # Ind_soft shape: [B*H*W, n_e] where B*H*W is flattened spatial dims
            avg_probs = Ind_soft.mean(dim=0)  # [n_e]

            # Target: uniform distribution over codebook entries
            uniform_dist = torch.ones_like(avg_probs) / self.n_e

            # Option 1: L1 loss (directly matches balance metric calculation)
            # balance = 1 - sum(|count_i - base|) / total / 2
            # We want to minimize: sum(|avg_probs_i - uniform|)
            balance_loss = torch.abs(avg_probs - uniform_dist).sum()

            # Option 2: MSE loss (larger gradient, easier to tune)
            # balance_loss = F.mse_loss(avg_probs, uniform_dist)

            # Option 3: KL divergence (commented out, can switch back if needed)
            # balance_loss = F.kl_div(
            #     (avg_probs + 1e-10).log(),
            #     uniform_dist,
            #     reduction='batchmean'
            # )

            # Option 4: Entropy regularization (maximize entropy)
            # entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
            # max_entropy = -torch.log(torch.tensor(1.0 / self.n_e))
            # balance_loss = max_entropy - entropy
            
            if threshold_reg_loss is not None:
                if balance_loss is None:
                    balance_loss = threshold_reg_loss * 0.01
                else:
                    balance_loss = balance_loss + threshold_reg_loss * 0.01
            
            
            # ====== Pure STE Implementation ======
            # If use_pure_ste is True, override gradient flow with identity STE
            if self.use_pure_ste:
                # Use hard indices (already computed above)
                x_q = self.embedding(indices).view(x.shape)
                
                # Still compute balance_loss for monitoring using softmax
                # (This doesn't affect gradients due to detach in code below)
                Ind_soft_for_balance = F.softmax(logits / tau, dim=-1)
                avg_probs = Ind_soft_for_balance.mean(dim=0)
                uniform_dist = torch.ones_like(avg_probs) / self.n_e
                balance_loss = torch.abs(avg_probs - uniform_dist).sum()
        else:
            # Standard VQ: use hard quantization
            x_q = self.embedding(indices).view(x.shape)
        
        # compute loss for embedding
        # IMPORTANT: Loss must be computed BEFORE STE is applied
        # This ensures gradients can flow back to codebook embeddings
        if self.dist.lower() == 'l2':
            codebook_loss = F.mse_loss(x_q, x.detach())
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = codebook_loss + self.beta * commitment_loss
        elif self.dist.lower() in ['dot', 'cos']:
            loss = self.beta * F.cross_entropy(-d, indices.detach())
        else:
            raise NotImplementedError

        # Apply STE AFTER loss computation (like traditional VQ-VAE)
        # This preserves gradient flow to codebook while using STE for encoder
        # When use_gumbel=True and use_pure_ste=True: Apply STE here
        # When use_gumbel=False (frozen mode): Also apply STE
        if (use_gumbel and self.use_pure_ste) or not use_gumbel:
            x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        logit = d

        return x_q, loss, indices, code_one_hot, logit, balance_loss, gate_reg_loss, sigma
