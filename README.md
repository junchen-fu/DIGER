# DIGER: Discrete Item Generation for Recommendation

A minimal, clean implementation for anonymous review.

## Overview

This repository contains the core implementation of our recommendation model with two training strategies:
1. **Adaptive Popularity-based Selection**: Dynamically switches between Gumbel sampling and deterministic indexing based on code usage frequency
2. **Simple Uncertainty Loss**: Uses learnable uncertainty to balance task loss

## Repository Structure

```
DIGER/
├── main.py                              # Main training entry point
├── vq.py                                # Vector Quantization (RQ-VAE) implementation
├── trainer.py                           # Training loop and loss computation
├── model.py                             # Recommender model architecture
├── data.py                              # Data loading utilities
├── utils.py                             # Helper functions
├── metrics.py                           # Evaluation metrics
├── layers.py                            # Neural network layers
├── config/
│   └── beauty_jo.yaml                  # Configuration file for Beauty dataset
├── accelerate_config.yaml              # Accelerate configuration
├── run_beauty_adaptive_popularity.sh   # Training script 1
└── run_beauty_simple_sigma.sh          # Training script 2
```

## Requirements

### Dependencies

```bash
pip install torch transformers accelerate pyyaml numpy faiss-cpu scikit-learn colorama tqdm
```

### Python Version
- Python 3.8+
- PyTorch 1.10+

## Data Preparation

### 1. Dataset Structure

Organize your dataset in the following structure:

```
dataset/
└── beauty/
    ├── beauty.train.inter
    ├── beauty.valid.inter
    ├── beauty.test.inter
    └── Beauty.emb-llama.npy    # Semantic embeddings
```

### 2. Data Format

- **Interaction files** (`.inter`): Tab-separated values with columns `user_id:token`, `item_id:token`, `timestamp:float`
- **Semantic embeddings** (`.npy`): NumPy array of shape `[num_items, embedding_dim]`

### 3. Pre-trained RQ-VAE Checkpoint

You need a pre-trained RQ-VAE checkpoint. The checkpoint should contain:
- Encoder weights
- Residual Quantization (RQ) codebooks
- Decoder weights (optional, can be frozen)

## Configuration

### Update Paths

Before running, update the placeholder paths in the following files:

1. **Shell scripts** (`run_beauty_adaptive_popularity.sh`, `run_beauty_simple_sigma.sh`):
   ```bash
   RQVAE_INIT="<PATH_TO_RQVAE_CHECKPOINT>"  # Update this
   ```

2. **Config file** (`config/beauty_jo.yaml`):
   ```yaml
   semantic_emb_path: <PATH_TO_DATASET>/beauty/Beauty.emb-llama.npy  # Update this
   rqvae_path: <PATH_TO_RQVAE_CHECKPOINT>  # Update this
   data_path: ./dataset  # Update if needed
   ```

### Key Configuration Parameters

#### Model Architecture
- `code_num`: 256 (codebook size)
- `code_length`: 4 (number of RQ layers + 1)
- `e_dim`: 256 (embedding dimension)
- `d_model`: 128 (transformer hidden size)
- `encoder_layers`: 6
- `decoder_layers`: 6

#### Training
- `epochs`: 200 (with early stopping)
- `early_stop`: 15
- `batch_size`: 256
- `lr_rec`: 0.005 (recommender learning rate)
- `lr_id`: 0.0005 (tokenizer learning rate)

#### Loss Weights
- `code_loss_weight`: 1.0
- `recon_loss_weight`: 1.0
- `vq_loss_weight`: 1.0

## Usage

### Training Script 1: Adaptive Popularity-based Selection

This script uses adaptive selection to dynamically choose between Gumbel sampling (for popular codes) and deterministic indexing (for rare codes).

```bash
bash run_beauty_adaptive_popularity.sh
```

**Key parameters:**
- `use_adaptive_selection=true`: Enable adaptive selection
- `hot_threshold_ratio=1.5`: Threshold for determining hot codes
- `usage_momentum=0.99`: EMA momentum for code usage tracking
- `gumbel_tau=2`: Temperature for Gumbel-Softmax

### Training Script 2: Simple Uncertainty Loss

This script uses a learnable uncertainty parameter to automatically balance task loss.

```bash
bash run_beauty_simple_sigma.sh
```

**Key parameters:**
- `use_learnable_sigma_gumbel=true`: Enable learnable uncertainty
- `use_simple_uncertainty_loss=true`: Use simple uncertainty formulation
- `sigma_lambda=1.7`: Lambda bias term
- `initial_std=1.0`: Initial standard deviation
- `lr_sigma=1e-3`: Learning rate for sigma

**Loss formula:**
```
L = L_task / (2*(σ+λ)²) + log(σ+λ)
```

At equilibrium: `σ = sqrt(L_task) - λ`

## Accelerate Configuration

The repository includes a minimal single-GPU configuration (`accelerate_config.yaml`). 

For multi-GPU training, modify the config:

```yaml
distributed_type: 'MULTI_GPU'
num_processes: 4  # Number of GPUs
```

Or use accelerate's config wizard:

```bash
accelerate config
```

## Output

### Logs

Training logs are saved to `./logs/<dataset>/` with timestamps.

### Checkpoints

Model checkpoints are saved to `./myckpt/<dataset>/` including:
- `best_model.pth`: Best model based on validation metric
- Training statistics and metrics

### Metrics

The model is evaluated on:
- Recall@1, Recall@5, Recall@10
- NDCG@5, NDCG@10

Validation metric: NDCG@10
