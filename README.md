# DIGER: Differentiable Semantic ID for Generative Recommendation

## Overview

This repository contains the core implementation of our recommendation model with two training strategies:
1. **Frequency-based Uncertainty Decay**: Dynamically switches between Gumbel sampling and deterministic indexing based on code usage frequency
2. **Standard Deviation Uncertainty Decay**: Uses learnable uncertainty to balance task loss

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
├── run_FrqUD.sh                        # Training script 1
└── run_SDUD.sh                         # Training script 2
```

## Requirements

### Dependencies

```bash
pip install torch transformers accelerate pyyaml numpy faiss-cpu scikit-learn colorama tqdm
```

### Python Version
- Python 3.12.11
- PyTorch 2.5.1

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

1. **Shell scripts** (`run_FrqUD.sh`, `run_SDUD.sh`):
   ```bash
   RQVAE_INIT="<PATH_TO_RQVAE_CHECKPOINT>"  # Update this
   ```

2. **Config file** (`config/beauty_jo.yaml`):
   ```yaml
   semantic_emb_path: <PATH_TO_DATASET>/beauty/Beauty.emb-llama.npy  # Update this
   rqvae_path: <PATH_TO_RQVAE_CHECKPOINT>  # Update this
   data_path: ./dataset  # Update if needed
   ```

## Usage

### Training Script 1: Frequency-based Uncertainty Decay

This script uses adaptive selection to dynamically choose between Gumbel sampling (for popular codes) and deterministic indexing (for rare codes).

```bash
bash run_FrqUD.sh
```

### Training Script 2: Standard Deviation Uncertainty Decay

This script uses a learnable uncertainty parameter to automatically balance task loss.

```bash
bash run_SDUD.sh
```

**Loss formula:**
```
L = L_task / (2*(σ+λ)²) + log(σ+λ)
```

At equilibrium: `σ = sqrt(L_task) - λ`

## Output

### Logs

Training logs are saved to `./logs/<dataset>/` with timestamps.

### Checkpoints

Model checkpoints are saved to `./myckpt/<dataset>/` including:
- `best_model.pth`: Best model based on validation metric
- Training statistics and metrics

### Metrics

The model is evaluated on:

-  Recall@5, Recall@10
- NDCG@5, NDCG@10

Validation metric: NDCG@10
