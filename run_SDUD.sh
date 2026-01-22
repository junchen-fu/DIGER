#!/bin/bash

# 
# 

set -e

DATASET=beauty
CONFIG=./config/${DATASET}_jo.yaml
ACCEL_CFG=accelerate_config.yaml
RQVAE_INIT="<PATH_TO_RQVAE_CHECKPOINT>"

STAGE1_EPOCHS=120
GUMBEL_HARD_SWITCH_EPOCH=1

echo "========================================================================"
echo "========================================================================"
echo "  - Loss Formula: L = L_task / (2*(σ+λ)^2) + log(σ+λ)"
echo "  - Equilibrium: σ = sqrt(L) - λ"
echo "  - Lambda (Bias): 1.0"
echo "  - Direct Std Operation: YES (No 2^x conversion)"
echo "  - Initial Std: 1.0"
echo "  - Sigma LR: 1e-3"
echo "========================================================================"

accelerate launch --config_file ${ACCEL_CFG} main.py \
    --config ${CONFIG} \
    --rqvae_path="${RQVAE_INIT}" \
    --lr_rec=1e-3 \
    --lr_id=1e-5 \
    --lr_sigma=1e-3 \
    --initial_std=1.0 \
    --noise_type=gumbel \
    --gumbel_tau=2 \
    --stop_gumbel_sampling_epoch=0 \
    --epochs=${STAGE1_EPOCHS} \
    --early_stop=15 \
    --freeze_semantic_embedding=true \
    --freeze_id_encoder=true \
    --freeze_id_encoder_layers=3 \
    --freeze_id_decoder=true \
    --freeze_id_epochs=0 \
    --freeze_rq=false \
    --gumbel_hard_switch_epoch=${GUMBEL_HARD_SWITCH_EPOCH} \
    --use_learnable_sigma_gumbel=true \
    --use_plain_code_loss=false \
    --code_loss_weight=1.0 \
    --recon_loss_weight=1.0 \
    --vq_loss_weight=1.0 \ \
    --use_adaptive_selection=false \
    --use_gate_network=false \
    --use_simple_uncertainty_loss=true \
    --sigma_lambda=1.7 \
    --usage_dir_suffix=sdud_17

echo ""
echo "========================================================================"
echo "========================================================================"
echo ""
echo "Target Equilibrium Check:"
echo "  If Loss = 2.0, Lambda = 1.0"
echo "  Target Sigma = sqrt(2.0) - 1.0 = 1.414 - 1.0 = 0.414"
echo "========================================================================"

