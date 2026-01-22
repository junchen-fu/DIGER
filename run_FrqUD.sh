#!/bin/bash


set -e

DATASET=beauty
CONFIG=./config/${DATASET}_jo.yaml
ACCEL_CFG=accelerate_config.yaml
RQVAE_INIT="<PATH_TO_RQVAE_CHECKPOINT>"

STAGE1_EPOCHS=120

echo "========================================================================"
echo "========================================================================"
echo "  - QS Loss: DISABLED (weight=0)"
echo "========================================================================"

accelerate launch --config_file ${ACCEL_CFG} main.py \
    --config ${CONFIG} \
    --rqvae_path="${RQVAE_INIT}" \
    --lr_rec=1e-3 \
    --lr_id=1e-5 \
    --gumbel_tau=2 \
    --epochs=${STAGE1_EPOCHS} \
    --early_stop=15 \
    --freeze_semantic_embedding=true \
    --freeze_id_encoder=true \
    --freeze_id_encoder_layers=0 \
    --freeze_id_decoder=true \
    --freeze_id_epochs=0 \
    --freeze_rq=false \
    --gumbel_hard_switch_epoch=0 \
    --use_learnable_sigma_gumbel=false \
    --stop_gumbel_sampling_epoch=0 \
    --code_loss_weight=1.0 \
    --recon_loss_weight=1.0 \
    --vq_loss_weight=1.0 \
    --balance_loss_weight=0 \
    --use_adaptive_selection=true \
    --hot_threshold_ratio=1.5 \
    --usage_momentum=0.99 \
    --use_soft_frequency=false \
    --use_gate_network=false

echo ""
echo "========================================================================"
echo ""
echo ""
echo "========================================================================"

