#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <beauty|instruments|yelp> <frqud|sdud|both>" >&2
  exit 2
fi

DATASET="$1"
VARIANT="$2"
GPU_SPEC="${GPU:-0}"

GPU_LIST="${GPU_SPEC}"
if [[ "${GPU_LIST}" == cuda:* ]]; then
  GPU_LIST="${GPU_LIST#cuda:}"
fi
GPU_LIST="${GPU_LIST//,/ }"
read -r -a _gpu_arr <<<"${GPU_LIST}"

if (( ${#_gpu_arr[@]} > 2 )); then
  echo "run_experiment supports at most 2 GPUs. Received: ${GPU_SPEC}" >&2
  exit 2
fi
if (( ${#_gpu_arr[@]} == 0 )) || [[ -z "${_gpu_arr[0]}" ]]; then
  echo "Invalid GPU spec: ${GPU_SPEC}" >&2
  exit 2
fi
if (( ${#_gpu_arr[@]} >= 1 )) && ! [[ "${_gpu_arr[0]}" =~ ^[0-9]+$ ]]; then
  echo "Invalid GPU index: ${_gpu_arr[0]}" >&2
  exit 2
fi
if (( ${#_gpu_arr[@]} == 2 )) && ! [[ "${_gpu_arr[1]}" =~ ^[0-9]+$ ]]; then
  echo "Invalid GPU index: ${_gpu_arr[1]}" >&2
  exit 2
fi
if (( ${#_gpu_arr[@]} == 2 )); then
  GPU="${_gpu_arr[0]},${_gpu_arr[1]}"
else
  GPU="${_gpu_arr[0]}"
fi
if [[ -n "${ACCEL_CFG:-}" ]]; then
  ACCEL_CFG="${ACCEL_CFG}"
elif (( ${#_gpu_arr[@]} == 2 )); then
  ACCEL_CFG="accelerate_config_multi_gpu.yaml"
else
  ACCEL_CFG="accelerate_config.yaml"
fi
CONFIG="config/${DATASET}_jo.yaml"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

if [ -n "${DIGER_ENV_BIN:-}" ]; then
  if [ ! -x "${DIGER_ENV_BIN}/accelerate" ]; then
    echo "DIGER_ENV_BIN is set but ${DIGER_ENV_BIN}/accelerate is not executable" >&2
    exit 2
  fi
  ACCELERATE_BIN="${DIGER_ENV_BIN}/accelerate"
fi

if [ ! -f "${CONFIG}" ]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi

COMMON_ARGS=(
  --config "${CONFIG}"
  --rqvae_path="${RQVAE_PATH:-./rqvae_ckpt/${DATASET}/best_collision_model.pth}"
  --lr_rec=0.001
  --lr_id=0.00001
  --weight_decay=0.05
  --freeze_semantic_embedding=true
  --freeze_id_encoder=true
  --freeze_id_encoder_layers=0
  --freeze_id_decoder=true
  --freeze_id_epochs=0
  --freeze_rq=false
  --stop_gumbel_sampling_epoch=0
  --code_loss_weight=1.0
  --recon_loss_weight=1.0
  --vq_loss_weight=1.0
  --qs_loss_weight=0
  --use_soft_frequency=false
  --use_gate_network=false
)

if [[ -n "${TRAIN_BATCH_SIZE:-}" ]]; then
  COMMON_ARGS+=(--batch_size="${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  COMMON_ARGS+=(--gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${GRADIENT_CHECKPOINTING:-}" ]]; then
  COMMON_ARGS+=(--gradient_checkpointing="${GRADIENT_CHECKPOINTING}")
fi
if [[ -n "${LOG_DIR:-}" ]]; then
  COMMON_ARGS+=(--log_dir="${LOG_DIR}")
fi

case "${DATASET}" in
  beauty)
    COMMON_ARGS+=(--epochs=120 --early_stop=15 --eval_batch_size="${EVAL_BATCH_SIZE:-32}" --num_beams=20 --gumbel_tau=2)
    ;;
  instruments)
    COMMON_ARGS+=(--epochs=100 --early_stop=15 --eval_batch_size="${EVAL_BATCH_SIZE:-32}" --num_beams=20 --gumbel_tau=2)
    ;;
  yelp)
    COMMON_ARGS+=(--epochs=200 --early_stop=20 --eval_batch_size="${EVAL_BATCH_SIZE:-16}" --num_beams=80 --gumbel_tau=1.5)
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

case "${DATASET}:${VARIANT}" in
  beauty:frqud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.5 --usage_momentum=0.99 --use_learnable_sigma_gumbel=false --gumbel_hard_switch_epoch=0)
    ;;
  beauty:sdud)
    RUN_ARGS=(--use_adaptive_selection=false --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.7 --gumbel_hard_switch_epoch=0)
    ;;
  beauty:both)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.5 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --lr_sigma=0.001 --sigma_reg_weight=2.0 --initial_std=1.0 --noise_type=gumbel --use_cosine_annealing=false --use_dynamic_sigma_lr=true --gate_loss_weight=0 --gumbel_hard_switch_epoch=0)
    ;;
  instruments:frqud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=2.0 --usage_momentum=0.99 --use_learnable_sigma_gumbel=false --gumbel_hard_switch_epoch=0)
    ;;
  instruments:sdud)
    RUN_ARGS=(--use_adaptive_selection=false --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=1.0 --noise_type=gumbel --sigma_lambda=1.8 --gumbel_hard_switch_epoch=0)
    ;;
  instruments:both)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.5 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_plain_code_loss=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=1.5 --noise_type=gumbel --sigma_lambda=1.8 --gumbel_hard_switch_epoch=0)
    ;;
  yelp:frqud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.1 --usage_momentum=0.99 --use_learnable_sigma_gumbel=false --use_simple_uncertainty_loss=true --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.0)
    ;;
  yelp:sdud)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=false --hot_threshold_ratio=1.1 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_simple_uncertainty_loss=true --use_plain_code_loss=false --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.7)
    ;;
  yelp:both)
    RUN_ARGS=(--balance_loss_weight=0 --use_adaptive_selection=true --hot_threshold_ratio=1.1 --usage_momentum=0.99 --use_learnable_sigma_gumbel=true --use_simple_uncertainty_loss=true --use_plain_code_loss=false --lr_sigma=0.001 --initial_std=2.0 --noise_type=gumbel --sigma_lambda=1.0)
    ;;
  *)
    echo "Unknown variant: ${DATASET}:${VARIANT}" >&2
    exit 2
    ;;
esac

REPRODUCTION_LOG_DIR="${REPRODUCTION_LOG_DIR:-reproduction_logs}"
mkdir -p "${REPRODUCTION_LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_LOG="${REPRODUCTION_LOG_DIR}/${DATASET}_${VARIANT}_${STAMP}.log"

echo "Running ${DATASET}:${VARIANT} on GPU ${GPU}"
echo "stdout log: ${OUT_LOG}"
echo "accelerate: ${ACCELERATE_BIN}"
echo "accelerate config: ${ACCEL_CFG}"

CUDA_VISIBLE_DEVICES="${GPU}" "${ACCELERATE_BIN}" launch --config_file "${ACCEL_CFG}" main.py "${COMMON_ARGS[@]}" "${RUN_ARGS[@]}" 2>&1 | tee "${OUT_LOG}"
