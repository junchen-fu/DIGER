#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <beauty|instruments|yelp> <frqud|sdud|both>" >&2
  exit 2
fi

RQVAE_PATH="${RQVAE_PATH:-}"
if [[ "$1:$2" == "beauty:sdud" && -z "${RQVAE_PATH}" ]]; then
  RQVAE_PATH="./rqvae_ckpt/beauty/best_collision_model_two_gpu_preview.pth"
fi

GPU="${GPU:-0,1}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}" \
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}" \
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
RQVAE_PATH="${RQVAE_PATH}" \
  bash scripts/run_experiment.sh "$1" "$2"
