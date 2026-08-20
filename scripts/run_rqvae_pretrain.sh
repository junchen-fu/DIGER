#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <beauty|instruments|yelp>" >&2
  exit 2
fi

DATASET="$1"
PROJECT_ROOT="${RQVAE_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_ROOT="${RQVAE_DATA_ROOT:-${PROJECT_ROOT}/dataset}"
RQVAE_DIR="${PROJECT_ROOT}/scripts/rqvae"
RQVAE_CKPT_ROOT="${RQVAE_CKPT_ROOT:-${PROJECT_ROOT}/rqvae_ckpt}"
GPU="${RQVAE_GPU:-0}"

case "${DATASET}" in
  beauty)
    EMB_PATH="${DATA_ROOT}/beauty/Beauty.emb-llama.npy"
    BATCH_SIZE=1024
    BETA=0.25
    DATASET_TAG="beauty_strong_sinkhorn"
    ;;
  instruments)
    EMB_PATH="${DATA_ROOT}/instruments/Instruments.emb-llama.npy"
    BATCH_SIZE=2048
    BETA=0.25
    DATASET_TAG="instruments_strong_sinkhorn"
    ;;
  yelp)
    EMB_PATH="${DATA_ROOT}/yelp/Yelp.emb-llama.npy"
    BATCH_SIZE=4096
    BETA=0.5
    DATASET_TAG="yelp_strong_sinkhorn"
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

if [ -n "${RQVAE_EMBEDDING_PATH:-}" ]; then
  EMB_PATH="${RQVAE_EMBEDDING_PATH}"
  if [ -z "${RQVAE_DATASET_TAG:-}" ]; then
    DATASET_TAG="${DATASET}"
  fi
fi

if [ -n "${RQVAE_DATASET_TAG:-}" ]; then
  DATASET_TAG="${RQVAE_DATASET_TAG}"
fi

BATCH_SIZE="${RQVAE_BATCH_SIZE:-${BATCH_SIZE}}"
BETA="${RQVAE_BETA:-${BETA}}"

OUTPUT_PARENT="${RQVAE_CKPT_ROOT}/${DATASET_TAG}"
TARGET_CKPT="${RQVAE_CKPT_ROOT}/${DATASET}/best_collision_model.pth"
mkdir -p "${TARGET_CKPT%/*}"

if [[ "${GPU}" == "cpu" ]]; then
  DEVICE="cpu"
elif [[ "${GPU}" == "cuda" ]]; then
  DEVICE="cuda"
else
  GPU_LIST="${GPU#cuda:}"
  GPU_LIST="${GPU_LIST//,/ }"
  read -r -a _gpu_arr <<<"${GPU_LIST}"
  if (( ${#_gpu_arr[@]} > 2 )); then
    echo "RQVAE pretrain supports at most 2 GPUs. Received: ${GPU}" >&2
    exit 2
  fi
  if (( ${#_gpu_arr[@]} == 0 )) || [[ -z "${_gpu_arr[0]}" ]]; then
    echo "Invalid GPU spec: ${GPU}" >&2
    exit 2
  fi
  if ! [[ "${_gpu_arr[0]}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: ${_gpu_arr[0]}" >&2
    exit 2
  fi
  if (( ${#_gpu_arr[@]} == 2 )) && ! [[ "${_gpu_arr[1]}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: ${_gpu_arr[1]}" >&2
    exit 2
  fi

  DEVICE="cuda:0"
  CUDA_VISIBLE_GPUS="${_gpu_arr[*]// /,}"
fi
EPOCHS="${RQVAE_EPOCHS:-10000}"
LR="${RQVAE_LR:-1e-3}"
WEIGHT_DECAY="${RQVAE_WEIGHT_DECAY:-1e-4}"
LR_SCHEDULER="${RQVAE_LR_SCHEDULER:-linear}"
DATA_E_DIM="${RQVAE_E_DIM:-256}"
LAYERS="${RQVAE_LAYERS:-2048 1024 512}"
NUM_EMB="${RQVAE_NUM_EMB:-256 256 256}"
SK_EPS="${RQVAE_SK_EPS:-0.003 0.003 0.003}"
SK_ITERS="${RQVAE_SK_ITERS:-50}"
RUN_LOG_DIR="${PROJECT_ROOT}/reproduction_logs/rqvae"
mkdir -p "${RUN_LOG_DIR}"

RUN_LOG="${RUN_LOG_DIR}/${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo "RQ-VAE checkpoint training run"
echo "Dataset: ${DATASET}"
echo "Embedding: ${EMB_PATH}"
echo "Output parent: ${OUTPUT_PARENT}"
echo "Log: ${RUN_LOG}"
echo

if [ ! -f "${EMB_PATH}" ]; then
  echo "Missing embedding file: ${EMB_PATH}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_PARENT}"
NUM_EMB_ARGS=(${NUM_EMB})
SK_EPS_ARGS=(${SK_EPS})
LAYERS_ARGS=(${LAYERS})

MAX_NUM_EMB=0
for n in "${NUM_EMB_ARGS[@]}"; do
  if [[ "${n}" -gt "${MAX_NUM_EMB}" ]]; then
    MAX_NUM_EMB="${n}"
  fi
done

if [[ "${BATCH_SIZE}" -lt "${MAX_NUM_EMB}" ]]; then
  echo "WARN: requested batch_size=${BATCH_SIZE} < max(num_emb)=${MAX_NUM_EMB}; auto-adjusting to ${MAX_NUM_EMB}"
  BATCH_SIZE="${MAX_NUM_EMB}"
fi

if [[ -n "${CUDA_VISIBLE_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_GPUS}"
fi

echo "GPU selector: ${GPU}"

python3 -u "${RQVAE_DIR}/main.py" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --lr_scheduler_type "${LR_SCHEDULER}" \
  --e_dim "${DATA_E_DIM}" \
  --quant_loss_weight 1.0 \
  --beta "${BETA}" \
  --num_emb_list "${NUM_EMB_ARGS[@]}" \
  --sk_epsilons "${SK_EPS_ARGS[@]}" \
  --sk_iters "${SK_ITERS}" \
  --layers "${LAYERS_ARGS[@]}" \
  --vq_type vq \
  --loss_type mse \
  --dist l2 \
  --device "${DEVICE}" \
  --kmeans_init True \
  --data_path "${EMB_PATH}" \
  --ckpt_dir "${OUTPUT_PARENT}" \
  2>&1 | tee "${RUN_LOG}"

NEW_CKPT="$(find "${OUTPUT_PARENT}" -type f -name best_collision_model.pth -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | awk '{print $2}')"

if [ -z "${NEW_CKPT}" ]; then
  echo "Could not locate new best_collision_model.pth under ${OUTPUT_PARENT}" >&2
  exit 3
fi

cp "${NEW_CKPT}" "${TARGET_CKPT}"
ls -l "${NEW_CKPT}" "${TARGET_CKPT}"
echo "Copied canonical ckpt -> ${TARGET_CKPT}"
