#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rqvae_from_embedding.sh --embedding /abs/path/to/xxx.npy [--dataset beauty|instruments|yelp] [--gpu 0|0,1]

Options:
  --embedding   Path to LLM semantic embedding (.npy)
  --dataset     Optional dataset hint (beauty / instruments / yelp)
  --gpu         GPU id(s) (e.g., 0 or 0,1) or cpu/cuda (default: 0). At most 2 ids supported.

The script will route to the canonical path:
  rqvae_ckpt/<dataset>/best_collision_model.pth
EOF
}

EMB_PATH=""
DATASET=""
GPU="${RQVAE_GPU:-${GPU:-0}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding|-e)
      EMB_PATH="$2"
      shift 2
      ;;
    --dataset|-d)
      DATASET="$2"
      shift 2
      ;;
    --gpu|-g)
      GPU="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
esac
done

if [[ "${GPU}" != "cpu" && "${GPU}" != "cuda" ]]; then
  GPU="${GPU#cuda:}"
  GPU_LIST="${GPU//,/ }"
  read -r -a _gpu_arr <<<"${GPU_LIST}"
  if (( ${#_gpu_arr[@]} > 2 )); then
    echo "RQ-VAE scripts support at most 2 GPUs. Received: ${GPU}" >&2
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
  GPU="$(IFS=,; echo "${_gpu_arr[*]}")"
fi

if [[ -z "${EMB_PATH}" ]]; then
  echo "--embedding is required" >&2
  usage
  exit 2
fi

if [[ ! -f "${EMB_PATH}" ]]; then
  echo "Missing embedding file: ${EMB_PATH}" >&2
  exit 2
fi

if [[ -z "${DATASET}" ]]; then
  EMB_BASENAME_LOWER="$(basename "${EMB_PATH}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${EMB_BASENAME_LOWER}" == *beauty* ]]; then
    DATASET="beauty"
  elif [[ "${EMB_BASENAME_LOWER}" == *instrument* ]]; then
    DATASET="instruments"
  elif [[ "${EMB_BASENAME_LOWER}" == *yelp* ]]; then
    DATASET="yelp"
  else
    # Auto-detect by number of rows (unique for this repository):
    # beauty=12101, instruments=9922, yelp=20033
    DATASET="$(python3 - "${EMB_PATH}" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
shape0 = int(np.load(path, mmap_mode='r').shape[0])
if shape0 == 12101:
    print('beauty')
elif shape0 == 9922:
    print('instruments')
elif shape0 == 20033:
    print('yelp')
else:
    print('', end='')
PY
)"
  fi
fi

if [[ "${DATASET}" != beauty && "${DATASET}" != instruments && "${DATASET}" != yelp ]]; then
  echo "Cannot infer dataset from embedding path (${EMB_PATH})." >&2
  echo "Please provide --dataset beauty|instruments|yelp explicitly." >&2
  exit 2
fi

PROJECT_ROOT="${RQVAE_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CKPT_ROOT="${RQVAE_CKPT_ROOT:-${PROJECT_ROOT}/rqvae_ckpt}"

echo "Preparing RQ-VAE checkpoint reproduction"
echo "Embedding: ${EMB_PATH}"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"

RQVAE_GPU="${GPU}" \
RQVAE_CKPT_ROOT="${CKPT_ROOT}" \
RQVAE_EMBEDDING_PATH="${EMB_PATH}" \
RQVAE_DATASET_TAG="${DATASET}" \
bash "${PROJECT_ROOT}/scripts/run_rqvae_pretrain.sh" "${DATASET}"

echo "Done. Canonical RQ-VAE checkpoint: ${CKPT_ROOT}/${DATASET}/best_collision_model.pth"
