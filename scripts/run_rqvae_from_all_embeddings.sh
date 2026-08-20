#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_rqvae_from_all_embeddings.sh [--emb-dir /path/to/dataset] [--gpu 0|0,1]

Options:
  --emb-dir   Directory containing Beauty.emb-llama.npy, Instruments.emb-llama.npy, Yelp.emb-llama.npy
              (default: dataset directory under repo)
  --gpu       GPU id(s) (e.g., 0 or 0,1) or cpu/cuda (default: 0). At most 2 ids supported.
EOF
}

EMB_DIR="${RQVAE_DATA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/dataset}"
GPU="${RQVAE_GPU:-${GPU:-0}}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_ROOT="${RQVAE_CKPT_ROOT:-${PROJECT_ROOT}/rqvae_ckpt}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --emb-dir)
      EMB_DIR="$2"
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

EMB_ROOT="$(cd "${EMB_DIR}" && pwd)"

BEAUTY_EMB="${EMB_ROOT}/beauty/Beauty.emb-llama.npy"
INSTR_EMB="${EMB_ROOT}/instruments/Instruments.emb-llama.npy"
YELP_EMB="${EMB_ROOT}/yelp/Yelp.emb-llama.npy"

for p in "${BEAUTY_EMB}" "${INSTR_EMB}" "${YELP_EMB}"; do
  if [[ ! -f "${p}" ]]; then
    echo "Missing expected embedding file: ${p}" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Running three-dataset RQ-VAE checkpoint reproduction..."
RQVAE_CKPT_ROOT="${CKPT_ROOT}" "${SCRIPT_DIR}/run_rqvae_from_embedding.sh" --embedding "${BEAUTY_EMB}" --dataset beauty --gpu "${GPU}"
RQVAE_CKPT_ROOT="${CKPT_ROOT}" "${SCRIPT_DIR}/run_rqvae_from_embedding.sh" --embedding "${INSTR_EMB}" --dataset instruments --gpu "${GPU}"
RQVAE_CKPT_ROOT="${CKPT_ROOT}" "${SCRIPT_DIR}/run_rqvae_from_embedding.sh" --embedding "${YELP_EMB}" --dataset yelp --gpu "${GPU}"

echo "Done. Check all three outputs under:"
echo "  ${CKPT_ROOT}/beauty/best_collision_model.pth"
echo "  ${CKPT_ROOT}/instruments/best_collision_model.pth"
echo "  ${CKPT_ROOT}/yelp/best_collision_model.pth"
