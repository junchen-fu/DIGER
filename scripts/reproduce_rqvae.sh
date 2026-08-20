#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/reproduce_rqvae.sh [--embedding /path/to/custom_emb.npy --dataset beauty|instruments|yelp]
  bash scripts/reproduce_rqvae.sh --emb-dir /path/to/dataset
  bash scripts/reproduce_rqvae.sh --all

Options:
  --embedding    Path to LLM embedding for a single dataset.
  --dataset      Optional dataset hint for --embedding (beauty | instruments | yelp).
  --emb-dir      Directory containing beauty/instruments/yelp embedding files.
  --gpu          GPU id(s), e.g. 0 or 0,1 or "0 1" (at most 2 supported).
  --ckpt-root    Where to place reproduced ckpts (default: ./rqvae_ckpt).
  --all          Reproduce all 3 datasets from repo default dataset directory.
  --no-verify    Skip verification step after reproduction.

Examples:
  bash scripts/reproduce_rqvae.sh --embedding ./dataset/beauty/Beauty.emb-llama.npy --dataset beauty
  bash scripts/reproduce_rqvae.sh --embedding /path/to/llm_embedding.npy --dataset yelp --gpu 0,1
  bash scripts/reproduce_rqvae.sh --all --gpu 0
EOF
}

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SINGLE_EMB=""
DATASET=""
EMB_DIR=""
DO_ALL=0
GPU_SPEC="${RQVAE_GPU:-${GPU:-0}}"
CKPT_ROOT="${RQVAE_CKPT_ROOT:-${PROJECT_ROOT}/rqvae_ckpt}"
DO_VERIFY=1

normalize_gpu() {
  local spec="$1"
  if [[ "${spec}" == "cpu" || "${spec}" == "cuda" ]]; then
    echo "${spec}"
    return 0
  fi
  if [[ "${spec}" == cuda:* ]]; then
    spec="${spec#cuda:}"
  fi

  local list="${spec//,/ }"
  local arr=()
  read -r -a arr <<<"${list}"
  if (( ${#arr[@]} > 2 )); then
    return 1
  fi
  if (( ${#arr[@]} == 0 )) || [[ -z "${arr[0]}" ]]; then
    return 1
  fi
  if ! [[ "${arr[0]}" =~ ^[0-9]+$ ]]; then
    return 2
  fi
  if (( ${#arr[@]} == 2 )) && ! [[ "${arr[1]}" =~ ^[0-9]+$ ]]; then
    return 2
  fi

  if (( ${#arr[@]} == 2 )); then
    echo "${arr[0]},${arr[1]}"
  else
    echo "${arr[0]}"
  fi
}

GPU_SPEC="$(normalize_gpu "${GPU_SPEC}")" || {
  status=$?
  if (( status == 1 )); then
    echo "Invalid or unsupported GPU spec: ${RQVAE_GPU:-${GPU:-0}}" >&2
  else
    echo "Invalid GPU index in spec: ${RQVAE_GPU:-${GPU:-0}}" >&2
  fi
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --embedding|-e)
      SINGLE_EMB="$2"
      shift 2
      ;;
    --dataset|-d)
      DATASET="$2"
      shift 2
      ;;
    --emb-dir)
      EMB_DIR="$2"
      shift 2
      ;;
    --gpu|-g)
      GPU_SPEC="$(normalize_gpu "$2")" || {
        status=$?
        if (( status == 1 )); then
          echo "Invalid or unsupported GPU spec: $2" >&2
        else
          echo "Invalid GPU index in spec: $2" >&2
        fi
        exit 2
      }
      shift 2
      ;;
    --ckpt-root)
      CKPT_ROOT="$2"
      shift 2
      ;;
    --all)
      DO_ALL=1
      shift
      ;;
    --verify)
      DO_VERIFY=1
      shift
      ;;
    --no-verify)
      DO_VERIFY=0
      shift
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

if [[ -z "${SINGLE_EMB}" && "${DO_ALL}" -eq 0 && -z "${EMB_DIR}" ]]; then
  echo "Need --embedding, --emb-dir, or --all." >&2
  usage
  exit 2
fi

if [[ -n "${SINGLE_EMB}" && -n "${EMB_DIR}" ]]; then
  echo "Pass only one of --embedding or --emb-dir, not both." >&2
  exit 2
fi

if [[ -n "${DATASET}" && "${SINGLE_EMB}" == "" ]]; then
  echo "--dataset requires --embedding." >&2
  exit 2
fi

if [[ -n "${DATASET}" ]]; then
  case "${DATASET}" in
    beauty|instruments|yelp)
      ;;
    *)
      echo "Invalid dataset: ${DATASET}" >&2
      exit 2
      ;;
  esac
fi

if [[ "${DO_ALL}" -eq 1 && "${SINGLE_EMB}" != "" ]]; then
  echo "--all cannot be used with --embedding." >&2
  exit 2
fi

if [[ "${DO_ALL}" -eq 1 && "${EMB_DIR}" != "" ]]; then
  echo "--all cannot be used with --emb-dir." >&2
  exit 2
fi

if [[ "${DO_ALL}" -eq 1 ]]; then
  echo "[Step] Reproduce all 3 datasets from repo embeddings"
  RQVAE_CKPT_ROOT="${CKPT_ROOT}" \
    RQVAE_GPU="${GPU_SPEC}" \
    bash "${PROJECT_ROOT}/scripts/run_rqvae_from_all_embeddings.sh"
else
  if [[ "${EMB_DIR}" != "" ]]; then
    echo "[Step] Reproduce 3 datasets from embedding directory"
    RQVAE_CKPT_ROOT="${CKPT_ROOT}" \
      RQVAE_GPU="${GPU_SPEC}" \
      bash "${PROJECT_ROOT}/scripts/run_rqvae_from_all_embeddings.sh" --emb-dir "${EMB_DIR}"
  else
    echo "[Step] Reproduce single dataset from provided embedding"
    SINGLE_ARGS=(--embedding "${SINGLE_EMB}" --gpu "${GPU_SPEC}")
    if [[ -n "${DATASET}" ]]; then
      SINGLE_ARGS+=(--dataset "${DATASET}")
    fi
    RQVAE_CKPT_ROOT="${CKPT_ROOT}" \
      bash "${PROJECT_ROOT}/scripts/run_rqvae_from_embedding.sh" \
        "${SINGLE_ARGS[@]}"
  fi
fi

if [[ "${DO_VERIFY}" -eq 0 ]]; then
  echo "Verification skipped by request."
  exit 0
fi

VERIFY_ARGS=(--ckpt_root "${CKPT_ROOT}" --strict)

if [[ "${SINGLE_EMB}" != "" ]]; then
  if [[ -z "${DATASET}" ]]; then
    DETECTED_DATASET="$(python3 - "$SINGLE_EMB" <<'PY'
import os
import sys
import numpy as np

path = sys.argv[1]
name = os.path.basename(path).lower()
if 'beauty' in name:
    print('beauty')
elif 'instrument' in name:
    print('instruments')
elif 'yelp' in name:
    print('yelp')
else:
    shape0 = int(np.load(path, mmap_mode='r').shape[0])
    if shape0 == 12101:
        print('beauty')
    elif shape0 == 9922:
        print('instruments')
    elif shape0 == 20033:
        print('yelp')
    else:
        print('')
PY)"
  else
    DETECTED_DATASET="${DATASET}"
  fi
  if [[ -n "${DETECTED_DATASET}" ]]; then
    VERIFY_ARGS+=(--datasets "${DETECTED_DATASET}")
  else
    echo "[WARN] Cannot infer single dataset for comparison, verifying all datasets if available."
  fi
fi

echo "[Step] Verify reproduced checkpoint structure and training configuration"
python3 "${PROJECT_ROOT}/scripts/rqvae/verify_rqvae_ckpt.py" "${VERIFY_ARGS[@]}"
echo "[DONE] RQ-VAE checkpoint reproduction run finished."
