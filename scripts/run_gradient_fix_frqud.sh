#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 <beauty|instruments|yelp> <physical-gpu-index> <run-label>" >&2
  exit 2
fi

DATASET="$1"
GPU_INDEX="$2"
RUN_LABEL="$3"

if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "GPU index must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${RUN_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run label contains unsupported characters" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DIGER_DATA_ROOT:-${PROJECT_ROOT}/dataset}"
RQVAE_ROOT="${DIGER_RQVAE_ROOT:-${PROJECT_ROOT}/rqvae_ckpt}"

if [[ -n "${DIGER_ENV_BIN:-}" ]]; then
  PYTHON_BIN="${DIGER_ENV_BIN}/python"
  ACCELERATE_BIN="${DIGER_ENV_BIN}/accelerate"
else
  PYTHON_BIN="$(command -v python || true)"
  ACCELERATE_BIN="$(command -v accelerate || true)"
fi

case "${DATASET}" in
  beauty)
    FRQ_RATIO=1.5
    ASSIGNMENT_TEMPERATURE=2.0
    LR_ID=0.00001
    EXPECTED_EPOCHS=120
    EXPECTED_EARLY_STOP=15
    EXPECTED_BEAMS=20
    ;;
  instruments)
    FRQ_RATIO=2.0
    ASSIGNMENT_TEMPERATURE=2.0
    LR_ID=0.00002
    EXPECTED_EPOCHS=100
    EXPECTED_EARLY_STOP=15
    EXPECTED_BEAMS=20
    ;;
  yelp)
    FRQ_RATIO=1.1
    ASSIGNMENT_TEMPERATURE=1.5
    LR_ID=0.000002
    EXPECTED_EPOCHS=200
    EXPECTED_EARLY_STOP=20
    EXPECTED_BEAMS=80
    ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

CONFIG="${PROJECT_ROOT}/config/${DATASET}_gradient_fix.yaml"
TRAIN_LOG_DIR="./logs/true_e2e_frqud/${RUN_LABEL}"
ARTIFACT_DIR="${PROJECT_ROOT}/reproduction_logs/true_e2e_frqud/${RUN_LABEL}"
TRAIN_LOG="${ARTIFACT_DIR}/${DATASET}_train.log"
STATUS_JSON="${ARTIFACT_DIR}/run_status.json"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Could not find a Python executable" >&2
  exit 2
fi
if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Could not find an Accelerate executable" >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}/${DATASET}" ]]; then
  echo "Missing dataset directory: ${DATA_ROOT}/${DATASET}" >&2
  exit 2
fi
if [[ ! -f "${RQVAE_ROOT}/${DATASET}/best_collision_model.pth" ]]; then
  echo "Missing RQ-VAE checkpoint for ${DATASET}" >&2
  exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing configuration: ${CONFIG}" >&2
  exit 2
fi
if [[ -e "${ARTIFACT_DIR}" ]]; then
  echo "Refusing to overwrite existing artifact directory: ${ARTIFACT_DIR}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}"
"${PYTHON_BIN}" - "${STATUS_JSON}" "${DATASET}" "${RUN_LABEL}" "${GPU_INDEX}" \
  "${FRQ_RATIO}" "${ASSIGNMENT_TEMPERATURE}" "${LR_ID}" "${EXPECTED_BEAMS}" \
  "${TRAIN_LOG}" "${TRAIN_LOG_DIR}" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, dataset, label, gpu, ratio, temperature, lr_id, beams, train_log, train_log_dir = sys.argv[1:]
payload = {
    "schema_version": 1,
    "dataset": dataset,
    "run_label": label,
    "gpu": int(gpu),
    "training_mode": "true_e2e_frqud",
    "hot_threshold_ratio": float(ratio),
    "assignment_temperature": float(temperature),
    "gumbel_noise_scale": 1.0,
    "lr_id": float(lr_id),
    "lr_rec": 0.001,
    "num_beams": int(beams),
    "freeze_id_encoder": True,
    "freeze_rq": False,
    "assignment_forward": "hard_st",
    "use_cached_sinkhorn_forward": True,
    "seed": 2020,
    "train_log": train_log,
    "train_log_dir": train_log_dir,
    "status": "running",
    "started_at": datetime.now(timezone.utc).isoformat(),
}
with open(path, "w", encoding="utf-8") as file_obj:
    json.dump(payload, file_obj, indent=2, sort_keys=True)
    file_obj.write("\n")
PY

on_exit() {
  local exit_code=$?
  if [[ "${exit_code}" -ne 0 && -f "${STATUS_JSON}" ]]; then
    STATUS_PATH="${STATUS_JSON}" EXIT_CODE="${exit_code}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone

path = os.environ["STATUS_PATH"]
with open(path, encoding="utf-8") as file_obj:
    payload = json.load(file_obj)
payload.update({
    "status": "failed",
    "exit_code": int(os.environ["EXIT_CODE"]),
    "failure_reason": f"runner exited with code {int(os.environ['EXIT_CODE'])}",
    "finished_at": datetime.now(timezone.utc).isoformat(),
})
with open(path, "w", encoding="utf-8") as file_obj:
    json.dump(payload, file_obj, indent=2, sort_keys=True)
    file_obj.write("\n")
PY
  fi
  exit "${exit_code}"
}
trap on_exit EXIT

echo "[GradientFix] dataset=${DATASET} gpu=${GPU_INDEX} label=${RUN_LABEL}"
echo "[GradientFix] FrQUD ratio=${FRQ_RATIO} tau=${ASSIGNMENT_TEMPERATURE} lr_id=${LR_ID}"
echo "[GradientFix] validation-only training; test validation-best checkpoint once"

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${ACCELERATE_BIN}" launch \
  --config_file "${PROJECT_ROOT}/accelerate_config.yaml" \
  "${PROJECT_ROOT}/main.py" \
  --config="${CONFIG}" \
  --training_mode=true_e2e_frqud \
  --log_dir="${TRAIN_LOG_DIR}" \
  --data_path="${DATA_ROOT}" \
  --rqvae_path="${RQVAE_ROOT}/${DATASET}/best_collision_model.pth" \
  --assignment_forward=hard_st \
  --use_cached_sinkhorn_forward=true \
  --assignment_temperature="${ASSIGNMENT_TEMPERATURE}" \
  --gumbel_noise_scale=1.0 \
  --use_gumbel=true \
  --use_adaptive_selection=false \
  --use_gate_network=false \
  --use_soft_frequency=false \
  --use_learnable_sigma_gumbel=false \
  --use_simple_uncertainty_loss=false \
  --hot_threshold_ratio="${FRQ_RATIO}" \
  --usage_momentum=0.99 \
  --lr_id="${LR_ID}" \
  --lr_rec=0.001 \
  --freeze_id_encoder=true \
  --freeze_rq=false \
  --balance_loss_weight=0.0 \
  --gate_loss_weight=0.0 \
  --qs_loss_weight=0.0 \
  --gradient_accumulation_steps=1 \
  --evaluate_test_at_end=false \
  --epochs="${EXPECTED_EPOCHS}" \
  --early_stop="${EXPECTED_EARLY_STOP}" \
  --num_beams="${EXPECTED_BEAMS}" \
  2>&1 | tee "${TRAIN_LOG}"

MANIFEST="$("${PYTHON_BIN}" - \
  "${PROJECT_ROOT}/myckpt/${DATASET}" "${TRAIN_LOG_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
log_dir = sys.argv[2]
matches = []
for path in root.glob('*/manifest.json'):
    try:
        with path.open(encoding='utf-8') as file_obj:
            manifest = json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        continue
    if manifest.get('resolved_config', {}).get('log_dir') == log_dir:
        matches.append(path)
if matches:
    print(max(matches, key=lambda path: path.stat().st_mtime))
PY
)"

if [[ -z "${MANIFEST}" || ! -f "${MANIFEST}" ]]; then
  echo "Could not locate manifest for ${TRAIN_LOG_DIR}" >&2
  exit 1
fi

RUN_METADATA="$("${PYTHON_BIN}" - \
  "${MANIFEST}" "${DATASET}" "${TRAIN_LOG_DIR}" "${FRQ_RATIO}" \
  "${ASSIGNMENT_TEMPERATURE}" "${LR_ID}" "${EXPECTED_EPOCHS}" \
  "${EXPECTED_EARLY_STOP}" "${EXPECTED_BEAMS}" <<'PY'
import json
import sys

(
    path, dataset, log_dir, ratio, temperature, lr_id, epochs,
    early_stop, beams,
) = sys.argv[1:]
with open(path, encoding='utf-8') as file_obj:
    manifest = json.load(file_obj)
config = manifest.get('resolved_config', {})
expected = {
    'log_dir': log_dir,
    'training_mode': 'true_e2e_frqud',
    'hot_threshold_ratio': float(ratio),
    'assignment_temperature': float(temperature),
    'gumbel_noise_scale': 1.0,
    'use_gumbel': True,
    'use_adaptive_selection': False,
    'use_gate_network': False,
    'use_soft_frequency': False,
    'use_learnable_sigma_gumbel': False,
    'use_simple_uncertainty_loss': False,
    'usage_momentum': 0.99,
    'lr_id': float(lr_id),
    'lr_rec': 0.001,
    'num_beams': int(beams),
    'freeze_id_encoder': True,
    'freeze_rq': False,
    'use_cached_sinkhorn_forward': True,
    'assignment_forward': 'hard_st',
    'balance_loss_weight': 0.0,
    'gate_loss_weight': 0.0,
    'qs_loss_weight': 0.0,
    'gradient_accumulation_steps': 1,
    'evaluate_test_at_end': False,
    'seed': 2020,
    'epochs': int(epochs),
    'early_stop': int(early_stop),
}
mismatches = {
    key: {'actual': config.get(key), 'expected': value}
    for key, value in expected.items()
    if config.get(key) != value
}
for key in ('max_eval_batches', 'stop_after_epoch'):
    if config.get(key) is not None:
        mismatches[key] = {'actual': config.get(key), 'expected': None}
if manifest.get('dataset') != dataset:
    mismatches['dataset'] = {
        'actual': manifest.get('dataset'), 'expected': dataset,
    }
if manifest.get('status') not in {'completed', 'stopped'}:
    mismatches['status'] = {
        'actual': manifest.get('status'), 'expected': 'completed or stopped',
    }
if mismatches:
    raise SystemExit(f'manifest validation failed: {mismatches}')
best_epoch = manifest.get('best_epoch')
best_checkpoint = manifest.get('best_checkpoint')
if not isinstance(best_epoch, int) or not isinstance(best_checkpoint, str):
    raise SystemExit('manifest does not contain a valid best checkpoint')
print(f'{best_epoch}\t{best_checkpoint}')
PY
)"
IFS=$'\t' read -r BEST_EPOCH BEST_CHECKPOINT <<<"${RUN_METADATA}"
if [[ "${BEST_CHECKPOINT}" != /* ]]; then
  BEST_CHECKPOINT="${PROJECT_ROOT}/${BEST_CHECKPOINT#./}"
fi
if [[ ! -f "${BEST_CHECKPOINT}" || ! -f "${BEST_CHECKPOINT}.rqvae" ]]; then
  echo "Best checkpoint bundle is incomplete: ${BEST_CHECKPOINT}" >&2
  exit 1
fi
CODE_PATH="${BEST_CHECKPOINT%.pt}.code.json"
if [[ ! -f "${CODE_PATH}" ]]; then
  echo "Best code map is missing: ${CODE_PATH}" >&2
  exit 1
fi

TEST_JSON="${ARTIFACT_DIR}/best_e${BEST_EPOCH}_single_test.json"
TEST_LOG="${ARTIFACT_DIR}/best_e${BEST_EPOCH}_single_test.log"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON_BIN}" \
  "${PROJECT_ROOT}/scripts/evaluate_checkpoint.py" \
  --checkpoint "${BEST_CHECKPOINT}" \
  --config "${CONFIG}" \
  --split test \
  --data-path "${DATA_ROOT}" \
  --device cuda \
  --output "${TEST_JSON}" \
  2>&1 | tee "${TEST_LOG}"

"${PYTHON_BIN}" - \
  "${TEST_JSON}" "${DATASET}" "${BEST_CHECKPOINT}" "${EXPECTED_BEAMS}" <<'PY'
import json
import sys

path, dataset, checkpoint, beams = sys.argv[1:]
with open(path, encoding='utf-8') as file_obj:
    result = json.load(file_obj)
expected = {
    'dataset': dataset,
    'split': 'test',
    'checkpoint': checkpoint,
    'num_beams': int(beams),
}
mismatches = {
    key: {'actual': result.get(key), 'expected': value}
    for key, value in expected.items()
    if result.get(key) != value
}
required_metrics = {'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10'}
if not isinstance(result.get('metrics'), dict):
    mismatches['metrics'] = {'actual': type(result.get('metrics')).__name__, 'expected': 'object'}
elif not required_metrics.issubset(result['metrics']):
    mismatches['metrics'] = {
        'actual': sorted(result['metrics']), 'expected': sorted(required_metrics),
    }
if mismatches:
    raise SystemExit(f'test result validation failed: {mismatches}')
PY

STATUS_PATH="${STATUS_JSON}" MANIFEST_PATH="${MANIFEST}" TEST_PATH="${TEST_JSON}" \
  TEST_LOG_PATH="${TEST_LOG}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from datetime import datetime, timezone

with open(os.environ["STATUS_PATH"], encoding="utf-8") as file_obj:
    status = json.load(file_obj)
with open(os.environ["MANIFEST_PATH"], encoding="utf-8") as file_obj:
    manifest = json.load(file_obj)
with open(os.environ["TEST_PATH"], encoding="utf-8") as file_obj:
    test = json.load(file_obj)
status.update({
    "status": "completed",
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "manifest": os.environ["MANIFEST_PATH"],
    "manifest_status": manifest.get("status"),
    "stop_reason": manifest.get("stop_reason"),
    "best_epoch": manifest.get("best_epoch"),
    "best_checkpoint": manifest.get("best_checkpoint"),
    "train_log": status["train_log"],
    "test_json": os.environ["TEST_PATH"],
    "test_log": os.environ["TEST_LOG_PATH"],
    "test_metrics": test.get("metrics"),
})
with open(os.environ["STATUS_PATH"], "w", encoding="utf-8") as file_obj:
    json.dump(status, file_obj, indent=2, sort_keys=True)
    file_obj.write("\n")
PY

trap - EXIT
echo "[GradientFix] complete: ${TEST_JSON}"
