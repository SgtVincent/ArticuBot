#!/usr/bin/env bash
set -eo pipefail

# Compare heuristic vs integrated_inverse demo generation for 100 valid demos.
# Outputs:
#  - logs: outputs/compare_sampling_logs/
#  - results: outputs/compare_sampling_results/microwave_7167_compare_100.json

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SIM_DATASET_DIR_DEFAULT="data/custom_objects_v2/sim"
OBJECT_TYPE_DEFAULT="microwave"
OBJECT_ID_DEFAULT="7167"
RM4D_MAP_DEFAULT="data/rm4d_franka_1M.npy"
NUM_DEMOS_DEFAULT=100
MAX_TRY_DEFAULT=200000
TIMEOUT_INIT_DEFAULT=180
TIMEOUT_EXEC_DEFAULT=300

SIM_DATASET_DIR="${SIM_DATASET_DIR:-$SIM_DATASET_DIR_DEFAULT}"
OBJECT_TYPE="${OBJECT_TYPE:-$OBJECT_TYPE_DEFAULT}"
OBJECT_ID="${OBJECT_ID:-$OBJECT_ID_DEFAULT}"
RM4D_MAP="${RM4D_MAP:-$RM4D_MAP_DEFAULT}"
NUM_DEMOS="${NUM_DEMOS:-$NUM_DEMOS_DEFAULT}"
MAX_TRY_TIMES="${MAX_TRY_TIMES:-$MAX_TRY_DEFAULT}"
TIMEOUT_INIT="${TIMEOUT_INIT:-$TIMEOUT_INIT_DEFAULT}"
TIMEOUT_EXEC="${TIMEOUT_EXEC:-$TIMEOUT_EXEC_DEFAULT}"

# Best-effort conda activation (only if not already active)
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_PREFIX##*/}" != "articubot" ]]; then
  if [[ -f /home/junting/miniforge3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/junting/miniforge3/etc/profile.d/conda.sh
    conda activate articubot
  fi
fi

if [[ -z "${PROJECT_DIR:-}" ]]; then
  echo "PROJECT_DIR not set; sourcing prepare.sh"
  # shellcheck disable=SC1091
  source "$ROOT_DIR/prepare.sh"
fi

mkdir -p "$ROOT_DIR/outputs/compare_sampling_logs"
mkdir -p "$ROOT_DIR/outputs/compare_sampling_results"

asset_dir="${SIM_DATASET_DIR}/${OBJECT_TYPE}_${OBJECT_ID}"

summary_json="$ROOT_DIR/outputs/compare_sampling_results/${OBJECT_TYPE}_${OBJECT_ID}_compare_${NUM_DEMOS}.json"

record_result() {
  local method="$1"
  local exp_name="$2"
  local log_path="$3"
  local start_ts="$4"
  local end_ts="$5"

  local duration_s=$((end_ts - start_ts))
  local mp_trials
  mp_trials=$(grep -a -c "\\[MP\\] try_idx" "$log_path" || true)

  local exp_dir="$ROOT_DIR/${SIM_DATASET_DIR}/${OBJECT_TYPE}_${OBJECT_ID}/experiment/${exp_name}"
  local counts
  counts=$(python scripts/experiments/count_valid_demos.py --experiment-dir "$exp_dir")
  local valid
  valid=$(python -c "import json,sys; print(json.loads(sys.argv[1]).get('valid', 0))" "$counts")
  local total
  total=$(python -c "import json,sys; print(json.loads(sys.argv[1]).get('total', 0))" "$counts")

  python - <<PY
import json
from pathlib import Path

summary_path = Path("$summary_json")
if summary_path.exists():
    data = json.loads(summary_path.read_text())
else:
    data = {}

data["$method"] = {
    "exp_name": "$exp_name",
    "sampling_method": "$method",
    "num_target": $NUM_DEMOS,
    "start_ts": $start_ts,
    "end_ts": $end_ts,
    "duration_s": $duration_s,
    "valid": $valid,
    "total": $total,
    "mp_trials": int("$mp_trials"),
    "log_path": "$log_path",
}
summary_path.write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2))
PY
}

run_method() {
  local method="$1"
  local exp_name="$2"
  local log_path="$3"
  local rm4d_flag="$4"

  echo "=== ${method} ===" | tee -a "$log_path"
  echo "exp_name=${exp_name}" | tee -a "$log_path"
  echo "asset_dir=${asset_dir}" | tee -a "$log_path"
  echo "num_to_generate=${NUM_DEMOS}" | tee -a "$log_path"

  local start_ts
  start_ts=$(date +%s)

  GRASP_SOURCE=ondemand PYTHONUNBUFFERED=1 python -u manipulation/gen_demo_custom.py \
    --asset-dir "$asset_dir" \
    --exp-name "$exp_name" \
    --sampling-method "$method" \
    --num-to-generate "$NUM_DEMOS" \
    --max-try-times "$MAX_TRY_TIMES" \
    --timeout-init "$TIMEOUT_INIT" \
    --timeout-exec "$TIMEOUT_EXEC" \
    $rm4d_flag \
    2>&1 | tee -a "$log_path"

  local end_ts
  end_ts=$(date +%s)

  record_result "$method" "$exp_name" "$log_path" "$start_ts" "$end_ts"
}

# Run integrated_inverse first, then heuristic
integrated_exp="${OBJECT_TYPE}_${OBJECT_ID}_compare_integrated_inverse_${NUM_DEMOS}"
heuristic_exp="${OBJECT_TYPE}_${OBJECT_ID}_compare_heuristic_${NUM_DEMOS}"

integrated_log="$ROOT_DIR/outputs/compare_sampling_logs/${integrated_exp}.log"
heuristic_log="$ROOT_DIR/outputs/compare_sampling_logs/${heuristic_exp}.log"

run_method "integrated_inverse" "$integrated_exp" "$integrated_log" "--rm4d-map $RM4D_MAP"
run_method "heuristic" "$heuristic_exp" "$heuristic_log" ""

echo "Comparison results saved to: $summary_json"
