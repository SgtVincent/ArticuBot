#!/usr/bin/env bash
set -eo pipefail

# Reproducible scaling experiment for microwave_7167:
# - Generate 1000 successful demos (keeps all.gif per demo)
# - Convert to training format for N demos
# - Finetune high-level policy per N
# - Evaluate with eval_sim_to_sim.py on the original mesh model (keeps evaluation GIFs)
# - Plot scaling curve

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Source environment
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  if [[ -f /mnt/wjb/software/miniconda3/etc/profile.d/conda.sh ]]; then
    source /mnt/wjb/software/miniconda3/etc/profile.d/conda.sh
  fi
  conda activate ./articubot || echo "Warning: Failed to activate ./articubot"
fi

SIM_DATASET_DIR_DEFAULT="data/custom_objects_v2/sim"
OBJECT_TYPE_DEFAULT="microwave"
OBJECT_ID_DEFAULT="7167"
DEMO_EXP_NAME_DEFAULT="microwave_7167_train1000"
PRETRAINED_HIGH_DEFAULT="data/high_level_200_obj_ckpt.pth"
LOW_LEVEL_EXP_DIR_DEFAULT="data/low-level-ckpt"
LOW_LEVEL_CKPT_DEFAULT="low-level.ckpt"
RM4D_MAP_DEFAULT="data/rm4d_franka_1M.npy"
SIZES_DEFAULT=(10 20 50 100 200)
CONVERT_WORKERS_DEFAULT=1
CONVERT_BATCH_SIZE_DEFAULT=0

usage() {
  cat <<EOF
Usage:
  scripts/experiments/microwave_7167_scaling.sh <command> [options]

Commands:
  generate   Generate 1000 successful demos (keeps all.gif)
  convert    Convert demos into training format for multiple N
  finetune   Finetune high-level policy for multiple N
  eval       Evaluate each finetuned checkpoint on the original model (keeps GIFs)
  plot       Plot scaling curve from evaluation outputs
  all        Run convert+finetune+eval+plot (assumes demos already exist)

Options (env vars):
  SIM_DATASET_DIR     (default: ${SIM_DATASET_DIR_DEFAULT})
  OBJECT_TYPE         (default: ${OBJECT_TYPE_DEFAULT})
  OBJECT_ID           (default: ${OBJECT_ID_DEFAULT})
  DEMO_EXP_NAME       (default: ${DEMO_EXP_NAME_DEFAULT})
  PRETRAINED_HIGH     (default: ${PRETRAINED_HIGH_DEFAULT})
  LOW_LEVEL_EXP_DIR   (default: ${LOW_LEVEL_EXP_DIR_DEFAULT})
  LOW_LEVEL_CKPT_NAME (default: ${LOW_LEVEL_CKPT_DEFAULT})
  RM4D_MAP            (default: ${RM4D_MAP_DEFAULT})
  SIZES               (space-separated, default: "${SIZES_DEFAULT[*]}")
  CONVERT_WORKERS     (default: ${CONVERT_WORKERS_DEFAULT})
  CONVERT_BATCH_SIZE  (default: ${CONVERT_BATCH_SIZE_DEFAULT})  # 0 = auto (num workers)
  NUM_EPOCHS          (default: 20)
  SAVE_FREQ           (default: 5)
  NUM_TRIALS          (default: 5)
  EVAL_HORIZON        (default: 70)
  GRASP_SOURCE        (default: ondemand)  # auto|precomputed|ondemand

Notes:
  - Run inside the project env:
      conda activate articubot && source prepare.sh
  - For finetuning without W&B login prompts:
      export WANDB_MODE=offline
EOF
}

require_env() {
  if [[ -z "${PROJECT_DIR:-}" ]]; then
    echo "ERROR: PROJECT_DIR is not set. Run: source prepare.sh" >&2
    exit 1
  fi
}

sizes_array() {
  if [[ -n "${SIZES:-}" ]]; then
    # shellcheck disable=SC2206
    echo ${SIZES}
  else
    echo ${SIZES_DEFAULT[*]}
  fi
}

find_latest_high_ckpt() {
  local exp_suffix="$1"   # e.g. microwave_7167_n50
  local num_epochs="$2"   # e.g. 20

  local exps_root="$ROOT_DIR/weighted_displacement_model/exps"
  local latest_dir
  latest_dir="$(ls -td "${exps_root}"/*"_${exp_suffix}" 2>/dev/null | head -1 || true)"

  if [[ -z "$latest_dir" ]]; then
    echo ""; return 0
  fi

  local ckpt="${latest_dir}/model_${num_epochs}.pth"
  if [[ -f "$ckpt" ]]; then
    echo "$ckpt"; return 0
  fi

  # Fallback: pick the latest model_*.pth
  ckpt="$(ls -t "${latest_dir}"/model_*.pth 2>/dev/null | head -1 || true)"
  echo "$ckpt"
}

cmd_generate() {
  require_env

  local sim_dataset_dir="${SIM_DATASET_DIR:-$SIM_DATASET_DIR_DEFAULT}"
  local object_type="${OBJECT_TYPE:-$OBJECT_TYPE_DEFAULT}"
  local object_id="${OBJECT_ID:-$OBJECT_ID_DEFAULT}"
  local demo_exp_name="${DEMO_EXP_NAME:-$DEMO_EXP_NAME_DEFAULT}"
  local rm4d_map="${RM4D_MAP:-$RM4D_MAP_DEFAULT}"
  local grasp_source="${GRASP_SOURCE:-ondemand}"

  local asset_dir="${sim_dataset_dir}/${object_type}_${object_id}"

  mkdir -p "$ROOT_DIR/outputs/demo_gen_logs"
  local log_path="$ROOT_DIR/outputs/demo_gen_logs/${object_type}_${object_id}_${demo_exp_name}.log"

  echo "Generating demos in: ${asset_dir}"
  echo "Experiment name: ${demo_exp_name}"
  echo "Log: ${log_path}"

  echo "==================== $(date -Is) generate ====================" | tee -a "${log_path}"

  # If the GraspGen container is running with /models mounted from the sibling
  # GraspGen checkout, point GRASPGEN_HOST_ROOT at that mount source so
  # on-demand grasps can run without docker-cp.
  if [[ -z "${GRASPGEN_HOST_ROOT:-}" ]]; then
    local graspgen_models_dir="$ROOT_DIR/../GraspGen/GraspGenModels"
    if [[ -d "$graspgen_models_dir" ]]; then
      export GRASPGEN_HOST_ROOT
      GRASPGEN_HOST_ROOT="$(cd "$graspgen_models_dir" && pwd)"
      echo "Using GRASPGEN_HOST_ROOT=${GRASPGEN_HOST_ROOT}" | tee -a "${log_path}"
    fi
  fi

  # IMPORTANT:
  # - If predicted_grasps.yml exists and grasp_source is auto/precomputed, gen_demo_custom.py will
  #   enforce fixed scale + fixed joint angle range to avoid grasp mismatches.
  # - Parallel generation with 16 workers, 4 GPUs, and random scale 0.4-0.5
  export ARTICUBOT_NUM_GPUS=4
  PYTHONUNBUFFERED=1 python -u manipulation/gen_demo_custom_parallel.py \
    --asset-dir "${asset_dir}" \
    --exp-name "${demo_exp_name}" \
    --sampling-method integrated_inverse \
    --rm4d-map "${rm4d_map}" \
    --num-workers 16 \
    --num-to-generate 1000 \
    --max-try-times 5000 \
    --timeout-init 180 \
    --timeout-exec 300 \
    --grasp-source "${grasp_source}" \
    --top-k-grasps 5 \
    --size-scale 0.4 0.6 \
    --object-z-offset 0.3 0.5 \
    2>&1 | tee -a "${log_path}"
}

cmd_convert() {
  require_env

  local sim_dataset_dir="${SIM_DATASET_DIR:-$SIM_DATASET_DIR_DEFAULT}"
  local object_type="${OBJECT_TYPE:-$OBJECT_TYPE_DEFAULT}"
  local object_id="${OBJECT_ID:-$OBJECT_ID_DEFAULT}"
  local demo_exp_name="${DEMO_EXP_NAME:-$DEMO_EXP_NAME_DEFAULT}"

  local convert_workers="${CONVERT_WORKERS:-$CONVERT_WORKERS_DEFAULT}"
  local convert_batch_size="${CONVERT_BATCH_SIZE:-$CONVERT_BATCH_SIZE_DEFAULT}"

  local asset_dir="${sim_dataset_dir}/${object_type}_${object_id}"
  local input_dir="${asset_dir}/experiment/${demo_exp_name}"

  if [[ ! -d "$input_dir" ]]; then
    echo "ERROR: Demo experiment directory not found: ${input_dir}" >&2
    exit 1
  fi

  local sizes
  sizes="$(sizes_array)"

  for n in $sizes; do
    local out_dir="$ROOT_DIR/data/custom_finetune/${object_type}_${object_id}_n${n}"
    echo "Converting N=${n} -> ${out_dir}"
    python scripts/custom_finetune/convert_custom_demos.py \
      --input-dir "${input_dir}" \
      --output-dir "${out_dir}" \
      --num-experiment "${n}" \
      --num-workers "${convert_workers}" \
      --batch-size "${convert_batch_size}"
  done
}

cmd_finetune() {
  require_env

  export WANDB_MODE="${WANDB_MODE:-offline}"

  local object_type="${OBJECT_TYPE:-$OBJECT_TYPE_DEFAULT}"
  local object_id="${OBJECT_ID:-$OBJECT_ID_DEFAULT}"
  local pretrained_high="${PRETRAINED_HIGH:-$PRETRAINED_HIGH_DEFAULT}"
  local num_epochs="${NUM_EPOCHS:-20}"
  local save_freq="${SAVE_FREQ:-5}"

  local sizes
  sizes="$(sizes_array)"

  mkdir -p "$ROOT_DIR/outputs/train_logs"

  for n in $sizes; do
    local ds_dir="$ROOT_DIR/data/custom_finetune/${object_type}_${object_id}_n${n}"
    local exp_name="${object_type}_${object_id}_n${n}"

    if [[ ! -d "$ds_dir" ]]; then
      echo "ERROR: Dataset directory not found: ${ds_dir}" >&2
      exit 1
    fi

    echo "Finetuning high-level for N=${n} (WANDB_MODE=${WANDB_MODE})"
    local log_path="$ROOT_DIR/outputs/train_logs/high_level_${exp_name}.log"
    echo "==================== $(date -Is) finetune_n${n} ====================" | tee -a "${log_path}"
    bash scripts/custom_finetune/finetune_high_level.sh \
      --dataset-path "${ds_dir}" \
      --pretrained-model "${pretrained_high}" \
      --exp-name "${exp_name}" \
      --num-epochs "${num_epochs}" \
      --save-freq "${save_freq}" \
      2>&1 | tee -a "${log_path}"
  done
}

cmd_eval() {
  require_env

  local sim_dataset_dir="${SIM_DATASET_DIR:-$SIM_DATASET_DIR_DEFAULT}"
  local object_type="${OBJECT_TYPE:-$OBJECT_TYPE_DEFAULT}"
  local object_id="${OBJECT_ID:-$OBJECT_ID_DEFAULT}"
  local low_level_exp_dir="${LOW_LEVEL_EXP_DIR:-$LOW_LEVEL_EXP_DIR_DEFAULT}"
  local low_level_ckpt="${LOW_LEVEL_CKPT_NAME:-$LOW_LEVEL_CKPT_DEFAULT}"
  local num_trials="${NUM_TRIALS:-5}"
  local horizon="${EVAL_HORIZON:-70}"
  local num_epochs="${NUM_EPOCHS:-20}"

  local sizes
  sizes="$(sizes_array)"

  mkdir -p "$ROOT_DIR/outputs/sim_to_sim_eval/microwave_7167_scaling"

  for n in $sizes; do
    local exp_suffix="${object_type}_${object_id}_n${n}"
    local ckpt
    ckpt="$(find_latest_high_ckpt "${exp_suffix}" "${num_epochs}")"

    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
      echo "ERROR: Could not find checkpoint for ${exp_suffix}. Did finetune run?" >&2
      exit 1
    fi

    local save_dir="$ROOT_DIR/outputs/sim_to_sim_eval/microwave_7167_scaling/n${n}"
    mkdir -p "$save_dir"

    echo "Evaluating N=${n} with ckpt: ${ckpt}"
    python eval_sim_to_sim.py \
      --sim-dataset-dir "${sim_dataset_dir}" \
      --object-type "${object_type}" \
      --object-id "${object_id}" \
      --low-level-exp-dir "${low_level_exp_dir}" \
      --low-level-ckpt-name "${low_level_ckpt}" \
      --high-level-ckpt "${ckpt}" \
      --num-trials "${num_trials}" \
      --horizon "${horizon}" \
      --save-dir "${save_dir}"
  done
}

cmd_plot() {
  require_env

  local root="$ROOT_DIR/outputs/sim_to_sim_eval/microwave_7167_scaling"
  python scripts/experiments/plot_sim_to_sim_scaling.py \
    --root "${root}" \
    --out "${root}/scaling_curve.png"
}

cmd_all() {
  cmd_generate
  cmd_convert
  cmd_finetune
  cmd_eval
  cmd_plot
}

cmd="${1:-}"
case "$cmd" in
  generate) shift; cmd_generate "$@";;
  convert) shift; cmd_convert "$@";;
  finetune) shift; cmd_finetune "$@";;
  eval) shift; cmd_eval "$@";;
  plot) shift; cmd_plot "$@";;
  all) shift; cmd_all "$@";;
  -h|--help|help|"") usage; exit 0;;
  *) echo "Unknown command: $cmd"; usage; exit 1;;
esac
