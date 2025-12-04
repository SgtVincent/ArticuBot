#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_DIR"

usage() {
  cat <<EOF
Usage: $0 [options]

Run the custom-object pipeline for a single asset directory.

Options:
  --asset-dir PATH        Path to the object asset directory (default: data/custom_objects/sim_microwave_good)
  --num-demos INT         Number of demos to generate (default: 5)
  --output-dir PATH       Directory to store training outputs (default: data/custom_pipeline_output)
  --dataset-dir PATH      Directory to store extracted datasets (default: data/custom_pipeline_dataset)
  --exp-name NAME         Experiment name under asset_dir/experiment (default: demo_gen)
  --joint-name NAME       Joint name passed to gen_demo_custom.py (default: joint1)
  --env-name NAME         Environment name passed to data generation scripts (default: articulated_custom_object)
  --timeout-exec SEC      Timeout (seconds) for demo execution (default: 600)
  --train-epochs INT      Number of training epochs (default: 2)
  -h, --help              Show this help message
EOF
}

# Defaults
ASSET_DIR="data/custom_objects/sim_microwave_good"
NUM_DEMOS=5
OUTPUT_DIR="data/custom_pipeline_output"
DATASET_DIR="data/custom_pipeline_dataset"
EXP_NAME="demo_gen"
JOINT_NAME="joint1"
ENV_NAME="articulated_custom_object"
TIMEOUT_EXEC=600
TRAIN_EPOCHS=2

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --asset-dir)
      ASSET_DIR="$2"; shift 2 ;;
    --num-demos)
      NUM_DEMOS="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --dataset-dir)
      DATASET_DIR="$2"; shift 2 ;;
    --exp-name)
      EXP_NAME="$2"; shift 2 ;;
    --joint-name)
      JOINT_NAME="$2"; shift 2 ;;
    --env-name)
      ENV_NAME="$2"; shift 2 ;;
    --timeout-exec)
      TIMEOUT_EXEC="$2"; shift 2 ;;
    --train-epochs)
      TRAIN_EPOCHS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

ASSET_DIR=$(readlink -f "$ASSET_DIR")
OUTPUT_DIR=$(readlink -f "$OUTPUT_DIR")
DATASET_DIR=$(readlink -f "$DATASET_DIR")

mkdir -p "$OUTPUT_DIR" "$DATASET_DIR"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PROJECT_DIR

echo "Running pipeline with:" \
  && echo "  Asset dir   : $ASSET_DIR" \
  && echo "  Num demos   : $NUM_DEMOS" \
  && echo "  Dataset dir : $DATASET_DIR" \
  && echo "  Output dir  : $OUTPUT_DIR" \
  && echo "  Exp name    : $EXP_NAME"

CONFIG_PATH="$ASSET_DIR/base_config.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Base config not found at $CONFIG_PATH. Nothing to do." >&2
  exit 0
fi

run_gen_demo() {
  echo "Step 1: Generating demos"
  python3 manipulation/gen_demo_custom.py \
    --asset-dir "$ASSET_DIR" \
    --num-to-generate "$NUM_DEMOS" \
    --exp-name "$EXP_NAME" \
    --env-name "$ENV_NAME" \
    --prepare-mobility \
    --joint-name "$JOINT_NAME" \
    --timeout-exec "$TIMEOUT_EXEC"
}

run_extract() {
  echo "Step 2: Extracting data"
  local parent_dir
  parent_dir=$(dirname "$ASSET_DIR")
  local extract_name
  extract_name=$(basename "$ASSET_DIR")

  python3 manipulation/extract_pcd_from_states.py \
    --folder_name "$parent_dir" \
    --extract_name "$extract_name" \
    --exp_name "$EXP_NAME" \
    --env_name "$ENV_NAME" \
    --save_path "$DATASET_DIR" \
    --num_experiment "$NUM_DEMOS"
}

run_training() {
  echo "Step 3: Training weighted displacement model"
  torchrun --standalone --nproc_per_node=1 weighted_displacement_model/train_ddp_weighted_displacement.py \
    --batch_size 2 \
    --num_epochs "$TRAIN_EPOCHS" \
    --exp_path "$OUTPUT_DIR/training_exps" \
    --num_train_objects custom \
    --dataset_prefix "$DATASET_DIR" \
    --exp_name "$EXP_NAME" \
    --use_all_data
}

run_gen_demo
run_extract
# run_training

echo "Pipeline completed successfully!"
