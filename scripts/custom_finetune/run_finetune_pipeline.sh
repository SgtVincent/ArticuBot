#!/bin/bash
# Complete finetuning pipeline for custom objects (high-level only)
#
# This script runs the complete pipeline:
# 1. Convert raw demos to training format
# 2. Finetune high-level policy
#
# Usage:
#   source prepare.sh
#   bash scripts/custom_finetune/run_finetune_pipeline.sh \
#       --input-dir data/custom_objects/sim_microwave_good/experiment/traj_100 \
#       --output-name sim_microwave \
#       --pretrained-high-level data/high_level_200_obj_ckpt.pth

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# Default values
INPUT_DIR=""
OUTPUT_NAME=""
PRETRAINED_HIGH=""
SKIP_CONVERT=0
HIGH_LEVEL_EPOCHS=20
BATCH_SIZE_HIGH=32

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --pretrained-high-level)
            PRETRAINED_HIGH="$2"
            shift 2
            ;;
        --skip-convert)
            SKIP_CONVERT=1
            shift
            ;;
        --high-level-epochs)
            HIGH_LEVEL_EPOCHS="$2"
            shift 2
            ;;
        --batch-size-high)
            BATCH_SIZE_HIGH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: run_finetune_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-dir PATH          Input demo directory (required)"
            echo "  --output-name NAME        Output name for experiment (required)"
            echo "  --pretrained-high-level   Path to pretrained high-level model"
            echo "  --skip-convert            Skip data conversion step"
            echo "  --high-level-epochs N     Number of high-level training epochs (default: 20)"
            echo "  --batch-size-high N       Batch size for high-level (default: 32)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input-dir is required"
    exit 1
fi

if [ -z "$OUTPUT_NAME" ]; then
    echo "Error: --output-name is required"
    exit 1
fi

# Define output path
OUTPUT_DIR="${PROJECT_DIR}/data/custom_finetune/${OUTPUT_NAME}"

echo "========================================"
echo "ArticuBot Custom Object Finetuning Pipeline"
echo "========================================"
echo "Input directory: $INPUT_DIR"
echo "Output name: $OUTPUT_NAME"
echo "Converted data will be at: $OUTPUT_DIR"
echo "========================================"

# Step 1: Convert demos
if [ $SKIP_CONVERT -eq 0 ]; then
    echo ""
    echo ">>> Step 1: Converting demos to training format..."
    echo ""
    
    cd "$PROJECT_DIR"
    python scripts/custom_finetune/convert_custom_demos.py \
        --input-dir "$INPUT_DIR" \
        --output-dir "$OUTPUT_DIR"
else
    echo ""
    echo ">>> Step 1: Skipping data conversion (--skip-convert)"
    echo ""
fi

# Step 2: Finetune high-level policy
echo ""
echo ">>> Step 2: Finetuning high-level policy..."
echo ""

HIGH_LEVEL_ARGS="--dataset-path $OUTPUT_DIR --exp-name $OUTPUT_NAME --batch-size $BATCH_SIZE_HIGH --num-epochs $HIGH_LEVEL_EPOCHS"
if [ -n "$PRETRAINED_HIGH" ]; then
    HIGH_LEVEL_ARGS="$HIGH_LEVEL_ARGS --pretrained-model $PRETRAINED_HIGH"
fi

bash "${SCRIPT_DIR}/finetune_high_level.sh" $HIGH_LEVEL_ARGS

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo ""
echo "Outputs:"
echo "  - Converted data: $OUTPUT_DIR"
echo "  - High-level checkpoints: ${PROJECT_DIR}/weighted_displacement_model/exps/*${OUTPUT_NAME}*/"
echo "========================================"
