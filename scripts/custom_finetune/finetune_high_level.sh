#!/bin/bash
# Finetune the high-level policy (weighted displacement model) on custom object demonstrations
#
# This script uses the original train_ddp_weighted_displacement.py with
# num_train_objects=custom to load data from a custom dataset path.
#
# Usage:
#   bash scripts/custom_finetune/finetune_high_level.sh [OPTIONS]
#
# Required environment variables (set via prepare.sh):
#   PROJECT_DIR - Path to ArticuBot repository root
#
# Example:
#   source prepare.sh
#   bash scripts/custom_finetune/finetune_high_level.sh \
#       --dataset-path data/custom_finetune/sim_microwave_traj_100 \
#       --pretrained-model data/high_level_200_obj_ckpt.pth \
#       --exp-name finetune_microwave

set -e

# Default values
DATASET_PATH=""
PRETRAINED_MODEL=""
EXP_NAME="custom_finetune"
BATCH_SIZE=32
NUM_EPOCHS=20
LR=0.0001
SAVE_FREQ=5
NUM_GPUS=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --pretrained-model)
            PRETRAINED_MODEL="$2"
            shift 2
            ;;
        --exp-name)
            EXP_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET_PATH" ]; then
    echo "Error: --dataset-path is required"
    echo "Usage: bash scripts/custom_finetune/finetune_high_level.sh --dataset-path <path> [OPTIONS]"
    exit 1
fi

if [ -z "$PROJECT_DIR" ]; then
    echo "Error: PROJECT_DIR not set. Run 'source prepare.sh' first."
    exit 1
fi

# Resolve paths
if [[ ! "$DATASET_PATH" = /* ]]; then
    DATASET_PATH="${PROJECT_DIR}/${DATASET_PATH}"
fi

if [ -n "$PRETRAINED_MODEL" ] && [[ ! "$PRETRAINED_MODEL" = /* ]]; then
    PRETRAINED_MODEL="${PROJECT_DIR}/${PRETRAINED_MODEL}"
fi

# Check dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Build pretrained model argument
LOAD_MODEL_ARG=""
if [ -n "$PRETRAINED_MODEL" ]; then
    if [ ! -f "$PRETRAINED_MODEL" ]; then
        echo "Error: Pretrained model not found: $PRETRAINED_MODEL"
        exit 1
    fi
    LOAD_MODEL_ARG="--load_model_path ${PRETRAINED_MODEL}"
    echo "Finetuning from pretrained model: $PRETRAINED_MODEL"
else
    echo "Training from scratch (no pretrained model specified)"
fi

echo "========================================"
echo "High-Level Policy Finetuning"
echo "========================================"
echo "Dataset path: $DATASET_PATH"
echo "Experiment name: $EXP_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LR"
echo "Save frequency: $SAVE_FREQ epochs"
echo "GPUs: $NUM_GPUS"
echo "========================================"

cd "${PROJECT_DIR}/weighted_displacement_model"

# Use the original training script with num_train_objects=custom
# This tells the dataset loader to scan the dataset_prefix directory
torchrun --standalone --nproc_per_node=${NUM_GPUS} train_ddp_weighted_displacement.py \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --save_freq ${SAVE_FREQ} \
    --exp_path "${PROJECT_DIR}/weighted_displacement_model/exps" \
    --num_train_objects custom \
    --dataset_prefix "${DATASET_PATH}" \
    --exp_name "_${EXP_NAME}" \
    --use_all_data \
    ${LOAD_MODEL_ARG}

echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: ${PROJECT_DIR}/weighted_displacement_model/exps/"
echo "========================================"
