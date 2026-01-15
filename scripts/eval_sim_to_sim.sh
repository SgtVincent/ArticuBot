#!/bin/bash
# Sim-to-Sim Evaluation Pipeline Script
#
# This script runs the complete sim-to-sim evaluation for ArticuBot,
# comparing policy performance on digital twins vs original models.
#
# Usage:
#   bash scripts/eval_sim_to_sim.sh --object microwave_7128 --ckpt path/to/ckpt.pth
#   bash scripts/eval_sim_to_sim.sh --all --ckpt path/to/ckpt.pth
#   bash scripts/eval_sim_to_sim.sh --object-type microwave --ckpt path/to/ckpt.pth

set -e

# Default values
SIM_DATASET_DIR="data/custom_objects_v2/sim"
LOW_LEVEL_EXP_DIR="data/low-level-ckpt"
LOW_LEVEL_CKPT="low-level.ckpt"
HIGH_LEVEL_CKPT="data/high_level_200_obj_ckpt.pth"
NUM_TRIALS=25
HORIZON=70
EVAL_MODE="both"  # both, twin, original

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --object)
            OBJECT_NAME="$2"
            shift 2
            ;;
        --object-type)
            OBJECT_TYPE="$2"
            shift 2
            ;;
        --object-id)
            OBJECT_ID="$2"
            shift 2
            ;;
        --ckpt|--high-level-ckpt)
            HIGH_LEVEL_CKPT="$2"
            shift 2
            ;;
        --low-level-ckpt)
            LOW_LEVEL_EXP_DIR="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --sim-dataset-dir)
            SIM_DATASET_DIR="$2"
            shift 2
            ;;
        --all)
            EVAL_ALL=true
            shift
            ;;
        --twin-only)
            EVAL_MODE="twin"
            shift
            ;;
        --original-only)
            EVAL_MODE="original"
            shift
            ;;
        --list)
            LIST_OBJECTS=true
            shift
            ;;
        --help)
            echo "Sim-to-Sim Evaluation Pipeline"
            echo ""
            echo "Usage: bash scripts/eval_sim_to_sim.sh [OPTIONS]"
            echo ""
            echo "Object Selection (choose one):"
            echo "  --object NAME           Object folder name (e.g., microwave_7128)"
            echo "  --object-type TYPE      Object type (e.g., microwave)"
            echo "  --object-id ID          Object ID (e.g., 7128)"
            echo "  --all                   Evaluate all available objects"
            echo "  --list                  List available objects and exit"
            echo ""
            echo "Checkpoints:"
            echo "  --ckpt PATH             High-level checkpoint path"
            echo "  --low-level-ckpt PATH   Low-level checkpoint directory"
            echo ""
            echo "Evaluation Settings:"
            echo "  --num-trials N          Number of trials per object (default: 25)"
            echo "  --horizon N             Steps per episode (default: 70)"
            echo "  --twin-only             Only evaluate on digital twin"
            echo "  --original-only         Only evaluate on original model"
            echo ""
            echo "Paths:"
            echo "  --sim-dataset-dir PATH  Sim dataset directory"
            echo ""
            echo "Examples:"
            echo "  bash scripts/eval_sim_to_sim.sh --object microwave_7128 --ckpt model.pth"
            echo "  bash scripts/eval_sim_to_sim.sh --all --ckpt model.pth --num-trials 10"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if prepare.sh was sourced
if [ -z "$PROJECT_DIR" ]; then
    echo "Warning: PROJECT_DIR not set. Sourcing prepare.sh..."
    source prepare.sh
fi

# List objects mode
if [ "$LIST_OBJECTS" = true ]; then
    python eval_sim_to_sim.py \
        --sim-dataset-dir "$SIM_DATASET_DIR" \
        --list-objects
    exit 0
fi

# Validate checkpoint
if [ ! -f "$HIGH_LEVEL_CKPT" ]; then
    echo "Error: High-level checkpoint not found: $HIGH_LEVEL_CKPT"
    exit 1
fi

# Build command
CMD="python eval_sim_to_sim.py"
CMD="$CMD --sim-dataset-dir $SIM_DATASET_DIR"
CMD="$CMD --high-level-ckpt $HIGH_LEVEL_CKPT"
CMD="$CMD --low-level-exp-dir $LOW_LEVEL_EXP_DIR"
CMD="$CMD --low-level-ckpt-name $LOW_LEVEL_CKPT"
CMD="$CMD --num-trials $NUM_TRIALS"
CMD="$CMD --horizon $HORIZON"

# Object selection
if [ "$EVAL_ALL" = true ]; then
    CMD="$CMD --eval-all"
elif [ -n "$OBJECT_NAME" ]; then
    # Parse object name to get type and id
    if [[ "$OBJECT_NAME" =~ ^([a-zA-Z]+[0-9]*)_([0-9]+)$ ]]; then
        CMD="$CMD --object-type ${BASH_REMATCH[1]} --object-id ${BASH_REMATCH[2]}"
    else
        CMD="$CMD --object-list $OBJECT_NAME"
    fi
elif [ -n "$OBJECT_TYPE" ] && [ -n "$OBJECT_ID" ]; then
    CMD="$CMD --object-type $OBJECT_TYPE --object-id $OBJECT_ID"
elif [ -n "$OBJECT_TYPE" ]; then
    # Find all objects of this type
    echo "Finding all objects of type: $OBJECT_TYPE"
    OBJECTS=$(python -c "
import os
for entry in os.listdir('$SIM_DATASET_DIR'):
    if entry.startswith('${OBJECT_TYPE}_') and os.path.isdir(os.path.join('$SIM_DATASET_DIR', entry)):
        print(entry, end=' ')
")
    if [ -z "$OBJECTS" ]; then
        echo "Error: No objects found of type $OBJECT_TYPE"
        exit 1
    fi
    CMD="$CMD --object-list $OBJECTS"
else
    echo "Error: Must specify --object, --object-type, or --all"
    echo "Run with --help for usage information"
    exit 1
fi

# Evaluation mode
case $EVAL_MODE in
    both)
        CMD="$CMD --eval-both"
        ;;
    twin)
        CMD="$CMD --eval-on-twin"
        ;;
    original)
        # Default mode, no flag needed
        ;;
esac

# Print configuration
echo "========================================"
echo "Sim-to-Sim Evaluation"
echo "========================================"
echo "Dataset:     $SIM_DATASET_DIR"
echo "Checkpoint:  $HIGH_LEVEL_CKPT"
echo "Trials:      $NUM_TRIALS"
echo "Eval Mode:   $EVAL_MODE"
echo "========================================"
echo ""
echo "Running: $CMD"
echo ""

# Run evaluation
eval $CMD

echo ""
echo "========================================"
echo "Evaluation Complete"
echo "========================================"
