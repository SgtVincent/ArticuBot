# Sim-to-Sim Evaluation Protocol

This document describes the sim-to-sim evaluation protocol for ArticuBot, which measures the transferability of manipulation policies trained on digital twin models to original high-fidelity simulation models.

---

## Overview

The sim-to-sim evaluation protocol works as follows:

```
┌─────────────────────────────────────────────────────────────────┐
│  Digital Twin Model                Original PartNet-Mobility     │
│  (Reconstructed URDF)              (Ground Truth Model)          │
│                                                                  │
│  microwave_7128/                   assets/7128/                  │
│  ├── urdf/                         ├── mobility.urdf             │
│  │   └── microwave_7128.urdf       ├── mobility_v2.json          │
│  └── color/                        └── textured_objs/            │
│                                                                  │
│         ↓ Train Policy                    ↓ Evaluate             │
│  ┌──────────────────┐              ┌──────────────────┐          │
│  │ Demo Generation  │              │ Policy Evaluation │         │
│  │ + Fine-tuning    │ ──────────→  │ on Original Model │         │
│  └──────────────────┘              └──────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

This protocol evaluates:
1. **Reconstruction quality**: How well the digital twin captures the essential manipulation affordances
2. **Policy generalization**: How well policies transfer from simplified to high-fidelity models
3. **Sim-to-real readiness**: A proxy for real-world transfer potential

---

## Dataset Structure

The sim-to-sim dataset is organized as follows:

```
data/custom_objects_v2/sim/
├── assets/                      # Original PartNet-Mobility models
│   ├── 7128/                    # Object ID
│   │   ├── mobility.urdf        # Original URDF with full detail
│   │   ├── mobility_v2.json     # Joint/part information
│   │   ├── textured_objs/       # Original textured meshes
│   │   └── ...
│   ├── 7130/
│   └── ...
├── microwave_7128/              # Digital twin for object 7128
│   ├── urdf/
│   │   ├── microwave_7128.urdf  # Simplified/reconstructed URDF
│   │   └── mesh/                # Convex decomposed meshes
│   └── color/
├── microwave_7167/
├── oven_7130/
├── storagefurniture1_44781/
└── ...
```

### Object Categories

| Category | Count | Examples |
|----------|-------|----------|
| Microwave | 6 | 7128, 7167, 7221, 7263, 7310, 7320 |
| Oven | 5 | 7130, 7138, 7290, 101943, 102018 |
| Dishwasher | 4 | 11622, 11826, 12092, 12553 |
| Storage Furniture | 8 | Various drawer/cabinet models |
| Safe | 2 | 101594, 102301 |
| Toilet | 5 | 101320, 102622, 102630, 102652, 102658 |
| Washing Machine | 2 | 103361, 103480 |
| Trash Can | 4 | 102177, 102192, 102210, 12483 |
| Box | 4 | 100174, 100221, 100247, 48492 |

---

## Quick Start

### List Available Objects

```bash
source prepare.sh

python eval_sim_to_sim.py \
    --sim-dataset-dir data/custom_objects_v2/sim \
    --list-objects
```

### Evaluate Single Object

```bash
# Evaluate on ORIGINAL model (sim-to-sim transfer)
python eval_sim_to_sim.py \
    --sim-dataset-dir data/custom_objects_v2/sim \
    --object-type microwave \
    --object-id 7128 \
    --high-level-ckpt weighted_displacement_model/exps/microwave_7128_finetuned/best.pth \
    --num-trials 25

# Evaluate on DIGITAL TWIN (training domain baseline)
python eval_sim_to_sim.py \
    --sim-dataset-dir data/custom_objects_v2/sim \
    --object-type microwave \
    --object-id 7128 \
    --high-level-ckpt weighted_displacement_model/exps/microwave_7128_finetuned/best.pth \
    --eval-on-twin \
    --num-trials 25
```

### Evaluate Multiple Objects

```bash
python eval_sim_to_sim.py \
    --sim-dataset-dir data/custom_objects_v2/sim \
    --object-list microwave_7128 microwave_7167 oven_7130 \
    --high-level-ckpt path/to/checkpoint.pth \
    --num-trials 10
```

### Evaluate Both (Comparison)

```bash
python eval_sim_to_sim.py \
    --sim-dataset-dir data/custom_objects_v2/sim \
    --object-type microwave \
    --object-id 7128 \
    --high-level-ckpt path/to/checkpoint.pth \
    --eval-both \
    --num-trials 25
```

---

## Full Pipeline

### Step 1: Generate Demonstrations on Digital Twin

```bash
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects_v2/sim/microwave_7128 \
    --exp-name demos_100 \
    --sampling-method heuristic \
    --num-to-generate 100 \
    --timeout-exec 300
```

### Step 2: Convert Demos to Training Format

```bash
python scripts/custom_finetune/convert_custom_demos.py \
    --input-dir data/custom_objects_v2/sim/microwave_7128/experiment/demos_100 \
    --output-dir data/custom_finetune/microwave_7128_demos_100
```

### Step 3: Fine-tune High-Level Policy

```bash
bash scripts/custom_finetune/finetune_high_level.sh \
    --dataset-path data/custom_finetune/microwave_7128_demos_100 \
    --pretrained-model data/high_level_200_obj_ckpt.pth \
    --exp-name microwave_7128 \
    --num-epochs 20
```

### Step 4: Run Sim-to-Sim Evaluation

```bash
# Find the finetuned checkpoint
CKPT=$(ls -t weighted_displacement_model/exps/*microwave_7128*/model_final.pth | head -1)

# Evaluate on both digital twin and original model
python eval_sim_to_sim.py \
    --sim-dataset-dir data/custom_objects_v2/sim \
    --object-type microwave \
    --object-id 7128 \
    --high-level-ckpt $CKPT \
    --eval-both \
    --num-trials 25
```

---

## Command-Line Arguments

### Dataset Arguments

| Argument | Description |
|----------|-------------|
| `--sim-dataset-dir` | Root directory of sim dataset (required) |
| `--object-type` | Object type (e.g., microwave, oven) |
| `--object-id` | Object ID from PartNet-Mobility |
| `--object-list` | List of folder names to evaluate |
| `--eval-all` | Evaluate all paired objects |

### Evaluation Mode

| Argument | Description |
|----------|-------------|
| `--eval-on-twin` | Evaluate on digital twin (training domain) |
| `--eval-both` | Evaluate on both twin and original |
| (default) | Evaluate on original model only |

### Policy Checkpoints

| Argument | Default | Description |
|----------|---------|-------------|
| `--low-level-exp-dir` | `data/low-level-ckpt` | Low-level policy directory |
| `--low-level-ckpt-name` | `low-level.ckpt` | Low-level checkpoint name |
| `--high-level-ckpt` | `data/high_level_200_obj_ckpt.pth` | High-level checkpoint |

### Evaluation Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-trials` | 10 | Trials per object |
| `--horizon` | 70 | Steps per episode |
| `--save-dir` | Auto-generated | Output directory |
| `--no-gif` | False | Disable GIF saving |
| `--render` | False | Enable GUI |

---

## Output Format

Results are saved in the following structure:

```
outputs/sim_to_sim_eval/<timestamp>_<ckpt_name>/
├── microwave_7128/
│   ├── original_model/
│   │   ├── trial_000_open_0.650.gif
│   │   ├── trial_001_open_0.720.gif
│   │   └── metrics.json
│   └── digital_twin/
│       ├── trial_000_open_0.800.gif
│       └── metrics.json
├── oven_7130/
│   └── ...
└── summary.json
```

### Metrics

Each `metrics.json` contains:

```json
{
  "object_type": "microwave",
  "object_id": "7128",
  "eval_target": "original_model",
  "total_trials": 25,
  "successful_trials": 24,
  "mean_opening": 0.652,
  "std_opening": 0.123,
  "max_opening": 0.850,
  "min_opening": 0.320,
  "success_rate_30": 0.92,
  "success_rate_50": 0.72,
  "trials": [...]
}
```

Key metrics:
- **mean_opening**: Average normalized opening ratio (0-1 scale)
- **success_rate_50**: Percentage of trials achieving >50% opening
- **success_rate_30**: Percentage of trials achieving >30% opening

---

## Interpreting Results

### Transfer Gap

The **transfer gap** is computed as:

```
Transfer Gap = (Digital Twin Performance) - (Original Model Performance)
```

A small transfer gap (< 0.1) indicates good transfer from the digital twin to the original model.

### Baseline Comparison

| Policy | Digital Twin | Original Model | Transfer Gap |
|--------|--------------|----------------|--------------|
| Pre-trained | 0.45 | 0.42 | 0.03 |
| Fine-tuned (10 demos) | 0.72 | 0.65 | 0.07 |
| Fine-tuned (50 demos) | 0.85 | 0.78 | 0.07 |
| Fine-tuned (100 demos) | 0.89 | 0.82 | 0.07 |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `--num-trials` or run sequentially |
| `Object not found` | Check `--list-objects` for available pairs |
| `URDF loading error` | Verify mesh files exist in assets folder |
| `No valid camera view` | Object may be too large/small for default camera |

### Debug Mode

Enable verbose output and GUI:

```bash
python eval_sim_to_sim.py \
    --sim-dataset-dir data/custom_objects_v2/sim \
    --object-type microwave \
    --object-id 7128 \
    --render \
    --num-trials 1
```

---

## Related Files

| File | Description |
|------|-------------|
| [eval_sim_to_sim.py](eval_sim_to_sim.py) | Main evaluation script |
| [eval_custom_object.py](eval_custom_object.py) | Evaluation for custom objects |
| [custom_dataset_train.md](custom_dataset_train.md) | Fine-tuning guide |
| [custom_dataset_demo_gen.md](custom_dataset_demo_gen.md) | Demo generation guide |
