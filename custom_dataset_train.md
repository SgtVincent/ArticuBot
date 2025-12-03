# Custom Dataset Training Guide

This guide explains how to finetune ArticuBot's high-level policy (weighted displacement model) on custom demonstration datasets.

---

## Prerequisites

### Environment Setup
```bash
# Activate the environment
conda activate articubot
source prepare.sh
```

### Required Checkpoints

Download pre-trained checkpoint from the [Google Drive link](https://drive.google.com/drive/folders/1lbpoo8SqNuLWTjMyvO5RWnBd0XpGq6C4?usp=sharing):

- **High-level policy**: `high_level_200_obj_ckpt.pth` → `data/high_level_200_obj_ckpt.pth`

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Generate Demos (see custom_dataset_demo_gen.md)        │
│  Output: Raw trajectories in experiment/<exp_name>/             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Convert Demos to Training Format                       │
│  Script: scripts/custom_finetune/convert_custom_demos.py        │
│  Output: Pickle files compatible with training                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Finetune High-Level Policy                             │
│  Script: scripts/custom_finetune/finetune_high_level.sh         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
source prepare.sh

# Step 1: Convert demos
python scripts/custom_finetune/convert_custom_demos.py \
    --input-dir data/custom_objects/sim_microwave_good/experiment/traj_100 \
    --output-dir data/custom_finetune/sim_microwave_traj_100

# Step 2: Finetune
bash scripts/custom_finetune/finetune_high_level.sh \
    --dataset-path data/custom_finetune/sim_microwave_traj_100 \
    --pretrained-model data/high_level_200_obj_ckpt.pth \
    --exp-name sim_microwave \
    --num-epochs 20
```

---

## Step-by-Step Guide

### Step 1: Generate Demonstrations

Generate demonstrations following [custom_dataset_demo_gen.md](custom_dataset_demo_gen.md).

**Required files for each demo:**
- `all.gif` - Indicates successful demo
- `stage_lengths.json` - Timing information  
- `task_config.yaml` - Environment configuration
- `states/` folder with `state_*.pkl` files

### Step 2: Convert Demos

Convert raw demos to training format:

```bash
python scripts/custom_finetune/convert_custom_demos.py \
    --input-dir data/custom_objects/sim_microwave_good/experiment/traj_100 \
    --output-dir data/custom_finetune/sim_microwave_traj_100 \
    --env-name articulated_custom_object
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | Required | Path to experiment folder with demos |
| `--output-dir` | Required | Output directory |
| `--env-name` | `articulated_custom_object` | Environment name |
| `--pointcloud-num` | 4500 | Points in point cloud |
| `--combine-action-steps` | 2 | Combine consecutive steps |
| `--min-opened-ratio` | 0.2 | Min ratio for success |

**Output Structure:**
```
data/custom_finetune/sim_microwave_traj_100/
├── 2025-12-01-03-13-58/
│   ├── 0.pkl
│   ├── 1.pkl
│   └── camera_params.npz
├── demo_rgbs/
├── all_demo_path.txt
└── meta_info.json
```

### Step 3: Finetune High-Level Policy

The high-level policy predicts goal gripper positions from point clouds:

```bash
bash scripts/custom_finetune/finetune_high_level.sh \
    --dataset-path data/custom_finetune/sim_microwave_traj_100 \
    --pretrained-model data/high_level_200_obj_ckpt.pth \
    --exp-name sim_microwave \
    --num-epochs 20
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-path` | Required | Path to converted dataset |
| `--pretrained-model` | None | Pre-trained model (for finetuning) |
| `--exp-name` | `custom_finetune` | Experiment name |
| `--batch-size` | 32 | Training batch size |
| `--num-epochs` | 20 | Training epochs |
| `--lr` | 0.0001 | Learning rate |
| `--save-freq` | 5 | Checkpoint save frequency |
| `--num-gpus` | 1 | Number of GPUs |

**Output:** `weighted_displacement_model/exps/<date>_*<exp_name>/`

---

## Training from Scratch

Omit `--pretrained-model` to train from scratch:

```bash
bash scripts/custom_finetune/finetune_high_level.sh \
    --dataset-path data/custom_finetune/sim_microwave_traj_100 \
    --exp-name sim_microwave_scratch \
    --num-epochs 60
```

> **Note:** Requires significantly more data and epochs.

---

## Example: Complete Workflow

```bash
# 1. Setup
conda activate articubot
source prepare.sh

# 2. Convert demos
python scripts/custom_finetune/convert_custom_demos.py \
    --input-dir data/custom_objects/sim_microwave_good/experiment/traj_100 \
    --output-dir data/custom_finetune/sim_microwave_traj_100

# 3. Finetune
bash scripts/custom_finetune/finetune_high_level.sh \
    --dataset-path data/custom_finetune/sim_microwave_traj_100 \
    --pretrained-model data/high_level_200_obj_ckpt.pth \
    --exp-name sim_microwave \
    --num-epochs 20

# 4. Find checkpoints
ls weighted_displacement_model/exps/*sim_microwave*/
```

---

## Tips

### Data Quality
- Use **10+ successful demonstrations** per object
- Ensure diverse initial positions
- Include variations in joint angles

### Hyperparameters
- **Finetuning:** lr=1e-4, 20-30 epochs
- **From scratch:** lr=1e-3, 60+ epochs

### Monitoring
- Training logs saved to [Weights & Biases](https://wandb.ai/)
- Check loss curves in wandb dashboard

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `--batch-size` |
| `Dataset not found` | Check `--dataset-path` |
| `Handle not observed` | Adjust camera in environment |

---

## Quick Data Verification

Check valid demos before conversion:

```bash
python -c "
from pathlib import Path
demo_dir = Path('data/custom_objects/sim_microwave_good/experiment/traj_100')
subdirs = [d for d in demo_dir.iterdir() if d.is_dir()]
valid = [d for d in subdirs if (d / 'all.gif').exists() 
         and (d / 'stage_lengths.json').exists()]
print(f'Valid demos: {len(valid)} / {len(subdirs)}')
"
```

---

## File Reference

| File | Description |
|------|-------------|
| [scripts/custom_finetune/convert_custom_demos.py](scripts/custom_finetune/convert_custom_demos.py) | Convert demos to training format |
| [scripts/custom_finetune/finetune_high_level.sh](scripts/custom_finetune/finetune_high_level.sh) | High-level policy finetuning |
| [manipulation/custom_object_utils/robogen_wrapper_custom.py](manipulation/custom_object_utils/robogen_wrapper_custom.py) | Custom object wrapper |
| [manipulation/custom_object_utils/extract_pcd_from_states_custom.py](manipulation/custom_object_utils/extract_pcd_from_states_custom.py) | Point cloud extraction |

---

## Related Documentation

- [custom_dataset_demo_gen.md](custom_dataset_demo_gen.md) - Generate demonstrations
- [readme.md](readme.md) - Main ArticuBot documentation
