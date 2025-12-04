# Evaluating Custom URDF Models

This guide explains how to evaluate ArticuBot policies (both pre-trained and finetuned) on custom articulated objects using `eval_custom_object.py`.

## Overview

The unified evaluation script supports:
- **Pre-trained checkpoints**: Evaluate the original ArticuBot policy on your custom objects
- **Finetuned checkpoints**: Evaluate finetuned policies after training on your custom demos
- **Two initialization modes**:
  - From saved demonstration states (recommended for fair comparison)
  - Random initialization (for stress testing)

## Prerequisites

1. **Environment Setup**:
   ```bash
   conda activate articubot
   source prepare.sh
   ```

2. **Pretrained Checkpoints** (download from [Google Drive](https://drive.google.com/drive/folders/1lbpoo8SqNuLWTjMyvO5RWnBd0XpGq6C4?usp=sharing)):
   - Low-level policy: `data/low-level-ckpt/low-level.ckpt`
   - High-level policy: `data/high_level_200_obj_ckpt.pth`

3. **Custom Object Assets** (see `custom_dataset_demo_gen.md` for setup):
   ```
   data/custom_objects/my_object/
   ├── *.urdf
   ├── base_config.yaml
   ├── mobility_v2.json
   ├── configs/
   │   ├── config_0.yaml
   │   └── ...
   ├── experiment/           # Generated demos
   │   └── my_demos/
   │       ├── trial_0000/
   │       └── ...
   └── mesh/
   ```

## Quick Start

### Evaluate Pre-trained Policy on Custom Object

```bash
# Using saved demo states
python eval_custom_object.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --experiment-name my_demos \
    --num-trials 10

# Using random initialization
python eval_custom_object.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --use-random-init \
    --config-id 0 \
    --num-trials 10
```

### Evaluate Finetuned Policy

```bash
python eval_custom_object.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --experiment-name my_demos \
    --high-level-ckpt weighted_displacement_model/exps/my_finetuned/best.pth \
    --num-trials 10
```

### Using the Shell Wrapper

```bash
# Evaluate with default pre-trained checkpoints
bash scripts/custom_finetune/eval_finetuned.sh \
    --asset-dir data/custom_objects/sim_microwave_good \
    --experiment-name my_demos \
    --num-trials 10

# Evaluate finetuned checkpoint
bash scripts/custom_finetune/eval_finetuned.sh \
    --asset-dir data/custom_objects/sim_microwave_good \
    --experiment-name my_demos \
    --high-level-ckpt weighted_displacement_model/exps/finetuned/best.pth \
    --num-trials 10

# Random initialization mode
bash scripts/custom_finetune/eval_finetuned.sh \
    --asset-dir data/custom_objects/sim_microwave_good \
    --use-random-init \
    --config-id 0 \
    --num-trials 10
```

## Command Line Arguments

### Required
| Argument | Description |
|----------|-------------|
| `--asset-dir` | Path to custom object asset directory |

### State Selection (one required)
| Argument | Description |
|----------|-------------|
| `--experiment-name` | Use saved states from this experiment folder |
| `--use-random-init` | Use random initialization instead |

### Checkpoints
| Argument | Default | Description |
|----------|---------|-------------|
| `--high-level-ckpt` | `data/high_level_200_obj_ckpt.pth` | High-level policy checkpoint |
| `--low-level-exp-dir` | `data/low-level-ckpt` | Low-level experiment directory |
| `--low-level-ckpt-name` | `low-level.ckpt` | Low-level checkpoint filename |

### Evaluation Settings
| Argument | Default | Description |
|----------|---------|-------------|
| `--num-trials` | `10` | Number of evaluation episodes |
| `--horizon` | `70` | Steps per episode |
| `--update-goal-freq` | `1` | Goal update frequency |
| `--config-id` | `0` | Config ID (for random init mode) |

### Output
| Argument | Default | Description |
|----------|---------|-------------|
| `--save-dir` | `data/eval_custom/<asset>` | Output directory |
| `--no-gif` | - | Disable GIF saving |

## Output Files

Results are saved to `{save_dir}/`:

```
data/eval_custom/sim_microwave_good/
├── metrics.json              # Aggregated results
├── trial_000_improve_0.1234.gif
├── trial_001_improve_0.2345.gif
└── ...
```

### metrics.json Structure
```json
{
  "config": {
    "asset_dir": "data/custom_objects/sim_microwave_good",
    "experiment_name": "my_demos",
    "high_level_ckpt": "data/high_level_200_obj_ckpt.pth",
    "num_trials": 10
  },
  "trials": [
    {
      "trial": 0,
      "success": true,
      "initial_angle": 0.0,
      "final_angle": 0.785,
      "improved_angle": 0.785
    },
    ...
  ]
}
```

## Understanding Results

- **improved_angle**: Change in joint angle (higher = more opening)
- **success**: Whether the trial completed without errors
- **GIF naming**: `trial_XXX_improve_Y.YYYY.gif` shows the improvement directly

## Comparing Pre-trained vs Finetuned

```bash
# Evaluate pre-trained
python eval_custom_object.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --experiment-name my_demos \
    --save-dir data/eval_custom/sim_microwave_good/pretrained \
    --num-trials 20

# Evaluate finetuned
python eval_custom_object.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --experiment-name my_demos \
    --high-level-ckpt weighted_displacement_model/exps/finetuned/best.pth \
    --save-dir data/eval_custom/sim_microwave_good/finetuned \
    --num-trials 20

# Compare metrics
python -c "
import json
pre = json.load(open('data/eval_custom/sim_microwave_good/pretrained/metrics.json'))
fine = json.load(open('data/eval_custom/sim_microwave_good/finetuned/metrics.json'))
pre_imp = [t['improved_angle'] for t in pre['trials'] if t['success']]
fine_imp = [t['improved_angle'] for t in fine['trials'] if t['success']]
print(f'Pre-trained: {sum(pre_imp)/len(pre_imp):.4f} mean improvement')
print(f'Finetuned:   {sum(fine_imp)/len(fine_imp):.4f} mean improvement')
"
```

## Troubleshooting

### "Experiment directory not found"
Ensure you've generated demos first:
```bash
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/my_object \
    --exp-name my_demos \
    --num-to-generate 20
```

### "Initial state not found"
The experiment structure may be incomplete. Check that each trial has:
- `task_config.yaml`
- `states/state_0.pkl`

### "Could not get handle point cloud"
The custom wrapper couldn't find handle points. Ensure:
- `annotation.json` exists and has valid points
- Handle mesh was generated (check for `handle_*.obj`)

### Low improvement scores
Try:
- More finetuning epochs
- More demo data
- Check if observation mode matches training


## Preparing Custom Assets and Demos

See `custom_dataset_demo_gen.md` for the full pipeline:
1. Prepare URDF + annotation
2. Generate `base_config.yaml` with `manipulation/generate_base_config_from_urdf_custom.py`
3. Generate demos with `manipulation/gen_demo_custom.py`
4. (Optional) Finetune with `scripts/custom_finetune/finetune_high_level.sh`
5. Evaluate with `eval_custom_object.py`

## Related Files

| File | Description |
|------|-------------|
| `eval_custom_object.py` | Unified evaluation script for custom objects |
| `custom_dataset_demo_gen.md` | Full demo generation pipeline |
| `custom_dataset_train.md` | High-level finetuning guide |
| `scripts/custom_finetune/eval_finetuned.sh` | Shell wrapper for evaluation |
| `manipulation/custom_object_utils/robogen_wrapper_custom.py` | Custom environment wrapper |
| `manipulation/gen_demo_custom.py` | Demo generation for custom objects |

