# Custom Dataset for Real-to-Sim-to-Real Transfer Learning

This guide explains how to prepare custom URDF objects and generate demonstration data for training ArticuBot policies.

---

## Step 0: Initial Directory Structure

Before running the pipeline, you need to prepare your raw URDF asset with minimal required files:

```
your_urdf_folder/                  # Your source asset
├── <object>.urdf                  # URDF model file 
├── annotation.json                # Affordance annotation 
└── mesh/                          # Mesh files referenced by URDF 
    ├── link0.obj
    ├── link1.obj
    └── ...
```

**Minimum required files:**
- `*.urdf` – Articulated object model with joints and mesh references
- `annotation.json` – 3D points defining the handle/graspable region
- `mesh/` – Directory containing all mesh files (OBJ/STL/DAE)

---

## Step 1: Create `base_config.yaml`

The base configuration defines object properties and paths. You can generate it automatically or create manually.

### Option A: Auto-generate with Script

```bash
python manipulation/generate_base_config_from_urdf_custom.py \
    --asset-folder /path/to/your/urdf_folder \
    --dataset-name my_object_v1 \
    --annotation /path/to/annotation.json \
    --joint-name joint_main \
    --handle-padding 0.01
```

This will:
- Copy assets to `data/custom_objects/my_object_v1/`
- Generate `mobility_v2.json` from URDF joint data
- Create handle mesh from annotation points
- Write `base_config.yaml`

### Generate `mobility_v2.json`

---

## Step 2: Generate Grasp Predictions (GraspGen)

ArticuBot uses GraspGen to predict grasp poses. There are two modes:

### Option A: Pre-generate Grasps (Fixed Joint State Only)

> ⚠️ **Important**: Pre-generated grasps are computed for a **specific object state** (joint angles, scale). If you use this option, you **must disable** joint randomization and scale randomization in the demo generation step to prevent grasp-pose mismatches.

#### Prepare GraspGen Docker (recommended)

ArticuBot integrates with GraspGen via a long-running Docker container (default name: `friendly_galileo`).

1) Build the image:

```bash
cd third_party/GraspGen
bash docker/build.sh
```

2) Start the container (detached) with the correct mounts:

```bash
cd /path/to/ArticuBot
scripts/run_graspgen_container.sh --recreate
```

Mount convention used by ArticuBot:
- GraspGen code is available at `/code`
- The ArticuBot repo is mounted at `/workspace`
- Your custom objects under `data/` are accessible inside the container as `/workspace/data/...`

If you do not have GPU / NVIDIA runtime available, start with:

```bash
scripts/run_graspgen_container.sh --recreate --no-gpu
```

#### Pre-generate grasps

Start the GraspGen Docker container (see steps above), then run grasp generation in the container:

```bash
docker exec -w /code friendly_galileo \
    python scripts/generate_grasps_from_urdf.py \
        --urdf-root /workspace/data/custom_objects/sim_microwave_good \
        --gripper-config GraspGenModels/checkpoints/graspgen_franka_panda.yml \
        --num-grasps 400 \
        --num-sample-points 2000 \
        --use-aabb-sampling \
        --skip-aabb-filter \
        --no-remove-outliers
```

This creates `predicted_grasps.yml` in the object folder.

**When using Option A for demo generation, disable randomization:**
```bash
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --exp-name sim_microwave_demos \
    --size-scale 1.0 1.0 \
    --num-to-generate 50
```

The joint angles will remain at the default state specified in `base_config.yaml` (typically 0.0 for all joints).

### Option B: On-Demand Grasp Generation

The demo generator will automatically call GraspGen in Docker for each randomized joint configuration. This is useful when joint angles are randomized during data generation.

**Requirements:**
- Docker container `friendly_galileo` (or configured name) must be running
- GraspGen code mounted at `/code`
- ArticuBot repo mounted at `/workspace` (so assets are accessible at `/workspace/data/...`)

Start the container once (recommended):

```bash
cd /path/to/ArticuBot
scripts/run_graspgen_container.sh
```

The on-demand system:
1. Randomizes object position and joint angles
2. Exports URDF state to a temp folder inside the asset directory
3. Calls GraspGen via `docker exec`
4. Uses the highest-confidence grasp for the demo

---

## Step 3: Generate Demonstrations

The demo generation script supports **two sampling methods**:

1. **Heuristic (Original)**: Random object placement + robot configuration sampling
2. **Integrated Inverse (New)**: Direct sampling from an integrated inverse map

### Using the Heuristic Method (Original)

The heuristic method samples object positions randomly and attempts to find valid robot configurations:

```bash
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --exp-name sim_microwave_demos_heuristic \
    --sampling-method heuristic \
    --num-to-generate 100 \
    --max-try-times 500 \
    --timeout-init 120 \
    --timeout-exec 300
```

### Integrated Inverse Method (recommended)

The integrated inverse method computes an aggregated inverse reachability
distribution along the manipulation trajectory and samples object poses
from that distribution directly. This reduces wasted attempts and is the
recommended approach over naive rejection sampling.

```bash
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/sim_microwave_good \
    --exp-name sim_microwave_demos_integrated \
    --sampling-method integrated_inverse \
    --rm4d-map data/rm4d_franka_1M.npy \
    --num-to-generate 100 \
    --max-try-times 300 \
    --timeout-init 120 \
    --timeout-exec 300
```

> **Note**: The integrated inverse method requires a pre-computed RM4D reachability map. See [RM4D Setup](#rm4d-setup) for details.

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--asset-dir` | Required | Path to object asset folder |
| `--exp-name` | `debug_custom` | Experiment name for output |
| `--sampling-method` | `heuristic` | Sampling method: `heuristic` or `integrated_inverse` |
| `--num-to-generate` | 10 | Number of successful demos to collect |
| `--max-try-times` | 200 | Maximum attempts before giving up |
| `--center-jitter` | 0.05 | Position randomization (meters) |
| `--size-scale` | (0.8, 1.1) | Scale randomization range |
| `--target-position` | (0.4, 0.0, 0.0) | Object placement in world frame |
| `--near-distance` | 0.15 | Min EEF-to-grasp distance |
| `--far-distance` | 0.5 | Max EEF-to-grasp distance |
| `--render` | False | Enable GUI visualization |
| `--timeout-init` | 60.0 | Timeout for init state generation |
| `--timeout-exec` | 300.0 | Timeout for demo execution |

#### RM4D / Integrated Inverse Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rm4d-map` | None | Path to RM4D reachability map (.npy) |
| `--approach-distance` | 0.15 | Gripper approach distance (meters) |
| `--target-ratio` | 0.8 | Target opening ratio for manipulation |

### Algorithm Comparison

#### Heuristic Method (Original)
```python
for each demo attempt:
    1. Sample random object position/orientation
    2. Sample random robot configuration
    3. Check if robot can reach the grasp pose
    4. If yes, attempt motion planning and execution
    # Problem: Many attempts wasted on infeasible placements
```

#### Integrated Inverse Method (New)
```python
for each demo attempt:
    1. Sample randomized object config (scale, joint angles)
    2. Predict grasps for the object state
    3. Compute in-contact trajectory in object frame
    4. Aggregate reachability along trajectory and form a pose distribution
    5. Sample poses from the distribution and execute on feasible ones
    # Advantage: Samples from feasible set directly, fewer wasted attempts
```

Using RM4D-based pre-filtering can improve success rate per attempt by rejecting infeasible placements before attempting expensive motion planning.
 



### What the Pipeline Does

1. **Creates variant configs** – Randomizes position, orientation, and scale
2. **Generates initial state** – Places object, finds valid robot configuration
3. **Predicts grasps** – Uses pre-loaded or on-demand GraspGen
4. **Executes manipulation** – Approaches handle, grasps, opens door/drawer
5. **Saves demos** – Stores trajectories in `experiment/<exp_name>/`

### Output Structure

```
experiment/<exp_name>/
├── meta_info.json           # Generation parameters
├── demo_0000.pkl            # Demo trajectory + observations
├── demo_0001.pkl
└── ...
```

---

## RM4D Setup

The RM4D (Reachability Map 4D) reachability map encodes inverse reachability information for the robot arm and is used by the integrated inverse method.

### Using Pre-computed Maps

Pre-computed maps are available in `data/`:

```bash
# 1M sample Franka map (recommended)
data/rm4d_franka_1M.npy

# Alternative sizes available (faster but less accurate)
data/rm4d_franka_1M.100000.npy   # 100K samples
data/rm4d_franka_1M.200000.npy   # 200K samples
```

### Creating a New Map

If you need to create a new map (e.g., for a different robot) you have two options: use the Python helper or run the bundled CLI script.

Python API (programmatic):

```python
from manipulation.rm4d_filtering import load_or_create_rm4d_map

# This will create a new map if it doesn't exist
rmap = load_or_create_rm4d_map(
    map_path="data/rm4d_franka_1M.npy",
    robot_type="franka",
    n_samples=1_000_000,
    voxel_res=0.05,
    n_bins_theta=36,
)
```

CLI script (recommended for reproducible builds):

```bash
# Build a 1M-sample RM4D map for Franka (runs in PyBullet, may take 10-60min)
source prepare.sh && conda activate articubot
python scripts/build_rm4d_map.py --robot franka --samples 1000000 --output data/rm4d_franka_1M.npy

# For a faster, low-resolution map (useful for testing)
python scripts/build_rm4d_map.py --robot franka --samples 100000 --output data/rm4d_franka_100K.npy
```

Notes:
- The builder uses PyBullet (DIRECT mode) and RM4D's joint-space sampler; run inside the project env (`source prepare.sh && conda activate articubot`).
- Use `--checkpoint-every N` to save intermediate map checkpoints while sampling large numbers of configurations.

### Visualizing an RM4D Map

A lightweight interactive visualizer is provided to inspect the map and its inverse-base predictions.

```bash
# Start the viser-based visualizer and point it at your map
source prepare.sh && conda activate articubot
python scripts/visualize_rm4d_viser.py --rm4d-map data/rm4d_franka_1M.npy --port 8080 --max-points 50000
```

Open `http://localhost:8080` in your browser to view:
- A sampled point-cloud of reachable EE positions (colorized by height)
- A robot URDF with joint sliders
- Controls to query inverse base positions for a target EE pose (toggle "Show base positions")

> **Note**: `visualize_rm4d_viser.py` depends on the `viser` package (see `environment.yaml` / conda env). Sampling fewer points via `--max-points` reduces memory and rendering time.

---

## Step 4: Verify Generated Demos

Check the number of successful demos:

```bash
ls data/custom_objects/sim_microwave_good/experiment/sim_microwave_demos/*.pkl | wc -l
```

Visualize a demo (if you have the viewer):

```bash
python scripts/visualize_demo.py \
    --demo-path data/custom_objects/sim_microwave_good/experiment/sim_microwave_demos/demo_0000.pkl
```

---

## Troubleshooting

### "No handle points found inside annotation bbox"
- Check that `annotation.json` points are in the correct coordinate frame
- Ensure `handle_name` in config matches the URDF link name
- Try increasing `--handle-dilate` parameter

### "GraspGen produced no grasps"
- Verify Docker container is running: `docker ps | grep graspgen`
- Check path mappings in `graspgen_client.py`
- Ensure URDF and meshes are accessible inside container

### "Failed to generate initial state" (timeout)
- Increase `--timeout-init` (e.g., 180 seconds)
- Check that object is within robot workspace (`--target-position`)
- Verify grasp predictions exist and are reachable

### Import/Module errors
- Run `source prepare.sh` to set environment variables
- Activate conda: `conda activate articubot`

## Optional: Visualize Trajectories and Mesh Orientation

You can inspect the URDF and overlay recorded end-effector (EEF) trajectories to verify mesh orientation and handle alignment.

- **Why:** Use the mesh viewer to confirm the canonical mesh orientation/scale from `base_config.yaml`, and overlay trajectories (transformed into object-local coordinates) to verify the gripper approaches the handle as expected.
- **Tools:** [mesh_viewer.py](mesh_viewer.py) (URDF/viewer with optional scaled overlay) and [visualize_trajectories.py](visualize_trajectories.py) (converts EEF positions from world to object/mesh coordinates).
- **Quick commands:**

```bash
# activate env
source prepare.sh
conda activate articubot

# preview URDF + scaled mesh (uses base_config.yaml for scale if present)
python mesh_viewer.py \
    --urdf-path data/custom_objects/sim_microwave_good/color_0025_0.urdf \
    --annotation-path data/custom_objects/sim_microwave_good/annotation.json

# visualize multiple trajectories transformed into mesh coordinates
python visualize_trajectories.py \
    --experiment-path data/custom_objects/sim_microwave_good/experiment/traj_100 \
    --urdf-path data/custom_objects/sim_microwave_good/color_0025_0.urdf \
    --config-path data/custom_objects/sim_microwave_good/base_config.yaml \
    --object-name "sim microwave good" \
    --max-trajectories 16 --subsample 5
```
---

## Related Files

| File | Description |
|------|-------------|
| `manipulation/gen_demo_custom.py` | Main demo generation script (supports heuristic & integrated_inverse) |
| `manipulation/generate_base_config_from_urdf_custom.py` | Config generator |
| `manipulation/custom_object_utils/demo_utils.py` | Demo utilities + on-demand GraspGen |
| `manipulation/custom_object_utils/contact_trajectory.py` | In-contact trajectory computation |
| `manipulation/custom_object_utils/graspgen_client.py` | GraspGen Docker client |
| `manipulation/custom_object_utils/object_utils.py` | URDF/annotation utilities |
| `manipulation/rm4d_filtering/trajectory_filter.py` | RM4D trajectory reachability filter |
| `test_custom_model.py` | Test pretrained policy on custom objects |

---

## Final Directory Structure

After running the full pipeline, your object folder will contain all generated files:

```
data/custom_objects/<object_name>/
├── <object>.urdf                  # Original URDF model
├── base_config.yaml               # Generated configuration
├── annotation.json                # Original affordance annotation
├── mobility_v2.json               # Auto-generated joint metadata
├── predicted_grasps.yml           # Grasp predictions (if pre-generated)
│
├── mesh/                          # Original mesh files
│   ├── link0.obj
│   ├── link1.obj
│   └── ...
│
├── parts_render/                  # Auto-generated handle point cloud
│   └── 1link1_points.npy          # Format: {semantic_id}{link_name}_points.npy
│
├── configs/                       # Auto-generated variant configs
│   ├── config_custom_0000.yaml    # Randomized position/orientation/scale
│   ├── config_custom_0001.yaml
│   ├── config_custom_0002.yaml
│   └── ...
│
├── experiment/                    # Output demonstrations
│   └── <exp_name>/
│       ├── meta_info.json         # Generation parameters & stats
│       ├── demo_0000.pkl          # Trajectory + observations
│       ├── demo_0001.pkl
│       └── ...
│
└── _tmp_graspgen_*/               # Temporary folders (can be deleted)
    ├── object.urdf                # URDF snapshot with frozen joint state
    ├── mesh/                      # Copied meshes
    └── predicted_grasps.yml       # On-demand grasp predictions
```

## Next Steps

After generating demos, you can:

1. **Train a policy** – Use the 3D Diffusion Policy training scripts
2. **Evaluate** – Test with `test_custom_model.py`
3. **Transfer to real robot** – Deploy the trained policy 
