# Viser Integration Summary

## Changes Made

This document summarizes the integration of interactive viser visualization into the ArticuBot demo generation pipeline.

### 1. New Files Created

#### `manipulation/custom_object_utils/viser_visualization.py` (~660 lines)
Interactive 3D visualization module using viser server.

**Key Features:**
- `IntegratedInverseMapVisualizer` class managing viser server and GUI
- Real-time 3D trajectory visualization with orientation arrows
- Per-waypoint inverse reachability heatmap display
- Integrated map visualization showing overall object placement distribution
- **Robot URDF visualization** with joint control sliders
- **Object mesh URDF visualization** at sampled pose
- Interactive GUI controls (sliders, dropdowns, buttons)
- Keyboard input support ('n' key to proceed)

**Main Methods:**
- `visualize_attempt()`: Main entry point, displays complete visualization
- `_load_robot_urdf()`: Loads Franka Panda robot with joint sliders
- `_load_object_urdf()`: Loads object mesh at specified pose
- `_visualize_trajectory()`: Renders 3D trajectory waypoints and orientations
- `_visualize_integrated_map()`: Shows aggregated reachability heatmap
- `_compute_per_waypoint_scores()`: Computes per-waypoint reachability
- `set_robot_joint_config()`: Updates robot to show sampled joint configuration

**Helper Functions:**
- `_resolve_urdf_packages()`: Resolves package:// URIs in URDF files to absolute paths

#### `VISER_VISUALIZATION_GUIDE.md`
User documentation explaining:
- How to use `--use-viser` flag
- GUI controls and keyboard shortcuts
- Troubleshooting tips
- Example workflows for debugging

### 2. Modified Files

#### `manipulation/gen_demo_custom.py`
**Added:**
- `--use-viser` command-line flag (enables interactive visualization)
- `--viser-port` argument (default: 8080)
- Import of `IntegratedInverseMapVisualizer`
- User instructions printed when viser enabled

**Modified Functions:**
- `run_integrated_inverse_generation()`: Passes `use_viser`, `viser_port`, and `attempt_number` to `custom_gen_init_state_integrated()`

#### `manipulation/custom_object_utils/demo_utils_integrated.py`
**Added Parameters to `custom_gen_init_state_integrated()`:**
- `use_viser: bool = False`
- `viser_port: int = 8080`
- `attempt_number: int = 0`

**Added Parameters to `_custom_gen_init_state_integrated()`:**
- Same three parameters as above
- Creates `IntegratedInverseMapVisualizer` instance when `use_viser=True`
- Passes visualizer to `integrated_sample_initial_state()`

**Modified:**
- Multiprocessing `Process()` call now passes all three new parameters
- Viser instance created inside subprocess (multiprocessing-safe)

#### `manipulation/custom_object_utils/integrated_sampling.py`
**Previously Modified (from earlier work):**
- `integrated_sample_initial_state()` already accepts `viser_visualizer` and `attempt_number`
- Calls `viser_visualizer.visualize_attempt()` after successful pose sampling
- Waits for user keyboard input before returning

### 3. Data Flow

```
gen_demo_custom.py (main loop)
  ↓ (--use-viser=True, --viser-port=8080, attempt_number=N)
custom_gen_init_state_integrated() [multiprocessing wrapper]
  ↓ (creates subprocess, passes use_viser, viser_port, attempt_number)
_custom_gen_init_state_integrated() [subprocess]
  ↓ (creates IntegratedInverseMapVisualizer if use_viser=True)
integrated_sample_initial_state()
  ↓ (after successful sampling)
viser_visualizer.visualize_attempt()
  ↓ (displays 3D visualization, blocks on keyboard input)
User presses 'n' → returns to main loop
```

### 4. Key Design Decisions

**Multiprocessing Compatibility:**
- Viser object cannot be pickled for multiprocessing
- **Solution**: Pass `use_viser` flag and `viser_port` integer instead
- Create viser instance inside subprocess after fork

**Blocking Behavior:**
- User requested manual stepping through attempts
- **Solution**: `_wait_for_next()` uses `threading.Event()` to block
- GUI button click or 'n' key press releases the block

**Visualization Timing:**
- Visualization happens immediately after pose sampling
- Before motion planning execution
- Allows user to inspect if sampling succeeded/failed before continuing

**Data Passing:**
- All visualization data computed from available variables:
  - `trajectory_world`: 3D waypoint positions and orientations
  - `sampler.integrated_map`: Pre-computed reachability distribution
  - `per_waypoint_scores`: Computed on-the-fly from RM4D map
- No additional data structures needed

### 5. Testing

**Import Test:**
```bash
cd /home/junting/repo/articulated_objects/ArticuBot
source prepare.sh
python -c "from manipulation.gen_demo_custom import main; print('Import successful')"
# Output: Import successful ✓
```

**Full Integration Test (Recommended):**
```bash
# Test with GIF visualization only (fast, no user interaction)
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/color_0025_0_v2 \
    --sampling-method integrated_inverse \
    --rm4d-map data/rm4d_franka_1M.npy \
    --debug-vis \
    --num-to-generate 1 \
    --max-try-times 5

# Test with interactive viser (manual stepping)
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/color_0025_0_v2 \
    --sampling-method integrated_inverse \
    --rm4d-map data/rm4d_franka_1M.npy \
    --use-viser \
    --viser-port 8080 \
    --num-to-generate 1 \
    --max-try-times 5
```

### 6. Usage Examples

**Debug Low Success Rate:**
```bash
# Run with viser to manually inspect each failed attempt
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/my_object \
    --sampling-method integrated_inverse \
    --rm4d-map data/rm4d_franka_1M.npy \
    --use-viser \
    --num-to-generate 10
```

**Batch Processing with GIF Archive:**
```bash
# Generate many demos with GIF visualization for later review
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/my_object \
    --sampling-method integrated_inverse \
    --rm4d-map data/rm4d_franka_1M.npy \
    --debug-vis \
    --num-to-generate 100 \
    --max-try-times 500
```

**Parameter Tuning:**
```bash
# Use viser to test different parameters interactively
python manipulation/gen_demo_custom.py \
    --asset-dir data/custom_objects/my_object \
    --sampling-method integrated_inverse \
    --rm4d-map data/rm4d_franka_1M.npy \
    --use-viser \
    --coverage-threshold 0.7 \  # Require 70% waypoints reachable
    --min-score-threshold 0.2 \  # Filter low-probability poses
    --aggregation-method mean \
    --num-to-generate 5
```

### 7. Known Limitations

1. **Single Viser Instance**: Only one viser server per port. Cannot run multiple concurrent demo generation processes with viser enabled.

2. **Multiprocessing Overhead**: Creating viser instance in each subprocess adds ~1 second startup time per attempt.

3. **Browser Required**: User must have browser access to visualize. No headless mode.

4. **Memory Usage**: Keeping viser server alive for many attempts can accumulate memory. Recommended to restart script every 50-100 attempts.

### 8. Future Enhancements

**Potential Improvements:**
- **Replay Mode**: Save visualization data to file, replay later without re-running
- **Comparison View**: Show multiple attempts side-by-side
- **Heatmap Filtering**: Allow user to set threshold in GUI and resample
- **Statistics Dashboard**: Show success rate, timing, reachability distributions
- **Recording**: Capture viser views as video for presentations

### 9. Related Documentation

- [VISER_VISUALIZATION_GUIDE.md](VISER_VISUALIZATION_GUIDE.md) - User guide
- [custom_dataset_demo_gen.md](custom_dataset_demo_gen.md) - Original demo generation docs
- [manipulation/custom_object_utils/viser_visualization.py](manipulation/custom_object_utils/viser_visualization.py) - Implementation
- [manipulation/custom_object_utils/debug_visualization.py](manipulation/custom_object_utils/debug_visualization.py) - GIF visualization

### 10. Dependencies

**Required:**
- `viser>=1.0.0` - 3D visualization server
- `numpy` - Array operations
- `scipy` - Spatial transformations
- Existing ArticuBot dependencies (PyBullet, RM4D, etc.)

**Installation:**
```bash
pip install viser
```
