#!/usr/bin/env python3
"""Generate demonstrations for custom URDF objects with affordance annotations.

This script implements two sampling methods:
1. HEURISTIC (original): Random object placement + robot configuration sampling
2. INTEGRATED_INVERSE (new): Direct sampling from an integrated inverse map

The integrated inverse method computes a distribution over object poses by
aggregating reachability along the manipulation trajectory and samples from
that distribution directly, improving sampling efficiency over naive
rejection sampling.

Usage:
    # Heuristic method (original)
    python gen_demo_custom.py --asset-dir data/custom_objects/my_object \
        --sampling-method heuristic
    
    # Integrated inverse method (new)
    python gen_demo_custom.py --asset-dir data/custom_objects/my_object \
        --sampling-method integrated_inverse \
        --rm4d-map data/rm4d_franka_1M.npy
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import multiprocessing as mp
import subprocess
import numpy as np
import time
from termcolor import cprint
from enum import Enum
from typing import Optional, Tuple, List

from manipulation.envs.articulated import handle_name_dict
from manipulation.gen_demo import get_current_num_demos

from manipulation.custom_object_utils.object_utils import (
    find_first_urdf,
    ensure_support_files,
    load_predicted_grasps,
    find_annotation_path,
)
from manipulation.custom_object_utils.demo_utils import (
    parse_config_metadata,
    create_variant_config,
    custom_gen_init_state,
    custom_execute,
    resolve_relative_path,
)


class SamplingMethod(Enum):
    """Available sampling methods for initial state generation."""
    HEURISTIC = "heuristic"  # Original random sampling method
    INTEGRATED_INVERSE = "integrated_inverse"  # Direct sampling from integrated inverse map


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate demos for custom objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Basic configuration
    _add_basic_args(parser)
    
    # Sampling method selection
    parser.add_argument("--sampling-method", type=str, default="heuristic",
                        choices=["heuristic", "integrated_inverse"],
                        help="Sampling method: 'heuristic' (original) or 'integrated_inverse' (direct sampling)")
    
    # Distance parameters
    parser.add_argument("--near-distance", type=float, default=0.15,
                        help="Min EEF-to-grasp distance")
    parser.add_argument("--far-distance", type=float, default=0.5,
                        help="Max EEF-to-grasp distance")
    
    # Rendering and debugging
    parser.add_argument("--render", action="store_true",
                        help="Enable GUI visualization")
    
    # Object configuration
    _add_object_agrs(parser)
    
    # Randomization parameters
    _add_randomization_args(parser)
    
    # Timeout parameters
    parser.add_argument("--timeout-init", type=float, default=60.0,
                        help="Timeout for init state generation")
    parser.add_argument("--timeout-exec", type=float, default=300.0,
                        help="Timeout for demo execution")
    
    # URDF and joint configuration
    _add_urdf_args(parser)
    
    # Object placement
    parser.add_argument("--target-position", type=float, nargs=3, default=(0.4, 0.0, 0.0),
                        help="Target world position (x, y, z) for object placement")
    parser.add_argument("--object-z-offset", type=float, nargs='+', default=[0.2, 0.4],
                        help="Z-offset to elevate object above ground (meters). "
                             "If two values provided, randomizes between them.")
    
    # RM4D parameters (used by integrated inverse sampling)
    parser.add_argument("--rm4d-map", type=pathlib.Path, default=None,
                        help="Path to RM4D reachability map (.npy)")
    
    # Integrated inverse map parameters
    _add_integrated_args(parser)
    
    # Trajectory computation parameters (used by integrated_inverse and general methods)
    parser.add_argument("--approach-distance", type=float, default=0.15,
                        help="Approach distance for gripper (meters)")
    parser.add_argument("--target-ratio", type=float, default=0.8,
                        help="Target opening ratio for door/drawer manipulation")

    # Simulation parameters
    # (Removed cheat options for strict physics compliance)
    
    # Debugging and visualization
    _add_debug_args(parser)
    
    args = parser.parse_args()

    if args.use_viser:
        # Avoid timing out while stepping through interactive visualization.
        args.timeout_init = 1e6
        args.timeout_exec = 1e6

    return args


def _add_basic_args(parser):
    parser.add_argument("--asset-dir", default="data/custom_objects/color_0025_0_v2",
                        help="Processed asset folder")
    parser.add_argument("--config-path", default=None,
                        help="Base config path (defaults to <asset-dir>/base_config.yaml)")
    parser.add_argument("--exp-name", default="debug_custom",
                        help="Experiment name under <solution_path>/experiment/")
    parser.add_argument("--num-to-generate", type=int, default=10,
                        help="Number of successful demos to collect")
    parser.add_argument("--max-try-times", type=int, default=200,
                        help="Maximum attempts before giving up")
    parser.add_argument("--env-name", default="articulated_custom_object",
                        help="Environment name for simulation")
    parser.add_argument("--process-index", type=int, default=None,
                        help="Optional process index: when set, used to derive per-process seed (base seed + index if --seed provided)")


def _add_object_agrs(parser):
    parser.add_argument("--annotation-path", type=pathlib.Path, default=None,
                        help="Path to affordance.txt file (Nx3 points)")
    parser.add_argument("--handle-part-id", type=int, default=1,
                        help="Part ID for handle mesh")
    parser.add_argument("--handle-padding", type=float, default=0.0,
                        help="Padding for handle bounding box")
    parser.add_argument("--handle-dilate", type=float, default=0.0,
                        help="Dilation (meters) to expand affordance bbox for handle extraction")


def _add_randomization_args(parser):
    parser.add_argument("--center-jitter", type=float, default=0.05,
                        help="Position randomization (meters)")
    parser.add_argument("--size-scale", type=float, nargs=2, default=(0.6, 0.8),
                        help="Scale randomization range")
    parser.add_argument("--joint-angle-range", type=float, nargs=2, default=(0.0, 0.2),
                        help="Object joint angle randomization range as fraction of joint limits. "
                             "Default (0.0, 0.2) means joint will be opened 0-20%% of its full range, "
                             "similar to the original gen_demo.py behavior.")
    parser.add_argument(
        "--grasp-source",
        type=str,
        default="ondemand",
        choices=["auto", "precomputed", "ondemand"],
        help=(
            "How to obtain grasps: 'ondemand' queries GraspGen for each randomized object state; "
            "'precomputed' forces predicted_grasps.yml (and disables scale/joint randomization to avoid mismatches); "
            "'auto' is treated as 'ondemand' (recommended) to avoid grasp mismatches under randomization."
        ),
    )
    parser.add_argument(
        "--top-k-grasps",
        type=int,
        default=1,
        help=(
            "Number of top-scoring grasps to try per variant config. "
            "All successful demos from these grasps are saved, improving efficiency. Default: 10"
        ),
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility (base seed for parallel runs)")


def _add_urdf_args(parser):
    parser.add_argument("--urdf-path", type=pathlib.Path, default=None,
                        help="Path to URDF file")
    parser.add_argument("--joint-name", default=None,
                        help="Specific joint name to use")
    parser.add_argument("--prepare-mobility", action="store_true",
                        help="Regenerate mobility_v2.json if missing")
    parser.add_argument("--overwrite-support", action="store_true",
                        help="Overwrite existing support files")


def _add_integrated_args(parser):
    parser.add_argument("--xy-resolution", type=float, default=0.02,
                        help="XY grid resolution for integrated inverse map (meters)")
    parser.add_argument("--theta-bins", type=int, default=24,
                        help="Number of orientation bins for integrated inverse map")
    parser.add_argument("--aggregation-method", type=str, default="mean",
                        choices=["min", "product", "mean"],
                        help="Method to aggregate waypoint scores: 'min', 'product', or 'mean'")
    parser.add_argument("--coverage-threshold", type=float, default=0.5,
                        help="Minimum fraction of reachable waypoints (for 'mean' aggregation)")
    parser.add_argument("--sampling-temperature", type=float, default=1.0,
                        help="Sampling temperature for integrated inverse map (higher=more uniform)")
    parser.add_argument("--min-score-threshold", type=float, default=0.1,
                        help="Minimum reachability score threshold for valid poses")


def _add_debug_args(parser):
    parser.add_argument("--debug-vis", action="store_true",
                        help="Save debug visualization as GIF showing trajectory poses and inverse map values")
    parser.add_argument("--debug-vis-output", type=pathlib.Path, default=None,
                        help="Output path for debug visualization GIF (defaults to <exp-dir>/debug_vis.gif)")
    parser.add_argument(
        "--save-heatmaps",
        action="store_true",
        help=(
            "Dump integrated-inverse heatmaps for offline inspection. Writes the full (x,y,theta) score grid "
            "as .npz and saves a PNG for every theta slice under a per-attempt folder."
        ),
    )
    parser.add_argument(
        "--heatmap-output-dir",
        type=pathlib.Path,
        default=None,
        help="Base output directory for heatmaps (default: <experiment-dir>/heatmaps)",
    )
    parser.add_argument(
        "--heatmap-per-waypoint",
        action="store_true",
        help=(
            "Also dump per-waypoint inverse reachability heatmaps (best-theta) and arrays. "
            "This can be slower and generate many files."
        ),
    )
    parser.add_argument("--init-only", action="store_true",
                        help="Only run initial-state sampling (and write config/init-state); skip full demo execution")
    parser.add_argument("--use-viser", action="store_true",
                        help="Enable interactive 3D visualization with viser (waits for keyboard input between attempts)")
    parser.add_argument("--viser-port", type=int, default=8080,
                        help="Port number for viser server (default: 8080)")
    
    # Filtering layer control (for benchmarking / ablation studies)
    parser.add_argument(
        "--disable-prefilter",
        action="store_true",
        help="Disable affordance pre-filter (geometric workspace + facing check). For benchmarking."
    )
    parser.add_argument(
        "--disable-occlusion-filter",
        action="store_true",
        help="Disable occlusion/facing filter during init state sampling."
    )


def _get_gpu_count() -> int:
    raw = os.environ.get("ARTICUBOT_NUM_GPUS")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        if result.returncode == 0:
            lines = [line for line in (result.stdout or "").splitlines() if line.strip().startswith("GPU")]
            return len(lines)
    except Exception:
        pass
    return 0


def _configure_graspgen_gpu(args: argparse.Namespace) -> None:
    if getattr(args, "process_index", None) is None:
        return

    num_gpus = _get_gpu_count()
    if num_gpus <= 0:
        cprint("No GPUs detected; skipping per-process GraspGen GPU assignment.", "yellow")
        return

    gpu_id = int(args.process_index) % num_gpus
    base_name = os.environ.get("GRASPGEN_CONTAINER_BASE_NAME", "friendly_galileo")
    container_name = f"{base_name}_gpu{gpu_id}"

    os.environ["GRASPGEN_CONTAINER_NAME"] = container_name
    os.environ["GRASPGEN_GPU_ID"] = str(gpu_id)
    os.environ.setdefault("GRASPGEN_AUTOSTART", "1")

    cprint(
        f"GraspGen GPU assignment: process {args.process_index} -> GPU {gpu_id} (container {container_name})",
        "cyan",
    )



def _resolve_annotation_path(args: argparse.Namespace) -> None:
    """Resolve annotation path logic extracted from validate_and_prepare_args."""
    config_parent = args.config_path.parent
    object_name, handle_name, annotation_hint, solution_path = parse_config_metadata(str(args.config_path))
    
    args._object_name = object_name
    args._handle_name = handle_name
    args._solution_path = solution_path
    
    resolved_annotation = None
    if args.annotation_path is not None:
        resolved_annotation = pathlib.Path(args.annotation_path).expanduser().resolve(strict=False)
        if resolved_annotation.suffix.lower() not in {".txt", ".pts", ".xyz"}:
            raise ValueError(
                f"Affordance file must be a text file (affordance.txt), got {resolved_annotation}"
            )
    elif annotation_hint:
        hint_path = pathlib.Path(str(annotation_hint))
        if hint_path.suffix.lower() in {".txt", ".pts", ".xyz"}:
            p_cwd = hint_path.resolve()
            p_rel = resolve_relative_path(annotation_hint, config_parent)

            if p_cwd.exists():
                resolved_annotation = p_cwd
            elif p_rel.exists():
                resolved_annotation = p_rel
            else:
                resolved_annotation = p_rel

    if resolved_annotation is None or not resolved_annotation.exists():
        resolved_annotation = find_annotation_path(args.asset_dir, resolved_annotation or annotation_hint)

    args.annotation_path = resolved_annotation

    if args.annotation_path is None or not args.annotation_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found at {args.annotation_path}")


def validate_and_prepare_args(args: argparse.Namespace) -> None:
    """Validate arguments and resolve paths.
    
    Args:
        args: Parsed command line arguments.
        
    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If arguments are invalid.
    """
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    # If a per-process index is supplied, derive a per-process seed from it.
    if getattr(args, "process_index", None) is not None:
        if args.seed is not None:
            derived_seed = int(args.seed + int(args.process_index))
        else:
            derived_seed = int(args.process_index)
        random.seed(derived_seed)
        np.random.seed(derived_seed % (2**32))
        os.environ["PYTHONHASHSEED"] = str(derived_seed)
        cprint(f"Using process index {args.process_index} to derive seed {derived_seed}", "cyan")

    _configure_graspgen_gpu(args)

    args.asset_dir = pathlib.Path(args.asset_dir).expanduser().resolve()
    if not args.asset_dir.exists():
        raise FileNotFoundError(f"Asset dir does not exist: {args.asset_dir}")
    
    args.config_path = pathlib.Path(args.config_path or args.asset_dir / "base_config.yaml")
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    # Resolve annotation path
    _resolve_annotation_path(args)

    if args.urdf_path is not None:
        args.urdf_path = pathlib.Path(args.urdf_path).expanduser().resolve(strict=False)
    else:
        args.urdf_path = find_first_urdf(args.asset_dir)

    if args.rm4d_map is not None:
        args.rm4d_map = pathlib.Path(args.rm4d_map).expanduser().resolve(strict=False)

    if args.heatmap_output_dir is not None:
        args.heatmap_output_dir = pathlib.Path(args.heatmap_output_dir).expanduser().resolve(strict=False)

    if not args.urdf_path.exists():
        raise FileNotFoundError(f"URDF path not found at {args.urdf_path}")

    _validate_grasp_source(args)
    _validate_integrated_inverse_args(args)

    if getattr(args, "disable_prefilter", False) and not getattr(args, "disable_occlusion_filter", False):
        cprint(
            "WARNING: --disable-prefilter is deprecated; using it to disable occlusion filter.",
            "yellow",
        )
        args.disable_occlusion_filter = True


def _validate_grasp_source(args):
    """Validate grasp source argument."""
    # Grasp-source policy:
    # For randomized scale/joint configs, pre-generated grasps generally mismatch.
    # Treat 'auto' as 'ondemand' to avoid accidentally using predicted_grasps.yml.
    if args.grasp_source == "auto":
        cprint(
            "WARNING: --grasp-source auto is treated as ondemand to ensure grasps match randomized object states.",
            "yellow",
        )
        args.grasp_source = "ondemand"


def _validate_integrated_inverse_args(args):
    """Validate arguments specific to integrated inverse sampling."""
    if args.sampling_method == "integrated_inverse":
        if args.rm4d_map is None:
            cprint("WARNING: --rm4d-map not provided for integrated_inverse method. "
                   "Will fall back to random sampling.", "yellow")


def load_grasps(args: argparse.Namespace) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load predicted grasps from file.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Tuple of (grasps, confidences) or None if not available.
    """
    predicted_grasps_path = args.asset_dir / "predicted_grasps.yml"
    predicted_grasps: Optional[Tuple[np.ndarray, np.ndarray]] = None

    if args.grasp_source == "ondemand":
        cprint(
            "Grasp source: ondemand (ignoring predicted_grasps.yml if present).", "cyan"
        )
        return None

    if args.grasp_source == "precomputed" and not predicted_grasps_path.exists():
        raise FileNotFoundError(
            f"--grasp-source precomputed requires {predicted_grasps_path}"
        )
    
    if predicted_grasps_path.exists():
        predicted_grasps = _load_and_validate_precomputed_grasps(args, predicted_grasps_path)
    else:
        if args.grasp_source == "precomputed":
            raise FileNotFoundError(
                f"--grasp-source precomputed requires {predicted_grasps_path}"
            )
        cprint(
            "No predicted_grasps.yml found. Will use on-demand GraspGen for each randomized state.",
            "cyan",
        )
    
    return predicted_grasps


def _load_and_validate_precomputed_grasps(args, path):
    """Load precomputed grasps and enforce fixed scale/joint configs."""
    print(f"Loading predicted grasps from {path}")
    predicted_grasps = load_predicted_grasps(path)
    
    if len(predicted_grasps[0]) == 0:
        print("No grasps found in predicted_grasps.yml, will use on-demand GraspGen")
        return None

    print(f"Loaded {len(predicted_grasps[0])} grasps")
    
    # Enforce fixed scale for pre-generated grasps
    scale_min, scale_max = args.size_scale
    if abs(scale_min - scale_max) > 1e-6 or abs(scale_min - 1.0) > 1e-6:
        _warn_fixed_config("size-scale", args.size_scale, "(1.0, 1.0)")
        args.size_scale = (1.0, 1.0)

    # Enforce fixed joint angle for pre-generated grasps
    jmin, jmax = tuple(args.joint_angle_range)
    if abs(jmin) > 1e-9 or abs(jmax) > 1e-9:
        _warn_fixed_config("joint-angle-range", args.joint_angle_range, "(0.0, 0.0)")
        args.joint_angle_range = (0.0, 0.0)
        
    return predicted_grasps


def _warn_fixed_config(param_name, current_val, enforced_val):
    cprint("=" * 70, "yellow")
    cprint("WARNING: Pre-generated grasps detected (predicted_grasps.yml exists).", "yellow")
    cprint("Grasps are computed for a FIXED object state (scale=1.0, default joints).", "yellow")
    cprint(f"Your --{param_name} {current_val} enables randomization.", "yellow")
    cprint(f"ENFORCING: Setting --{param_name} to {enforced_val} to prevent grasp mismatches.", "yellow")
    cprint("=" * 70, "yellow")


def _handle_init_only(args: argparse.Namespace, total_success: int) -> tuple[bool, int]:
    if not args.init_only:
        return False, total_success
    total_success += 1
    cprint(f"Init-only success. Total init states: {total_success}/{args.num_to_generate}", "green")
    return True, total_success


def setup_experiment_directory(args: argparse.Namespace) -> pathlib.Path:
    """Set up experiment directory and save metadata.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Path to experiment directory.
    """
    solution_root = pathlib.Path(args._solution_path).expanduser().resolve(strict=False)
    experiment_path = solution_root / "experiment" / args.exp_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings for JSON serialization
    args_dict = vars(args).copy()
    for k, v in list(args_dict.items()):
        if isinstance(v, pathlib.Path):
            args_dict[k] = str(v)
        elif k.startswith('_'):  # Remove internal attributes
            del args_dict[k]
            
    with (experiment_path / "meta_info.json").open("w", encoding="utf-8") as fp:
        json.dump(args_dict, fp, indent=4)
    
    return experiment_path


def run_heuristic_generation(args: argparse.Namespace, 
                              experiment_path: pathlib.Path,
                              predicted_grasps: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Run demo generation using the heuristic (original) method.
    
    This method uses random object placement and robot configuration sampling.
    It's the original approach kept for comparison.
    
    Args:
        args: Parsed command line arguments.
        experiment_path: Path to experiment directory.
        predicted_grasps: Pre-loaded grasps or None for on-demand generation.
    """
    cprint("=" * 70, "cyan")
    cprint("Using HEURISTIC sampling method (original)", "cyan")
    cprint("=" * 70, "cyan")
    
    num_existing = get_current_num_demos(str(experiment_path))
    cprint(f"Existing successful demos: {num_existing}", "yellow")

    attempt = 0
    total_success = num_existing
    
    while total_success < args.num_to_generate and attempt < args.max_try_times:
        t_config_start = time.time()
        variant_path = create_variant_config(
            str(args.config_path),
            args.asset_dir,
            attempt,
            args._handle_name,
            args.annotation_path,
            args.center_jitter,
            args.size_scale,
            urdf_path=args.urdf_path,
            predicted_grasps=predicted_grasps,
            target_position=tuple(args.target_position),
            joint_angle_range=tuple(args.joint_angle_range),
        )
        
        cprint(f"[TIME] create_variant_config: {time.time() - t_config_start:.4f}s", "blue")
        cprint(f"[HEURISTIC] Attempt {attempt+1}/{args.max_try_times}: {variant_path.name}", "cyan")
        
        # 1. Generate initial state
        if not _attempt_heuristic_init(args, variant_path):
            attempt += 1
            continue

        handled, total_success = _handle_init_only(args, total_success)
        if handled:
            attempt += 1
            continue
            
        # 2. Execute demo
        if _attempt_heuristic_execution(args, variant_path, experiment_path):
            total_success += 1
            cprint(f"Success! Total demos: {total_success}/{args.num_to_generate}", "green")
        else:
            cprint("Execution failed", "red")
            
        attempt += 1


def _attempt_heuristic_init(args, variant_path):
    """Run initial state generation for heuristic method."""
    t_init_start = time.time()
    success_init = custom_gen_init_state(
        str(variant_path),
        args.env_name,
        args.render,
        near_distance=args.near_distance,
        far_distance=args.far_distance,
        timeout=args.timeout_init,
        rm4d_map_path=None,  # Disable RM4D-based filtering for heuristic
        object_traj_path=None,
        enable_occlusion_filter=not args.disable_occlusion_filter,
    )
    cprint(f"[TIME] custom_gen_init_state: {time.time() - t_init_start:.4f}s", "blue")
    
    if not success_init:
        cprint("Failed to generate initial state", "red")
        return False
    return True


def _attempt_heuristic_execution(args, variant_path, experiment_path):
    """Run demo execution for heuristic method."""
    t_exec_start = time.time()
    success_exec = custom_execute(
        str(variant_path),
        args.env_name,
        str(args.asset_dir),
        str(experiment_path),
        timeout=args.timeout_exec
    )
    cprint(f"[TIME] custom_execute: {time.time() - t_exec_start:.4f}s", "blue")
    return success_exec
    

    cprint(f"\nHEURISTIC method completed: {total_success} demos from {attempt} attempts", "green")

def _get_z_offset(args: argparse.Namespace) -> float:
    """Determine Z-offset for the current configuration."""
    if isinstance(args.object_z_offset, list) and len(args.object_z_offset) == 2:
        return random.uniform(args.object_z_offset[0], args.object_z_offset[1])
    elif isinstance(args.object_z_offset, list) and len(args.object_z_offset) == 1:
        return args.object_z_offset[0]
    else:
        return args.object_z_offset if not isinstance(args.object_z_offset, list) else args.object_z_offset[0]


def _generate_grasps(args: argparse.Namespace, variant_path: pathlib.Path) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generate grasps for a specific variant config using on-demand GraspGen."""
    from manipulation.custom_object_utils.graspgen_client import (
        GraspGenConfig,
        predict_grasps_for_urdf_folder,
        prepare_urdf_with_joint_state,
    )
    from scipy.spatial.transform import Rotation as R
    import shutil
    import uuid
    import yaml

    t_grasp_start = time.time()
    grasps_list = []

    try:
        # Load config to get object info
        config = yaml.safe_load(open(str(variant_path), "r"))
        object_scale = 1.0
        urdf_path_for_graspgen = None
        
        for config_dict in config:
            if 'size' in config_dict:
                object_scale = float(config_dict['size'])
            if 'urdf_path' in config_dict:
                urdf_path_for_graspgen = config_dict['urdf_path']
        
        # Determine URDF path
        if urdf_path_for_graspgen is None:
            cfg_parent = pathlib.Path(variant_path).resolve().parent.parent
            candidates = list(cfg_parent.glob('*.urdf'))
            urdf_path_for_graspgen = str(candidates[0]) if candidates else None
        
        if urdf_path_for_graspgen is None:
            cprint(f"No URDF path available for GraspGen, skipping config", "red")
            return []
        
        # Prepare temporary URDF folder
        joint_states = {}
        gg_cfg = GraspGenConfig()
        tmp_root = pathlib.Path(gg_cfg.host_graspgen_root).resolve() / "_articubot_tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_dir = str(tmp_root / f"_tmp_{uuid.uuid4().hex[:8]}")
        pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        
        prepared = prepare_urdf_with_joint_state(urdf_path_for_graspgen, joint_states, tmp_dir, scale=object_scale)
        grasps, confidences = predict_grasps_for_urdf_folder(prepared, gg_cfg)
        
        if len(confidences) == 0:
            cprint(f"GraspGen returned no grasps, skipping config", "red")
            shutil.rmtree(tmp_dir)
            return []
        
        # Sort by confidence and take top-k
        order = np.argsort(-np.asarray(confidences))
        top_k = min(args.top_k_grasps, len(order))
        
        for j in range(top_k):
            gi = int(order[j])
            g = grasps[gi]
            g_pos = np.asarray(g[:3, 3], dtype=float)
            g_quat = R.from_matrix(g[:3, :3]).as_quat().astype(float)  # xyzw
            g_conf = float(confidences[gi])
            grasps_list.append((g_pos, g_quat, g_conf))
        
        cprint(f"[MULTI-GRASP] GraspGen found {len(grasps)} grasps, trying top {top_k}", "cyan")
        shutil.rmtree(tmp_dir)

        t_grasp_end = time.time()
        cprint(f"[TIME] GraspGen and prep: {t_grasp_end - t_grasp_start:.4f}s", "blue")
        
    except Exception as e:
        cprint(f"GraspGen preparation failed: {e}", "red")
        return []

    return grasps_list



def _update_and_save_variant_config(
    variant_path: pathlib.Path,
    grasp_pos: np.ndarray,
    grasp_quat: np.ndarray,
    config_attempt: int,
    grasp_idx: int
) -> pathlib.Path:
    import yaml
    # Update variant config
    config = yaml.safe_load(open(str(variant_path), "r"))
    for config_dict in config:
        if 'center' in config_dict:
            config_dict['predicted_grasp_position'] = str(tuple(grasp_pos.tolist()))
            config_dict['predicted_grasp_orientation'] = str(tuple(grasp_quat.tolist()))
    
    # Save specific config
    grasp_variant_path = variant_path.parent / f"config_custom_{config_attempt:04d}_g{grasp_idx:02d}.yaml"
    with open(str(grasp_variant_path), 'w') as f:
        yaml.dump(config, f, indent=4)
    return grasp_variant_path


def _get_vis_paths(args: argparse.Namespace, experiment_path: pathlib.Path, attempt: int) -> Tuple[Optional[str], Optional[str]]:
    debug_vis_path = None
    if args.debug_vis:
        if args.debug_vis_output:
            debug_vis_path = str(args.debug_vis_output)
        else:
            debug_vis_path = str(experiment_path / f"debug_vis_attempt_{attempt:04d}.gif")

    heatmap_dir = None
    if getattr(args, "save_heatmaps", False):
        base_dir = args.heatmap_output_dir if args.heatmap_output_dir is not None else (experiment_path / "heatmaps")
        heatmap_dir = str(pathlib.Path(base_dir) / f"attempt_{attempt:04d}")
    return debug_vis_path, heatmap_dir


def _execute_grasp_attempt(
    args: argparse.Namespace,
    variant_path: pathlib.Path,
    grasp_data: Tuple[np.ndarray, np.ndarray, float],
    experiment_path: pathlib.Path,
    attempt_info: dict
) -> bool:
    """Execute a single grasp attempt."""
    from manipulation.custom_object_utils.demo_utils_integrated import (
        custom_gen_init_state_integrated,
    )
    import yaml

    grasp_pos, grasp_quat, grasp_conf = grasp_data
    attempt = attempt_info['attempt']
    grasp_idx = attempt_info['grasp_idx']
    total_grasps = attempt_info['total_grasps']
    current_z_offset = attempt_info['z_offset']
    config_attempt = attempt_info['config_attempt']

    cprint(f"  [Grasp {grasp_idx+1}/{total_grasps}] conf={grasp_conf:.3f}, attempt #{attempt}", "cyan")
    
    grasp_variant_path = _update_and_save_variant_config(
        variant_path, grasp_pos, grasp_quat, config_attempt, grasp_idx
    )
    
    debug_vis_path, heatmap_dir = _get_vis_paths(args, experiment_path, attempt)
    
    # Generate init state
    t_init_start = time.time()
    success_init = custom_gen_init_state_integrated(
        str(grasp_variant_path),
        args.env_name,
        args.render,
        near_distance=args.near_distance,
        far_distance=args.far_distance,
        timeout=args.timeout_init,
        rm4d_map_path=str(args.rm4d_map) if args.rm4d_map else None,
        xy_resolution=args.xy_resolution,
        theta_bins=args.theta_bins,
        aggregation_method=args.aggregation_method,
        coverage_threshold=args.coverage_threshold,
        temperature=args.sampling_temperature,
        min_score_threshold=args.min_score_threshold,
        approach_distance=args.approach_distance,
        target_ratio=args.target_ratio,
        asset_dir=str(args.asset_dir),
        debug_vis_path=debug_vis_path,
        object_z_offset=current_z_offset,
        use_viser=args.use_viser,
        viser_port=args.viser_port,
        attempt_number=attempt,
        save_heatmaps=bool(getattr(args, "save_heatmaps", False)),
        heatmap_dir=heatmap_dir,
        heatmap_per_waypoint=bool(getattr(args, "heatmap_per_waypoint", False)),
        enable_occlusion_filter=not args.disable_occlusion_filter,
        disable_prefilter=(args.disable_prefilter or args.disable_occlusion_filter),
    )
    t_init_end = time.time()
    cprint(f"[TIME] custom_gen_init_state_integrated: {t_init_end - t_init_start:.4f}s", "blue")
    
    if not success_init:
        cprint(f"    Init failed for grasp {grasp_idx+1}", "yellow")
        return False
        
    if args.init_only:
        cprint(f"Init-only success.", "green")
        return True
    
    # Execute
    t_exec_start = time.time()
    success_exec = custom_execute(
        str(grasp_variant_path),
        args.env_name,
        str(args.asset_dir),
        str(experiment_path),
        timeout=args.timeout_exec
    )
    t_exec_end = time.time()
    cprint(f"[TIME] custom_execute: {t_exec_end - t_exec_start:.4f}s", "blue")
    
    if success_exec:
        cprint(f"    ✓ Success!", "green")
        return True
    else:
        cprint(f"    ✗ Execution failed", "red")
        return False


def run_integrated_inverse_generation(args: argparse.Namespace,
                                       experiment_path: pathlib.Path,
                                       predicted_grasps: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Run demo generation using integrated inverse map sampling with multi-grasp retry.
    
    This method implements an efficient sampling approach with multi-grasp optimization:
    1. For each randomized variant config, call GraspGen ONCE to get top-k grasps
    2. Try ALL top-k grasps for sampling + execution
    3. Save ALL successful demos (not just the first one)
    
    This dramatically improves efficiency by:
    - Avoiding redundant GraspGen calls for failed attempts
    - Trying diverse grasps that may have different reachability profiles
    - Maximizing demos per variant config iteration
    
    Args:
        args: Parsed command line arguments.
        experiment_path: Path to experiment directory.
        predicted_grasps: Pre-loaded grasps or None for on-demand generation.
    """
    cprint("Using INTEGRATED_INVERSE sampling method with MULTI-GRASP RETRY", "magenta")
    cprint(f"Top-k grasps per config: {args.top_k_grasps}", "magenta")

    from manipulation.custom_object_utils.demo_utils_integrated import (
        custom_gen_init_state_integrated,
    )
    from manipulation.custom_object_utils.graspgen_client import (
        GraspGenConfig,
        predict_grasps_for_urdf_folder,
        prepare_urdf_with_joint_state,
    )
    from scipy.spatial.transform import Rotation as R
    import shutil
    import uuid
    import pybullet as p
    import yaml
    
    num_existing = get_current_num_demos(str(experiment_path))
    cprint(f"Existing successful demos: {num_existing}", "yellow")

    attempt = 0
    config_attempt = 0  # Tracks how many variant configs we've created
    total_success = num_existing
    
    while total_success < args.num_to_generate and config_attempt < args.max_try_times:
        demos_found, attempts_made = _process_single_config_integrated(
            args, experiment_path, config_attempt, attempt, total_success
        )
        
        total_success += demos_found
        attempt += attempts_made
        config_attempt += 1
    
    cprint(f"\nINTEGRATED_INVERSE (multi-grasp) completed: {total_success} demos from {attempt} attempts ({config_attempt} configs)", "magenta")


def _process_single_config_integrated(args, experiment_path, config_attempt, current_attempt_count, total_success_count):
    """Process a single variant configuration with integrated inverse sampling."""
    # Step 1: Create a new variant config (randomized scale, joint angle, z-offset)
    t_config_start = time.time()
    variant_path = create_variant_config(
        str(args.config_path),
        args.asset_dir,
        config_attempt,
        args._handle_name,
        args.annotation_path,
        args.center_jitter,
        args.size_scale,
        urdf_path=args.urdf_path,
        predicted_grasps=None,
        target_position=tuple(args.target_position),
        joint_angle_range=tuple(args.joint_angle_range),
    )
    cprint(f"[TIME] create_variant_config: {time.time() - t_config_start:.4f}s", "blue")
    
    cprint(f"\n[MULTI-GRASP] Config {config_attempt+1}/{args.max_try_times}: {variant_path.name}", "magenta")
    
    current_z_offset = _get_z_offset(args)
    
    # Step 2: Generate grasps ONCE
    grasps_list = _generate_grasps(args, variant_path)
    
    if not grasps_list:
        return 0, 0
    
    # Step 3: Try ALL top-k grasps for this config
    success_this_config = 0
    attempts_made = 0
    
    for grasp_idx, grasp_data in enumerate(grasps_list):
        if total_success_count + success_this_config >= args.num_to_generate:
            break
        
        attempts_made += 1
        current_global_attempt = current_attempt_count + attempts_made
        
        attempt_info = {
            'attempt': current_global_attempt,
            'grasp_idx': grasp_idx,
            'total_grasps': len(grasps_list),
            'z_offset': current_z_offset,
            'config_attempt': config_attempt
        }
        
        success = _execute_grasp_attempt(args, variant_path, grasp_data, experiment_path, attempt_info)
        
        if success:
            success_this_config += 1
            current_total = total_success_count + success_this_config
            cprint(f"    ✓ Success! Total demos: {current_total}/{args.num_to_generate}", "green")
    
    cprint(f"[MULTI-GRASP] Config {config_attempt+1} completed: {success_this_config} demos from {len(grasps_list)} grasps", "magenta")
    return success_this_config, attempts_made
    




def main() -> None:
    """Main entry point for demo generation."""
    args = parse_args()
    
    # Validate and prepare arguments
    validate_and_prepare_args(args)
    
    # Register handle name
    handle_name_dict[args._object_name] = args._handle_name
    
    # Ensure support files exist
    ensure_support_files(args.asset_dir, args, args._handle_name)
    
    # Load predicted grasps
    predicted_grasps = load_grasps(args)
    
    # Setup experiment directory
    experiment_path = setup_experiment_directory(args)
    
    # Select and run sampling method
    if args.sampling_method == "heuristic":
        run_heuristic_generation(args, experiment_path, predicted_grasps)
    elif args.sampling_method == "integrated_inverse":
        run_integrated_inverse_generation(args, experiment_path, predicted_grasps)
    else:
        raise ValueError(f"Unknown sampling method: {args.sampling_method}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
