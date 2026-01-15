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
import pathlib
import random
import multiprocessing as mp
import numpy as np
from termcolor import cprint
from enum import Enum
from typing import Optional, Tuple

from manipulation.envs.articulated import handle_name_dict
from manipulation.gen_demo import get_current_num_demos

from manipulation.custom_object_utils.object_utils import (
    find_first_urdf,
    ensure_support_files,
    load_predicted_grasps,
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
    parser.add_argument("--annotation-path", type=pathlib.Path, default=None,
                        help="Path to annotation.json file")
    parser.add_argument("--handle-part-id", type=int, default=1,
                        help="Part ID for handle mesh")
    parser.add_argument("--handle-padding", type=float, default=0.0,
                        help="Padding for handle bounding box")
    parser.add_argument("--handle-dilate", type=float, default=0.01,
                        help="Dilation (meters) to expand annotation bbox for handle extraction")
    
    # Randomization parameters
    parser.add_argument("--center-jitter", type=float, default=0.05,
                        help="Position randomization (meters)")
    parser.add_argument("--size-scale", type=float, nargs=2, default=(0.8, 1.1),
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
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    # Timeout parameters
    parser.add_argument("--timeout-init", type=float, default=60.0,
                        help="Timeout for init state generation")
    parser.add_argument("--timeout-exec", type=float, default=300.0,
                        help="Timeout for demo execution")
    
    # URDF and joint configuration
    parser.add_argument("--urdf-path", type=pathlib.Path, default=None,
                        help="Path to URDF file")
    parser.add_argument("--joint-name", default=None,
                        help="Specific joint name to use")
    parser.add_argument("--prepare-mobility", action="store_true",
                        help="Regenerate mobility_v2.json if missing")
    parser.add_argument("--overwrite-support", action="store_true",
                        help="Overwrite existing support files")
    
    # Object placement
    parser.add_argument("--target-position", type=float, nargs=3, default=(0.4, 0.0, 0.0),
                        help="Target world position (x, y, z) for object placement")
    parser.add_argument("--object-z-offset", type=float, default=0.0,
                        help="Z-offset to elevate object above ground (e.g., 0.7 for table height)")
    
    # RM4D parameters (used by integrated inverse sampling)
    parser.add_argument("--rm4d-map", type=pathlib.Path, default=None,
                        help="Path to RM4D reachability map (.npy)")
    
    # Integrated inverse map parameters
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
    
    # Trajectory computation parameters (used by integrated_inverse and general methods)
    parser.add_argument("--approach-distance", type=float, default=0.15,
                        help="Approach distance for gripper (meters)")
    parser.add_argument("--target-ratio", type=float, default=0.8,
                        help="Target opening ratio for door/drawer manipulation")
    
    # Debugging and visualization
    parser.add_argument("--debug-vis", action="store_true",
                        help="Save debug visualization as GIF showing trajectory poses and inverse map values")
    parser.add_argument("--debug-vis-output", type=pathlib.Path, default=None,
                        help="Output path for debug visualization GIF (defaults to <exp-dir>/debug_vis.gif)")
    parser.add_argument("--init-only", action="store_true",
                        help="Only run initial-state sampling (and write config/init-state); skip full demo execution")
    parser.add_argument("--use-viser", action="store_true",
                        help="Enable interactive 3D visualization with viser (waits for keyboard input between attempts)")
    parser.add_argument("--viser-port", type=int, default=8080,
                        help="Port number for viser server (default: 8080)")
    
    args = parser.parse_args()

    if args.use_viser:
        # Avoid timing out while stepping through interactive visualization.
        args.timeout_init = 1e6
        args.timeout_exec = 1e6
    
    return args


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

    args.asset_dir = pathlib.Path(args.asset_dir).expanduser().resolve()
    if not args.asset_dir.exists():
        raise FileNotFoundError(f"Asset dir does not exist: {args.asset_dir}")
    
    args.config_path = pathlib.Path(args.config_path or args.asset_dir / "base_config.yaml")
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    # Resolve annotation path
    config_parent = args.config_path.parent
    object_name, handle_name, annotation_hint, solution_path = parse_config_metadata(str(args.config_path))
    
    # Store these for later use
    args._object_name = object_name
    args._handle_name = handle_name
    args._solution_path = solution_path
    
    if args.annotation_path is not None:
        args.annotation_path = pathlib.Path(args.annotation_path).expanduser().resolve(strict=False)
    elif annotation_hint:
        p_cwd = pathlib.Path(annotation_hint).resolve()
        p_rel = resolve_relative_path(annotation_hint, config_parent)
        
        if p_cwd.exists():
            args.annotation_path = p_cwd
        elif p_rel.exists():
            args.annotation_path = p_rel
        else:
            args.annotation_path = p_rel
    else:
        args.annotation_path = (args.asset_dir / "annotation.json").resolve(strict=False)

    if args.urdf_path is not None:
        args.urdf_path = pathlib.Path(args.urdf_path).expanduser().resolve(strict=False)
    else:
        args.urdf_path = find_first_urdf(args.asset_dir)

    if args.rm4d_map is not None:
        args.rm4d_map = pathlib.Path(args.rm4d_map).expanduser().resolve(strict=False)

    if not args.annotation_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found at {args.annotation_path}")
    if not args.urdf_path.exists():
        raise FileNotFoundError(f"URDF path not found at {args.urdf_path}")

    # Grasp-source policy:
    # For randomized scale/joint configs, pre-generated grasps generally mismatch.
    # Treat 'auto' as 'ondemand' to avoid accidentally using predicted_grasps.yml.
    if args.grasp_source == "auto":
        cprint(
            "WARNING: --grasp-source auto is treated as ondemand to ensure grasps match randomized object states.",
            "yellow",
        )
        args.grasp_source = "ondemand"
    
    # Validate integrated_inverse specific requirements
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
        print(f"Loading predicted grasps from {predicted_grasps_path}")
        predicted_grasps = load_predicted_grasps(predicted_grasps_path)
        if len(predicted_grasps[0]) == 0:
            print("No grasps found in predicted_grasps.yml, will use on-demand GraspGen")
            predicted_grasps = None
        else:
            print(f"Loaded {len(predicted_grasps[0])} grasps")
            
            # Enforce fixed scale for pre-generated grasps
            scale_min, scale_max = args.size_scale
            if abs(scale_min - scale_max) > 1e-6 or abs(scale_min - 1.0) > 1e-6:
                cprint("=" * 70, "yellow")
                cprint("WARNING: Pre-generated grasps detected (predicted_grasps.yml exists).", "yellow")
                cprint("Grasps are computed for a FIXED object state (scale=1.0, default joints).", "yellow")
                cprint(f"Your --size-scale ({scale_min}, {scale_max}) enables scale randomization.", "yellow")
                cprint("ENFORCING: Setting --size-scale to (1.0, 1.0) to prevent grasp mismatches.", "yellow")
                cprint("=" * 70, "yellow")
                args.size_scale = (1.0, 1.0)

            # Enforce fixed joint angle for pre-generated grasps
            jmin, jmax = tuple(args.joint_angle_range)
            if abs(jmin) > 1e-9 or abs(jmax) > 1e-9:
                cprint("=" * 70, "yellow")
                cprint("WARNING: Pre-generated grasps detected (predicted_grasps.yml exists).", "yellow")
                cprint("Grasps are computed for a FIXED object state (scale=1.0, default joints).", "yellow")
                cprint(f"Your --joint-angle-range ({jmin}, {jmax}) enables joint randomization.", "yellow")
                cprint(
                    "ENFORCING: Setting --joint-angle-range to (0.0, 0.0) to prevent grasp mismatches.",
                    "yellow",
                )
                cprint("=" * 70, "yellow")
                args.joint_angle_range = (0.0, 0.0)
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
        
        cprint(f"[HEURISTIC] Attempt {attempt+1}/{args.max_try_times}: {variant_path.name}", "cyan")
        
        # Generate initial state using heuristic method
        # Pass rm4d_map and object_traj as None to use heuristic sampling
        success_init = custom_gen_init_state(
            str(variant_path),
            args.env_name,
            args.render,
            near_distance=args.near_distance,
            far_distance=args.far_distance,
            timeout=args.timeout_init,
            rm4d_map_path=None,  # Disable RM4D-based filtering for heuristic
            object_traj_path=None,
        )
        
        if not success_init:
            cprint("Failed to generate initial state", "red")
            attempt += 1
            continue

        handled, total_success = _handle_init_only(args, total_success)
        if handled:
            attempt += 1
            continue
            
        success_exec = custom_execute(
            str(variant_path),
            args.env_name,
            str(args.asset_dir),
            str(experiment_path),
            timeout=args.timeout_exec
        )
        
        if success_exec:
            total_success += 1
            cprint(f"Success! Total demos: {total_success}/{args.num_to_generate}", "green")
        else:
            cprint("Execution failed", "red")
            
        attempt += 1
    
    cprint(f"\nHEURISTIC method completed: {total_success} demos from {attempt} attempts", "green")


def run_integrated_inverse_generation(args: argparse.Namespace,
                                       experiment_path: pathlib.Path,
                                       predicted_grasps: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Run demo generation using integrated inverse map sampling.
    
    This method implements the most efficient sampling approach:
    1. Compute the distribution over object poses by integrating reachability
       along the full manipulation trajectory
    2. Sample directly from this distribution (not rejection sampling)
    3. Check robot configuration feasibility for sampled poses
    
    Key insight: Instead of sampling random poses and rejecting infeasible ones,
    we pre-compute which poses are feasible and sample from them directly.
    
    Args:
        args: Parsed command line arguments.
        experiment_path: Path to experiment directory.
        predicted_grasps: Pre-loaded grasps or None for on-demand generation.
    """
    cprint("Using INTEGRATED_INVERSE sampling method (direct sampling)", "magenta")

    from manipulation.custom_object_utils.demo_utils_integrated import (
        custom_gen_init_state_integrated,
    )
    
    num_existing = get_current_num_demos(str(experiment_path))
    cprint(f"Existing successful demos: {num_existing}", "yellow")

    attempt = 0
    total_success = num_existing
    
    while total_success < args.num_to_generate and attempt < args.max_try_times:
        # Step 1: Sample randomized object config
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
        
        cprint(f"[INTEGRATED_INVERSE] Attempt {attempt+1}/{args.max_try_times}: {variant_path.name}", "magenta")
        
        # Determine debug visualization path
        debug_vis_path = None
        if args.debug_vis:
            if args.debug_vis_output:
                debug_vis_path = str(args.debug_vis_output)
            else:
                debug_vis_path = str(experiment_path / f"debug_vis_attempt_{attempt:04d}.gif")
        
        # Generate initial state using integrated inverse map method
        success_init = custom_gen_init_state_integrated(
            str(variant_path),
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
            object_z_offset=args.object_z_offset,
            use_viser=args.use_viser,
            viser_port=args.viser_port,
            attempt_number=attempt,
        )
        
        if not success_init:
            cprint("Failed to generate initial state (integrated inverse)", "red")
            attempt += 1
            continue

        handled, total_success = _handle_init_only(args, total_success)
        if handled:
            attempt += 1
            continue
            
        success_exec = custom_execute(
            str(variant_path),
            args.env_name,
            str(args.asset_dir),
            str(experiment_path),
            timeout=args.timeout_exec
        )
        
        if success_exec:
            total_success += 1
            cprint(f"Success! Total demos: {total_success}/{args.num_to_generate}", "green")
        else:
            cprint("Execution failed", "red")
            
        attempt += 1
    
    cprint(f"\nINTEGRATED_INVERSE method completed: {total_success} demos from {attempt} attempts", "magenta")


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
