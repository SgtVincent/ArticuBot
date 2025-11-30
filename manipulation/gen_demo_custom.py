#!/usr/bin/env python3
"""Generate demonstrations for custom URDF objects with affordance annotations."""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import multiprocessing as mp
import numpy as np
from termcolor import cprint

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demos for custom objects")
    parser.add_argument("--asset-dir", default="data/custom_objects/color_0025_0_v2", help="Processed asset folder")
    parser.add_argument("--config-path", default=None, help="Base config path (defaults to <asset-dir>/base_config.yaml)")
    parser.add_argument("--exp-name", default="debug_custom", help="Experiment name under <solution_path>/experiment/")
    parser.add_argument("--num-to-generate", type=int, default=10)
    parser.add_argument("--max-try-times", type=int, default=200)
    parser.add_argument("--env-name", default="articulated_custom_object")
    parser.add_argument("--near-distance", type=float, default=0.15)
    parser.add_argument("--far-distance", type=float, default=0.5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--annotation-path", type=pathlib.Path, default=None)
    parser.add_argument("--handle-part-id", type=int, default=1)
    parser.add_argument("--handle-padding", type=float, default=0.0)
    parser.add_argument("--handle-dilate", type=float, default=0.01, help="Dilation (meters) to expand annotation bbox for handle extraction")
    parser.add_argument("--center-jitter", type=float, default=0.05)
    parser.add_argument("--size-scale", type=float, nargs=2, default=(0.8, 1.1))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout-init", type=float, default=60.0)
    parser.add_argument("--timeout-exec", type=float, default=300.0)
    parser.add_argument("--urdf-path", type=pathlib.Path, default=None)
    parser.add_argument("--joint-name", default=None)
    parser.add_argument("--prepare-mobility", action="store_true", help="Regenerate mobility_v2.json if missing")
    parser.add_argument("--overwrite-support", action="store_true")
    parser.add_argument("--target-position", type=float, nargs=3, default=(0.4, 0.0, 0.0),
                        help="Target world position (x, y, z) for object placement. Default: (0.4, 0.0, 0.0)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    asset_dir = pathlib.Path(args.asset_dir).expanduser().resolve()
    if not asset_dir.exists():
        raise FileNotFoundError(f"Asset dir does not exist: {asset_dir}")
    config_path = pathlib.Path(args.config_path or asset_dir / "base_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    object_name, handle_name, annotation_hint, solution_path = parse_config_metadata(str(config_path))
    handle_name_dict[object_name] = handle_name

    config_parent = config_path.parent
    if args.annotation_path is not None:
        args.annotation_path = pathlib.Path(args.annotation_path).expanduser().resolve(strict=False)
    elif annotation_hint:
        # Try resolving relative to CWD first (common for paths in config generated from root)
        p_cwd = pathlib.Path(annotation_hint).resolve()
        # Try resolving relative to config file
        p_rel = resolve_relative_path(annotation_hint, config_parent)
        
        if p_cwd.exists():
            args.annotation_path = p_cwd
        elif p_rel.exists():
            args.annotation_path = p_rel
        else:
            # Default to p_rel for error reporting
            args.annotation_path = p_rel
    else:
        args.annotation_path = (asset_dir / "annotation.json").resolve(strict=False)

    if args.urdf_path is not None:
        args.urdf_path = pathlib.Path(args.urdf_path).expanduser().resolve(strict=False)
    else:
        args.urdf_path = find_first_urdf(asset_dir)

    if not args.annotation_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found at {args.annotation_path}")
    if not args.urdf_path.exists():
        raise FileNotFoundError(f"URDF path not found at {args.urdf_path}")

    ensure_support_files(asset_dir, args, handle_name)
    
    # Load predicted grasps if available
    predicted_grasps_path = asset_dir / "predicted_grasps.yml"
    predicted_grasps = None
    
    if predicted_grasps_path.exists():
        print(f"Loading predicted grasps from {predicted_grasps_path}")
        predicted_grasps = load_predicted_grasps(predicted_grasps_path)
        if len(predicted_grasps[0]) == 0:
            print("No grasps found in predicted_grasps.yml, will use on-demand GraspGen")
            predicted_grasps = None
        else:
            print(f"Loaded {len(predicted_grasps[0])} grasps")
            
            # ENFORCEMENT: Pre-generated grasps require fixed scale (no randomization)
            # Grasps are predicted for a specific object state; scale/joint changes invalidate them
            scale_min, scale_max = args.size_scale
            if abs(scale_min - scale_max) > 1e-6 or abs(scale_min - 1.0) > 1e-6:
                cprint("=" * 70, "yellow")
                cprint("WARNING: Pre-generated grasps detected (predicted_grasps.yml exists).", "yellow")
                cprint("Grasps are computed for a FIXED object state (scale=1.0, default joints).", "yellow")
                cprint(f"Your --size-scale ({scale_min}, {scale_max}) enables scale randomization.", "yellow")
                cprint("ENFORCING: Setting --size-scale to (1.0, 1.0) to prevent grasp mismatches.", "yellow")
                cprint("To use scale/joint randomization, delete predicted_grasps.yml for on-demand GraspGen.", "yellow")
                cprint("=" * 70, "yellow")
                args.size_scale = (1.0, 1.0)
    else:
        cprint("No predicted_grasps.yml found. Will use on-demand GraspGen for each randomized state.", "cyan")

    solution_root = pathlib.Path(solution_path).expanduser().resolve(strict=False)
    experiment_path = solution_root / "experiment" / args.exp_name
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    # Convert Path objects to strings for JSON serialization
    args_dict = vars(args).copy()
    for k, v in args_dict.items():
        if isinstance(v, pathlib.Path):
            args_dict[k] = str(v)
            
    with (experiment_path / "meta_info.json").open("w", encoding="utf-8") as fp:
        json.dump(args_dict, fp, indent=4)

    num_existing = get_current_num_demos(str(experiment_path))
    cprint(f"Existing successful demos: {num_existing}", "yellow")

    attempt = 0
    total_success = num_existing
    while total_success < args.num_to_generate and attempt < args.max_try_times:
        variant_path = create_variant_config(
            str(config_path),
            asset_dir,
            attempt,
            handle_name,
            args.annotation_path,
            args.center_jitter,
            args.size_scale,
            urdf_path=args.urdf_path,
            predicted_grasps=predicted_grasps,
            target_position=tuple(args.target_position),
        )
        
        cprint(f"Attempt {attempt+1}/{args.max_try_times}: {variant_path.name}", "cyan")
        
        # Generate initial state
        success_init = custom_gen_init_state(
            str(variant_path),
            args.env_name,
            args.render,
            near_distance=args.near_distance,
            far_distance=args.far_distance,
            timeout=args.timeout_init
        )
        
        if not success_init:
            cprint("Failed to generate initial state", "red")
            attempt += 1
            continue
            
        success_exec = custom_execute(
            str(variant_path),
            args.env_name,
            str(asset_dir),
            str(experiment_path),
            timeout=args.timeout_exec
        )
        
        if success_exec:
            total_success += 1
            cprint(f"Success! Total demos: {total_success}", "green")
        else:
            cprint("Execution failed", "red")
            
        attempt += 1

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
