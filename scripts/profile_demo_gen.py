#!/usr/bin/env python3
"""Profile the demo generation pipeline to identify performance bottlenecks.

This script instruments a small number of demo generation attempts and
reports timing breakdowns for each stage:
  1. create_variant_config (config YAML generation)
  2. _generate_grasps (GraspGen via Docker)
  3. custom_gen_init_state_integrated (subprocess: env setup + integrated sampling)
  4. custom_execute (subprocess: env setup + simulation execution)

Usage:
    source prepare.sh
    python scripts/profile_demo_gen.py \
        --asset-dir data/custom_objects_v2/sim/microwave_7167 \
        --num-attempts 3
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pathlib
import pstats
import random
import sys
import time

import numpy as np

# Ensure project root is on PYTHONPATH
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from termcolor import cprint


def parse_args():
    parser = argparse.ArgumentParser(description="Profile demo generation pipeline")
    parser.add_argument("--asset-dir", default="data/custom_objects_v2/sim/microwave_7167")
    parser.add_argument("--num-attempts", type=int, default=3,
                        help="Number of config iterations to profile")
    parser.add_argument("--sampling-method", default="integrated_inverse",
                        choices=["heuristic", "integrated_inverse"])
    parser.add_argument("--rm4d-map", default="data/rm4d_franka_1M.npy")
    parser.add_argument("--top-k-grasps", type=int, default=3)
    parser.add_argument("--timeout-init", type=float, default=120.0)
    parser.add_argument("--timeout-exec", type=float, default=180.0)
    parser.add_argument("--profile-cprofile", action="store_true",
                        help="Also run cProfile on init_state subprocess internals")
    parser.add_argument("--output", default="profile_results.json",
                        help="Output JSON file for timing results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup (mirrors gen_demo_custom.py)
    from manipulation.envs.articulated import handle_name_dict
    from manipulation.gen_demo import get_current_num_demos
    from manipulation.custom_object_utils.object_utils import (
        find_first_urdf, ensure_support_files, find_annotation_path,
    )
    from manipulation.custom_object_utils.demo_utils import (
        parse_config_metadata, create_variant_config, custom_gen_init_state,
        custom_execute,
    )

    asset_dir = pathlib.Path(args.asset_dir).expanduser().resolve()
    config_path = asset_dir / "base_config.yaml"
    urdf_path = find_first_urdf(asset_dir)

    object_name, handle_name, annotation_hint, solution_path = parse_config_metadata(str(config_path))
    handle_name_dict[object_name] = handle_name

    # Resolve annotation
    annotation_path = find_annotation_path(asset_dir, annotation_hint)

    # Ensure support files
    fake_args = argparse.Namespace(
        handle_part_id=1,
        handle_padding=0.0,
        handle_dilate=0.0,
        joint_name=None,
        prepare_mobility=False,
        overwrite_support=False,
        annotation_path=annotation_path,
        urdf_path=urdf_path,
    )

    ensure_support_files(asset_dir, fake_args, handle_name)

    # Setup experiment dir
    experiment_path = pathlib.Path(solution_path).resolve() / "experiment" / "profile_test"
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Timing results
    results = {
        "attempts": [],
        "summary": {},
    }

    cprint("=" * 70, "cyan")
    cprint("PROFILING DEMO GENERATION PIPELINE", "cyan")
    cprint(f"Asset: {asset_dir}", "cyan")
    cprint(f"Method: {args.sampling_method}", "cyan")
    cprint(f"Attempts: {args.num_attempts}", "cyan")
    cprint("=" * 70, "cyan")

    for attempt_idx in range(args.num_attempts):
        attempt_result = {"attempt": attempt_idx, "stages": {}}
        cprint(f"\n--- Attempt {attempt_idx + 1}/{args.num_attempts} ---", "yellow")

        # Stage 1: create_variant_config
        t0 = time.time()
        variant_path = create_variant_config(
            str(config_path),
            asset_dir,
            attempt_idx + 1000,  # offset to not collide with running experiment
            handle_name,
            annotation_path,
            0.05,  # center_jitter
            (0.6, 0.8),  # size_scale
            urdf_path=urdf_path,
            predicted_grasps=None,
            target_position=(0.4, 0.0, 0.0),
            joint_angle_range=(0.0, 0.2),
        )
        t1 = time.time()
        attempt_result["stages"]["create_variant_config"] = t1 - t0
        cprint(f"  create_variant_config: {t1-t0:.3f}s", "blue")

        # Stage 2: _generate_grasps (GraspGen Docker call)
        if args.sampling_method == "integrated_inverse":
            t0 = time.time()
            # Import and call grasp generation
            from manipulation.gen_demo_custom import _generate_grasps

            class GraspArgs:
                top_k_grasps = args.top_k_grasps

            grasps_list = _generate_grasps(GraspArgs(), variant_path)
            t1 = time.time()
            attempt_result["stages"]["generate_grasps"] = t1 - t0
            attempt_result["num_grasps"] = len(grasps_list)
            cprint(f"  generate_grasps (Docker): {t1-t0:.3f}s ({len(grasps_list)} grasps)", "blue")

            if not grasps_list:
                cprint("  No grasps generated, skipping remaining stages", "red")
                results["attempts"].append(attempt_result)
                continue

            # Stage 3: custom_gen_init_state_integrated (for first grasp)
            grasp_pos, grasp_quat, grasp_conf = grasps_list[0]

            # Update variant config with grasp
            from manipulation.gen_demo_custom import _update_and_save_variant_config
            grasp_variant_path = _update_and_save_variant_config(
                variant_path, grasp_pos, grasp_quat, attempt_idx + 1000, 0
            )

            z_offset = random.uniform(0.3, 0.5)

            t0 = time.time()
            from manipulation.custom_object_utils.demo_utils_integrated import (
                custom_gen_init_state_integrated,
            )
            success_init = custom_gen_init_state_integrated(
                str(grasp_variant_path),
                "articulated_custom_object",
                False,  # render
                near_distance=0.15,
                far_distance=0.5,
                timeout=args.timeout_init,
                rm4d_map_path=args.rm4d_map,
                xy_resolution=0.02,
                theta_bins=24,
                aggregation_method="mean",
                coverage_threshold=0.5,
                temperature=1.0,
                min_score_threshold=0.1,
                approach_distance=0.15,
                target_ratio=0.8,
                asset_dir=str(asset_dir),
                object_z_offset=z_offset,
                trajectory_occlusion_rate=0.5,
            )
            t1 = time.time()
            attempt_result["stages"]["init_state_integrated"] = t1 - t0
            attempt_result["init_success"] = bool(success_init)
            cprint(f"  init_state_integrated: {t1-t0:.3f}s (success={success_init})", "blue")

            # Stage 4: custom_execute (only if init succeeded)
            if success_init:
                t0 = time.time()
                success_exec = custom_execute(
                    str(grasp_variant_path),
                    "articulated_custom_object",
                    str(asset_dir),
                    str(experiment_path),
                    timeout=args.timeout_exec,
                )
                t1 = time.time()
                attempt_result["stages"]["custom_execute"] = t1 - t0
                attempt_result["exec_success"] = bool(success_exec)
                cprint(f"  custom_execute: {t1-t0:.3f}s (success={success_exec})", "blue")
            else:
                attempt_result["stages"]["custom_execute"] = 0.0
                attempt_result["exec_success"] = False

        results["attempts"].append(attempt_result)

    # Compute summary
    all_stages = {}
    for attempt in results["attempts"]:
        for stage, duration in attempt["stages"].items():
            if stage not in all_stages:
                all_stages[stage] = []
            all_stages[stage].append(duration)

    cprint("\n" + "=" * 70, "green")
    cprint("PROFILING SUMMARY", "green")
    cprint("=" * 70, "green")

    total_times = []
    for stage, durations in all_stages.items():
        avg = sum(durations) / len(durations)
        results["summary"][stage] = {
            "mean": avg,
            "min": min(durations),
            "max": max(durations),
            "count": len(durations),
            "total": sum(durations),
        }
        cprint(f"  {stage:30s}: avg={avg:.2f}s  min={min(durations):.2f}s  max={max(durations):.2f}s  (n={len(durations)})", "green")

    # Per-attempt total
    for attempt in results["attempts"]:
        total = sum(attempt["stages"].values())
        total_times.append(total)

    if total_times:
        avg_total = sum(total_times) / len(total_times)
        cprint(f"\n  {'TOTAL per attempt':30s}: avg={avg_total:.2f}s  min={min(total_times):.2f}s  max={max(total_times):.2f}s", "green")

        # Percentage breakdown
        cprint("\n  Stage breakdown (% of total):", "green")
        for stage, info in results["summary"].items():
            pct = (info["mean"] / avg_total * 100) if avg_total > 0 else 0
            cprint(f"    {stage:30s}: {pct:5.1f}%", "green")

    # Bottleneck analysis
    cprint("\n" + "=" * 70, "yellow")
    cprint("BOTTLENECK ANALYSIS", "yellow")
    cprint("=" * 70, "yellow")

    if results["summary"]:
        sorted_stages = sorted(results["summary"].items(), key=lambda x: x[1]["mean"], reverse=True)
        cprint(f"  Slowest stage: {sorted_stages[0][0]} ({sorted_stages[0][1]['mean']:.2f}s avg)", "yellow")

        # Key observations
        cprint("\n  Key observations:", "yellow")

        if "generate_grasps" in results["summary"]:
            gg_time = results["summary"]["generate_grasps"]["mean"]
            cprint(f"  - GraspGen Docker call: {gg_time:.1f}s avg (GPU-bound, runs in Docker)", "yellow")
            cprint(f"    → This blocks the main process; could be overlapped with other work", "yellow")

        if "init_state_integrated" in results["summary"]:
            init_time = results["summary"]["init_state_integrated"]["mean"]
            cprint(f"  - Init state (subprocess): {init_time:.1f}s avg", "yellow")
            cprint(f"    → Spawns mp.Process each time: env setup + PyBullet load overhead repeated", "yellow")
            cprint(f"    → RM4D map loaded fresh in each subprocess", "yellow")

        if "custom_execute" in results["summary"]:
            exec_time = results["summary"]["custom_execute"]["mean"]
            cprint(f"  - Execution (subprocess): {exec_time:.1f}s avg", "yellow")
            cprint(f"    → Also spawns mp.Process: another env setup + PyBullet load", "yellow")

        cprint("\n  Parallelization opportunities:", "yellow")
        cprint("  1. Run multiple demo attempts in parallel (each is independent)", "yellow")
        cprint("  2. Each attempt's init_state + execute is sequential but independent across attempts", "yellow")
        cprint("  3. GraspGen Docker calls could be batched or pipelined", "yellow")
        cprint("  4. RM4D map loading could be shared (load once, pass via shared memory)", "yellow")

    # Save results
    output_path = pathlib.Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    cprint(f"\nResults saved to {output_path}", "cyan")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
