#!/usr/bin/env python3
"""Parallel demo generation for a single custom object.

This script spawns multiple independent worker processes to generate demos
in parallel, dramatically improving throughput on multi-core machines.

Each worker runs the full demo generation pipeline independently:
1. Create variant config (randomized scale, joint angle)
2. Call GraspGen for grasps
3. Run integrated inverse sampling for init state
4. Execute simulation

Workers share:
- A counter for total successful demos (via multiprocessing.Value)
- The experiment directory (each demo gets unique timestamp folder)

Performance improvement:
- On 128-core machine with 1TB RAM: from ~2.5 min/demo (sequential) to ~20 sec/demo (parallel)
- Linear scaling up to ~16-32 workers (limited by GraspGen Docker bottleneck)

Usage:
    source prepare.sh
    python manipulation/gen_demo_custom_parallel.py \
        --asset-dir data/custom_objects_v2/sim/microwave_7167 \
        --exp-name microwave_7167_parallel_test \
        --num-workers 16 \
        --num-to-generate 200 \
        --sampling-method integrated_inverse \
        --rm4d-map data/rm4d_franka_1M.npy \
        --top-k-grasps 5 \
        --object-z-offset 0.3 0.5
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import random
import subprocess
import signal
import sys
import time
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
from termcolor import cprint

# Ensure project root is on PYTHONPATH
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Parallel demo generation for custom objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Parallel execution parameters
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel worker processes")
    parser.add_argument("--worker-id", type=int, default=None,
                        help="(Internal) Worker ID when running as subprocess")

    # Basic configuration (same as gen_demo_custom.py)
    parser.add_argument("--asset-dir", required=True,
                        help="Processed asset folder")
    parser.add_argument("--config-path", default=None,
                        help="Base config path (defaults to <asset-dir>/base_config.yaml)")
    parser.add_argument("--exp-name", default="parallel_demo",
                        help="Experiment name under <solution_path>/experiment/")
    parser.add_argument("--num-to-generate", type=int, default=100,
                        help="Total number of successful demos to collect")
    parser.add_argument("--max-try-times", type=int, default=10000,
                        help="Maximum attempts across all workers")

    # Sampling method
    parser.add_argument("--sampling-method", type=str, default="integrated_inverse",
                        choices=["heuristic", "integrated_inverse"])

    # RM4D and integrated inverse parameters
    parser.add_argument("--rm4d-map", type=pathlib.Path, default=None)
    parser.add_argument("--xy-resolution", type=float, default=0.02)
    parser.add_argument("--theta-bins", type=int, default=24)
    parser.add_argument("--aggregation-method", default="mean")
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--min-score-threshold", type=float, default=0.1)

    # Grasp parameters
    parser.add_argument("--grasp-source", default="ondemand",
                        choices=["ondemand", "precomputed", "auto"])
    parser.add_argument("--top-k-grasps", type=int, default=5)

    # Object configuration
    parser.add_argument("--center-jitter", type=float, default=0.05)
    parser.add_argument("--size-scale", type=float, nargs=2, default=(0.6, 0.8))
    parser.add_argument("--joint-angle-range", type=float, nargs=2, default=(0.0, 0.2))
    parser.add_argument("--target-position", type=float, nargs=3, default=(0.4, 0.0, 0.0))
    parser.add_argument("--object-z-offset", type=float, nargs='+', default=[0.2, 0.4])

    # Distance parameters
    parser.add_argument("--near-distance", type=float, default=0.15)
    parser.add_argument("--far-distance", type=float, default=0.5)
    parser.add_argument("--approach-distance", type=float, default=0.15)
    parser.add_argument("--target-ratio", type=float, default=0.8)
    parser.add_argument("--trajectory-occlusion-rate", type=float, default=0.5)

    # Timeout parameters
    parser.add_argument("--timeout-init", type=float, default=120.0)
    parser.add_argument("--timeout-exec", type=float, default=300.0)

    # Environment
    parser.add_argument("--env-name", default="articulated_custom_object")
    parser.add_argument("--render", action="store_true")

    # Support file options
    parser.add_argument("--handle-part-id", type=int, default=1)
    parser.add_argument("--handle-padding", type=float, default=0.0)
    parser.add_argument("--handle-dilate", type=float, default=0.0)
    parser.add_argument("--joint-name", type=str, default=None)
    parser.add_argument("--urdf-path", type=pathlib.Path, default=None)
    parser.add_argument("--annotation-path", type=pathlib.Path, default=None)
    parser.add_argument("--prepare-mobility", action="store_true")
    parser.add_argument("--overwrite-support", action="store_true")

    # Debug
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed for reproducibility (worker seeds are derived from this)")
    parser.add_argument("--debug-vis", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--use-viser", action="store_true")
    parser.add_argument("--viser-port", type=int, default=8888)

    return parser.parse_args()


def get_current_num_demos(experiment_path: str) -> int:
    """Count existing demos in experiment directory."""
    exp_dir = pathlib.Path(experiment_path)
    if not exp_dir.exists():
        return 0
    # Count directories matching timestamp pattern
    count = sum(1 for d in exp_dir.iterdir()
                if d.is_dir() and d.name[:4].isdigit())
    return count


def _compute_worker_seed(base_seed: Optional[int], worker_id: int) -> int:
    if base_seed is None:
        base_seed = int(time.time_ns() % (2**31 - 1))
    return int(base_seed + worker_id * 10007)


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


def _configure_graspgen_gpu_for_worker(worker_id: int) -> Optional[int]:
    num_gpus = _get_gpu_count()
    if num_gpus <= 0:
        return None

    gpu_id = int(worker_id) % num_gpus
    base_name = os.environ.get("GRASPGEN_CONTAINER_BASE_NAME", "friendly_galileo")
    container_name = f"{base_name}_gpu{gpu_id}"

    os.environ["GRASPGEN_CONTAINER_NAME"] = container_name
    os.environ["GRASPGEN_GPU_ID"] = str(gpu_id)
    os.environ.setdefault("GRASPGEN_AUTOSTART", "1")
    return gpu_id


def worker_main(
    worker_id: int,
    args: argparse.Namespace,
    success_counter: mp.Value,
    attempt_counter: mp.Value,
    stop_event: mp.Event,
) -> None:
    """Main function for a single worker process.

    Each worker runs independently and checks shared counters to determine
    when to stop.
    """
    # Set unique random seed per worker
    worker_seed = _compute_worker_seed(args.seed, worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(worker_seed)

    gpu_id = _configure_graspgen_gpu_for_worker(worker_id)

    # Import here to avoid issues with multiprocessing
    from manipulation.envs.articulated import handle_name_dict
    from manipulation.custom_object_utils.object_utils import (
        find_first_urdf, ensure_support_files, find_annotation_path,
    )
    from manipulation.custom_object_utils.demo_utils import (
        parse_config_metadata, create_variant_config, custom_execute,
    )
    from manipulation.custom_object_utils.demo_utils_integrated import (
        custom_gen_init_state_integrated,
    )
    from manipulation.gen_demo_custom import (
        _generate_grasps, _update_and_save_variant_config, _get_z_offset,
    )

    # Setup paths
    asset_dir = pathlib.Path(args.asset_dir).expanduser().resolve()
    config_path = pathlib.Path(args.config_path or asset_dir / "base_config.yaml")
    urdf_path = args.urdf_path or find_first_urdf(asset_dir)

    # Parse config metadata
    object_name, handle_name, annotation_hint, solution_path = parse_config_metadata(str(config_path))
    handle_name_dict[object_name] = handle_name

    # Resolve annotation
    annotation_path = find_annotation_path(asset_dir, annotation_hint)

    # Setup experiment directory
    experiment_path = pathlib.Path(solution_path).resolve() / "experiment" / args.exp_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    gpu_info = f", GPU: {gpu_id}" if gpu_id is not None else ""
    cprint(
        f"[Worker {worker_id}] Started. Target: {args.num_to_generate} demos. Seed: {worker_seed}{gpu_info}",
        "cyan",
    )

    local_success = 0
    local_attempts = 0

    while not stop_event.is_set():
        # Check if we've reached the target
        with success_counter.get_lock():
            if success_counter.value >= args.num_to_generate:
                break

        with attempt_counter.get_lock():
            if attempt_counter.value >= args.max_try_times:
                break
            current_attempt = attempt_counter.value
            attempt_counter.value += 1

        local_attempts += 1

        # Create unique config index for this worker/attempt
        config_idx = worker_id * 100000 + current_attempt

        try:
            # Step 1: Create variant config
            variant_path = create_variant_config(
                str(config_path),
                asset_dir,
                config_idx,
                handle_name,
                annotation_path,
                args.center_jitter,
                tuple(args.size_scale),
                urdf_path=urdf_path,
                predicted_grasps=None,
                target_position=tuple(args.target_position),
                joint_angle_range=tuple(args.joint_angle_range),
            )

            # Step 2: Generate grasps
            # Create a namespace-like object for _generate_grasps
            class GraspArgs:
                top_k_grasps = args.top_k_grasps

            grasps_list = _generate_grasps(GraspArgs(), variant_path)

            if not grasps_list:
                continue

            # Get z-offset for this attempt
            if isinstance(args.object_z_offset, list) and len(args.object_z_offset) == 2:
                current_z_offset = random.uniform(args.object_z_offset[0], args.object_z_offset[1])
            elif isinstance(args.object_z_offset, list):
                current_z_offset = args.object_z_offset[0]
            else:
                current_z_offset = args.object_z_offset

            # Step 3: Try each grasp
            for grasp_idx, (grasp_pos, grasp_quat, grasp_conf) in enumerate(grasps_list):
                # Check stop condition again
                with success_counter.get_lock():
                    if success_counter.value >= args.num_to_generate:
                        break

                # Update config with grasp
                grasp_variant_path = _update_and_save_variant_config(
                    variant_path, grasp_pos, grasp_quat, config_idx, grasp_idx
                )

                # Step 4: Generate init state
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
                    asset_dir=str(asset_dir),
                    object_z_offset=current_z_offset,
                    trajectory_occlusion_rate=args.trajectory_occlusion_rate,
                )

                if not success_init:
                    continue

                if args.init_only:
                    with success_counter.get_lock():
                        success_counter.value += 1
                        local_success += 1
                        total = success_counter.value
                    cprint(f"[Worker {worker_id}] Init-only success #{local_success} (total: {total}/{args.num_to_generate})", "green")
                    continue

                # Step 5: Execute
                success_exec = custom_execute(
                    str(grasp_variant_path),
                    args.env_name,
                    str(asset_dir),
                    str(experiment_path),
                    timeout=args.timeout_exec,
                )

                if success_exec:
                    with success_counter.get_lock():
                        success_counter.value += 1
                        local_success += 1
                        total = success_counter.value
                    cprint(f"[Worker {worker_id}] Success #{local_success} (total: {total}/{args.num_to_generate})", "green")

        except Exception as e:
            cprint(f"[Worker {worker_id}] Error: {e}", "red")
            continue

    cprint(f"[Worker {worker_id}] Finished. Local: {local_success} demos from {local_attempts} attempts", "cyan")


def main():
    """Main entry point for parallel demo generation."""
    args = parse_args()

    # Resolve paths
    args.asset_dir = pathlib.Path(args.asset_dir).expanduser().resolve()
    if not args.asset_dir.exists():
        raise FileNotFoundError(f"Asset dir does not exist: {args.asset_dir}")

    if args.rm4d_map is not None:
        args.rm4d_map = pathlib.Path(args.rm4d_map).expanduser().resolve()

    # Setup experiment directory
    solution_path = args.asset_dir  # For custom objects, solution_path is the asset_dir
    experiment_path = solution_path / "experiment" / args.exp_name
    experiment_path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    args_dict = {k: str(v) if isinstance(v, pathlib.Path) else v
                 for k, v in vars(args).items()}
    with (experiment_path / "meta_info_parallel.json").open("w") as f:
        json.dump(args_dict, f, indent=2)

    # Check existing demos
    num_existing = get_current_num_demos(str(experiment_path))

    cprint("=" * 70, "magenta")
    cprint("PARALLEL DEMO GENERATION", "magenta")
    cprint("=" * 70, "magenta")
    cprint(f"Asset:           {args.asset_dir}", "cyan")
    cprint(f"Experiment:      {args.exp_name}", "cyan")
    cprint(f"Workers:         {args.num_workers}", "cyan")
    cprint(f"Target demos:    {args.num_to_generate}", "cyan")
    cprint(f"Existing demos:  {num_existing}", "cyan")
    cprint(f"Method:          {args.sampling_method}", "cyan")
    cprint(f"Base seed:       {args.seed if args.seed is not None else 'auto'}", "cyan")
    cprint("=" * 70, "magenta")

    if num_existing >= args.num_to_generate:
        cprint(f"Already have {num_existing} demos, nothing to do.", "green")
        return

    # Create shared counters
    success_counter = mp.Value('i', num_existing)
    attempt_counter = mp.Value('i', 0)
    stop_event = mp.Event()

    # Signal handler for graceful shutdown
    def signal_handler(signum, frame):
        cprint("\nReceived interrupt signal, stopping workers...", "yellow")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start workers
    workers = []
    start_time = time.time()

    for i in range(args.num_workers):
        p = mp.Process(
            target=worker_main,
            args=(i, args, success_counter, attempt_counter, stop_event),
        )
        p.start()
        workers.append(p)
        time.sleep(0.5)  # Stagger starts slightly

    cprint(f"Started {args.num_workers} workers", "green")

    # Monitor progress
    try:
        while any(p.is_alive() for p in workers):
            time.sleep(10)
            with success_counter.get_lock():
                current = success_counter.value
            with attempt_counter.get_lock():
                attempts = attempt_counter.value

            elapsed = time.time() - start_time
            rate = (current - num_existing) / elapsed * 60 if elapsed > 0 else 0

            running = sum(1 for p in workers if p.is_alive())
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] "
                   f"Demos: {current}/{args.num_to_generate} | "
                   f"Attempts: {attempts} | "
                   f"Rate: {rate:.1f}/min | "
                   f"Workers: {running}/{args.num_workers}", "yellow")

            if current >= args.num_to_generate:
                stop_event.set()
                break

    except KeyboardInterrupt:
        stop_event.set()

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    # Final summary
    elapsed = time.time() - start_time
    final_count = get_current_num_demos(str(experiment_path))
    new_demos = final_count - num_existing

    cprint("=" * 70, "green")
    cprint("PARALLEL GENERATION COMPLETE", "green")
    cprint("=" * 70, "green")
    cprint(f"Total demos:     {final_count}", "green")
    cprint(f"New demos:       {new_demos}", "green")
    cprint(f"Total time:      {elapsed/60:.1f} min", "green")
    cprint(f"Rate:            {new_demos/elapsed*60:.1f} demos/min", "green")
    cprint(f"Time per demo:   {elapsed/max(new_demos,1):.1f} sec", "green")
    cprint("=" * 70, "green")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
