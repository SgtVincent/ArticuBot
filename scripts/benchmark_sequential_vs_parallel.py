#!/usr/bin/env python3
"""Benchmark sequential vs parallel demo generation.

Runs a controlled experiment to generate a fixed number of SUCCESSFUL demos
and measures wall-clock time for:
  1. Sequential: gen_demo_custom.py (single process)
  2. Parallel:   gen_demo_custom_parallel.py (N workers)

All logs are written to local files for tracking.

Usage:
    source prepare.sh
    python scripts/benchmark_sequential_vs_parallel.py

Output:
    - benchmark_logs/bench_*.log  (individual experiment logs)
    - benchmark_results.json      (summary JSON)
    - benchmark_summary.txt       (human-readable summary table)
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import argparse
import sys
import time
from datetime import datetime

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "benchmark_logs"

# --------------- experiment parameters ---------------
ASSET_DIR = "data/custom_objects_v2/sim/microwave_7167"
BASE_SEED = 12345
COMMON_ARGS = [
    "--sampling-method", "integrated_inverse",
    "--rm4d-map", "data/rm4d_franka_1M.npy",
    "--top-k-grasps", "3",
    "--object-z-offset", "0.3", "0.5",
    "--timeout-init", "120",
    "--timeout-exec", "180",
    "--grasp-source", "ondemand",
    "--seed", str(BASE_SEED),
]

# Target: generate 10 successful demonstrations
# With ~32% success rate, need ~35 attempts to get 10 successes
NUM_TO_GENERATE = 10        # target successful demos
MAX_TRY_TIMES = 100         # max attempts (enough buffer for 10 successes @ 32%)

PARALLEL_WORKER_COUNTS = [8]
# ---------------------------------------------------------

def _get_gpu_count() -> int:
    """Return the number of GPUs available.

    Priority order:
      1) `ARTICUBOT_NUM_GPUS` env var (for testing/overrides)
      2) parse `nvidia-smi -L`
    """
    raw = os.environ.get("ARTICUBOT_NUM_GPUS")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass

    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5.0)
        if result.returncode == 0:
            lines = [line for line in (result.stdout or "").splitlines() if line.strip().startswith("GPU")]
            return len(lines)
    except Exception:
        pass
    return 0


def _prewarm_graspgen_containers(base_name: str = "friendly_galileo", num_gpus: int = 0, recreate: bool = False, timeout_s: int = 900) -> bool:
    """Start one GraspGen container per GPU and wait until each container can import grasp_gen.

    Returns True if all containers reported ready within timeout, False otherwise.
    """
    if num_gpus <= 0:
        print("[Prewarm] No GPUs detected; skipping pre-warm")
        return False

    script = PROJECT_ROOT / "scripts" / "run_graspgen_container.sh"
    if not script.exists():
        print(f"[Prewarm] Script not found: {script}; skipping pre-warm")
        return False

    # Start containers (detached)
    for i in range(num_gpus):
        name = f"{base_name}_gpu{i}"
        cmd = ["bash", str(script), "--name", name, "--gpu-id", str(i)]
        if recreate:
            cmd.append("--recreate")
        print(f"[Prewarm] Starting container: {name}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[Prewarm] Start failed for {name}: {res.stderr or res.stdout}")
        except Exception as e:
            print(f"[Prewarm] Exception while starting {name}: {e}")

    # Wait for readiness
    start = time.time()
    ready = set()
    while time.time() - start < timeout_s:
        for i in range(num_gpus):
            name = f"{base_name}_gpu{i}"
            if name in ready:
                continue
            try:
                r = subprocess.run(["docker", "exec", name, "python", "-c", "import grasp_gen"], capture_output=True, text=True, timeout=15.0)
                if r.returncode == 0:
                    ready.add(name)
                    print(f"[Prewarm] {name} ready")
            except Exception:
                pass
        if len(ready) >= num_gpus:
            print(f"[Prewarm] All {num_gpus} containers ready")
            return True
        time.sleep(5)
    print(f"[Prewarm] Timeout: {len(ready)}/{num_gpus} ready after {timeout_s}s")
    return False

# ---------------------------------------------------------


def run_experiment_group(label: str, cmds: list, exp_dir: pathlib.Path, log_prefix: pathlib.Path, timeout: int = 7200) -> dict:
    """Run a group of commands (parallel processes) and monitor experiment directory.

    Each command is launched as an independent process whose stdout/stderr is
    redirected to its own log file. The function monitors the experiment
    directory and will terminate remaining processes once we reach the target
    number of successful demos or a timeout occurs.
    """
    # Clean and prepare
    clean_exp(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    header = f"""
{'='*70}
  BENCHMARK EXPERIMENT GROUP: {label}
  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Commands: {len(cmds)} parallel processes
  Log prefix: {log_prefix}
{'='*70}
"""
    print(header)

    procs = []
    log_files = []
    t0 = time.time()

    try:
        for i, cmd in enumerate(cmds):
            log_file = LOG_DIR / f"{log_prefix.name}_{i}.log"
            lf = open(log_file, "w")
            lf.write(header + "\n")
            lf.flush()
            p = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=lf, stderr=subprocess.STDOUT, text=True)
            procs.append((p, lf))

        # Polling loop
        start = time.time()
        while True:
            time.sleep(5)
            total, success = count_demos(exp_dir)
            elapsed = time.time() - start
            print(f"  [{label}] {elapsed:.0f}s | demos: {success}/{NUM_TO_GENERATE} | procs running: {sum(1 for p, _ in procs if p.poll() is None)}")

            if success >= NUM_TO_GENERATE:
                print("Target reached — terminating remaining processes")
                break

            # All processes exited
            if all(p.poll() is not None for p, _ in procs):
                print("All processes completed")
                break

            if elapsed > timeout:
                print("Timeout reached — terminating remaining processes")
                break

        # Terminate remaining processes
        for p, lf in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    p.kill()

        # Wait for processes to finish cleanup
        for p, lf in procs:
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()
            try:
                lf.close()
            except Exception:
                pass

        elapsed_total = time.time() - t0
        total_dirs, success_dirs = count_demos(exp_dir)

        result = {
            "label": label,
            "wall_time_sec": round(elapsed_total, 1),
            "total_attempts": total_dirs,
            "successful_demos": success_dirs,
            "return_code": 0,
        }

        summary = f"""
{'='*70}
  EXPERIMENT GROUP COMPLETE: {label}
  Wall time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)
  Total attempts: {total_dirs}
  Successful demos: {success_dirs}
  Success rate: {100*success_dirs/max(total_dirs,1):.1f}%
  Time per success: {elapsed_total/max(success_dirs,1):.1f}s
{'='*70}
"""
        print(summary)
        with open(LOG_DIR / f"{log_prefix.name}_summary.log", "w") as sf:
            sf.write(summary)

        return result

    except Exception as e:
        # Cleanup on unexpected error
        for p, lf in procs:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:
                pass
            try:
                lf.close()
            except Exception:
                pass
        raise



def count_demos(exp_dir: pathlib.Path) -> tuple[int, int]:
    """Return (total_dirs, successful_dirs) under exp_dir."""
    if not exp_dir.exists():
        return 0, 0
    total = 0
    success = 0
    for d in exp_dir.iterdir():
        if d.is_dir() and d.name[:4].isdigit():
            total += 1
            if (d / "last_state_files.txt").exists():
                success += 1
    return total, success


def clean_exp(exp_dir: pathlib.Path):
    """Remove experiment directory if it exists."""
    if exp_dir.exists():
        shutil.rmtree(exp_dir)


def run_experiment(label: str, cmd: list[str], exp_dir: pathlib.Path, log_file: pathlib.Path, timeout: int = 7200) -> dict:
    """Run a single experiment and collect timing + results. All output logged to file."""
    clean_exp(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    header = f"""
{'='*70}
  BENCHMARK EXPERIMENT: {label}
  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Log file: {log_file}
  CMD: {' '.join(cmd)}
{'='*70}
"""
    print(header)

    with open(log_file, "w") as lf:
        lf.write(header + "\n")
        lf.flush()

        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output to both console (brief) and log file (full)
        line_count = 0
        try:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                line_count += 1
                # Print periodic progress to console
                if line_count % 50 == 0 or "Success" in line or "SUCCESS" in line:
                    total, success = count_demos(exp_dir)
                    elapsed = time.time() - t0
                    print(f"  [{label}] {elapsed:.0f}s | demos: {success}/{NUM_TO_GENERATE} | {line.strip()[:60]}")
        except Exception as e:
            lf.write(f"\nException during output read: {e}\n")

        proc.wait(timeout=timeout)
        elapsed = time.time() - t0

        total_dirs, success_dirs = count_demos(exp_dir)

        result = {
            "label": label,
            "wall_time_sec": round(elapsed, 1),
            "total_attempts": total_dirs,
            "successful_demos": success_dirs,
            "return_code": proc.returncode,
        }

        summary = f"""
{'='*70}
  EXPERIMENT COMPLETE: {label}
  Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)
  Total attempts: {total_dirs}
  Successful demos: {success_dirs}
  Success rate: {100*success_dirs/max(total_dirs,1):.1f}%
  Time per success: {elapsed/max(success_dirs,1):.1f}s
{'='*70}
"""
        lf.write(summary)
        print(summary)

    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark: Sequential vs Parallel demo generation')
    parser.add_argument('--prewarm-graspgen', action='store_true', help='Start & wait for per-GPU GraspGen containers before running benchmarks')
    parser.add_argument('--prewarm-recreate', action='store_true', help='Pass --recreate to container startup')
    parser.add_argument('--graspgen-base-name', type=str, default=os.environ.get('GRASPGEN_CONTAINER_BASE_NAME','friendly_galileo'), help='Container base name')
    args = parser.parse_args()

    # Optionally pre-warm GraspGen containers
    if args.prewarm_graspgen:
        num_gpus = _get_gpu_count()
        if num_gpus <= 0:
            print('[Prewarm] No GPUs detected; skipping pre-warm')
        else:
            print(f"[Prewarm] Starting pre-warm of {num_gpus} containers (base={args.graspgen_base_name})")
            ok = _prewarm_graspgen_containers(base_name=args.graspgen_base_name, num_gpus=num_gpus, recreate=args.prewarm_recreate, timeout_s=900)
            if not ok:
                print('[Prewarm] WARNING: Not all containers became ready; check container logs')

    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    asset_path = PROJECT_ROOT / ASSET_DIR
    results = []

    print(f"""
{'#'*70}
  BENCHMARK: Sequential vs Parallel Demo Generation
  Target: {NUM_TO_GENERATE} successful demos
  Asset: {ASSET_DIR}
  Logs: {LOG_DIR}/
{'#'*70}
""")

    # ---- 1. Sequential baseline ----
    exp_name_seq = "bench_sequential"
    exp_dir_seq = asset_path / "experiment" / exp_name_seq
    log_seq = LOG_DIR / f"bench_sequential_{timestamp}.log"

    cmd_seq = [
        sys.executable, "-u", "manipulation/gen_demo_custom.py",
        "--asset-dir", ASSET_DIR,
        "--exp-name", exp_name_seq,
        "--num-to-generate", str(NUM_TO_GENERATE),
        "--max-try-times", str(MAX_TRY_TIMES),
        *COMMON_ARGS,
    ]
    results.append(run_experiment("sequential (1 worker)", cmd_seq, exp_dir_seq, log_seq))

    # ---- 2. Parallel runs ----
    for nw in PARALLEL_WORKER_COUNTS:
        exp_name_par = f"bench_parallel_{nw}w"
        exp_dir_par = asset_path / "experiment" / exp_name_par
        log_par = LOG_DIR / f"bench_parallel_{nw}w_{timestamp}"

        # Build one gen_demo_custom.py command per worker and pass process-index
        cmds = []
        for i in range(nw):
            cmd = [
                sys.executable, "-u", "manipulation/gen_demo_custom.py",
                "--asset-dir", ASSET_DIR,
                "--exp-name", exp_name_par,
                "--num-to-generate", str(NUM_TO_GENERATE),
                "--max-try-times", str(MAX_TRY_TIMES),
                *COMMON_ARGS,
                "--process-index", str(i),
            ]
            cmds.append(cmd)

        results.append(run_experiment_group(f"parallel ({nw} workers)", cmds, exp_dir_par, log_par))

    # ---- Summary table ----
    baseline_time = results[0]["wall_time_sec"] if results else 1.0

    summary_lines = []
    summary_lines.append("")
    summary_lines.append("=" * 90)
    summary_lines.append("BENCHMARK RESULTS: Sequential vs Parallel Demo Generation")
    summary_lines.append(f"Target: {NUM_TO_GENERATE} successful demonstrations")
    summary_lines.append("=" * 90)
    summary_lines.append(f"{'Configuration':<25} {'Wall Time':>12} {'Attempts':>10} {'Success':>10} {'Time/Demo':>12} {'Speedup':>10}")
    summary_lines.append("-" * 90)

    for r in results:
        t = r["wall_time_sec"]
        att = r["total_attempts"]
        suc = r["successful_demos"]
        per_demo = t / max(suc, 1)
        speedup = baseline_time / max(t, 0.1)
        summary_lines.append(f"{r['label']:<25} {t:>10.1f}s {att:>10} {suc:>10} {per_demo:>10.1f}s {speedup:>9.2f}x")

    summary_lines.append("=" * 90)
    summary_lines.append("")
    summary_lines.append("CONCLUSION:")
    if len(results) > 1 and results[0]["wall_time_sec"] > 0:
        best_parallel = min(results[1:], key=lambda x: x["wall_time_sec"])
        speedup = results[0]["wall_time_sec"] / max(best_parallel["wall_time_sec"], 0.1)
        summary_lines.append(f"  Best parallel config: {best_parallel['label']}")
        summary_lines.append(f"  Speedup vs sequential: {speedup:.1f}x")
        summary_lines.append(f"  Time saved: {results[0]['wall_time_sec'] - best_parallel['wall_time_sec']:.0f}s")
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save summary to file
    summary_path = PROJECT_ROOT / "benchmark_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Summary saved to: {summary_path}")

    # Save JSON
    json_path = PROJECT_ROOT / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
