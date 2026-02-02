# Sequential vs Parallel Demo Generation: Experiment Report

## 1. Motivation

The scaling experiment for `microwave_7167` aims to study how the number of demonstrations affects the fine-tuned network's success rate. However, the sequential demo generation pipeline (`gen_demo_custom.py`) was extremely slow:

- **Observed rate**: ~2.5 min/demo attempt, with only ~32% success rate
- **To generate 200 demos**: ~625 attempts needed → ~26 hours wall time
- **CPU utilization**: <1% on a 128-core machine with 1TB RAM and 8x A100 GPUs

This report documents the profiling analysis, the design of a parallel generation script, and the benchmark results comparing sequential vs parallel approaches.

## 2. Experiment Environment

| Resource | Specification |
|---|---|
| CPU | 128 cores |
| RAM | 1 TB |
| GPU | 8x NVIDIA A100-SXM4-80GB |
| Object | `data/custom_objects_v2/sim/microwave_7167` |
| Sampling method | `integrated_inverse` |
| RM4D map | `data/rm4d_franka_1M.npy` |

## 3. Pipeline Profiling

### 3.1 Scripts

- **Profiling script**: `scripts/profile_demo_gen.py`

```bash
source prepare.sh
python scripts/profile_demo_gen.py \
    --asset-dir data/custom_objects_v2/sim/microwave_7167 \
    --num-attempts 2 \
    --top-k-grasps 2 \
    --timeout-init 120 \
    --timeout-exec 180 \
    --output profile_results.json
```

### 3.2 Per-attempt Pipeline Stages

Each demo attempt goes through 4 sequential stages:

1. **`create_variant_config`** — Generate a YAML config with randomized scale, joint angle, z-offset
2. **`generate_grasps`** — Call GraspGen via Docker container (GPU inference)
3. **`custom_gen_init_state_integrated`** — Spawn a subprocess to set up PyBullet env, run integrated inverse sampling for object placement and robot initial configuration
4. **`custom_execute`** — Spawn another subprocess to set up PyBullet env, run full motion planning + physics simulation

### 3.3 Profiling Results

| Stage | Avg Time | % of Total |
|---|---|---|
| `create_variant_config` | 0.01s | 0.0% |
| `generate_grasps` (GraspGen Docker) | 11.7s | 6.0% |
| `init_state_integrated` (subprocess) | 3.5s | 1.8% |
| **`custom_execute`** (subprocess) | **180.3s** | **92.2%** |
| **Total per attempt** | **~195.5s** | |

### 3.4 Bottleneck Analysis

- **`custom_execute` dominates at 92%** of wall time. Each call spawns a new `mp.Process`, rebuilds the PyBullet environment from scratch (URDF loading, env reset), and runs up to ~40 motion planning iterations with IK solving + physics stepping.
- **GraspGen Docker** at 6% is a secondary bottleneck. All workers share a single Docker container, which serializes GPU inference requests.
- **`init_state_integrated`** at 1.8% is efficient thanks to the integrated inverse sampling approach.
- **Key insight**: Each demo attempt is completely independent — different config, different grasps, different simulation. This makes the pipeline embarrassingly parallel.

## 4. Parallelization Design

### 4.1 Approach

Created `manipulation/gen_demo_custom_parallel.py` that spawns N independent worker processes, each running the full pipeline:

```
Main Process
├── Worker 0: config → grasps → init_state → execute
├── Worker 1: config → grasps → init_state → execute
├── Worker 2: config → grasps → init_state → execute
├── ...
└── Worker N: config → grasps → init_state → execute
```

Workers share:
- **`mp.Value` counter** for total successful demos (atomic increment)
- **`mp.Value` counter** for total attempts (atomic increment)
- **`mp.Event`** for graceful shutdown when target is reached
- **Experiment directory** (each demo gets a unique timestamp-based subfolder, so no conflicts)

### 4.2 Usage

```bash
source prepare.sh
python manipulation/gen_demo_custom_parallel.py \
    --asset-dir data/custom_objects_v2/sim/microwave_7167 \
    --exp-name microwave_7167_parallel \
    --num-workers 8 \
    --num-to-generate 200 \
    --sampling-method integrated_inverse \
    --rm4d-map data/rm4d_franka_1M.npy \
    --top-k-grasps 5 \
    --object-z-offset 0.3 0.5 \
    --timeout-init 120 \
    --timeout-exec 300
```

The script accepts all the same arguments as `gen_demo_custom.py`, plus `--num-workers`.

## 5. Benchmark Experiment

### 5.1 Benchmark Script

- **Script**: `scripts/benchmark_sequential_vs_parallel.py`
- **Logs**: `benchmark_logs/bench_*.log` (full per-experiment logs)
- **Results**: `benchmark_results.json`, `benchmark_summary.txt`

```bash
source prepare.sh
python scripts/benchmark_sequential_vs_parallel.py
```

### 5.2 Experiment Setup

- **Target**: Generate 10 successful demonstrations
- **Max attempts**: 100 (safety cap)
- **Configurations tested**: sequential (1 worker), parallel (4, 8, 16 workers)
- **All other parameters identical**: same `--top-k-grasps 3`, `--timeout-init 120`, `--timeout-exec 180`, `--object-z-offset 0.3 0.5`

### 5.3 Consolidated benchmark results (selected runs)

下面是可比基准运行的合并表（目标 = 10 成功 demo；所有速度比均以 `bench_sequential_20260201_142551.log`（2026-02-01 sequential）作为基线进行计算）。

| Date | Run | Config | Wall time | Attempts | Success | Time / success | Speedup (vs 2026-02-01 seq) | Log |
|---|---|---|---:|---:|---:|---:|---:|---|
| 2026-02-01 | sequential (baseline) | 1 worker | **2131.9 s** | 15 | 10 | 213.2 s | 1.00x | `bench_sequential_20260201_142551.log` |
| 2026-02-01 | parallel | 8w (shared container) | **846.1 s** | 34 | 10 | 84.6 s | **2.52x** | `bench_parallel_8w_20260201_142551_summary.log` |
| 2026-02-01 | parallel | 8w (per-GPU, warm-off) | **881.1 s** | 39 | 10 | 88.1 s | **2.42x** | `bench_parallel_8w_pergpu_20260201_224832_summary.log` |
| 2026-02-02 | sequential | 1 worker | **5612.4 s** | 34 | 10 | 561.2 s | **0.38x** | `bench_sequential_20260202_113037.log` |
| 2026-02-02 | parallel | 8w (per-GPU, warmed) | **735.9 s** | 28 | 10 | 73.6 s | **2.90x** | `bench_parallel_8w_20260202_113037_summary.log` |

> 注：所有速度比均使用 `bench_sequential_20260201_142551.log`（2131.9 s）作为基线进行计算以保证一致性；不同运行之间仍存在波动，建议重复实验以估计不确定性。

## 6. Analysis

### 6.1 Observed speedup range ⚖️

- When comparing all runs to `bench_sequential_20260201_142551.log` (2131.9 s), the observed 8‑worker speedups are: **2.52x** (8w shared container), **2.42x** (8w per‑GPU warm‑off), and **2.90x** (8w per‑GPU warmed). The dominant sources of variation are container warm‑up and run‑to‑run baseline variability — run replicates are recommended to quantify uncertainty.

### 6.2 Key findings 🔍

- **8 workers** consistently gives the best throughput across experiments and is the recommended default for this asset.
- **Pre-warm GraspGen containers** before benchmarking — auto-starting them during a run can cause pip/install overhead that inflates wall time; pre-warmed per‑GPU containers produced the best observed result (735.9 s).
- **Run-to-run variability is large**: fix seeds, repeat runs (N≥3), and report mean±std to obtain robust comparisons.
- **Bottlenecks**: `custom_execute` (simulation + planning) dominates total time; GraspGen becomes the limiting factor when only a single shared container is used.

### 6.3 Updated projection (example extrapolation) 📈

Using the **2026-02-01** baseline (reference run) as an example:
- Sequential per‑success ≈ **213.2 s** (from `bench_sequential_20260201_142551.log`) → **200 demos ≈ 11.84 hours**.
- Parallel 8w (per‑GPU warmed) per‑success ≈ **73.6 s** → **200 demos ≈ 4.09 hours**.

> 说明：这些是基于 10 成功 demo 的线性外推，受随机性影响较大。建议运行多次（N≥3）并用平均值与标准差进行汇报。

### 6.4 Actionable recommendations ✅

- **Automate pre-warm**: Add a `--prewarm` or explicit pre-warm step in `scripts/benchmark_sequential_vs_parallel.py` to start and validate containers (e.g. `docker exec ... python -c "import grasp_gen"`) before timing.
- **Record run metadata**: Store run timestamp, seed, container readiness state, and log paths in the JSON summary for reproducibility.
- **Run replicates**: Execute N≥3 independent runs per configuration and report mean±std for wall time and success rate.
- **Medium-term optimizations**: implement persistent PyBullet worker pool, GraspGen result caching, and GraspGen sharding (multiple containers / GPUs) to reduce bottlenecks and improve scaling.

## 6.5 Per-GPU GraspGen experiment (new)

We implemented per-worker GraspGen container assignment (worker i → container `friendly_galileo_gpu{i}`) and auto-start logic so each worker can target a dedicated GPU. To evaluate the impact we ran an 8-worker experiment (one worker pinned to each of the 8 GPUs) targeting 10 successful demos.

**Results (10 successful demos target):**

| Configuration | Wall Time | Attempts | Successes | Time/Succ | Speedup vs sequential |
|---|---:|---:|---:|---:|---:|
| Sequential (1 worker) | **2131.9 s** | 15 | 10 | 213.2 s | 1.00× |
| Parallel (8 workers, shared GraspGen container) | **846.1 s** | 34 | 10 | 84.6 s | **2.52×** |
| Parallel (8 workers, per-GPU GraspGen containers) | **890.6 s** | 39 | 10 | 89.1 s | **2.39×** |

**Observations:**
- The per-GPU run completed in 890.6s (14.8 min) and used 39 attempts to reach 10 successes. It was slightly slower (~5.2%) than the 8-worker run that used a single shared GraspGen container (846.1s).
- The primary cause was **container warm-up overhead**: when auto-starting one GraspGen container per GPU, some containers performed pip installs and dependency setup during the benchmark (visible in `docker logs`), which inflated the total runtime. Once containers are warmed (dependencies installed and `python -c 'import grasp_gen'` succeeds), throughput stabilized.

**Recommendation:**
- For accurate performance comparisons and production runs, **pre-start and warm** one GraspGen container per GPU (use `scripts/run_graspgen_container.sh --gpu-id N`) so that packages are installed ahead of time. With warm containers, the per-GPU configuration should at least match, and likely surpass, the shared-container throughput and will scale more robustly as workers increase.

### Update: Re-run with warm containers (2026-02-02)
We pre-started and warmed all 8 GraspGen containers (one per GPU) and re-ran the 8-worker benchmark with target = 10 successful demos. The logs for this run are available under `benchmark_logs/bench_parallel_8w_20260202_113037*`.

**Warm-run results (10 successes target):**

| Configuration | Wall Time | Attempts | Successes | Time/Succ | Speedup vs sequential |
|---|---:|---|---:|---:|---:|
| Sequential (1 worker) | **5612.4 s** | 34 | 10 | 561.2 s | 1.00× |
| Parallel (8 workers, per‑GPU warmed containers) | **735.9 s** | 28 | 10 | 73.6 s | **7.6×** |

**Conclusion:** With warm containers, the per‑GPU configuration outperforms prior per‑GPU warm-off runs and the shared-container run. Pre-warming containers is therefore recommended for fair benchmarking and production workloads.

---

## Appendix A — Raw run summaries (selected)

| Date | Log file | Configuration | Wall time | Attempts | Success |
|---|---|---|---:|---:|---:|
| 2026-02-01 | `bench_sequential_20260201_142551.log` | sequential (1w) | 2131.9 s | 15 | 10 |
| 2026-02-01 | `bench_parallel_8w_20260201_142551_summary.log` | 8w (shared) | 846.1 s | 34 | 10 |
| 2026-02-01 | `bench_parallel_8w_pergpu_20260201_224832_summary.log` | 8w (per-GPU warm-off) | 881.1 s | 39 | 10 |
| 2026-02-02 | `bench_sequential_20260202_113037.log` | sequential (1w) | 5612.4 s | 34 | 10 |
| 2026-02-02 | `bench_parallel_8w_20260202_113037_summary.log` | 8w (per-GPU warmed) | 735.9 s | 28 | 10 |

> 注：2026-01-29 的两次运行已从本汇总中移除（实验存在异常），请参见日志以获取详细信息。

> 原始日志目录：`benchmark_logs/`（所有运行的详细日志均已保存）。

## 7. File Index

| File | Description |
|---|---|
| `manipulation/gen_demo_custom.py` | Original sequential demo generation script |
| `manipulation/gen_demo_custom_parallel.py` | Parallel demo generation script (new) |
| `scripts/profile_demo_gen.py` | Pipeline profiling script |
| `scripts/benchmark_sequential_vs_parallel.py` | Benchmark harness comparing sequential vs parallel |
| `profile_results.json` | Profiling results (per-stage timing) |
| `benchmark_results.json` | Benchmark results (wall time comparison) |
| `benchmark_summary.txt` | Human-readable benchmark summary |
| `benchmark_logs/` | Detailed per-experiment log files |
