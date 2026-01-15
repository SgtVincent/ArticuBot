#!/usr/bin/env python3
"""Plot sim-to-sim scaling curves from eval_sim_to_sim.py outputs.

Expected layout (created by scripts/experiments/microwave_7167_scaling.sh):
  outputs/sim_to_sim_eval/microwave_7167_scaling/
    n50/summary.json
    n100/summary.json
    ...

The plot uses the 'original_model' entries only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_points(summary: dict[str, Any]) -> dict[str, float]:
    results = summary.get("results", [])
    original = [r for r in results if r.get("eval_target") == "original_model"]
    if not original:
        return {}

    # Single-object mode -> one summary entry
    s = original[0]
    return {
        "mean_opening": float(s.get("mean_opening", 0.0)),
        "success_rate_50": float(s.get("success_rate_50", 0.0)),
        "successful_trials": float(s.get("successful_trials", 0.0)),
        "total_trials": float(s.get("total_trials", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Root folder containing n*/summary.json")
    parser.add_argument("--out", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    points: list[tuple[int, dict[str, float]]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        m = re.fullmatch(r"n(\d+)", child.name)
        if not m:
            continue
        n = int(m.group(1))
        summary_path = child / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load_summary(summary_path)
        metrics = _extract_points(summary)
        if metrics:
            points.append((n, metrics))

    points.sort(key=lambda x: x[0])
    if not points:
        raise RuntimeError(f"No evaluation summaries found under: {root}")

    xs = [n for n, _ in points]
    mean_open = [m["mean_opening"] for _, m in points]
    success50 = [100.0 * m["success_rate_50"] for _, m in points]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5), dpi=140)

    ax1.plot(xs, mean_open, marker="o", linewidth=2, label="Mean normalized opening")
    ax1.set_xlabel("# training demos")
    ax1.set_ylabel("Mean normalized opening")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(xs, success50, marker="s", linewidth=2, color="#d95f02", label="Success@50% (pct)")
    ax2.set_ylabel("Success@50% (%)")
    ax2.set_ylim(0, 100)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")

    title = "Sim-to-sim scaling (original model)"
    ax1.set_title(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)

    # Also dump a small machine-readable table.
    table_path = out_path.with_suffix(".json")
    table = {
        "points": [
            {
                "n": n,
                "mean_opening": m["mean_opening"],
                "success_rate_50": m["success_rate_50"],
                "successful_trials": m["successful_trials"],
                "total_trials": m["total_trials"],
            }
            for n, m in points
        ]
    }
    table_path.write_text(json.dumps(table, indent=2), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {table_path}")


if __name__ == "__main__":
    main()
