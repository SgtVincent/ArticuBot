#!/usr/bin/env python3
"""Count valid demos inside an experiment folder.

A demo is considered valid (training-ready) if it contains:
- all.gif
- stage_lengths.json
- task_config.yaml
- states/state_*.pkl (at least one)

This matches the assumptions used by downstream conversion/training.

Prints a JSON object to stdout:
  {"total": <int>, "valid": <int>}

Notes:
- Skips the "configs" subfolder.
- If the experiment directory doesn't exist, returns {"total": 0, "valid": 0}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = (
    "all.gif",
    "stage_lengths.json",
    "task_config.yaml",
)


def _is_valid_demo_dir(demo_dir: Path) -> bool:
    for name in REQUIRED_FILES:
        if not (demo_dir / name).exists():
            return False

    states_dir = demo_dir / "states"
    if not states_dir.exists() or not states_dir.is_dir():
        return False

    if not any(states_dir.glob("state_*.pkl")):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()

    exp_dir = args.experiment_dir.expanduser().resolve()
    if not exp_dir.exists():
        print(json.dumps({"total": 0, "valid": 0}))
        return

    run_dirs = [
        d
        for d in sorted(exp_dir.iterdir())
        if d.is_dir() and d.name != "configs"
    ]

    total = len(run_dirs)
    valid = sum(1 for d in run_dirs if _is_valid_demo_dir(d))

    print(json.dumps({"total": total, "valid": valid}))


if __name__ == "__main__":
    main()
