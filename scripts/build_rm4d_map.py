#!/usr/bin/env python3
"""Build an RM4D reachability map for the Franka Panda robot.

This script creates a 4D reachability map using RM4D's efficient dimensionality
reduction approach. The map can be used for both forward queries (is a pose
reachable?) and inverse queries (where should the base be for a given EE pose?).

Example usage:
    python scripts/build_rm4d_map.py --robot franka --samples 1000000 --output data/rm4d_franka_1M.npy

For trajectory filtering during demo generation, use with gen_demo_custom.py:
    python manipulation/gen_demo_custom.py ... --rm4d-map data/rm4d_franka_1M.npy --object-traj <traj.npz>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add RM4D to path
_rm4d_path = Path(__file__).resolve().parent.parent / "third_party" / "rm4d"
if str(_rm4d_path) not in sys.path:
    sys.path.insert(0, str(_rm4d_path))

from rm4d import ReachabilityMap4D, JointSpaceConstructor
from rm4d.robots import Simulator, Franka, UR5E


def main():
    parser = argparse.ArgumentParser(description="Build RM4D reachability map")
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        choices=["franka", "ur5e"],
        help="Robot type (default: franka)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1_000_000,
        help="Number of joint-space samples (default: 1M)",
    )
    parser.add_argument(
        "--voxel-res",
        type=float,
        default=0.05,
        help="Voxel resolution in meters (default: 0.05)",
    )
    parser.add_argument(
        "--n-bins-theta",
        type=int,
        default=36,
        help="Number of theta angle bins (default: 36)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for .npy file (default: data/rm4d_<robot>_<samples>.npy)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100_000,
        help="Save checkpoint every N samples (0 to disable)",
    )
    args = parser.parse_args()

    # Set up limits based on robot
    if args.robot == "franka":
        xy_limits = [-1.05, 1.05]
        z_limits = [0.0, 1.35]
    elif args.robot == "ur5e":
        xy_limits = [-1.0, 1.0]
        z_limits = [0.0, 1.2]
    else:
        raise ValueError(f"Unknown robot: {args.robot}")

    # Default output path
    if args.output is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
        samples_str = f"{args.samples // 1_000_000}M" if args.samples >= 1_000_000 else f"{args.samples // 1_000}K"
        args.output = str(data_dir / f"rm4d_{args.robot}_{samples_str}.npy")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building RM4D map for {args.robot}")
    print(f"  Samples: {args.samples:,}")
    print(f"  Resolution: {args.voxel_res}m")
    print(f"  Theta bins: {args.n_bins_theta}")
    print(f"  Output: {args.output}")

    # Create map
    rmap = ReachabilityMap4D(
        xy_limits=xy_limits,
        z_limits=z_limits,
        voxel_res=args.voxel_res,
        n_bins_theta=args.n_bins_theta,
    )
    rmap.print_structure()

    # Create simulator and robot
    print("\nInitializing PyBullet simulation...")
    sim = Simulator(with_gui=False)
    
    if args.robot == "franka":
        robot = Franka(sim)
    else:
        robot = UR5E(sim)
    print(f"Robot loaded: {args.robot}")

    # Build map using joint space sampling
    print(f"\nSampling {args.samples:,} configurations...")
    constructor = JointSpaceConstructor(rmap, robot)
    
    if args.checkpoint_every > 0 and args.samples > args.checkpoint_every:
        # Sample in batches with checkpoints
        samples_done = 0
        while samples_done < args.samples:
            batch_size = min(args.checkpoint_every, args.samples - samples_done)
            constructor.sample(n_samples=batch_size, prevent_collisions=True)
            samples_done += batch_size
            
            # Save checkpoint
            checkpoint_path = output_path.with_suffix(f".{samples_done}.npy")
            rmap.to_file(str(checkpoint_path))
            print(f"  Checkpoint saved: {checkpoint_path} ({samples_done:,}/{args.samples:,})")
    else:
        constructor.sample(n_samples=args.samples, prevent_collisions=True)

    # Save final map
    rmap.to_file(str(output_path))
    print(f"\nRM4D map saved to {output_path}")

    # Print statistics
    print("\nMap Statistics:")
    print(f"  Non-zero cells: {rmap.map.sum():,} / {rmap.map.size:,} ({100*rmap.map.sum()/rmap.map.size:.2f}%)")
    print(f"  Memory: {rmap.map.nbytes / 1024 / 1024:.2f} MB")

    # Cleanup
    robot.remove_from_sim()
    print("\nDone!")


if __name__ == "__main__":
    main()
