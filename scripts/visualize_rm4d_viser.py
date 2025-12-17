#!/usr/bin/env python3
"""Visualize an RM4D reachability map with viser.

This script provides an interactive 3D visualization of the RM4D reachability map
along with the robot URDF. You can:
- Visualize the reachable workspace as a point cloud
- Move a target end-effector pose and see the inverse reachability (base positions)
- Adjust robot joint positions via sliders

Example usage:
    python scripts/visualize_rm4d_viser.py --rm4d-map data/rm4d_franka_1M.npy
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import viser
from viser.extras import ViserUrdf

# Add RM4D to path
_rm4d_path = Path(__file__).resolve().parent.parent / "third_party" / "rm4d"
if str(_rm4d_path) not in sys.path:
    sys.path.insert(0, str(_rm4d_path))

from rm4d import ReachabilityMap4D


def _resolve_urdf_packages(src_path: Path) -> Path:
    """Resolve package:// URDF paths to absolute asset paths."""
    if not src_path.exists():
        raise FileNotFoundError(f"URDF not found: {src_path}")

    text = src_path.read_text()
    replacements = {
        "package://meshes/": (src_path.parent / "meshes").as_posix() + "/",
        "package://franka_description/": (
            Path(__file__).resolve().parent.parent / "pybullet_ompl/models/franka_description"
        ).as_posix() + "/",
    }

    updated = text
    for token, prefix in replacements.items():
        if token in updated:
            updated = updated.replace(token, prefix)

    tmp_path = Path(tempfile.gettempdir()) / f"{src_path.stem}_viser_resolved.urdf"
    tmp_path.write_text(updated)
    return tmp_path


def _sample_reachable_points(rmap: ReachabilityMap4D, max_points: int = 100_000) -> np.ndarray:
    """Sample points from the reachable workspace.
    
    Returns (N, 3) array of xyz positions in base frame.
    """
    # Get all reachable cells
    reachable = rmap.map > 0  # shape: (n_z, n_theta, n_xy, n_xy)
    
    # Find indices where reachable
    z_idxs, theta_idxs, x_idxs, y_idxs = np.where(reachable)
    
    if len(z_idxs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Subsample if too many
    if len(z_idxs) > max_points:
        indices = np.random.choice(len(z_idxs), max_points, replace=False)
        z_idxs = z_idxs[indices]
        theta_idxs = theta_idxs[indices]
        x_idxs = x_idxs[indices]
        y_idxs = y_idxs[indices]
    
    # Convert indices to positions
    # Note: RM4D uses (z, theta, x*, y*) where x*, y* are canonical base positions
    # For visualization, we'll show the EE positions that are reachable
    points = []
    
    for z_idx, theta_idx, x_idx, y_idx in zip(z_idxs, theta_idxs, x_idxs, y_idxs):
        # Get z from index
        z = rmap.z_limits[0] + (z_idx + 0.5) * rmap.voxel_res
        
        # Get theta from index
        theta = rmap.theta_limits[0] + (theta_idx + 0.5) * rmap.theta_res
        
        # Get canonical positions
        x_star = rmap.xy_limits[0] + (x_idx + 0.5) * rmap.voxel_res
        y_star = rmap.xy_limits[0] + (y_idx + 0.5) * rmap.voxel_res
        
        # Approximate x, y in base frame (assuming psi=0 for simplicity)
        # This is approximate since we don't know the exact EE orientation
        x = -x_star
        y = -y_star
        
        points.append([x, y, z])
    
    return np.array(points, dtype=np.float32)


def _load_robot(server: viser.ViserServer, urdf_path: Path) -> ViserUrdf | None:
    """Load robot URDF into viser."""
    try:
        resolved = _resolve_urdf_packages(urdf_path)
        robot = ViserUrdf(server, urdf_or_path=resolved, load_meshes=True, load_collision_meshes=False)
    except Exception as exc:
        print(f"[WARN] Failed to load robot URDF: {exc}")
        return None

    joint_limits = robot.get_actuated_joint_limits()
    sliders = []
    initial_cfg = []
    with server.gui.add_folder("Joint Control"):
        for joint_name, (lower, upper) in joint_limits.items():
            lower_val = lower if lower is not None else -np.pi
            upper_val = upper if upper is not None else np.pi
            init = 0.0 if lower_val < -0.1 and upper_val > 0.1 else (lower_val + upper_val) / 2.0
            slider = server.gui.add_slider(label=joint_name, min=lower_val, max=upper_val, step=1e-3, initial_value=init)
            sliders.append(slider)
            initial_cfg.append(init)

    def _update_cfg(_=None) -> None:
        if not sliders:
            return
        cfg = np.array([s.value for s in sliders], dtype=float)
        robot.update_cfg(cfg)

    for s in sliders:
        s.on_update(_update_cfg)

    reset_btn = server.gui.add_button("Reset joints")

    @reset_btn.on_click
    def _(_: object) -> None:
        for slider, init in zip(sliders, initial_cfg):
            slider.value = init

    if sliders:
        robot.update_cfg(np.array(initial_cfg, dtype=float))

    return robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize RM4D map with viser")
    parser.add_argument(
        "--rm4d-map",
        type=Path,
        required=True,
        help="Path to RM4D map (.npy)",
    )
    parser.add_argument(
        "--robot-urdf",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "manipulation/assets/panda_bullet/panda.urdf",
        help="Robot URDF path",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--point-size", type=float, default=0.008)
    parser.add_argument("--max-points", type=int, default=50_000)
    args = parser.parse_args()

    # Load RM4D map
    print(f"Loading RM4D map from {args.rm4d_map}...")
    rmap = ReachabilityMap4D.from_file(str(args.rm4d_map))

    # Sample reachable points for visualization
    print(f"Sampling up to {args.max_points} reachable points...")
    points = _sample_reachable_points(rmap, max_points=args.max_points)
    print(f"  Got {len(points)} points")

    # Create colors (gradient based on z height)
    if len(points) > 0:
        z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = (255 * (1 - z_norm)).astype(np.uint8)  # Red for low
        colors[:, 2] = (255 * z_norm).astype(np.uint8)  # Blue for high
        colors[:, 1] = 128  # Some green
    else:
        colors = np.zeros((0, 3), dtype=np.uint8)

    # Start viser server
    server = viser.ViserServer(port=args.port)
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.003)
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.05, position=(0, 0, 0))

    # Load robot
    robot = _load_robot(server, args.robot_urdf)

    # Add reachable workspace point cloud
    with server.gui.add_folder("Workspace"):
        show_ws = server.gui.add_checkbox("Show reachable", initial_value=True)
        
    if len(points) > 0:
        cloud_handle = server.scene.add_point_cloud(
            "/workspace",
            points=points,
            colors=colors,
            point_size=args.point_size,
        )
    else:
        cloud_handle = None

    @show_ws.on_update
    def _(_: object) -> None:
        if cloud_handle is not None:
            cloud_handle.visible = show_ws.value

    # Add target EE pose control
    with server.gui.add_folder("Target EE"):
        target_x = server.gui.add_slider("x", min=-1.0, max=1.0, step=0.01, initial_value=0.4)
        target_y = server.gui.add_slider("y", min=-1.0, max=1.0, step=0.01, initial_value=0.0)
        target_z = server.gui.add_slider("z", min=0.0, max=1.35, step=0.01, initial_value=0.4)
        show_base_pos = server.gui.add_checkbox("Show base positions", initial_value=False)

    target_frame = server.scene.add_frame("/target_ee", axes_length=0.1, axes_radius=0.005)
    base_cloud_handle = None

    def _update_target(_=None) -> None:
        nonlocal base_cloud_handle
        pos = np.array([target_x.value, target_y.value, target_z.value])
        target_frame.position = tuple(pos)
        
        if show_base_pos.value:
            # Create a simple EE transform (pointing down)
            tf_ee = np.eye(4)
            tf_ee[:3, 3] = pos
            tf_ee[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # Z pointing down
            
            try:
                base_positions = rmap.get_base_positions(tf_ee, as_3d=True)
                # Filter to non-zero scores
                mask = base_positions[:, 3] > 0
                base_pts = base_positions[mask, :3]
                if len(base_pts) > 0:
                    base_colors = np.zeros((len(base_pts), 3), dtype=np.uint8)
                    base_colors[:, 1] = 200  # Green
                    
                    if base_cloud_handle is not None:
                        try:
                            server.scene.remove("/base_positions")
                        except:
                            pass
                    base_cloud_handle = server.scene.add_point_cloud(
                        "/base_positions",
                        points=base_pts.astype(np.float32),
                        colors=base_colors,
                        point_size=args.point_size * 1.5,
                    )
            except IndexError:
                # Target outside map bounds
                if base_cloud_handle is not None:
                    try:
                        server.scene.remove("/base_positions")
                    except:
                        pass
                    base_cloud_handle = None
        else:
            if base_cloud_handle is not None:
                try:
                    server.scene.remove("/base_positions")
                except:
                    pass
                base_cloud_handle = None

    for slider in [target_x, target_y, target_z]:
        slider.on_update(_update_target)
    show_base_pos.on_update(_update_target)
    
    _update_target()  # Initial update

    print(f"Viser server running on http://localhost:{args.port}")
    try:
        import time
        while True:
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == "__main__":
    main()
