"""Debug visualization utilities for integrated inverse map sampling.

This module provides functions to create GIF animations showing:
1. In-contact trajectory 6D poses in the world frame
2. Per-waypoint inverse map reachability values (color-coded 0=red to 1=green)
3. The integrated inverse map distribution over object placements

The visualization helps debug low success rates in the IntegratedInverseMapSampler
by showing:
- Whether each waypoint's 6D EE pose is reachable from the robot base
- The per-waypoint inverse reachability heatmaps showing which object placements
  make each waypoint reachable
- The aggregated/integrated heatmap used for sampling
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for subprocess
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Sequence
from pathlib import Path

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

from scipy.spatial.transform import Rotation as R_scipy


def create_coordinate_frame(
    ax,  # Axes3D
    R: np.ndarray,
    p: np.ndarray,
    scale: float = 0.05,
    alpha: float = 0.8,
) -> None:
    """Draw a 3D coordinate frame (RGB = XYZ axes).
    
    Args:
        ax: Matplotlib 3D axes.
        R: (3, 3) rotation matrix.
        p: (3,) position vector.
        scale: Length of axis arrows.
        alpha: Transparency.
    """
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        direction = R[:, i] * scale
        ax.quiver(
            p[0], p[1], p[2],
            direction[0], direction[1], direction[2],
            color=color, alpha=alpha, arrow_length_ratio=0.2, linewidth=1.5
        )


def score_to_color(score: float) -> Tuple[float, float, float]:
    """Convert reachability score to RGB color (0=red, 1=green).
    
    Args:
        score: Reachability score in [0, 1].
        
    Returns:
        (R, G, B) tuple with values in [0, 1].
    """
    # Red to green gradient via yellow
    if score < 0.5:
        r = 1.0
        g = 2.0 * score
    else:
        r = 2.0 * (1.0 - score)
        g = 1.0
    return (r, g, 0.0)


def plot_trajectory_with_scores(
    trajectory: Sequence[Tuple[np.ndarray, np.ndarray]],
    scores: Sequence[float],
    object_pos: np.ndarray = None,
    object_quat: np.ndarray = None,
    title: str = "Trajectory with Reachability Scores",
    show_legend: bool = True,
    highlight_waypoint: int = None,
) -> Figure:
    """Plot trajectory waypoints colored by reachability score.
    
    Args:
        trajectory: List of (R, p) tuples for waypoints in world frame.
        scores: Reachability score for each waypoint.
        object_pos: Optional object position for reference.
        object_quat: Optional object quaternion for reference.
        title: Plot title.
        show_legend: Whether to show colorbar legend.
        highlight_waypoint: Optional index of waypoint to highlight.
        
    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = np.array([wp[1] for wp in trajectory])
    
    # Plot trajectory line
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'k-', alpha=0.3, linewidth=1, label='Trajectory')
    
    # Plot waypoints colored by score
    for i, ((R, p), score) in enumerate(zip(trajectory, scores)):
        color = score_to_color(score)
        marker_size = 150 if (highlight_waypoint is not None and i == highlight_waypoint) else 50
        ax.scatter(p[0], p[1], p[2], c=[color], s=marker_size, alpha=0.7)
        
        # Draw coordinate frame for every 5th waypoint or highlighted
        if i % 5 == 0 or (highlight_waypoint is not None and i == highlight_waypoint):
            frame_scale = 0.05 if (highlight_waypoint is not None and i == highlight_waypoint) else 0.03
            create_coordinate_frame(ax, R, p, scale=frame_scale, alpha=0.5)
    
    # Plot object position if provided
    if object_pos is not None:
        ax.scatter(object_pos[0], object_pos[1], object_pos[2], 
                   c='purple', s=200, marker='*', label='Object', zorder=5)
    
    # Plot robot base (origin)
    ax.scatter(0, 0, 0, c='black', s=100, marker='^', label='Robot Base')
    
    # Labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set axis limits for better view
    all_pts = np.vstack([positions, np.zeros((1, 3))])
    if object_pos is not None:
        all_pts = np.vstack([all_pts, object_pos.reshape(1, 3)])
    center = all_pts.mean(axis=0)
    max_range = np.max(np.abs(all_pts - center)) * 1.2
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(0, max(center[2] + max_range, 1.0))
    
    if show_legend:
        ax.legend()
        
        # Add colorbar using matplotlib.cm
        import matplotlib.cm as mcm
        sm = mcm.ScalarMappable(cmap='RdYlGn', norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Reachability Score')
    
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    
    return fig


def plot_integrated_inverse_map(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    scores: np.ndarray,
    theta_index: int = 0,
    object_pos: np.ndarray = None,
    sampled_poses: List[Tuple[np.ndarray, np.ndarray, float]] = None,
    title: str = "Integrated Inverse Map",
) -> Figure:
    """Plot 2D slice of integrated inverse map for a specific theta.
    
    Args:
        x_grid: X coordinate grid.
        y_grid: Y coordinate grid.
        theta_grid: Theta values.
        scores: 3D array of scores [nx, ny, ntheta].
        theta_index: Index into theta_grid for the slice.
        object_pos: Optional object position to mark.
        sampled_poses: Optional list of (pos, quat, score) sampled poses.
        title: Plot title.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get 2D slice
    score_slice = scores[:, :, theta_index].T  # Transpose for correct orientation
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Plot heatmap
    im = ax.pcolormesh(X, Y, score_slice, cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Reachability Score')
    
    # Mark robot base
    ax.scatter(0, 0, c='black', s=100, marker='^', label='Robot Base', zorder=5)
    
    # Mark object position if provided
    if object_pos is not None:
        ax.scatter(object_pos[0], object_pos[1], c='purple', s=200, 
                   marker='*', label='Sampled Object', zorder=6)
    
    # Mark sampled poses if provided
    if sampled_poses is not None:
        for i, (pos, quat, score) in enumerate(sampled_poses[:10]):  # Limit to 10
            ax.scatter(pos[0], pos[1], c='blue', s=50, marker='o', 
                       alpha=0.5, zorder=4)
    
    # Labels
    theta_deg = np.degrees(theta_grid[theta_index])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # Add statistics
    max_score = np.max(score_slice)
    mean_score = np.mean(score_slice[score_slice > 0]) if np.any(score_slice > 0) else 0
    n_valid = np.sum(score_slice > 0)
    
    ax.set_title(f"{title}\nθ = {theta_deg:.1f}° | max={max_score:.3f}, mean(>0)={mean_score:.3f}, valid={n_valid}")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def compute_per_waypoint_scores_grid(
    rmap,
    trajectory_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    z_height: float,
    base_euler: np.ndarray,
) -> np.ndarray:
    """Compute per-waypoint reachability scores for each grid cell.
    
    Returns a 4D array [n_waypoints, nx, ny, ntheta] where each slice
    shows which object placements make that waypoint reachable.
    
    Args:
        rmap: RM4D ReachabilityMap4D instance.
        trajectory_obj: Object-frame trajectory as list of (R, p) tuples.
        x_grid: X coordinate grid.
        y_grid: Y coordinate grid.
        theta_grid: Theta grid.
        z_height: Object z-height.
        base_euler: Base euler angles for object orientation.
        
    Returns:
        4D array [n_waypoints, nx, ny, ntheta] of binary reachability.
    """
    n_waypoints = len(trajectory_obj)
    nx, ny, ntheta = len(x_grid), len(y_grid), len(theta_grid)
    
    per_waypoint_scores = np.zeros((n_waypoints, nx, ny, ntheta), dtype=np.float32)
    
    # Pre-compute trajectory waypoints as 4x4 transforms
    traj_obj_tf = []
    for R_o, p_o in trajectory_obj:
        tf = np.eye(4)
        tf[:3, :3] = np.asarray(R_o)
        tf[:3, 3] = np.asarray(p_o)
        traj_obj_tf.append(tf)
    
    # Pre-compute rotation matrices for all theta values
    R_wo_all = np.zeros((ntheta, 3, 3), dtype=np.float64)
    for ith, theta in enumerate(theta_grid):
        euler = base_euler.copy()
        euler[2] += theta
        R_wo_all[ith] = R_scipy.from_euler('xyz', euler).as_matrix()
    
    # Compute scores for each waypoint
    for wp_idx in range(n_waypoints):
        tf_obj = traj_obj_tf[wp_idx]
        
        for ith in range(ntheta):
            R_wo = R_wo_all[ith]
            
            for ix, x in enumerate(x_grid):
                for iy, y in enumerate(y_grid):
                    # Build object-to-world transform
                    T_wo = np.eye(4)
                    T_wo[:3, :3] = R_wo
                    T_wo[:3, 3] = [x, y, z_height]
                    
                    # Transform waypoint to world frame
                    tf_world = T_wo @ tf_obj
                    
                    # Check reachability
                    try:
                        indices = rmap.get_indices_for_ee_pose(tf_world)
                        is_reachable = float(rmap.is_reachable(indices))
                    except (IndexError, ValueError):
                        is_reachable = 0.0
                    
                    per_waypoint_scores[wp_idx, ix, iy, ith] = is_reachable
    
    return per_waypoint_scores


def plot_per_waypoint_heatmap(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    waypoint_scores: np.ndarray,
    waypoint_idx: int,
    theta_index: int = 0,
    object_pos: np.ndarray = None,
    ee_pos_world: np.ndarray = None,
    title: str = "Per-Waypoint Inverse Map",
) -> Figure:
    """Plot heatmap showing which object placements make a waypoint reachable.
    
    Args:
        x_grid: X coordinate grid.
        y_grid: Y coordinate grid.
        theta_grid: Theta grid.
        waypoint_scores: 3D scores for this waypoint [nx, ny, ntheta].
        waypoint_idx: Index of the waypoint (for title).
        theta_index: Theta slice to visualize.
        object_pos: Optional object position.
        ee_pos_world: Optional EE position in world frame.
        title: Plot title prefix.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get 2D slice
    score_slice = waypoint_scores[:, :, theta_index].T
    
    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Plot heatmap
    im = ax.pcolormesh(X, Y, score_slice, cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Reachable (1) / Not Reachable (0)')
    
    # Mark robot base
    ax.scatter(0, 0, c='black', s=100, marker='^', label='Robot Base', zorder=5)
    
    # Mark object position if provided
    if object_pos is not None:
        ax.scatter(object_pos[0], object_pos[1], c='purple', s=200, 
                   marker='*', label='Object', zorder=6)
    
    # Mark EE position if provided
    if ee_pos_world is not None:
        ax.scatter(ee_pos_world[0], ee_pos_world[1], c='cyan', s=100, 
                   marker='x', label=f'EE waypoint {waypoint_idx}', zorder=7)
    
    # Statistics
    n_reachable = np.sum(score_slice > 0)
    total_cells = score_slice.size
    coverage = n_reachable / total_cells * 100
    
    theta_deg = np.degrees(theta_grid[theta_index])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"{title}\nWaypoint {waypoint_idx} | θ={theta_deg:.1f}° | Coverage: {coverage:.1f}% ({n_reachable}/{total_cells})")
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def fig_to_array(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array.
    
    Args:
        fig: Matplotlib figure.
        
    Returns:
        RGB array of shape (H, W, 3).
    """
    # Draw the figure
    fig.canvas.draw()
    
    # Convert to numpy array using renderer
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = buf.reshape(h, w, 3)
    
    return img


def compute_waypoint_scores(
    rmap,
    trajectory_world: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> List[float]:
    """Compute reachability score for each waypoint at current object pose.
    
    Args:
        rmap: RM4D ReachabilityMap4D instance.
        trajectory_world: Trajectory in world frame as (R, p) tuples.
        
    Returns:
        List of scores, one per waypoint.
    """
    scores = []
    for R_wp, p_wp in trajectory_world:
        tf = np.eye(4)
        tf[:3, :3] = R_wp
        tf[:3, 3] = p_wp
        
        try:
            indices = rmap.get_indices_for_ee_pose(tf)
            is_reachable = float(rmap.is_reachable(indices))
            scores.append(is_reachable)
        except (IndexError, ValueError):
            scores.append(0.0)
    
    return scores


def create_comprehensive_debug_gif(
    trajectory_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    sampler,  # IntegratedInverseMapSampler
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    integrated_scores: np.ndarray,
    base_euler: np.ndarray,
    z_height: float,
    output_path: str | Path,
    fps: int = 2,
    include_per_waypoint: bool = True,
    max_waypoint_frames: int = 10,
) -> None:
    """Create comprehensive debug visualization GIF.
    
    The GIF contains:
    1. Overview frame: 3D trajectory with all waypoints color-coded
    2. Per-waypoint frames: Heatmap for each waypoint showing valid object placements
    3. Integrated map frames: Final aggregated distribution for different theta values
    
    Args:
        trajectory_obj: Object-frame trajectory.
        sampler: IntegratedInverseMapSampler instance.
        object_pos: Sampled object position.
        object_quat: Sampled object quaternion.
        x_grid, y_grid, theta_grid: Grids for the distribution.
        integrated_scores: 3D integrated scores array.
        base_euler: Base euler angles for object.
        z_height: Object z-height.
        output_path: Path to save GIF.
        fps: Frames per second.
        include_per_waypoint: Whether to include per-waypoint heatmaps.
        max_waypoint_frames: Maximum number of waypoint frames to include.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames = []
    
    # Transform trajectory to world frame for visualization
    R_obj = R_scipy.from_quat(object_quat).as_matrix()
    trajectory_world = []
    for R_wp, p_wp in trajectory_obj:
        R_world = R_obj @ R_wp
        p_world = R_obj @ p_wp + object_pos
        trajectory_world.append((R_world, p_world))
    
    # Compute waypoint scores at sampled object pose
    if sampler is not None and hasattr(sampler, 'rmap'):
        waypoint_scores = compute_waypoint_scores(sampler.rmap, trajectory_world)
    else:
        waypoint_scores = [0.0] * len(trajectory_world)
    
    # Count reachable waypoints
    n_reachable = sum(1 for s in waypoint_scores if s > 0)
    total_waypoints = len(waypoint_scores)
    
    print(f"  [DEBUG VIS] Waypoint reachability at sampled pose: {n_reachable}/{total_waypoints}")
    
    # Frame 1: Overview with 3D trajectory
    fig = plot_trajectory_with_scores(
        trajectory_world, waypoint_scores, object_pos, object_quat,
        title=f"Full Trajectory - {n_reachable}/{total_waypoints} Waypoints Reachable"
    )
    frames.append(fig_to_array(fig))
    plt.close(fig)
    
    # Find best theta index (one with highest integrated score)
    theta_sums = np.sum(integrated_scores, axis=(0, 1))
    best_theta_idx = np.argmax(theta_sums)
    
    # Per-waypoint heatmaps (optional, but helpful for debugging)
    if include_per_waypoint and sampler is not None and hasattr(sampler, 'rmap'):
        print(f"  [DEBUG VIS] Computing per-waypoint heatmaps...")
        
        # Compute per-waypoint scores
        per_wp_scores = compute_per_waypoint_scores_grid(
            sampler.rmap, trajectory_obj, x_grid, y_grid, theta_grid,
            z_height, base_euler
        )
        
        # Select waypoints to visualize (evenly spaced)
        n_waypoints = len(trajectory_obj)
        step = max(1, n_waypoints // max_waypoint_frames)
        waypoint_indices = list(range(0, n_waypoints, step))[:max_waypoint_frames]
        
        for wp_idx in waypoint_indices:
            ee_pos = trajectory_world[wp_idx][1]
            
            fig = plot_per_waypoint_heatmap(
                x_grid, y_grid, theta_grid,
                per_wp_scores[wp_idx],
                waypoint_idx=wp_idx,
                theta_index=best_theta_idx,
                object_pos=object_pos,
                ee_pos_world=ee_pos,
                title="Per-Waypoint Inverse Reachability"
            )
            frames.append(fig_to_array(fig))
            plt.close(fig)
    
    # Integrated map slices for different theta values
    n_theta_frames = min(8, len(theta_grid))
    theta_indices = np.linspace(0, len(theta_grid) - 1, n_theta_frames, dtype=int)
    
    for theta_idx in theta_indices:
        fig = plot_integrated_inverse_map(
            x_grid, y_grid, theta_grid, integrated_scores,
            theta_index=theta_idx, object_pos=object_pos,
            title="Integrated Inverse Map (Aggregated)"
        )
        frames.append(fig_to_array(fig))
        plt.close(fig)
    
    # Summary statistics frame
    fig = create_summary_frame(
        trajectory_obj, trajectory_world, waypoint_scores,
        integrated_scores, x_grid, y_grid, theta_grid,
        object_pos, z_height
    )
    frames.append(fig_to_array(fig))
    plt.close(fig)
    
    # Save GIF
    imageio.mimsave(str(output_path), frames, fps=fps)
    print(f"  [DEBUG VIS] Saved to: {output_path}")


def create_summary_frame(
    trajectory_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    trajectory_world: Sequence[Tuple[np.ndarray, np.ndarray]],
    waypoint_scores: List[float],
    integrated_scores: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    object_pos: np.ndarray,
    z_height: float,
) -> Figure:
    """Create a summary statistics frame.
    
    Args:
        trajectory_obj: Object-frame trajectory.
        trajectory_world: World-frame trajectory.
        waypoint_scores: Per-waypoint reachability at sampled pose.
        integrated_scores: 3D integrated scores.
        x_grid, y_grid, theta_grid: Grids.
        object_pos: Sampled object position.
        z_height: Object z-height.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Waypoint score histogram
    ax = axes[0, 0]
    ax.bar(range(len(waypoint_scores)), waypoint_scores, color=['green' if s > 0 else 'red' for s in waypoint_scores])
    ax.set_xlabel('Waypoint Index')
    ax.set_ylabel('Reachable (0/1)')
    ax.set_title(f'Per-Waypoint Reachability at Sampled Pose\n{sum(1 for s in waypoint_scores if s > 0)}/{len(waypoint_scores)} reachable')
    ax.set_ylim(0, 1.1)
    
    # Top-right: Z-coordinate of waypoints in object frame
    ax = axes[0, 1]
    obj_z_coords = [p[2] for R, p in trajectory_obj]
    world_z_coords = [p[2] for R, p in trajectory_world]
    ax.plot(obj_z_coords, 'b-o', label='Object frame z', markersize=3)
    ax.plot(world_z_coords, 'g-o', label='World frame z', markersize=3)
    ax.axhline(y=z_height, color='r', linestyle='--', label=f'Object z_height={z_height:.3f}')
    ax.set_xlabel('Waypoint Index')
    ax.set_ylabel('Z Coordinate (m)')
    ax.set_title('Waypoint Z-Coordinates')
    ax.legend()
    
    # Bottom-left: Integrated score per theta
    ax = axes[1, 0]
    theta_deg = np.degrees(theta_grid)
    theta_sums = np.sum(integrated_scores, axis=(0, 1))
    theta_maxs = np.max(integrated_scores, axis=(0, 1))
    ax.plot(theta_deg, theta_sums, 'b-', label='Sum')
    ax.plot(theta_deg, theta_maxs, 'g--', label='Max')
    ax.set_xlabel('Object Theta (deg)')
    ax.set_ylabel('Score')
    ax.set_title('Integrated Score vs Object Orientation')
    ax.legend()
    
    # Bottom-right: Text summary
    ax = axes[1, 1]
    ax.axis('off')
    
    n_valid = np.sum(integrated_scores > 0)
    max_score = np.max(integrated_scores)
    mean_nonzero = np.mean(integrated_scores[integrated_scores > 0]) if n_valid > 0 else 0
    n_reachable_wp = sum(1 for s in waypoint_scores if s > 0)
    
    summary_text = f"""
    INTEGRATED INVERSE MAP SUMMARY
    ==============================
    
    Object Placement:
      Position: ({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f})
      Z-height: {z_height:.3f} m
    
    Trajectory:
      Total waypoints: {len(trajectory_obj)}
      Object-frame z range: [{min(obj_z_coords):.3f}, {max(obj_z_coords):.3f}]
      World-frame z range: [{min(world_z_coords):.3f}, {max(world_z_coords):.3f}]
    
    Waypoint Reachability (at sampled pose):
      Reachable: {n_reachable_wp}/{len(waypoint_scores)} ({100*n_reachable_wp/len(waypoint_scores):.1f}%)
    
    Integrated Distribution:
      Grid size: {len(x_grid)} x {len(y_grid)} x {len(theta_grid)}
      Valid cells (score > 0): {n_valid}
      Max score: {max_score:.4f}
      Mean of nonzero: {mean_nonzero:.4f}
      
    Potential Issues:
      {'⚠️ No valid cells in distribution!' if n_valid == 0 else ''}
      {'⚠️ Low waypoint reachability at sampled pose!' if n_reachable_wp < len(waypoint_scores) * 0.5 else ''}
      {'⚠️ World-frame z may be too high!' if max(world_z_coords) > 1.3 else ''}
      {'⚠️ World-frame z may be too low!' if min(world_z_coords) < 0.1 else ''}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig


def save_debug_visualization(
    trajectory_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    sampler,  # IntegratedInverseMapSampler or None
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    integrated_scores: np.ndarray,
    output_path: str | Path,
    base_euler: np.ndarray = None,
    z_height: float = None,
    include_per_waypoint: bool = True,
) -> None:
    """High-level function to create and save debug visualization.
    
    Args:
        trajectory_obj: Object-frame trajectory.
        sampler: IntegratedInverseMapSampler instance.
        object_pos: Final object position.
        object_quat: Final object quaternion.
        x_grid, y_grid, theta_grid: Grid arrays.
        integrated_scores: 3D scores array.
        output_path: Path to save GIF.
        base_euler: Base euler angles (optional, derived from quat if not provided).
        z_height: Object z-height (optional, uses object_pos[2] if not provided).
        include_per_waypoint: Whether to include per-waypoint heatmaps.
    """
    # Derive base_euler from quaternion if not provided
    if base_euler is None:
        base_euler = R_scipy.from_quat(object_quat).as_euler('xyz')
    
    # Use object z-position if z_height not provided
    if z_height is None:
        z_height = object_pos[2]
    
    # Create comprehensive GIF
    create_comprehensive_debug_gif(
        trajectory_obj=trajectory_obj,
        sampler=sampler,
        object_pos=object_pos,
        object_quat=object_quat,
        x_grid=x_grid,
        y_grid=y_grid,
        theta_grid=theta_grid,
        integrated_scores=integrated_scores,
        base_euler=base_euler,
        z_height=z_height,
        output_path=output_path,
        fps=2,
        include_per_waypoint=include_per_waypoint,
    )


# Legacy compatibility
def create_debug_gif(
    trajectory_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    waypoint_scores: Sequence[float],
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    theta_grid: np.ndarray,
    integrated_scores: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    output_path: str | Path,
    fps: int = 2,
) -> None:
    """Legacy function for backward compatibility."""
    save_debug_visualization(
        trajectory_obj=trajectory_obj,
        sampler=None,
        object_pos=object_pos,
        object_quat=object_quat,
        x_grid=x_grid,
        y_grid=y_grid,
        theta_grid=theta_grid,
        integrated_scores=integrated_scores,
        output_path=output_path,
        include_per_waypoint=False,
    )
