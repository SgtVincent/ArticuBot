"""Trajectory reachability filtering using RM4D inverse reachability maps.

This module provides utilities for checking whether a given articulated object
manipulation trajectory is reachable given a sampled object-robot base relative
transformation. It uses RM4D's efficient 4D reachability representation to
quickly filter out infeasible object placements.

Key Concepts:
- Object Trajectory: A sequence of end-effector poses (SE3) in the object's local frame.
- Object Pose: The transform from the robot base frame to the object frame (T_bo).
- Base Position Grid: RM4D's inverse map gives candidate base positions for each EE pose.
- Trajectory Reachability: Score computed by intersecting feasible base positions
  across all waypoints in the trajectory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

# Add RM4D to path
import sys
_rm4d_path = Path(__file__).resolve().parent.parent.parent / "third_party" / "rm4d"
if str(_rm4d_path) not in sys.path:
    sys.path.insert(0, str(_rm4d_path))

from rm4d import ReachabilityMap4D, JointSpaceConstructor


@dataclass
class TrajectoryReachabilityFilter:
    """Filter object poses based on trajectory reachability using RM4D.
    
    This class uses RM4D's inverse reachability map to efficiently filter
    candidate object poses. For each waypoint in the object-frame trajectory,
    it computes the set of feasible robot base positions and scores the pose
    based on the intersection of these sets.
    
    Attributes:
        rmap: The RM4D ReachabilityMap4D instance.
        coverage_threshold: Minimum fraction of trajectory waypoints that must
            be reachable (in map bounds and marked reachable).
        intersection_threshold: Minimum intersection score for base positions
            across trajectory waypoints.
    """
    
    rmap: ReachabilityMap4D
    coverage_threshold: float = 0.8
    intersection_threshold: float = 0.1
    
    def transform_trajectory_to_base(
        self,
        T_bo: Tuple[np.ndarray, np.ndarray],
        traj_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> List[np.ndarray]:
        """Transform object-frame trajectory to base-frame 4x4 transforms.
        
        Args:
            T_bo: (R, p) tuple representing transform from base to object frame.
                  R is (3,3) rotation matrix, p is (3,) position vector.
            traj_obj: List of (R, p) tuples for each waypoint in object frame.
            
        Returns:
            List of 4x4 homogeneous transforms for each waypoint in base frame.
        """
        R_bo, p_bo = T_bo
        R_bo = np.asarray(R_bo, dtype=np.float64)
        p_bo = np.asarray(p_bo, dtype=np.float64)
        
        tf_list = []
        for R_o, p_o in traj_obj:
            R_o = np.asarray(R_o, dtype=np.float64)
            p_o = np.asarray(p_o, dtype=np.float64)
            
            # Transform to base frame: T_b = T_bo @ T_o
            R_b = R_bo @ R_o
            p_b = R_bo @ p_o + p_bo
            
            # Build 4x4 transform
            tf = np.eye(4, dtype=np.float64)
            tf[:3, :3] = R_b
            tf[:3, 3] = p_b
            tf_list.append(tf)
            
        return tf_list
    
    def check_single_pose_reachable(self, tf_ee: np.ndarray) -> Tuple[bool, bool]:
        """Check if a single EE pose is reachable using RM4D.
        
        Args:
            tf_ee: 4x4 homogeneous transform for end-effector pose.
            
        Returns:
            Tuple of (in_bounds, is_reachable).
            - in_bounds: True if pose falls within map bounds.
            - is_reachable: True if pose is marked reachable in map.
        """
        try:
            indices = self.rmap.get_indices_for_ee_pose(tf_ee)
            in_bounds = True
            is_reachable = bool(self.rmap.is_reachable(indices))
            return in_bounds, is_reachable
        except IndexError:
            # Pose outside map bounds
            return False, False
    
    def score_trajectory(
        self, 
        traj_base: Sequence[np.ndarray],
    ) -> Tuple[float, float, int]:
        """Score a trajectory based on reachability.
        
        Args:
            traj_base: List of 4x4 transforms in base frame.
            
        Returns:
            Tuple of (coverage_score, intersection_score, n_reachable).
            - coverage_score: Fraction of waypoints that are in bounds.
            - intersection_score: Fraction of waypoints marked reachable in map.
            - n_reachable: Number of waypoints with reachable flag True.
        """
        if len(traj_base) == 0:
            return 0.0, 0.0, 0
        
        n_in_bounds = 0
        n_reachable = 0
        
        for tf_ee in traj_base:
            in_bounds, is_reachable = self.check_single_pose_reachable(tf_ee)
            if in_bounds:
                n_in_bounds += 1
                if is_reachable:
                    n_reachable += 1
        
        coverage = n_in_bounds / len(traj_base)
        # Intersection score: fraction of in-bounds waypoints that are reachable
        intersection = n_reachable / n_in_bounds if n_in_bounds > 0 else 0.0
        
        return coverage, intersection, n_reachable
    
    def filter_pose(
        self,
        T_bo: Tuple[np.ndarray, np.ndarray],
        traj_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[bool, float, float]:
        """Check if an object pose passes the reachability filter.
        
        Args:
            T_bo: (R, p) tuple for object pose in base frame.
            traj_obj: Object-frame trajectory as list of (R, p) tuples.
            
        Returns:
            Tuple of (accepted, coverage_score, intersection_score).
        """
        traj_base = self.transform_trajectory_to_base(T_bo, traj_obj)
        coverage, intersection, _ = self.score_trajectory(traj_base)
        
        accepted = (
            coverage >= self.coverage_threshold and 
            intersection >= self.intersection_threshold
        )
        
        return accepted, coverage, intersection


def load_or_create_rm4d_map(
    map_path: Optional[str | Path] = None,
    robot_type: str = "franka",
    n_samples: int = 1_000_000,
    voxel_res: float = 0.05,
    n_bins_theta: int = 36,
    force_rebuild: bool = False,
) -> ReachabilityMap4D:
    """Load an existing RM4D map or create a new one.
    
    Args:
        map_path: Path to existing map file (.npy). If None, uses default location.
        robot_type: Robot type ('franka' or 'ur5e').
        n_samples: Number of samples for map construction.
        voxel_res: Voxel resolution in meters.
        n_bins_theta: Number of theta angle bins.
        force_rebuild: If True, rebuild even if map exists.
        
    Returns:
        ReachabilityMap4D instance.
    """
    # Default map path
    if map_path is None:
        data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        map_path = data_dir / f"rm4d_{robot_type}_{n_samples // 1_000_000}M.npy"
    else:
        map_path = Path(map_path)
    
    # Try to load existing map
    if map_path.exists() and not force_rebuild:
        return ReachabilityMap4D.from_file(str(map_path))
    
    # Need to create the map - requires simulation
    from rm4d.robots import Simulator, Franka, UR5E
    
    # Set up limits based on robot
    if robot_type == "franka":
        xy_limits = [-1.05, 1.05]
        z_limits = [0.0, 1.35]
    elif robot_type == "ur5e":
        xy_limits = [-1.0, 1.0]
        z_limits = [0.0, 1.2]
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")
    
    # Create map
    rmap = ReachabilityMap4D(
        xy_limits=xy_limits,
        z_limits=z_limits,
        voxel_res=voxel_res,
        n_bins_theta=n_bins_theta,
    )
    
    # Create simulator and robot
    sim = Simulator(with_gui=False)
    if robot_type == "franka":
        robot = Franka(sim)
    else:
        robot = UR5E(sim)
    
    # Build map using joint space sampling
    constructor = JointSpaceConstructor(rmap, robot)
    constructor.sample(n_samples=n_samples, prevent_collisions=True)
    
    # Save map
    map_path.parent.mkdir(parents=True, exist_ok=True)
    rmap.to_file(str(map_path))
    
    # Cleanup
    robot.remove_from_sim()
    
    return rmap


def score_trajectory_reachability(
    rmap: ReachabilityMap4D,
    T_bo: Tuple[np.ndarray, np.ndarray],
    traj_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, float, int]:
    """Score a trajectory's reachability for a given object pose.
    
    Convenience function wrapping TrajectoryReachabilityFilter.
    
    Args:
        rmap: RM4D reachability map.
        T_bo: Object pose (R, p) in base frame.
        traj_obj: Object-frame trajectory as list of (R, p) tuples.
        
    Returns:
        Tuple of (coverage_score, intersection_score, n_reachable).
    """
    filter_obj = TrajectoryReachabilityFilter(rmap=rmap)
    traj_base = filter_obj.transform_trajectory_to_base(T_bo, traj_obj)
    return filter_obj.score_trajectory(traj_base)


def filter_object_pose_by_trajectory(
    rmap: ReachabilityMap4D,
    T_bo: Tuple[np.ndarray, np.ndarray],
    traj_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
    coverage_threshold: float = 0.8,
    intersection_threshold: float = 0.5,
) -> Tuple[bool, float, float]:
    """Check if an object pose is acceptable for the given trajectory.
    
    Args:
        rmap: RM4D reachability map.
        T_bo: Object pose (R, p) in base frame.
        traj_obj: Object-frame trajectory as list of (R, p) tuples.
        coverage_threshold: Minimum trajectory coverage (fraction in bounds).
        intersection_threshold: Minimum reachability (fraction reachable).
        
    Returns:
        Tuple of (accepted, coverage_score, intersection_score).
    """
    filter_obj = TrajectoryReachabilityFilter(
        rmap=rmap,
        coverage_threshold=coverage_threshold,
        intersection_threshold=intersection_threshold,
    )
    return filter_obj.filter_pose(T_bo, traj_obj)


def load_object_trajectory(path: str | Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load an object-frame trajectory from file.
    
    Supports:
    - npz with 'R' (N,3,3) and 'p' (N,3) arrays
    - npz with 'transforms' (N,4,4) array
    
    Args:
        path: Path to trajectory file.
        
    Returns:
        List of (R, p) tuples for each waypoint.
    """
    path = Path(path)
    data = np.load(path)
    
    if 'R' in data and 'p' in data:
        R_arr = np.asarray(data['R'])
        p_arr = np.asarray(data['p'])
    elif 'transforms' in data:
        tfs = np.asarray(data['transforms'])
        R_arr = tfs[:, :3, :3]
        p_arr = tfs[:, :3, 3]
    else:
        raise ValueError(f"Unknown trajectory format in {path}")
    
    traj = []
    for R_i, p_i in zip(R_arr, p_arr):
        traj.append((R_i.astype(np.float64), p_i.astype(np.float64)))
    
    return traj


def generate_straight_line_trajectory(
    start_pos: np.ndarray,
    direction: np.ndarray,
    length: float,
    n_steps: int = 20,
    approach_axis: str = 'z',
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate a simple straight-line trajectory in object frame.
    
    Useful for testing and for simple manipulation tasks.
    
    Args:
        start_pos: Starting position in object frame.
        direction: Unit direction vector for motion.
        length: Total length of trajectory.
        n_steps: Number of waypoints.
        approach_axis: EE approach direction ('x', 'y', 'z', '-x', '-y', '-z').
        
    Returns:
        List of (R, p) tuples.
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    
    # Generate rotation based on approach axis
    if approach_axis == 'z':
        R = np.eye(3)
    elif approach_axis == '-z':
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    elif approach_axis == 'x':
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
    elif approach_axis == '-x':
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    elif approach_axis == 'y':
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    elif approach_axis == '-y':
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown approach axis: {approach_axis}")
    
    traj = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        p = start_pos + t * length * direction
        traj.append((R.copy(), p.astype(np.float64)))
    
    return traj


@dataclass
class IntegratedInverseMapSampler:
    """Sample object poses from integrated inverse reachability map.
    
    This class implements a more efficient sampling approach than rejection 
    sampling. Instead of sampling object poses and checking reachability,
    it integrates the inverse reachability distributions along the full
    trajectory to create a probability distribution over object placements.
    
    Algorithm:
    1. For each candidate object pose (x, y, θ), transform the trajectory 
       to world frame
    2. For each waypoint, query if the EE pose is reachable from base at origin
    3. Aggregate scores across all waypoints using min/product
    4. Normalize to get a probability distribution
    5. Sample object poses from this distribution
    
    This is more efficient because we pre-compute the feasibility map once,
    then sample from it repeatedly without expensive rejection.
    
    Attributes:
        rmap: The RM4D ReachabilityMap4D instance.
        xy_resolution: Resolution for XY grid (meters).
        theta_bins: Number of bins for object orientation around z-axis.
        aggregation_method: How to combine waypoint scores ('min', 'product', or 'mean').
        coverage_threshold: Minimum fraction of waypoints that must be reachable (0-1).
            Only used when aggregation_method='mean'. If set to 0, all poses with 
            at least one reachable waypoint are valid.
    """
    
    rmap: ReachabilityMap4D
    xy_resolution: float = 0.02
    theta_bins: int = 24
    aggregation_method: str = 'mean'  # 'min', 'product', or 'mean'
    coverage_threshold: float = 0.8  # Minimum fraction of waypoints that must be reachable
    
    def compute_object_pose_distribution(
        self,
        traj_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
        xy_center: Tuple[float, float] = (0.4, 0.0),
        xy_range: float = 0.3,
        z_height: float = 0.0,
        base_euler: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute integrated reachability distribution over object poses.
        
        Creates a 3D grid over (x, y, θ) for object placement and computes
        the reachability score for each cell by integrating scores along
        the full trajectory.
        
        Args:
            traj_obj: Object-frame trajectory as list of (R, p) tuples.
            xy_center: Center of the XY sampling region (world frame).
            xy_range: Half-width of XY sampling region.
            z_height: Object z-coordinate (ground plane).
            base_euler: Base euler angles for object (default: identity).
            
        Returns:
            Tuple of (x_grid, y_grid, theta_grid, scores) where:
            - x_grid: 1D array of x coordinates
            - y_grid: 1D array of y coordinates  
            - theta_grid: 1D array of theta values (z-rotation)
            - scores: 3D array [nx, ny, ntheta] of reachability scores
        """
        if base_euler is None:
            base_euler = np.array([0.0, 0.0, 0.0])
        base_euler = np.asarray(base_euler, dtype=np.float64)
        
        # Create grid - round to avoid floating point precision issues
        x_min, x_max = xy_center[0] - xy_range, xy_center[0] + xy_range
        y_min, y_max = xy_center[1] - xy_range, xy_center[1] + xy_range
        
        x_grid = np.arange(x_min, x_max + self.xy_resolution, self.xy_resolution)
        y_grid = np.arange(y_min, y_max + self.xy_resolution, self.xy_resolution)
        theta_grid = np.linspace(-np.pi, np.pi, self.theta_bins, endpoint=False)
        
        # Round grids to avoid floating point precision issues at bin boundaries
        # Use precision slightly higher than the grid resolution
        decimals = int(-np.log10(self.xy_resolution)) + 3  # e.g., 0.05 -> 4 decimals
        x_grid = np.round(x_grid, decimals)
        y_grid = np.round(y_grid, decimals)
        
        nx, ny, ntheta = len(x_grid), len(y_grid), len(theta_grid)
        
        # Pre-compute trajectory waypoints as 4x4 transforms in object frame
        n_waypoints = len(traj_obj)
        traj_obj_tf = np.zeros((n_waypoints, 4, 4), dtype=np.float64)
        for i, (R_o, p_o) in enumerate(traj_obj):
            traj_obj_tf[i, :3, :3] = np.asarray(R_o)
            traj_obj_tf[i, :3, 3] = np.asarray(p_o)
            traj_obj_tf[i, 3, 3] = 1.0
        
        # Use vectorized computation
        scores = self._compute_scores_vectorized(
            x_grid, y_grid, theta_grid, z_height, base_euler, traj_obj_tf
        )
        
        return x_grid, y_grid, theta_grid, scores
    
    def _compute_scores_vectorized(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        theta_grid: np.ndarray,
        z_height: float,
        base_euler: np.ndarray,
        traj_obj_tf: np.ndarray,
    ) -> np.ndarray:
        """Compute scores for all grid cells.

        For each grid cell (object placement), we check if ALL waypoints in the
        trajectory are reachable from the robot at origin. The key insight is that
        we evaluate EVERY waypoint for EVERY grid cell consistently.

        Aggregation methods:
        - 'mean': Score = fraction of reachable waypoints. Apply coverage_threshold.
        - 'min': Score = 1 only if ALL waypoints are reachable (AND logic).
        - 'product': Same as 'min' for binary reachability.

        Out-of-bounds handling:
        - If a waypoint's EE pose falls outside the RM4D map (z or theta out of
          range), that waypoint counts as UNREACHABLE for all grid cells.
        - If the canonical base position (x*, y*) falls outside the map for a
          specific grid cell, that waypoint is UNREACHABLE for that cell.

        This ensures consistent evaluation across all waypoints and all grid cells.
        """
        nx, ny, ntheta = len(x_grid), len(y_grid), len(theta_grid)
        n_waypoints = int(traj_obj_tf.shape[0])

        if n_waypoints == 0:
            return np.zeros((nx, ny, ntheta), dtype=np.float32)

        # Precompute XY mesh for broadcasting (world/base frame).
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="ij")  # (nx, ny)

        # RM4D map parameters
        voxel_res = float(self.rmap.voxel_res)
        xy_min = float(self.rmap.xy_limits[0])
        z_min = float(self.rmap.z_limits[0])
        z_max = float(self.rmap.z_limits[1])
        theta_min = float(self.rmap.theta_limits[0])
        theta_res = float(self.rmap.theta_res)
        n_bins_z = int(self.rmap.n_bins_z)
        n_bins_theta = int(self.rmap.n_bins_theta)
        n_bins_xy = int(self.rmap.n_bins_xy)

        # For 'mean', count reachable waypoints per cell.
        # For 'min'/'product', track if ALL waypoints so far are reachable.
        reachable_counts = np.zeros((nx, ny, ntheta), dtype=np.int32)
        
        # For 'min'/'product', we need to track: has this cell seen any
        # unreachable waypoint? We'll use a "still valid" mask that starts True
        # and becomes False when any waypoint is unreachable.
        all_reachable = np.ones((nx, ny, ntheta), dtype=bool)

        # Pre-compute object rotation matrices for all sampled yaw bins.
        R_bo_all = np.zeros((ntheta, 3, 3), dtype=np.float64)
        for ith, yaw_delta in enumerate(theta_grid):
            euler = base_euler.copy()
            euler[2] += yaw_delta
            R_bo_all[ith] = Rotation.from_euler("xyz", euler).as_matrix()

        # Iterate over waypoints and yaw bins; vectorize over XY grid.
        for wp_idx in range(n_waypoints):
            tf_obj = traj_obj_tf[wp_idx]
            R_o = tf_obj[:3, :3]
            p_o = tf_obj[:3, 3]

            for ith in range(ntheta):
                # Early exit optimization: if all cells are already unreachable
                # for min/product, skip this theta bin.
                if self.aggregation_method != "mean":
                    if not np.any(all_reachable[:, :, ith]):
                        continue

                R_bo = R_bo_all[ith]

                # EE orientation in base frame for this waypoint and sampled yaw.
                R_b = R_bo @ R_o
                rz = R_b[:, 2]
                rz_x, rz_y, rz_z = float(rz[0]), float(rz[1]), float(rz[2])

                # Theta index depends only on EE z-axis tilt.
                theta_val = float(np.arccos(np.clip(rz_z, -1.0, 1.0)))
                if np.isclose(theta_val, np.pi):
                    theta_idx = n_bins_theta - 1
                else:
                    theta_idx = int((theta_val - theta_min) / theta_res)
                
                # Check if theta is out of RM4D bounds
                theta_out_of_bounds = (theta_idx < 0 or theta_idx >= n_bins_theta)

                # EE position offset from object translation (x, y are variable, z fixed).
                p_off = R_bo @ p_o
                p_z = float(z_height + p_off[2])
                z_idx = int((p_z - z_min) / voxel_res)
                
                # Check if z is out of RM4D bounds
                z_out_of_bounds = (p_z < z_min or p_z >= z_max or 
                                   z_idx < 0 or z_idx >= n_bins_z)

                # If theta or z is out of bounds, this waypoint is unreachable
                # for ALL grid cells in this theta bin.
                if theta_out_of_bounds or z_out_of_bounds:
                    # For min/product: mark all cells as having an unreachable waypoint
                    if self.aggregation_method != "mean":
                        all_reachable[:, :, ith] = False
                    # For mean: reachable_counts stays 0 for this waypoint (already 0)
                    continue

                # 2D rotation used by RM4D canonical base position.
                psi = float(np.arctan2(rz_y, rz_x))
                cpsi = float(np.cos(psi))
                spsi = float(np.sin(psi))

                # EE position in base frame for all XY grid points.
                p_x = x_mesh + float(p_off[0])
                p_y = y_mesh + float(p_off[1])

                # Canonical base position (x*, y*) = rot2d @ [-p_x, -p_y]
                x_star = -(cpsi * p_x + spsi * p_y)
                y_star = (spsi * p_x - cpsi * p_y)

                x_idx = ((x_star - xy_min) / voxel_res).astype(np.int32)
                y_idx = ((y_star - xy_min) / voxel_res).astype(np.int32)

                # Check which cells have valid (x*, y*) indices
                valid_xy = (
                    (x_idx >= 0)
                    & (x_idx < n_bins_xy)
                    & (y_idx >= 0)
                    & (y_idx < n_bins_xy)
                )

                # Look up reachability in RM4D map
                map_slice = self.rmap.map[z_idx, theta_idx]  # (n_bins_xy, n_bins_xy)
                
                # Initialize reachable to False (unreachable by default)
                reachable = np.zeros((nx, ny), dtype=bool)
                
                # Only cells with valid indices can be reachable
                if np.any(valid_xy):
                    reachable[valid_xy] = map_slice[x_idx[valid_xy], y_idx[valid_xy]]

                # Update aggregation
                if self.aggregation_method == "mean":
                    # Count reachable waypoints
                    reachable_counts[:, :, ith] += reachable.astype(np.int32)
                else:
                    # For min/product: a cell becomes unreachable if ANY waypoint 
                    # is unreachable. Use AND logic.
                    all_reachable[:, :, ith] &= reachable

        # Compute final scores
        if self.aggregation_method == "mean":
            scores = reachable_counts.astype(np.float32) / float(n_waypoints)
            scores[scores < float(self.coverage_threshold)] = 0.0
        else:
            # For min/product: score is 1.0 if all waypoints reachable, else 0.0
            scores = all_reachable.astype(np.float32)

        return scores
    
    def _score_object_pose(
        self,
        T_wo: np.ndarray,
        traj_obj_tf: List[np.ndarray],
    ) -> float:
        """Score a single object pose based on trajectory reachability.
        
        Args:
            T_wo: 4x4 object pose in world frame.
            traj_obj_tf: List of 4x4 transforms in object frame.
            
        Returns:
            Reachability score (0 to 1).
        """
        waypoint_scores = []
        
        for tf_obj in traj_obj_tf:
            # Transform to world frame: T_world = T_wo @ T_obj
            tf_world = T_wo @ tf_obj
            
            # Check reachability using RM4D
            try:
                indices = self.rmap.get_indices_for_ee_pose(tf_world)
                is_reachable = float(self.rmap.is_reachable(indices))
                waypoint_scores.append(is_reachable)
            except IndexError:
                # Pose outside map bounds
                waypoint_scores.append(0.0)
        
        if len(waypoint_scores) == 0:
            return 0.0
        
        # Aggregate scores
        if self.aggregation_method == 'min':
            return min(waypoint_scores)
        elif self.aggregation_method == 'product':
            score = 1.0
            for s in waypoint_scores:
                score *= s
            return score
        else:
            # Default to mean
            return np.mean(waypoint_scores)
    
    def sample_object_pose(
        self,
        scores: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        theta_grid: np.ndarray,
        z_height: float = 0.0,
        base_euler: np.ndarray = None,
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Sample an object pose from the distribution.
        
        Args:
            scores: 3D array of reachability scores.
            x_grid: X coordinate grid.
            y_grid: Y coordinate grid.
            theta_grid: Theta (z-rotation) grid.
            z_height: Object z-coordinate.
            base_euler: Base euler angles for object.
            temperature: Sampling temperature (higher = more uniform).
            
        Returns:
            Tuple of (position, quaternion, score) for sampled pose.
        """
        if base_euler is None:
            base_euler = np.array([0.0, 0.0, 0.0])
        base_euler = np.asarray(base_euler, dtype=np.float64)
        
        # Flatten scores for sampling
        flat_scores = scores.flatten()
        
        # Apply temperature and normalize to probability distribution
        if np.max(flat_scores) > 0:
            probs = np.power(flat_scores, 1.0 / temperature)
            probs = probs / np.sum(probs)
        else:
            # No valid poses, return failure
            return np.zeros(3), np.array([0, 0, 0, 1]), 0.0
        
        # Sample index
        flat_idx = np.random.choice(len(probs), p=probs)
        
        # Convert flat index to 3D indices
        nx, ny, ntheta = scores.shape
        ix = flat_idx // (ny * ntheta)
        iy = (flat_idx % (ny * ntheta)) // ntheta
        ith = flat_idx % ntheta
        
        # Get sampled values
        x = x_grid[ix]
        y = y_grid[iy]
        theta = theta_grid[ith]
        score = scores[ix, iy, ith]
        
        # Construct pose
        position = np.array([x, y, z_height], dtype=np.float64)
        euler = base_euler.copy()
        euler[2] += theta
        quaternion = Rotation.from_euler('xyz', euler).as_quat()
        
        return position, quaternion, score
    
    def sample_object_poses_batch(
        self,
        scores: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        theta_grid: np.ndarray,
        n_samples: int = 10,
        z_height: float = 0.0,
        base_euler: np.ndarray = None,
        temperature: float = 1.0,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Sample multiple object poses from the distribution.
        
        Args:
            scores: 3D array of reachability scores.
            x_grid, y_grid, theta_grid: Coordinate grids.
            n_samples: Number of poses to sample.
            z_height: Object z-coordinate.
            base_euler: Base euler angles for object.
            temperature: Sampling temperature.
            
        Returns:
            List of (position, quaternion, score) tuples.
        """
        if base_euler is None:
            base_euler = np.array([0.0, 0.0, 0.0])
        base_euler = np.asarray(base_euler, dtype=np.float64)
        
        # Flatten and create probability distribution
        flat_scores = scores.flatten()
        
        if np.max(flat_scores) == 0:
            return []
        
        probs = np.power(flat_scores, 1.0 / temperature)
        probs = probs / np.sum(probs)
        
        # Sample indices
        flat_indices = np.random.choice(len(probs), size=n_samples, p=probs)
        
        nx, ny, ntheta = scores.shape
        results = []
        
        for flat_idx in flat_indices:
            ix = flat_idx // (ny * ntheta)
            iy = (flat_idx % (ny * ntheta)) // ntheta
            ith = flat_idx % ntheta
            
            x, y, theta = x_grid[ix], y_grid[iy], theta_grid[ith]
            score = scores[ix, iy, ith]
            
            position = np.array([x, y, z_height], dtype=np.float64)
            euler = base_euler.copy()
            euler[2] += theta
            quaternion = Rotation.from_euler('xyz', euler).as_quat()
            
            results.append((position, quaternion, score))
        
        return results
    
    def get_best_object_pose(
        self,
        scores: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        theta_grid: np.ndarray,
        z_height: float = 0.0,
        base_euler: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get the object pose with highest reachability score.
        
        Args:
            scores: 3D array of reachability scores.
            x_grid, y_grid, theta_grid: Coordinate grids.
            z_height: Object z-coordinate.
            base_euler: Base euler angles for object.
            
        Returns:
            Tuple of (position, quaternion, score) for best pose.
        """
        if base_euler is None:
            base_euler = np.array([0.0, 0.0, 0.0])
        base_euler = np.asarray(base_euler, dtype=np.float64)
        
        # Find argmax
        flat_idx = np.argmax(scores)
        
        nx, ny, ntheta = scores.shape
        ix = flat_idx // (ny * ntheta)
        iy = (flat_idx % (ny * ntheta)) // ntheta
        ith = flat_idx % ntheta
        
        x, y, theta = x_grid[ix], y_grid[iy], theta_grid[ith]
        score = scores[ix, iy, ith]
        
        position = np.array([x, y, z_height], dtype=np.float64)
        euler = base_euler.copy()
        euler[2] += theta
        quaternion = Rotation.from_euler('xyz', euler).as_quat()
        
        return position, quaternion, score


def create_integrated_inverse_sampler(
    rmap: ReachabilityMap4D,
    xy_resolution: float = 0.02,
    theta_bins: int = 24,
    aggregation_method: str = 'mean',
    coverage_threshold: float = 0.8,
) -> IntegratedInverseMapSampler:
    """Create an integrated inverse map sampler.
    
    Convenience function for creating the sampler.
    
    Args:
        rmap: RM4D reachability map.
        xy_resolution: Grid resolution in meters.
        theta_bins: Number of orientation bins.
        aggregation_method: 'min', 'product', or 'mean'.
        coverage_threshold: Minimum fraction of reachable waypoints (for 'mean').
        
    Returns:
        IntegratedInverseMapSampler instance.
    """
    return IntegratedInverseMapSampler(
        rmap=rmap,
        xy_resolution=xy_resolution,
        theta_bins=theta_bins,
        aggregation_method=aggregation_method,
        coverage_threshold=coverage_threshold,
    )
