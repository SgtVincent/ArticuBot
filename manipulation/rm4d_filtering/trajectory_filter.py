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
