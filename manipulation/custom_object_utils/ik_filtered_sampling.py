"""IK-filtered initial state sampling for efficient demo generation.

This module implements the new algorithm for sampling object initial states
using inverse kinematics (IK) map filtering. The key insight is:

1. Compute the in-contact gripper trajectory in object frame
2. Use RM4D inverse reachability map to score candidate object placements
3. Only attempt motion planning for placements that pass the IK filter

This significantly reduces wasted attempts compared to the heuristic method
which samples randomly and only discovers infeasible placements after
expensive motion planning fails.
"""
from __future__ import annotations

import time
import pathlib
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
from typing import Optional, Tuple, List

from manipulation.custom_object_utils.contact_trajectory import (
    parse_joint_kinematics_from_urdf,
    parse_joint_kinematics_from_mobility,
    compute_full_manipulation_trajectory,
)

from manipulation.rm4d_filtering import (
    TrajectoryReachabilityFilter,
    generate_straight_line_trajectory,
)


@dataclass
class SamplingConfig:
    """Configuration for IK-filtered sampling."""
    # Object placement bounds
    target_position: Tuple[float, float, float] = (0.4, 0.0, 0.0)
    position_jitter: float = 0.1
    orientation_jitter: float = np.pi / 6  # radians around z-axis
    
    # IK filtering thresholds
    coverage_threshold: float = 0.8
    reachability_threshold: float = 0.5
    
    # Robot arm workspace bounds (for distance checks)
    near_distance: float = 0.15
    far_distance: float = 0.5
    
    # Trajectory parameters
    approach_distance: float = 0.15
    target_ratio: float = 0.8
    approach_steps: int = 20
    contact_steps: int = 50
    
    # Sampling limits
    max_pose_samples: int = 100
    max_robot_samples: int = 100
    timeout: float = 60.0


def sample_object_pose(
    base_position: Tuple[float, float, float],
    base_z: float,
    base_euler: np.ndarray,
    config: SamplingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a random object pose around the target position.
    
    Args:
        base_position: Target (x, y, z) position.
        base_z: Object z-coordinate after placement on ground.
        base_euler: Base euler angles of object.
        config: Sampling configuration.
        
    Returns:
        Tuple of (position, quaternion).
    """
    pos = np.array([
        base_position[0] + np.random.uniform(-config.position_jitter, config.position_jitter),
        base_position[1] + np.random.uniform(-config.position_jitter, config.position_jitter),
        base_z,
    ])
    
    euler = np.array([
        base_euler[0],
        base_euler[1],
        base_euler[2] + np.random.uniform(-config.orientation_jitter, config.orientation_jitter),
    ])
    
    rot = R.from_euler('xyz', euler)
    quat = rot.as_quat()  # x, y, z, w
    
    return pos, quat


def filter_pose_with_trajectory(
    object_pos: np.ndarray,
    object_quat: np.ndarray,
    trajectory_obj: List[Tuple[np.ndarray, np.ndarray]],
    traj_filter: TrajectoryReachabilityFilter,
) -> Tuple[bool, float, float]:
    """Check if object pose passes the trajectory reachability filter.
    
    Args:
        object_pos: Object position in world frame.
        object_quat: Object quaternion in world frame.
        trajectory_obj: Gripper trajectory in object frame.
        traj_filter: RM4D trajectory filter.
        
    Returns:
        Tuple of (accepted, coverage, reachability).
    """
    R_bo = R.from_quat(object_quat).as_matrix()
    T_bo = (R_bo, object_pos)
    
    return traj_filter.filter_pose(T_bo, trajectory_obj)


def transform_grasp_to_world(
    grasp_pos_obj: np.ndarray,
    grasp_quat_obj: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform grasp pose from object frame to world frame.
    
    Args:
        grasp_pos_obj: Grasp position in object frame.
        grasp_quat_obj: Grasp quaternion in object frame.
        object_pos: Object position in world frame.
        object_quat: Object quaternion in world frame.
        
    Returns:
        Tuple of (world_pos, world_quat).
    """
    r_obj = R.from_quat(object_quat)
    world_pos = r_obj.apply(grasp_pos_obj) + object_pos
    
    r_grasp = R.from_quat(grasp_quat_obj)
    r_world = r_obj * r_grasp
    world_quat = r_world.as_quat()
    
    return world_pos, world_quat


def check_robot_configuration(
    env,
    object_id: int,
    grasp_pos_world: np.ndarray,
    config: SamplingConfig,
) -> Tuple[bool, np.ndarray]:
    """Sample valid robot configuration for reaching the grasp.
    
    Args:
        env: PyBullet environment.
        object_id: Object body ID.
        grasp_pos_world: Target grasp position in world frame.
        config: Sampling configuration.
        
    Returns:
        Tuple of (success, joint_angles).
    """
    # Robot arm joint limits (reduced range for safety)
    low = [-2.9, -1.8, -2.9, -3.1, -2.9, -0.0, -2.9]
    high = [2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9]
    
    # Reduce range by 20%
    for i in range(7):
        joint_range = high[i] - low[i]
        low[i] += joint_range * 0.2
        high[i] -= joint_range * 0.2
    
    for _ in range(config.max_robot_samples):
        # Sample random joint angles
        joint_angles = [np.random.uniform(low[i], high[i]) for i in range(7)]
        env.robot.set_joint_angles(env.robot.right_arm_joint_indices, joint_angles)
        
        for _ in range(5):
            p.stepSimulation(physicsClientId=env.id)
        
        # Check collisions
        contact_points = p.getContactPoints(env.robot.body, object_id, physicsClientId=env.id)
        closest_points = p.getClosestPoints(env.robot.body, object_id, distance=0.01, physicsClientId=env.id)
        if len(contact_points) > 0 or len(closest_points) > 0:
            continue
        
        # Check link-ground collisions
        link_collision = False
        num_links = p.getNumJoints(env.robot.body, physicsClientId=env.id)
        for link_idx in range(1, num_links):
            contact = p.getClosestPoints(
                bodyA=env.robot.body, linkIndexA=link_idx,
                bodyB=env.urdf_ids['plane'], distance=0.01,
                physicsClientId=env.id
            )
            if len(contact) > 0:
                link_collision = True
                break
        
        if link_collision:
            continue
        
        # Check EEF distance to grasp
        robot_eef_pos, _ = env.robot.get_pos_orient(env.robot.right_end_effector)
        distance = np.linalg.norm(np.array(grasp_pos_world) - np.array(robot_eef_pos))
        
        if config.near_distance < distance < config.far_distance:
            return True, np.array(joint_angles)
    
    return False, np.zeros(7)


def ik_filtered_sample_initial_state(
    env,
    object_name: str,
    grasp_pos_obj: np.ndarray,
    grasp_quat_obj: np.ndarray,
    trajectory_obj: List[Tuple[np.ndarray, np.ndarray]],
    traj_filter: Optional[TrajectoryReachabilityFilter],
    base_euler: np.ndarray,
    config: SamplingConfig,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Sample initial state using IK-filtered approach.
    
    This implements the new algorithm:
    1. Sample candidate object poses
    2. Filter with trajectory reachability (if available)
    3. Check robot configuration feasibility
    
    Args:
        env: PyBullet environment.
        object_name: Name of the object in env.urdf_ids.
        grasp_pos_obj: Grasp position in object frame.
        grasp_quat_obj: Grasp quaternion in object frame.
        trajectory_obj: Full manipulation trajectory in object frame.
        traj_filter: Optional RM4D trajectory filter.
        base_euler: Base euler angles for object orientation.
        config: Sampling configuration.
        
    Returns:
        Tuple of (success, object_pos, object_quat, robot_joints).
    """
    object_id = env.urdf_ids[object_name]
    
    # Get object's z-coordinate after placement (already computed by env.reset)
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    object_z = init_pos[2]
    
    start_time = time.time()
    
    for pose_idx in range(config.max_pose_samples):
        if time.time() - start_time > config.timeout:
            print(f"IK-filtered sampling timed out after {config.timeout}s")
            break
        
        # Sample object pose
        obj_pos, obj_quat = sample_object_pose(
            config.target_position, object_z, base_euler, config
        )
        
        # Apply to simulation
        p.resetBasePositionAndOrientation(object_id, obj_pos, obj_quat, physicsClientId=env.id)
        
        # Filter with trajectory reachability
        if traj_filter is not None:
            accepted, coverage, reachability = filter_pose_with_trajectory(
                obj_pos, obj_quat, trajectory_obj, traj_filter
            )
            if not accepted:
                continue
            print(f"  Pose {pose_idx}: passed IK filter (cov={coverage:.2f}, reach={reachability:.2f})")
        
        # Transform grasp to world frame
        grasp_world_pos, grasp_world_quat = transform_grasp_to_world(
            grasp_pos_obj, grasp_quat_obj, obj_pos, obj_quat
        )
        
        # Check robot configuration
        success, joint_angles = check_robot_configuration(
            env, object_id, grasp_world_pos, config
        )
        
        if success:
            print(f"  Found valid configuration at pose {pose_idx}")
            return True, obj_pos, obj_quat, joint_angles
    
    print(f"Failed to find valid initial state after {config.max_pose_samples} pose samples")
    return False, np.zeros(3), np.zeros(4), np.zeros(7)


def load_or_compute_trajectory(
    grasp_pose: np.ndarray,
    asset_dir: pathlib.Path,
    scale: float = 1.0,
    approach_distance: float = 0.15,
    target_ratio: float = 0.8,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load cached trajectory or compute from joint kinematics.
    
    Args:
        grasp_pose: (4,4) grasp pose in object frame.
        asset_dir: Asset directory containing URDF/mobility files.
        scale: Object scale factor.
        approach_distance: Approach distance for pre-grasp.
        target_ratio: Target manipulation ratio.
        
    Returns:
        Full manipulation trajectory in object frame.
    """
    # Try to load joint kinematics from mobility or URDF
    mobility_path = asset_dir / "mobility_v2.json"
    urdf_files = list(asset_dir.glob("*.urdf"))
    
    joint_kin = None
    if mobility_path.exists():
        try:
            joint_kin = parse_joint_kinematics_from_mobility(mobility_path)
        except Exception:
            pass  # Will try URDF next
    
    if joint_kin is None and urdf_files:
        try:
            joint_kin = parse_joint_kinematics_from_urdf(urdf_files[0])
        except Exception:
            pass  # Will use fallback
    
    if joint_kin is None:
        # Fall back to straight line approach trajectory
        grasp_p = grasp_pose[:3, 3] * scale
        return generate_straight_line_trajectory(
            grasp_p,
            direction=np.array([0, 0, -1]),
            length=approach_distance,
            n_steps=20,
            approach_axis='z',
        )
    
    return compute_full_manipulation_trajectory(
        grasp_pose,
        joint_kin,
        approach_distance=approach_distance,
        target_ratio=target_ratio,
        scale=scale,
    )
