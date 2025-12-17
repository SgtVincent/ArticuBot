"""Compute in-contact gripper trajectories in object frame for articulated objects.

This module computes the expected gripper trajectory when manipulating an
articulated object (e.g., opening a door/drawer). The trajectory is computed
in the object's local coordinate frame, which allows it to be used for:

1. Pre-filtering object placements using inverse reachability maps (RM4D)
2. Motion planning constraints
3. Verification of manipulation feasibility

The key insight is that once a grasp pose is selected, the gripper's motion
during manipulation is determined by the object's joint kinematics.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET


@dataclass
class JointKinematics:
    """Kinematic parameters for an articulated joint."""
    joint_name: str
    joint_type: str  # 'revolute', 'prismatic', 'continuous'
    axis_origin: np.ndarray  # (3,) joint axis origin in object frame
    axis_direction: np.ndarray  # (3,) normalized axis direction
    limit_lower: float
    limit_upper: float
    parent_link: str
    child_link: str


def parse_joint_kinematics_from_urdf(urdf_path: str | Path, joint_name: Optional[str] = None) -> JointKinematics:
    """Parse joint kinematics from a URDF file.
    
    Args:
        urdf_path: Path to URDF file.
        joint_name: Optional specific joint name. If None, uses first non-fixed joint.
        
    Returns:
        JointKinematics dataclass with parsed parameters.
    """
    path = Path(urdf_path)
    tree = ET.parse(path)
    root = tree.getroot()
    
    for joint in root.iter("joint"):
        j_type = joint.get("type", "fixed").lower()
        if j_type not in {"revolute", "continuous", "prismatic"}:
            continue
        if joint_name and joint.get("name") != joint_name:
            continue
            
        parent_tag = joint.find("parent")
        child_tag = joint.find("child")
        if parent_tag is None or child_tag is None:
            continue
            
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        
        # Parse origin xyz
        xyz = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz_str = origin.get("xyz")
            if xyz_str:
                xyz = [float(x) for x in xyz_str.split()]
        
        # Parse axis direction
        axis_dir = [0.0, 0.0, 1.0]  # Default Z-axis
        if axis is not None:
            axis_str = axis.get("xyz")
            if axis_str:
                axis_dir = [float(x) for x in axis_str.split()]
        axis_dir = np.array(axis_dir)
        if np.linalg.norm(axis_dir) > 0:
            axis_dir = axis_dir / np.linalg.norm(axis_dir)
            
        # Parse limits
        lower = -0.5
        upper = 0.5
        if limit is not None:
            lower = float(limit.get("lower", "-0.5"))
            upper = float(limit.get("upper", "0.5"))
            
        return JointKinematics(
            joint_name=joint.get("name", "joint"),
            joint_type=j_type,
            axis_origin=np.array(xyz),
            axis_direction=axis_dir,
            limit_lower=lower,
            limit_upper=upper,
            parent_link=parent_tag.get("link", "base"),
            child_link=child_tag.get("link", "child"),
        )
    
    raise RuntimeError(f"No suitable joint found in {urdf_path}")


def parse_joint_kinematics_from_mobility(mobility_path: str | Path) -> JointKinematics:
    """Parse joint kinematics from mobility_v2.json.
    
    Args:
        mobility_path: Path to mobility_v2.json file.
        
    Returns:
        JointKinematics dataclass with parsed parameters.
    """
    path = Path(mobility_path)
    with open(path, "r") as f:
        data = json.load(f)
    
    for part in data:
        if part.get("joint") in {"hinge", "slider"}:
            joint_data = part.get("jointData", {})
            axis_data = joint_data.get("axis", {})
            limit_data = joint_data.get("limit", {})
            
            return JointKinematics(
                joint_name=part.get("name", "joint"),
                joint_type="revolute" if part["joint"] == "hinge" else "prismatic",
                axis_origin=np.array(axis_data.get("origin", [0, 0, 0])),
                axis_direction=np.array(axis_data.get("direction", [0, 0, 1])),
                limit_lower=float(limit_data.get("a", -0.5)),
                limit_upper=float(limit_data.get("b", 0.5)),
                parent_link=str(part.get("parent", 0)),
                child_link=str(part.get("id", 1)),
            )
    
    raise RuntimeError(f"No hinge/slider joint found in {mobility_path}")


def compute_revolute_transform(
    axis_origin: np.ndarray,
    axis_direction: np.ndarray,
    angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the transform for a revolute joint at given angle.
    
    Args:
        axis_origin: (3,) joint axis origin point.
        axis_direction: (3,) normalized axis direction.
        angle: Joint angle in radians.
        
    Returns:
        Tuple of (R, t) where R is (3,3) rotation and t is (3,) translation.
    """
    # Rodrigues' rotation formula
    rot = R.from_rotvec(axis_direction * angle)
    R_mat = rot.as_matrix()
    
    # Translation due to rotation about a point not at origin
    # t = axis_origin - R @ axis_origin
    t = axis_origin - R_mat @ axis_origin
    
    return R_mat, t


def compute_prismatic_transform(
    axis_direction: np.ndarray,
    distance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the transform for a prismatic joint at given distance.
    
    Args:
        axis_direction: (3,) normalized axis direction.
        distance: Joint displacement.
        
    Returns:
        Tuple of (R, t) where R is identity and t is (3,) translation.
    """
    R_mat = np.eye(3)
    t = axis_direction * distance
    return R_mat, t


def transform_pose(
    grasp_R: np.ndarray,
    grasp_p: np.ndarray,
    joint_R: np.ndarray,
    joint_t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply joint transform to a grasp pose.
    
    Args:
        grasp_R: (3,3) grasp rotation in object frame.
        grasp_p: (3,) grasp position in object frame.
        joint_R: (3,3) joint rotation.
        joint_t: (3,) joint translation.
        
    Returns:
        Tuple of (R_new, p_new) transformed pose.
    """
    # The grasp is attached to the child link, which moves with the joint
    R_new = joint_R @ grasp_R
    p_new = joint_R @ grasp_p + joint_t
    return R_new, p_new


def compute_contact_trajectory_revolute(
    grasp_R: np.ndarray,
    grasp_p: np.ndarray,
    joint_kin: JointKinematics,
    start_angle: float = 0.0,
    target_ratio: float = 0.8,
    n_steps: int = 50,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute gripper trajectory for revolute joint manipulation.
    
    The trajectory represents the gripper poses in object frame as the
    door/lid is opened from start_angle to target_ratio of the joint range.
    
    Args:
        grasp_R: (3,3) initial grasp rotation in object frame.
        grasp_p: (3,) initial grasp position in object frame.
        joint_kin: Joint kinematics parameters.
        start_angle: Starting joint angle (radians).
        target_ratio: Target opening ratio (0.0 to 1.0).
        n_steps: Number of trajectory waypoints.
        
    Returns:
        List of (R, p) tuples for each waypoint.
    """
    target_angle = joint_kin.limit_lower + target_ratio * (joint_kin.limit_upper - joint_kin.limit_lower)
    
    trajectory = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        angle = start_angle + t * (target_angle - start_angle)
        
        # Compute joint transform relative to start
        delta_angle = angle - start_angle
        joint_R, joint_t = compute_revolute_transform(
            joint_kin.axis_origin,
            joint_kin.axis_direction,
            delta_angle,
        )
        
        # Apply to grasp pose
        R_new, p_new = transform_pose(grasp_R, grasp_p, joint_R, joint_t)
        trajectory.append((R_new.copy(), p_new.copy()))
    
    return trajectory


def compute_contact_trajectory_prismatic(
    grasp_R: np.ndarray,
    grasp_p: np.ndarray,
    joint_kin: JointKinematics,
    start_position: float = 0.0,
    target_ratio: float = 0.8,
    n_steps: int = 50,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute gripper trajectory for prismatic joint manipulation.
    
    Args:
        grasp_R: (3,3) initial grasp rotation in object frame.
        grasp_p: (3,) initial grasp position in object frame.
        joint_kin: Joint kinematics parameters.
        start_position: Starting joint position.
        target_ratio: Target opening ratio (0.0 to 1.0).
        n_steps: Number of trajectory waypoints.
        
    Returns:
        List of (R, p) tuples for each waypoint.
    """
    target_position = joint_kin.limit_lower + target_ratio * (joint_kin.limit_upper - joint_kin.limit_lower)
    
    trajectory = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        position = start_position + t * (target_position - start_position)
        
        # Compute joint transform relative to start
        delta_position = position - start_position
        joint_R, joint_t = compute_prismatic_transform(
            joint_kin.axis_direction,
            delta_position,
        )
        
        # Apply to grasp pose
        R_new, p_new = transform_pose(grasp_R, grasp_p, joint_R, joint_t)
        trajectory.append((R_new.copy(), p_new.copy()))
    
    return trajectory


def compute_contact_trajectory(
    grasp_pose: np.ndarray,
    joint_kin: JointKinematics,
    start_state: float = 0.0,
    target_ratio: float = 0.8,
    n_steps: int = 50,
    scale: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute in-contact gripper trajectory in object frame.
    
    This is the main entry point for computing manipulation trajectories.
    Given a grasp pose and joint kinematics, computes the expected gripper
    trajectory during manipulation.
    
    Args:
        grasp_pose: (4,4) homogeneous grasp pose in object frame.
        joint_kin: Joint kinematics parameters.
        start_state: Starting joint state (angle or position).
        target_ratio: Target manipulation ratio (0.0 to 1.0).
        n_steps: Number of trajectory waypoints.
        scale: Object scale factor (applies to positions).
        
    Returns:
        List of (R, p) tuples for each waypoint in object frame.
    """
    # Extract rotation and position from grasp pose
    grasp_R = grasp_pose[:3, :3].copy()
    grasp_p = grasp_pose[:3, 3].copy() * scale
    
    # Scale joint axis origin
    scaled_kin = JointKinematics(
        joint_name=joint_kin.joint_name,
        joint_type=joint_kin.joint_type,
        axis_origin=joint_kin.axis_origin * scale,
        axis_direction=joint_kin.axis_direction.copy(),
        limit_lower=joint_kin.limit_lower,
        limit_upper=joint_kin.limit_upper,
        parent_link=joint_kin.parent_link,
        child_link=joint_kin.child_link,
    )
    
    if joint_kin.joint_type in {"revolute", "continuous"}:
        return compute_contact_trajectory_revolute(
            grasp_R, grasp_p, scaled_kin, start_state, target_ratio, n_steps
        )
    elif joint_kin.joint_type == "prismatic":
        return compute_contact_trajectory_prismatic(
            grasp_R, grasp_p, scaled_kin, start_state, target_ratio, n_steps
        )
    else:
        raise ValueError(f"Unknown joint type: {joint_kin.joint_type}")


def compute_approach_trajectory(
    grasp_pose: np.ndarray,
    approach_distance: float = 0.15,
    n_steps: int = 20,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute approach trajectory before grasping.
    
    The approach trajectory moves along the gripper's z-axis (approach direction).
    
    Args:
        grasp_pose: (4,4) target grasp pose.
        approach_distance: Distance to back off for approach.
        n_steps: Number of trajectory waypoints.
        
    Returns:
        List of (R, p) tuples from pre-grasp to grasp pose.
    """
    grasp_R = grasp_pose[:3, :3]
    grasp_p = grasp_pose[:3, 3]
    
    # Approach direction is negative z-axis of gripper frame
    approach_dir = grasp_R[:, 2]  # z-axis
    
    trajectory = []
    for i in range(n_steps):
        t = i / max(n_steps - 1, 1)
        # Move from far to close
        offset = approach_distance * (1 - t)
        p_new = grasp_p - approach_dir * offset
        trajectory.append((grasp_R.copy(), p_new.copy()))
    
    return trajectory


def combine_trajectories(
    *trajectories: List[Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Combine multiple trajectory segments into one.
    
    Args:
        *trajectories: Variable number of trajectory lists.
        
    Returns:
        Combined trajectory list.
    """
    combined = []
    for traj in trajectories:
        combined.extend(traj)
    return combined


def save_trajectory_npz(
    trajectory: List[Tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
) -> None:
    """Save trajectory to npz format for use with RM4D filtering.
    
    Args:
        trajectory: List of (R, p) tuples.
        output_path: Path to save npz file.
    """
    R_arr = np.array([t[0] for t in trajectory])
    p_arr = np.array([t[1] for t in trajectory])
    np.savez(output_path, R=R_arr, p=p_arr)


def load_trajectory_npz(input_path: str | Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load trajectory from npz format.
    
    Args:
        input_path: Path to npz file.
        
    Returns:
        List of (R, p) tuples.
    """
    data = np.load(input_path)
    R_arr = data["R"]
    p_arr = data["p"]
    return [(R_arr[i], p_arr[i]) for i in range(len(R_arr))]


def compute_full_manipulation_trajectory(
    grasp_pose: np.ndarray,
    joint_kin: JointKinematics,
    approach_distance: float = 0.15,
    start_state: float = 0.0,
    target_ratio: float = 0.8,
    approach_steps: int = 20,
    contact_steps: int = 50,
    scale: float = 1.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Compute full manipulation trajectory: approach + contact.
    
    Args:
        grasp_pose: (4,4) grasp pose in object frame.
        joint_kin: Joint kinematics parameters.
        approach_distance: Distance to back off for approach.
        start_state: Starting joint state.
        target_ratio: Target manipulation ratio.
        approach_steps: Number of approach waypoints.
        contact_steps: Number of contact (manipulation) waypoints.
        scale: Object scale factor.
        
    Returns:
        Combined trajectory list.
    """
    # Scale grasp position
    scaled_grasp = grasp_pose.copy()
    scaled_grasp[:3, 3] *= scale
    
    approach_traj = compute_approach_trajectory(
        scaled_grasp, approach_distance, approach_steps
    )
    
    contact_traj = compute_contact_trajectory(
        grasp_pose, joint_kin, start_state, target_ratio, contact_steps, scale
    )
    
    return combine_trajectories(approach_traj, contact_traj)

