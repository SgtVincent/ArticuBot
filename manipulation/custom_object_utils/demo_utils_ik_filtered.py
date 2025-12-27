"""IK-filtered demo utilities for the new sampling algorithm.

This module provides process-wrapped IK-filtered initial state generation,
implementing the new algorithm that:
1. Computes the in-contact trajectory in object frame
2. Uses RM4D inverse reachability maps to filter object placements
3. Only attempts motion planning for filtered placements
"""
from __future__ import annotations

import multiprocessing as mp
import pathlib
import numpy as np
import pybullet as p
import yaml
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, List

from manipulation.utils import build_up_env_gen, parse_center

from manipulation.custom_object_utils.ik_filtered_sampling import (
    SamplingConfig,
    ik_filtered_sample_initial_state,
    load_or_compute_trajectory,
)
from manipulation.rm4d_filtering import (
    TrajectoryReachabilityFilter,
    load_or_create_rm4d_map,
)


def _custom_gen_init_state_ik_filtered(
    q,
    config_path: str,
    env_name: str,
    render: bool,
    far_distance: float = 0.7,
    near_distance: float = 0.3,
    rm4d_map_path: Optional[str] = None,
    coverage_threshold: float = 0.8,
    reachability_threshold: float = 0.5,
    approach_distance: float = 0.15,
    target_ratio: float = 0.8,
    asset_dir: Optional[str] = None,
):
    """Generate initial state using IK-filtered sampling algorithm.
    
    This implements the new algorithm:
    1. Load grasp from config
    2. Compute in-contact trajectory in object frame
    3. Load RM4D map and create trajectory filter
    4. Sample object poses with IK filtering
    5. Save valid configuration to config
    
    Args:
        q: Multiprocessing queue for returning result.
        config_path: Path to variant config YAML.
        env_name: Environment name for simulation.
        render: Whether to enable GUI rendering.
        far_distance: Max EEF-to-grasp distance.
        near_distance: Min EEF-to-grasp distance.
        rm4d_map_path: Path to RM4D reachability map.
        coverage_threshold: Minimum trajectory coverage.
        reachability_threshold: Minimum reachability fraction.
        approach_distance: Gripper approach distance.
        target_ratio: Target opening ratio for manipulation.
        asset_dir: Path to asset directory.
    """
    # Parse config
    config = yaml.safe_load(open(config_path, "r"))
    object_name = None
    base_euler = None
    target_position = None
    predicted_grasp_pos = None
    predicted_grasp_orn = None
    object_scale = 1.0
    
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'euler' in config_dict:
            base_euler = parse_center(config_dict['euler'])
        if 'target_position' in config_dict:
            target_position = parse_center(config_dict['target_position'])
        if 'predicted_grasp_position' in config_dict:
            predicted_grasp_pos = parse_center(config_dict['predicted_grasp_position'])
            predicted_grasp_orn = parse_center(config_dict['predicted_grasp_orientation'])
        if 'size' in config_dict:
            object_scale = float(config_dict['size'])
            
    if object_name is None:
        q.put(False)
        return False
        
    if target_position is None:
        target_position = np.array([0.4, 0.0, 0.0])
        
    if base_euler is None:
        base_euler = np.array([0.0, 0.0, 0.0])

    # Load or create RM4D trajectory filter
    traj_filter = None
    if rm4d_map_path is not None:
        try:
            rm4d_map = load_or_create_rm4d_map(rm4d_map_path)
            traj_filter = TrajectoryReachabilityFilter(
                rmap=rm4d_map,
                coverage_threshold=coverage_threshold,
                intersection_threshold=reachability_threshold,
            )
        except Exception:
            traj_filter = None

    # Create grasp pose matrix from position and quaternion
    if predicted_grasp_pos is None or predicted_grasp_orn is None:
        # IK-filtered method requires a predicted grasp
        q.put(False)
        return False

    # Configs store predicted_grasp_position in the *scaled* object frame
    # (see create_variant_config). Avoid double-scaling when generating trajectories.
    grasp_pos_obj_scaled = np.array(predicted_grasp_pos, dtype=float)
    grasp_quat = np.array(predicted_grasp_orn, dtype=float)

    grasp_pose_unscaled = np.eye(4)
    grasp_pose_unscaled[:3, :3] = R.from_quat(grasp_quat).as_matrix()
    grasp_pose_unscaled[:3, 3] = grasp_pos_obj_scaled / float(object_scale)

    # Compute in-contact trajectory in object frame
    trajectory_obj = None
    if asset_dir is not None:
        asset_path = pathlib.Path(asset_dir)
        try:
            trajectory_obj = load_or_compute_trajectory(
                grasp_pose_unscaled,
                asset_path,
                scale=object_scale,
                approach_distance=approach_distance,
                target_ratio=target_ratio,
            )
        except Exception:
            trajectory_obj = None

    if trajectory_obj is None:
        # Fall back to straight-line approach
        grasp_p = grasp_pos_obj_scaled
        grasp_R = R.from_quat(grasp_quat).as_matrix()
        approach_dir = grasp_R[:, 2]
        trajectory_obj = []
        for i in range(20):
            t = i / 19.0
            offset = approach_distance * (1 - t)
            p_new = grasp_p - approach_dir * offset
            trajectory_obj.append((grasp_R.copy(), p_new.copy()))

    # Create environment
    env, _ = build_up_env_gen(config_path, env_name, render=render)
    env.reset()

    # Get object's z-coordinate after placement
    object_id = env.urdf_ids[object_name]
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    
    # Create sampling config
    sampling_config = SamplingConfig(
        target_position=tuple(target_position),
        position_jitter=0.1,
        orientation_jitter=np.pi / 6,
        coverage_threshold=coverage_threshold,
        reachability_threshold=reachability_threshold,
        near_distance=near_distance,
        far_distance=far_distance,
        approach_distance=approach_distance,
        target_ratio=target_ratio,
        max_pose_samples=100,
        max_robot_samples=100,
        timeout=60.0,
    )

    # Run IK-filtered sampling
    # predicted_grasp_pos is guaranteed non-None here due to earlier check
    grasp_pos_scaled = grasp_pos_obj_scaled
    
    success, obj_pos, obj_quat, joint_angles = ik_filtered_sample_initial_state(
        env,
        object_name,
        grasp_pos_scaled,
        grasp_quat,
        trajectory_obj,
        traj_filter,
        base_euler,
        sampling_config,
    )

    if not success:
        env.close()
        q.put(False)
        return False

    # Save state
    initial_finger_angle = np.random.uniform(
        env.robot.finger_fully_close_joint_angle,
        env.robot.finger_fully_open_joint_angle
    )
    env.robot.set_joint_angles(env.robot.right_arm_joint_indices, joint_angles.tolist())
    env.robot.set_gripper_open_position(
        env.robot.right_gripper_indices,
        [initial_finger_angle, initial_finger_angle],
        set_instantly=True
    )
    p.stepSimulation(physicsClientId=env.id)

    # Update config with the found configuration
    config = yaml.safe_load(open(config_path, "r"))
    for config_dict in config:
        if 'center' in config_dict:
            saved_pos = [obj_pos[0], obj_pos[1], 0.0]
            config_dict['center'] = str(tuple(saved_pos))
            config_dict['orientation'] = str(tuple(obj_quat.tolist()))
            # Keep euler consistent with quaternion (many utilities read 'euler').
            try:
                config_dict['euler'] = str(tuple(R.from_quat(np.array(obj_quat, dtype=float)).as_euler('xyz').tolist()))
            except Exception:
                pass
            config_dict['is_crop_size'] = False
            config_dict['initial_joint_angles'] = str(tuple(joint_angles.tolist()))
            config_dict['initial_finger_angle'] = initial_finger_angle
             
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)
        
    env.close()
    q.put(True)
    return True


def custom_gen_init_state_ik_filtered(
    config_path: str,
    env_name: str,
    render: bool,
    far_distance: float = 0.7,
    near_distance: float = 0.3,
    timeout: float = 60.0,
    rm4d_map_path: Optional[str] = None,
    coverage_threshold: float = 0.8,
    reachability_threshold: float = 0.5,
    approach_distance: float = 0.15,
    target_ratio: float = 0.8,
    asset_dir: Optional[str] = None,
) -> bool:
    """Generate initial state using IK-filtered sampling (process wrapper).
    
    This wraps _custom_gen_init_state_ik_filtered in a subprocess with
    timeout protection to prevent hangs.
    
    Args:
        config_path: Path to variant config YAML.
        env_name: Environment name for simulation.
        render: Whether to enable GUI rendering.
        far_distance: Max EEF-to-grasp distance.
        near_distance: Min EEF-to-grasp distance.
        timeout: Maximum time to wait for worker.
        rm4d_map_path: Path to RM4D reachability map.
        coverage_threshold: Minimum trajectory coverage.
        reachability_threshold: Minimum reachability fraction.
        approach_distance: Gripper approach distance.
        target_ratio: Target opening ratio for manipulation.
        asset_dir: Path to asset directory.
        
    Returns:
        True if initialization succeeded, False otherwise.
    """
    q = mp.Queue()
    proc = mp.Process(
        target=_custom_gen_init_state_ik_filtered,
        args=(
            q,
            config_path,
            env_name,
            render,
            far_distance,
            near_distance,
            rm4d_map_path,
            coverage_threshold,
            reachability_threshold,
            approach_distance,
            target_ratio,
            asset_dir,
        ),
    )
    proc.start()
    proc.join(timeout)
    
    if proc.is_alive():
        proc.terminate()
        proc.join()
        try:
            success = q.get_nowait()
        except Exception:
            success = False
        return success
    else:
        try:
            success = q.get(timeout=1)
        except Exception:
            success = False
        return success

