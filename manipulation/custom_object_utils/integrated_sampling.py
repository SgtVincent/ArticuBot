"""Integrated inverse map sampling for efficient demo generation.

This module implements a direct sampling approach that integrates the inverse
reachability distributions along the full manipulation trajectory. Instead of
rejection sampling (sample pose → check feasibility → reject/accept), this
approach:

1. Pre-computes a distribution over object placements
2. For each candidate (x, y, θ), computes reachability score along trajectory
3. Normalizes scores to create a probability distribution
4. Samples directly from this distribution

This is more efficient than rejection sampling when:
- The acceptance rate is low (many rejections)
- The trajectory is complex (expensive to check each pose)
- We need multiple samples from similar distributions

Key Insight:
For each 6D end-effector pose along the trajectory, RM4D provides an inverse
map showing which base positions can reach that pose. By integrating (taking
minimum or product) these maps across all waypoints, we get a distribution
of object placements that satisfy the full trajectory.
"""
from __future__ import annotations

import time
import pathlib
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from manipulation.custom_object_utils.contact_trajectory import (
    parse_joint_kinematics_from_urdf,
    parse_joint_kinematics_from_mobility,
    compute_full_manipulation_trajectory,
)

from manipulation.rm4d_filtering import (
    IntegratedInverseMapSampler,
    create_integrated_inverse_sampler,
    generate_straight_line_trajectory,
    load_or_create_rm4d_map,
)


@dataclass
class IntegratedSamplingConfig:
    """Configuration for integrated inverse map sampling."""
    # Object placement bounds
    target_position: Tuple[float, float, float] = (0.4, 0.0, 0.0)
    xy_range: float = 0.3  # Half-width of sampling region
    
    # Grid resolution for distribution computation
    xy_resolution: float = 0.02  # meters
    theta_bins: int = 24  # orientations around z-axis
    
    # Sampling parameters
    aggregation_method: str = 'min'  # 'min' or 'product'
    temperature: float = 1.0  # Higher = more uniform sampling
    min_score_threshold: float = 0.1  # Minimum score to consider valid
    
    # Robot arm workspace bounds (for distance checks)
    near_distance: float = 0.15
    far_distance: float = 0.5
    
    # Trajectory parameters
    approach_distance: float = 0.15
    target_ratio: float = 0.8
    approach_steps: int = 20
    contact_steps: int = 50
    
    # Sampling limits
    max_pose_samples: int = 50  # Number of poses to try from distribution
    max_robot_samples: int = 100
    timeout: float = 60.0
    
    # Cached distribution (computed once per trajectory)
    _cached_distribution: Optional[Tuple] = field(default=None, repr=False)


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
    config: IntegratedSamplingConfig,
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


def integrated_sample_initial_state(
    env,
    object_name: str,
    grasp_pos_obj: np.ndarray,
    grasp_quat_obj: np.ndarray,
    trajectory_obj: List[Tuple[np.ndarray, np.ndarray]],
    sampler: Optional[IntegratedInverseMapSampler],
    base_euler: np.ndarray,
    config: IntegratedSamplingConfig,
    debug_vis_path: Optional[str] = None,
    object_z_offset: float = 0.0,
    viser_visualizer = None,
    attempt_number: int = 0,
    object_urdf_path: Optional[str] = None,
    object_scale: float = 1.0,
    grasp_candidates_obj: Optional[List[Tuple[np.ndarray, np.ndarray, float]]] = None,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Sample initial state using integrated inverse map approach.
    
    This implements the integrated inverse map algorithm:
    1. Compute distribution over object poses based on trajectory reachability
    2. Sample candidate poses from this distribution
    3. For each sampled pose, check robot configuration feasibility
    
    Args:
        env: PyBullet environment.
        object_name: Name of the object in env.urdf_ids.
        grasp_pos_obj: Grasp position in object frame.
        grasp_quat_obj: Grasp quaternion in object frame.
        trajectory_obj: Full manipulation trajectory in object frame.
        sampler: IntegratedInverseMapSampler instance (or None for fallback).
        base_euler: Base euler angles for object orientation.
        config: Sampling configuration.
        debug_vis_path: Optional path to save debug visualization GIF.
        object_z_offset: Z-offset to elevate object above ground (for table height).
        viser_visualizer: Optional viser visualizer for interactive debugging.
        attempt_number: Current attempt number for viser tracking.
        object_urdf_path: Optional path to object URDF for mesh visualization in viser.
        object_scale: Uniform scale factor used when loading the object in PyBullet.
        
    Returns:
        Tuple of (success, object_pos, object_quat, robot_joints).
    """
    object_id = env.urdf_ids[object_name]
    
    # Get object's z-coordinate after placement (already computed by env.reset)
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    # Apply z-offset to elevate object (e.g., for table height)
    object_z = init_pos[2] + object_z_offset
    print(f"  Object z-height: {init_pos[2]:.3f} + offset {object_z_offset:.3f} = {object_z:.3f}")
    
    start_time = time.time()
    
    # Initialize sampled_poses
    sampled_poses: List[Tuple[np.ndarray, np.ndarray, float]] = []
    
    # Store distribution data for debug visualization
    x_grid, y_grid, theta_grid, scores = None, None, None, None
    
    # Compute distribution if sampler is available
    if sampler is not None:
        print("  Computing integrated inverse map distribution...")
        t0 = time.time()
        x_grid, y_grid, theta_grid, scores = sampler.compute_object_pose_distribution(
            traj_obj=trajectory_obj,
            xy_center=(config.target_position[0], config.target_position[1]),
            xy_range=config.xy_range,
            z_height=object_z,
            base_euler=base_euler,
        )
        
        # Report statistics
        n_valid = np.sum(scores > config.min_score_threshold)
        max_score = np.max(scores)
        mean_score = np.mean(scores[scores > 0]) if np.any(scores > 0) else 0
        print(f"  Distribution computed in {time.time() - t0:.2f}s: "
              f"{n_valid} valid cells (max: {max_score:.3f}, mean of nonzero: {mean_score:.3f})")
        
        if max_score < config.min_score_threshold:
            print(f"  WARNING: No valid poses found (max score {max_score:.3f} < threshold {config.min_score_threshold})")
            print(f"    Consider lowering --min-score-threshold or --coverage-threshold")
            # sampled_poses remains empty, will trigger fallback
        else:
            # Pre-sample poses from distribution
            sampled_poses = sampler.sample_object_poses_batch(
                scores=scores,
                x_grid=x_grid,
                y_grid=y_grid,
                theta_grid=theta_grid,
                n_samples=config.max_pose_samples,
                z_height=object_z,
                base_euler=base_euler,
                temperature=config.temperature,
            )
            print(f"  Pre-sampled {len(sampled_poses)} candidate poses")
    
    # Try sampled poses first (from integrated distribution)
    for pose_idx, (obj_pos, obj_quat, score) in enumerate(sampled_poses):
        if time.time() - start_time > config.timeout:
            print(f"Integrated sampling timed out after {config.timeout}s")
            break
        
        # Apply to simulation
        p.resetBasePositionAndOrientation(object_id, obj_pos, obj_quat, physicsClientId=env.id)
        
        print(f"  Pose {pose_idx}: sampled with score {score:.3f}")
        
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
            
            # Save debug visualization if requested (GIF)
            if debug_vis_path is not None and scores is not None:
                try:
                    from manipulation.custom_object_utils.debug_visualization import (
                        save_debug_visualization,
                    )
                    save_debug_visualization(
                        trajectory_obj=trajectory_obj,
                        sampler=sampler,
                        object_pos=obj_pos,
                        object_quat=obj_quat,
                        x_grid=x_grid,
                        y_grid=y_grid,
                        theta_grid=theta_grid,
                        integrated_scores=scores,
                        output_path=debug_vis_path,
                        base_euler=base_euler,
                        z_height=object_z,
                        include_per_waypoint=True,
                    )
                except Exception as e:
                    import traceback
                    print(f"  Warning: Failed to save debug visualization: {e}")
                    traceback.print_exc()
            
            # Viser interactive visualization
            if viser_visualizer is not None and scores is not None:
                try:
                    robot_base_pos, robot_base_quat = p.getBasePositionAndOrientation(
                        env.robot.body, physicsClientId=env.id
                    )
                    result = viser_visualizer.visualize_attempt(
                        trajectory_obj=trajectory_obj,
                        sampler=sampler,
                        object_pos=obj_pos,
                        object_quat=obj_quat,
                        x_grid=x_grid,
                        y_grid=y_grid,
                        theta_grid=theta_grid,
                        integrated_scores=scores,
                        base_euler=base_euler,
                        z_height=object_z,
                        attempt_number=attempt_number,
                        object_urdf_path=object_urdf_path,
                        robot_joint_angles=joint_angles,
                        object_scale=float(object_scale),
                        robot_base_pos=np.asarray(robot_base_pos, dtype=float),
                        robot_base_quat=np.asarray(robot_base_quat, dtype=float),
                        grasp_pos_obj=np.asarray(grasp_pos_obj, dtype=float),
                        grasp_quat_obj=np.asarray(grasp_quat_obj, dtype=float),
                        grasp_candidates_obj=grasp_candidates_obj,
                    )
                    if result == 'quit':
                        print("[VISER] User requested quit")
                        return False, np.zeros(3), np.zeros(4), np.zeros(7)
                except Exception as e:
                    import traceback
                    print(f"  Warning: Failed viser visualization: {e}")
                    traceback.print_exc()
            
            return True, obj_pos, obj_quat, joint_angles
    
    # Fallback: random sampling if integrated sampling failed
    if len(sampled_poses) == 0:
        print("  Falling back to random sampling (no sampler available)")
        remaining_attempts = config.max_pose_samples
        
        for pose_idx in range(remaining_attempts):
            if time.time() - start_time > config.timeout:
                print(f"Random sampling timed out after {config.timeout}s")
                break
            
            # Random object pose
            obj_pos = np.array([
                config.target_position[0] + np.random.uniform(-config.xy_range, config.xy_range),
                config.target_position[1] + np.random.uniform(-config.xy_range, config.xy_range),
                object_z,
            ])
            
            euler = base_euler.copy()
            euler[2] += np.random.uniform(-np.pi, np.pi)
            obj_quat = R.from_euler('xyz', euler).as_quat()
            
            # Apply to simulation
            p.resetBasePositionAndOrientation(object_id, obj_pos, obj_quat, physicsClientId=env.id)
            
            # Transform grasp to world frame
            grasp_world_pos, grasp_world_quat = transform_grasp_to_world(
                grasp_pos_obj, grasp_quat_obj, obj_pos, obj_quat
            )
            
            # Check robot configuration
            success, joint_angles = check_robot_configuration(
                env, object_id, grasp_world_pos, config
            )
            
            if success:
                print(f"  Found valid configuration (random fallback) at pose {pose_idx}")
                # For fallback, create minimal debug visualization
                if debug_vis_path is not None:
                    try:
                        from manipulation.custom_object_utils.debug_visualization import (
                            save_debug_visualization,
                        )
                        # Create dummy grid for fallback case
                        dummy_x = np.array([obj_pos[0]])
                        dummy_y = np.array([obj_pos[1]])
                        dummy_theta = np.array([0.0])
                        dummy_scores = np.array([[[1.0]]])
                        save_debug_visualization(
                            trajectory_obj=trajectory_obj,
                            sampler=sampler,
                            object_pos=obj_pos,
                            object_quat=obj_quat,
                            x_grid=dummy_x,
                            y_grid=dummy_y,
                            theta_grid=dummy_theta,
                            integrated_scores=dummy_scores,
                            output_path=debug_vis_path,
                            base_euler=euler,
                            z_height=object_z,
                            include_per_waypoint=False,  # Skip per-waypoint for fallback
                        )
                    except Exception as e:
                        import traceback
                        print(f"  Warning: Failed to save debug visualization: {e}")
                        traceback.print_exc()
                return True, obj_pos, obj_quat, joint_angles
    
    print(f"Failed to find valid initial state after {config.max_pose_samples} attempts")
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


def create_sampler_from_map(
    rm4d_map_path: Optional[str],
    xy_resolution: float = 0.02,
    theta_bins: int = 24,
    aggregation_method: str = 'mean',
    coverage_threshold: float = 0.5,
) -> Optional[IntegratedInverseMapSampler]:
    """Create an IntegratedInverseMapSampler from an RM4D map file.
    
    Args:
        rm4d_map_path: Path to RM4D reachability map (.npy).
        xy_resolution: Grid resolution in meters.
        theta_bins: Number of orientation bins.
        aggregation_method: 'min', 'product', or 'mean'.
        coverage_threshold: Minimum fraction of reachable waypoints (for 'mean').
        
    Returns:
        IntegratedInverseMapSampler or None if map not available.
    """
    if rm4d_map_path is None:
        return None
    
    try:
        rmap = load_or_create_rm4d_map(rm4d_map_path)
        return create_integrated_inverse_sampler(
            rmap=rmap,
            xy_resolution=xy_resolution,
            theta_bins=theta_bins,
            aggregation_method=aggregation_method,
            coverage_threshold=coverage_threshold,
        )
    except Exception as e:
        print(f"Failed to load RM4D map: {e}")
        return None
