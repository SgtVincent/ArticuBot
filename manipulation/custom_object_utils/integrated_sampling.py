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
    heatmap_dir: Optional[str] = None,
    save_heatmaps: bool = False,
    heatmap_per_waypoint: bool = False,
    object_z_offset: float = 0.0,
    viser_visualizer = None,
    attempt_number: int = 0,
    object_urdf_path: Optional[str] = None,
    object_scale: float = 1.0,
    grasp_candidates_obj: Optional[List[Tuple[np.ndarray, np.ndarray, float]]] = None,
    enable_occlusion_filter: bool = True,
    trajectory_occlusion_rate: float = 0.5,
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

    # Keep semantics consistent with heuristic sampling:
    # - config.target_position[2] acts as an additional Z offset
    # - object_z_offset remains as an explicit override (legacy/extra)
    try:
        target_z_offset = float(config.target_position[2])
    except Exception:
        target_z_offset = 0.0

    object_z = float(init_pos[2]) + float(target_z_offset) + float(object_z_offset)
    print(
        f"  Object z-height: {float(init_pos[2]):.3f} + target_z_offset {target_z_offset:.3f} "
        f"+ object_z_offset {float(object_z_offset):.3f} = {object_z:.3f}"
    )
    
    # NOTE: Scale consistency is ensured by the caller (demo_utils_integrated.py)
    # which uses get_object_runtime_scale() to get the correct runtime scale.
    # The grasp_pos_obj and trajectory_obj passed here are already in the correct scale.

    
    start_time = time.time()
    
    # Initialize sampled_poses
    sampled_poses: List[Tuple[np.ndarray, np.ndarray, float]] = []
    
    # Store distribution data for debug visualization
    x_grid, y_grid, theta_grid, scores = None, None, None, None
    
    # 1. Compute distribution if sampler is available
    if sampler is not None:
        x_grid, y_grid, theta_grid, scores, sampled_poses = _compute_integrated_distribution(
            sampler, trajectory_obj, config, object_z, base_euler, 
            save_heatmaps, heatmap_dir, heatmap_per_waypoint, attempt_number
        )

    # 2. Try sampled poses from integrated distribution
    success, obj_pos, obj_quat, joint_angles = _validate_candidate_poses(
        env, object_id, sampled_poses, grasp_pos_obj, grasp_quat_obj,
        trajectory_obj, sampler, base_euler, object_z,
        x_grid, y_grid, theta_grid, scores, 
        config, start_time, enable_occlusion_filter, trajectory_occlusion_rate,
        debug_vis_path, save_heatmaps, heatmap_dir, attempt_number,
        viser_visualizer, object_urdf_path, object_scale, grasp_candidates_obj
    )
    
    if success:
        return True, obj_pos, obj_quat, joint_angles
    
    # 3. Fallback: random sampling
    return _fallback_random_sampling(
        env, object_id, grasp_pos_obj, grasp_quat_obj, trajectory_obj, sampler,
        config, object_z, base_euler, start_time, config.max_pose_samples,
        debug_vis_path
    )


def _compute_integrated_distribution(
    sampler, trajectory_obj, config, object_z, base_euler, 
    save_heatmaps, heatmap_dir, heatmap_per_waypoint, attempt_number
):
    """Compute distribution over object poses using integrated inverse map."""
    print("  Computing integrated inverse map distribution...")
    t0 = time.time()
    x_grid, y_grid, theta_grid, scores = sampler.compute_object_pose_distribution(
        traj_obj=trajectory_obj,
        xy_center=(config.target_position[0], config.target_position[1]),
        xy_range=config.xy_range,
        z_height=object_z,
        base_euler=base_euler,
    )
    
    sampled_poses = []
    
    # Report statistics
    n_valid = np.sum(scores > config.min_score_threshold)
    max_score = np.max(scores)
    mean_score = np.mean(scores[scores > 0]) if np.any(scores > 0) else 0
    print(f"  Distribution computed in {time.time() - t0:.2f}s: "
          f"{n_valid} valid cells (max: {max_score:.3f}, mean of nonzero: {mean_score:.3f})")
    
    if max_score < config.min_score_threshold:
        print(f"  WARNING: No valid poses found (max score {max_score:.3f} < threshold {config.min_score_threshold})")
        print(f"    Consider lowering --min-score-threshold or --coverage-threshold")
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

    # Dump heatmaps for offline inspection
    if save_heatmaps or heatmap_dir is not None:
        try:
            from manipulation.custom_object_utils.debug_visualization import (
                dump_integrated_inverse_heatmaps,
            )
            dump_integrated_inverse_heatmaps(
                heatmap_dir if heatmap_dir is not None else f"heatmaps_attempt_{attempt_number:04d}",
                x_grid=np.asarray(x_grid),
                y_grid=np.asarray(y_grid),
                theta_grid=np.asarray(theta_grid),
                integrated_scores=np.asarray(scores),
                sampler=sampler,
                trajectory_obj=trajectory_obj if heatmap_per_waypoint else None,
                base_euler=np.asarray(base_euler, dtype=float),
                z_height=float(object_z),
                object_pos=None,
                save_per_waypoint=bool(heatmap_per_waypoint),
                write_npz=True,
                write_png=True,
            )
        except Exception as e:
            import traceback
            print(f"  Warning: Failed to dump heatmaps: {e}")
            traceback.print_exc()
            
    return x_grid, y_grid, theta_grid, scores, sampled_poses


def _validate_candidate_poses(
    env, object_id, sampled_poses, grasp_pos_obj, grasp_quat_obj,
    trajectory_obj, sampler, base_euler, object_z,
    x_grid, y_grid, theta_grid, scores, 
    config, start_time, enable_occlusion_filter, trajectory_occlusion_rate,
    debug_vis_path, save_heatmaps, heatmap_dir, attempt_number,
    viser_visualizer, object_urdf_path, object_scale, grasp_candidates_obj
):
    """Validate sampled poses against collision and robot reachability."""
    from manipulation.custom_object_utils.occlusion_utils import check_trajectory_occlusion, check_handle_facing

    rejection_stats = {
        "z_below_ground": 0,
        "trajectory_occlusion": 0,
        "trajectory_facing": 0,
        "handle_facing": 0,
    }
    total_checked = 0

    def _log_rejection_stats(tag: str) -> None:
        if enable_occlusion_filter:
            print(
                f"[INTEGRATED][{tag}] candidates={total_checked}, "
                f"z_below_ground={rejection_stats['z_below_ground']}, "
                f"trajectory_occlusion={rejection_stats['trajectory_occlusion']}, "
                f"trajectory_facing={rejection_stats['trajectory_facing']}, "
                f"handle_facing={rejection_stats['handle_facing']}"
            )
        else:
            print(f"[INTEGRATED][{tag}] candidates={total_checked} (occlusion filter disabled)")
    
    # Identify movable joint (heuristic: first non-fixed)
    movable_joint_id = None
    for ji in range(p.getNumJoints(object_id, physicsClientId=env.id)):
        if p.getJointInfo(object_id, ji, physicsClientId=env.id)[2] != p.JOINT_FIXED:
            movable_joint_id = ji
            break

    for pose_idx, (obj_pos, obj_quat, score) in enumerate(sampled_poses):
        if time.time() - start_time > config.timeout:
            print(f"Integrated sampling timed out after {config.timeout}s")
            break
        
        # Apply to simulation
        p.resetBasePositionAndOrientation(object_id, obj_pos, obj_quat, physicsClientId=env.id)

        print(f"  Pose {pose_idx}: sampled with score {score:.3f}")
        total_checked += 1

        # Check if object is above ground (AABB min z > 0)
        aabb_min, _ = p.getAABB(object_id, physicsClientId=env.id)
        if aabb_min[2] <= 1e-4:
            rejection_stats["z_below_ground"] += 1
            continue

        # Check occlusion
        if enable_occlusion_filter:
            aabb_min, aabb_max = p.getAABB(object_id, physicsClientId=env.id)
            obj_center = (np.array(aabb_min) + np.array(aabb_max)) / 2.0
            try:
                robot_pos, _ = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)
                robot_pos = np.array(robot_pos, dtype=float)
            except Exception:
                robot_pos = None

            if robot_pos is not None and trajectory_obj:
                vec_base = robot_pos - obj_center
                pos_count = 0
                total_count = 0
                r_obj = R.from_quat(obj_quat)
                for _, traj_pos_obj in trajectory_obj:
                    traj_pos_world = r_obj.apply(np.array(traj_pos_obj, dtype=float)) + np.array(obj_pos, dtype=float)
                    vec_traj = traj_pos_world - obj_center
                    if np.dot(vec_traj[:2], vec_base[:2]) > 0.0:
                        pos_count += 1
                    total_count += 1
                if total_count > 0:
                    ratio = pos_count / float(total_count)
                    if ratio < 0.5:
                        rejection_stats["trajectory_facing"] += 1
                        print(
                            f"  Pose {pose_idx}: rejected by trajectory-facing check (ratio={ratio:.2f})"
                        )
                        continue

            handle_pos_world = None
            handle_joint_for_filter = movable_joint_id
            try:
                handle_pts, handle_joint_ids, _, _ = env.get_handle_pos(return_median=True)
                if handle_pts:
                    handle_pos_world = np.array(handle_pts[0], dtype=float)
                if handle_joint_ids:
                    handle_joint_for_filter = handle_joint_ids[0]
            except Exception:
                handle_pos_world = None

            if not check_trajectory_occlusion(
                env,
                object_id,
                handle_joint_for_filter,
                threshold=trajectory_occlusion_rate,
            ):
                rejection_stats["trajectory_occlusion"] += 1
                print(f"  Pose {pose_idx}: rejected by occlusion filter")
                continue
            if not check_handle_facing(
                env,
                object_id,
                handle_joint_for_filter,
                handle_pos_world=handle_pos_world,
            ):
                rejection_stats["handle_facing"] += 1
                print(f"  Pose {pose_idx}: rejected by handle facing check")
                continue
        
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
            _log_rejection_stats("success")

            # Record selected pose in heatmap folder.
            if (save_heatmaps or heatmap_dir is not None) and scores is not None:
                try:
                    import json
                    out_dir = pathlib.Path(heatmap_dir) if heatmap_dir is not None else pathlib.Path(f"heatmaps_attempt_{attempt_number:04d}")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / "selected_pose.json").write_text(
                        json.dumps(
                            {
                                "pose_idx": int(pose_idx),
                                "score": float(score),
                                "object_pos": np.asarray(obj_pos, dtype=float).tolist(),
                                "object_quat": np.asarray(obj_quat, dtype=float).tolist(),
                            },
                            indent=2,
                        )
                    )
                except Exception:
                    pass
            
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
                    viser_visualizer.visualize_attempt(
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
                except Exception as e:
                    import traceback
                    print(f"  Warning: Failed viser visualization: {e}")
                    traceback.print_exc()
            
            return True, obj_pos, obj_quat, joint_angles
            
    # If heatmaps were requested and we computed a distribution, mark failure.
    if (save_heatmaps or heatmap_dir is not None) and scores is not None:
        try:
            import json
            out_dir = pathlib.Path(heatmap_dir) if heatmap_dir is not None else pathlib.Path(f"heatmaps_attempt_{attempt_number:04d}")
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "result.json").write_text(
                json.dumps(
                    {
                        "success": False,
                        "attempt_number": int(attempt_number),
                        "n_sampled_poses": int(len(sampled_poses)),
                    },
                    indent=2,
                )
            )
        except Exception:
            pass
    _log_rejection_stats("failure")
    return False, np.zeros(3), np.zeros(4), np.zeros(7)


def _fallback_random_sampling(
    env, object_id, grasp_pos_obj, grasp_quat_obj, trajectory_obj, sampler,
    config, object_z, base_euler, start_time, max_attempts,
    debug_vis_path
):
    """Fallback to random sampling if integrated sampling failed."""
    print("  Falling back to random sampling or no candidates found")
    
    for pose_idx in range(max_attempts):
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
    
    print(f"Failed to find valid initial state after {max_attempts} attempts")
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
    # Try to load joint kinematics from URDF first.
    # Rationale: many generated URDFs insert fixed joints (often with 90deg rotations)
    # between the URDF root and the articulated joint's parent link. URDF parsing can
    # account for these constant frame offsets, while mobility_v2.json typically cannot.
    mobility_path = asset_dir / "mobility_v2.json"
    urdf_files = list(asset_dir.glob("*.urdf"))

    joint_kin = None
    if urdf_files:
        try:
            joint_kin = parse_joint_kinematics_from_urdf(urdf_files[0])
        except Exception:
            joint_kin = None

    if joint_kin is None and mobility_path.exists():
        try:
            joint_kin = parse_joint_kinematics_from_mobility(mobility_path)
        except Exception:
            joint_kin = None
    
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
