"""Integrated inverse map demo utilities for direct sampling approach.

This module provides process-wrapped initial state generation using the
integrated inverse map algorithm. Unlike rejection sampling (heuristic methods), this approach:

1. Pre-computes a distribution over object placements based on trajectory reachability
2. Samples directly from this distribution
3. Only checks robot configuration feasibility for sampled poses

This is more efficient when the acceptance rate for rejection sampling is low.
"""
from __future__ import annotations

import multiprocessing as mp
import pathlib
import shutil
import uuid
import numpy as np
import pybullet as p
import yaml
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, List
from termcolor import cprint

from manipulation.utils import build_up_env_gen, parse_center

from manipulation.custom_object_utils.integrated_sampling import (
    IntegratedSamplingConfig,
    integrated_sample_initial_state,
    load_or_compute_trajectory,
    create_sampler_from_map,
)

from manipulation.custom_object_utils.graspgen_client import (
    GraspGenConfig,
    predict_grasps_for_urdf_folder,
    prepare_urdf_with_joint_state,
)

from manipulation.custom_object_utils.env_utils import get_object_runtime_scale


def _custom_gen_init_state_integrated(
    q,
    config_path: str,
    env_name: str,
    render: bool,
    far_distance: float = 0.7,
    near_distance: float = 0.3,
    rm4d_map_path: Optional[str] = None,
    xy_resolution: float = 0.02,
    theta_bins: int = 24,
    aggregation_method: str = 'mean',
    coverage_threshold: float = 0.5,
    temperature: float = 1.0,
    min_score_threshold: float = 0.1,
    approach_distance: float = 0.15,
    target_ratio: float = 0.8,
    asset_dir: Optional[str] = None,
    debug_vis_path: Optional[str] = None,
    object_z_offset: float = 0.0,
    save_heatmaps: bool = False,
    heatmap_dir: Optional[str] = None,
    heatmap_per_waypoint: bool = False,
    use_viser: bool = False,
    viser_port: int = 8080,
    attempt_number: int = 0,
    trajectory_occlusion_rate: float = 0.5,
    enable_occlusion_filter: bool = True,
):
    """Generate initial state using integrated inverse map sampling.
    
    This implements the integrated inverse map algorithm:
    1. Load grasp from config
    2. Compute in-contact trajectory in object frame
    3. Create integrated inverse map sampler
    4. Compute distribution over object poses
    5. Sample directly from distribution
    6. Check robot configuration feasibility
    
    Args:
        q: Multiprocessing queue for returning result.
        config_path: Path to variant config YAML.
        env_name: Environment name for simulation.
        render: Whether to enable GUI rendering.
        far_distance: Max EEF-to-grasp distance.
        near_distance: Min EEF-to-grasp distance.
        rm4d_map_path: Path to RM4D reachability map.
        xy_resolution: Grid resolution in meters.
        theta_bins: Number of orientation bins.
        aggregation_method: 'min', 'product', or 'mean'.
        coverage_threshold: Minimum fraction of reachable waypoints (for 'mean').
        temperature: Sampling temperature.
        min_score_threshold: Minimum score for valid poses.
        approach_distance: Gripper approach distance.
        target_ratio: Target opening ratio for manipulation.
        asset_dir: Path to asset directory.
        debug_vis_path: Optional path to save debug visualization GIF.
        save_heatmaps: Whether to dump heatmaps (png + npz) for offline inspection.
        heatmap_dir: Output directory to dump heatmaps for this attempt.
        heatmap_per_waypoint: Whether to also dump per-waypoint heatmaps (best-theta).
        object_z_offset: Z-offset to elevate object above ground.
        use_viser: Whether to enable interactive viser visualization.
        viser_port: Port number for viser server.
        attempt_number: Current attempt number for tracking.
    """
    cprint(f"[INTEGRATED] Parsing config: {config_path}", "cyan")
    
    # 1. Parse config
    task_config = _parse_task_config(config_path)
    if task_config is None:
        cprint("[INTEGRATED] ERROR: object_name is None in config", "red")
        q.put(False)
        return False

    (object_name, base_euler, target_position, predicted_grasp_pos, 
     predicted_grasp_orn, object_scale, urdf_path_for_graspgen, 
     urdf_orientation_quat) = task_config
        
    if target_position is None:
        target_position = np.array([0.4, 0.0, 0.0])
        cprint(f"[INTEGRATED] Using default target_position: {target_position}", "yellow")
        
    if base_euler is None:
        base_euler = np.array([0.0, 0.0, 0.0])
        
    cprint(f"[INTEGRATED] Object: {object_name}, scale: {object_scale}", "cyan")
    cprint(f"[INTEGRATED] Target position: {target_position}", "cyan")
    cprint(f"[INTEGRATED] Grasp in config: pos={predicted_grasp_pos}, orn={predicted_grasp_orn}", "cyan")
    cprint(f"[INTEGRATED] Object Z offset: {object_z_offset}", "cyan")

    # 2. Setup environment and get object state
    env, object_id, init_pos, init_orient, runtime_scale = _setup_environment_and_object(
        config_path, env_name, render, object_name, object_scale
    )
    object_scale = runtime_scale # Enforce runtime scale

    # Validate orientation match
    _validate_orientation_consistency(urdf_orientation_quat, base_euler, init_orient)

    # 3. Resolve grasp configuration (maybe via GraspGen)
    grasp_result = _resolve_grasp_configuration(
        env, object_id, config_path, urdf_path_for_graspgen, 
        predicted_grasp_pos, predicted_grasp_orn, object_scale
    )
    
    if grasp_result is None:
        env.close()
        q.put(False)
        return False
        
    predicted_grasp_pos, predicted_grasp_orn, grasp_candidates_obj = grasp_result

    # 4. Create integrated inverse map sampler
    sampler = None
    if rm4d_map_path is not None:
        sampler = create_sampler_from_map(
            rm4d_map_path,
            xy_resolution=xy_resolution,
            theta_bins=theta_bins,
            aggregation_method=aggregation_method,
            coverage_threshold=coverage_threshold,
        )

    # 5. Compute in-contact trajectory
    trajectory_obj = _compute_manipulation_trajectory(
        asset_dir, predicted_grasp_pos, predicted_grasp_orn, object_scale,
        approach_distance, target_ratio
    )

    # 6. Run integrated sampling
    sampling_config = IntegratedSamplingConfig(
        target_position=tuple(target_position),
        xy_range=0.3,
        xy_resolution=xy_resolution,
        theta_bins=theta_bins,
        aggregation_method=aggregation_method,
        temperature=temperature,
        min_score_threshold=min_score_threshold,
        near_distance=near_distance,
        far_distance=far_distance,
        approach_distance=approach_distance,
        target_ratio=target_ratio,
        max_pose_samples=50,
        max_robot_samples=100,
        timeout=60.0,
    )

    viser_visualizer = None
    if use_viser:
        from manipulation.custom_object_utils.viser_visualization import IntegratedInverseMapVisualizer
        viser_visualizer = IntegratedInverseMapVisualizer(port=viser_port)
        cprint(f"[INTEGRATED] Viser server started on port {viser_port}", "cyan")

    # Determine URDF path for visualization
    object_urdf_path = urdf_path_for_graspgen
    if object_urdf_path is None and asset_dir is not None:
        candidates = list(pathlib.Path(asset_dir).glob("*.urdf"))
        object_urdf_path = str(candidates[0]) if candidates else None

    grasp_pos_scaled = np.array(predicted_grasp_pos, dtype=float)
    grasp_quat = np.array(predicted_grasp_orn, dtype=float)

    cprint(f"[INTEGRATED] Starting integrated sampling with target_position={target_position}", "cyan")
    
    success, obj_pos, obj_quat, joint_angles = integrated_sample_initial_state(
        env,
        object_name,
        grasp_pos_scaled,
        grasp_quat,
        trajectory_obj,
        sampler,
        base_euler,
        sampling_config,
        debug_vis_path=debug_vis_path,
        object_z_offset=object_z_offset,
        heatmap_dir=heatmap_dir,
        save_heatmaps=bool(save_heatmaps),
        heatmap_per_waypoint=bool(heatmap_per_waypoint),
        viser_visualizer=viser_visualizer,
        attempt_number=attempt_number,
        object_urdf_path=object_urdf_path,
        object_scale=float(object_scale),
        grasp_candidates_obj=grasp_candidates_obj,
        enable_occlusion_filter=enable_occlusion_filter,
        trajectory_occlusion_rate=trajectory_occlusion_rate,
    )

    if not success:
        cprint("[INTEGRATED] Failed to find valid initial state", "red")
        env.close()
        q.put(False)
        return False
    
    cprint(f"[INTEGRATED] SUCCESS: obj_pos={obj_pos}, joint_angles={joint_angles[:3]}...", "green")

    # 7. Apply state and save
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

    _update_config_with_result(
        config_path, obj_pos, obj_quat, joint_angles, 
        initial_finger_angle, grasp_pos_scaled, grasp_quat, object_z_offset
    )
        
    env.close()
    q.put(True)
    return True


def _parse_task_config(config_path):
    """Parse task configuration file."""
    config = yaml.safe_load(open(config_path, "r"))
    object_name = None
    base_euler = None
    target_position = None
    predicted_grasp_pos = None
    predicted_grasp_orn = None
    object_scale = 1.0
    urdf_path_for_graspgen = None
    urdf_orientation_quat = None
    
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'euler' in config_dict:
            base_euler = parse_center(config_dict['euler'])
        if 'orientation' in config_dict:
            urdf_orientation_quat = parse_center(config_dict['orientation'])
        if 'target_position' in config_dict:
            target_position = parse_center(config_dict['target_position'])
        if 'predicted_grasp_position' in config_dict:
            predicted_grasp_pos = parse_center(config_dict['predicted_grasp_position'])
            predicted_grasp_orn = parse_center(config_dict['predicted_grasp_orientation'])
        if 'size' in config_dict:
            object_scale = float(config_dict['size'])
        if 'urdf_path' in config_dict:
            urdf_path_for_graspgen = config_dict['urdf_path']
            
    if object_name is None:
        return None
        
    return (object_name, base_euler, target_position, predicted_grasp_pos, 
            predicted_grasp_orn, object_scale, urdf_path_for_graspgen, 
            urdf_orientation_quat)


def _setup_environment_and_object(config_path, env_name, render, object_name, object_scale):
    """Initialize environment and retrieve object properties."""
    env, _ = build_up_env_gen(config_path, env_name, render=render)
    env.reset()
    
    object_id = env.urdf_ids[object_name]
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    
    runtime_scale = get_object_runtime_scale(env, object_name)
    if abs(runtime_scale - float(object_scale)) > 1e-6:
        cprint(
            f"[INTEGRATED] Using runtime scale {runtime_scale:.6f} (config specified {float(object_scale):.6f})",
            "yellow",
        )
    return env, object_id, init_pos, init_orient, runtime_scale


def _validate_orientation_consistency(urdf_orientation_quat, base_euler, init_orient):
    """Validate that runtime orientation matches configuration."""
    if urdf_orientation_quat is not None:
        try:
            cfg_euler = R.from_quat(np.array(urdf_orientation_quat, dtype=float)).as_euler('xyz')
            if base_euler is not None and np.linalg.norm((cfg_euler - np.array(base_euler, dtype=float))) > 1e-3:
                cprint(
                    "[INTEGRATED] Note: config 'euler' differs from 'orientation'; using runtime quaternion-derived euler",
                    "yellow",
                )

            r_cfg = R.from_quat(np.array(urdf_orientation_quat, dtype=float))
            r_rt = R.from_quat(np.array(init_orient, dtype=float))
            r_rel = r_cfg.inv() * r_rt
            angle_deg = float(np.linalg.norm(r_rel.as_rotvec()) * 180.0 / np.pi)
            if angle_deg > 1.0:
                cprint(
                    f"[INTEGRATED] Warning: runtime object orientation differs from config by ~{angle_deg:.2f} deg",
                    "yellow",
                )
        except Exception:
            pass


def _resolve_grasp_configuration(
    env, object_id, config_path, urdf_path_for_graspgen, 
    predicted_grasp_pos, predicted_grasp_orn, object_scale
):
    """Resolve grasp configuration, running on-demand GraspGen if needed."""
    if predicted_grasp_pos is not None and predicted_grasp_orn is not None:
        return predicted_grasp_pos, predicted_grasp_orn, None

    print("No predicted grasp in config, attempting on-demand GraspGen...")
    
    # Determine URDF path
    urdf_path = urdf_path_for_graspgen
    if urdf_path is None:
        cfg_parent = pathlib.Path(config_path).resolve().parent.parent
        candidates = list(pathlib.Path(cfg_parent).glob('*.urdf'))
        urdf_path = str(candidates[0]) if candidates else None
    
    if urdf_path is None:
        print("No URDF path available for on-demand GraspGen.")
        return None
    
    # Collect joint states
    joint_states = {}
    num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
    for ji in range(num_joints):
        jinfo = p.getJointInfo(object_id, ji, physicsClientId=env.id)
        if jinfo[2] == p.JOINT_FIXED:
            continue
        jstate = p.getJointState(object_id, ji, physicsClientId=env.id)[0]
        joint_states[jinfo[1].decode('utf-8')] = float(jstate)
    
    # Prepare temporary folder
    gg_cfg = GraspGenConfig()
    tmp_root = pathlib.Path(gg_cfg.host_graspgen_root).resolve() / "_articubot_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = str(tmp_root / f"_tmp_{uuid.uuid4().hex[:8]}")
    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        prepared = prepare_urdf_with_joint_state(urdf_path, joint_states, tmp_dir, scale=object_scale)
        grasps, confidences = predict_grasps_for_urdf_folder(prepared, gg_cfg)
        
        if len(confidences) == 0:
            print("GraspGen returned no grasps")
            shutil.rmtree(tmp_dir)
            return None
        
        order = np.argsort(-np.asarray(confidences))
        idx = int(order[0])
        sel = grasps[idx]
        best_pos = sel[:3, 3].tolist()
        best_orn = R.from_matrix(sel[:3, :3]).as_quat().tolist()

        # Top-k candidates
        top_k = int(min(10, len(order)))
        candidates = []
        for j in range(top_k):
            gi = int(order[j])
            g = grasps[gi]
            g_pos = np.asarray(g[:3, 3], dtype=float)
            g_quat = R.from_matrix(g[:3, :3]).as_quat().astype(float)
            g_conf = float(confidences[gi])
            candidates.append((g_pos, g_quat, g_conf))

        print(f"GraspGen found {len(grasps)} grasps, using best one (top_k={top_k} for viser)")
        shutil.rmtree(tmp_dir)
        return best_pos, best_orn, candidates
        
    except Exception as e:
        print(f"On-demand GraspGen failed: {e}")
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return None


def _compute_manipulation_trajectory(
    asset_dir, predicted_grasp_pos, predicted_grasp_orn, object_scale,
    approach_distance, target_ratio
):
    """Compute in-contact trajectory in object frame."""
    grasp_pos_scaled = np.array(predicted_grasp_pos, dtype=float)
    grasp_quat = np.array(predicted_grasp_orn, dtype=float)

    grasp_pose_unscaled = np.eye(4)
    grasp_pose_unscaled[:3, :3] = R.from_quat(grasp_quat).as_matrix()
    grasp_pose_unscaled[:3, 3] = grasp_pos_scaled / float(object_scale)

    trajectory_obj = None
    if asset_dir is not None:
        try:
            trajectory_obj = load_or_compute_trajectory(
                grasp_pose_unscaled,
                pathlib.Path(asset_dir),
                scale=object_scale,
                approach_distance=approach_distance,
                target_ratio=target_ratio,
            )
            print(f"Computed trajectory with {len(trajectory_obj)} waypoints")
        except Exception as e:
            print(f"Failed to compute trajectory: {e}")
            trajectory_obj = None

    if trajectory_obj is None:
        print("Falling back to straight-line approach trajectory")
        grasp_R = R.from_quat(grasp_quat).as_matrix()
        approach_dir = grasp_R[:, 2]
        trajectory_obj = []
        for i in range(20):
            t = i / 19.0
            offset = approach_distance * (1 - t)
            p_new = grasp_pos_scaled - approach_dir * offset
            trajectory_obj.append((grasp_R.copy(), p_new.copy()))
            
    return trajectory_obj


def _update_config_with_result(
    config_path, obj_pos, obj_quat, joint_angles, 
    initial_finger_angle, grasp_pos_scaled, grasp_quat, object_z_offset
):
    """Update configuration file with found parameters."""
    config = yaml.safe_load(open(config_path, "r"))
    for config_dict in config:
        if 'center' in config_dict:
            saved_pos = [obj_pos[0], obj_pos[1], 0.0]
            config_dict['center'] = str(tuple(saved_pos))
            config_dict['orientation'] = str(tuple(obj_quat.tolist()))
            try:
                config_dict['euler'] = str(tuple(R.from_quat(np.array(obj_quat, dtype=float)).as_euler('xyz').tolist()))
            except Exception:
                pass
            config_dict['is_crop_size'] = False
            config_dict['initial_joint_angles'] = str(tuple(joint_angles.tolist()))
            config_dict['initial_finger_angle'] = initial_finger_angle
            config_dict['predicted_grasp_position'] = str(tuple(grasp_pos_scaled.tolist()))
            config_dict['predicted_grasp_orientation'] = str(tuple(grasp_quat.tolist()))
            config_dict['object_z_offset'] = float(object_z_offset)
             
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)


def custom_gen_init_state_integrated(
    config_path: str,
    env_name: str,
    render: bool,
    far_distance: float = 0.7,
    near_distance: float = 0.3,
    timeout: float = 60.0,
    rm4d_map_path: Optional[str] = None,
    xy_resolution: float = 0.02,
    theta_bins: int = 24,
    aggregation_method: str = 'mean',
    coverage_threshold: float = 0.5,
    temperature: float = 1.0,
    min_score_threshold: float = 0.1,
    approach_distance: float = 0.15,
    target_ratio: float = 0.8,
    asset_dir: Optional[str] = None,
    debug_vis_path: Optional[str] = None,
    object_z_offset: float = 0.0,
    save_heatmaps: bool = False,
    heatmap_dir: Optional[str] = None,
    heatmap_per_waypoint: bool = False,
    use_viser: bool = False,
    viser_port: int = 8080,
    attempt_number: int = 0,
    trajectory_occlusion_rate: float = 0.5,
    enable_occlusion_filter: bool = True,
    disable_prefilter: bool = False,
) -> bool:
    """Generate initial state using integrated inverse map (process wrapper).
    
    This wraps _custom_gen_init_state_integrated in a subprocess with
    timeout protection to prevent hangs.
    
    Args:
        config_path: Path to variant config YAML.
        env_name: Environment name for simulation.
        render: Whether to enable GUI rendering.
        far_distance: Max EEF-to-grasp distance.
        near_distance: Min EEF-to-grasp distance.
        timeout: Maximum time to wait for worker.
        rm4d_map_path: Path to RM4D reachability map.
        xy_resolution: Grid resolution in meters.
        theta_bins: Number of orientation bins.
        aggregation_method: 'min', 'product', or 'mean'.
        coverage_threshold: Minimum fraction of reachable waypoints (for 'mean').
        temperature: Sampling temperature.
        min_score_threshold: Minimum score for valid poses.
        approach_distance: Gripper approach distance.
        target_ratio: Target opening ratio for manipulation.
        asset_dir: Asset directory for custom object.
        debug_vis_path: Path to save debug visualization GIF.
        object_z_offset: Vertical offset for object placement.
        use_viser: Whether to enable interactive viser visualization.
        viser_port: Port number for viser server.
        attempt_number: Current attempt number for tracking in visualization.
        asset_dir: Path to asset directory.
        debug_vis_path: Optional path to save debug visualization GIF.
        object_z_offset: Z-offset to elevate object above ground.
        
    Returns:
        True if initialization succeeded, False otherwise.
    """
    q = mp.Queue()
    effective_occlusion_filter = bool(enable_occlusion_filter) and not bool(disable_prefilter)
    proc = mp.Process(
        target=_custom_gen_init_state_integrated,
        args=(
            q,
            config_path,
            env_name,
            render,
            far_distance,
            near_distance,
            rm4d_map_path,
            xy_resolution,
            theta_bins,
            aggregation_method,
            coverage_threshold,
            temperature,
            min_score_threshold,
            approach_distance,
            target_ratio,
            asset_dir,
            debug_vis_path,
            object_z_offset,
            bool(save_heatmaps),
            heatmap_dir,
            bool(heatmap_per_waypoint),
            use_viser,
            viser_port,
            attempt_number,
            trajectory_occlusion_rate,
            effective_occlusion_filter,
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
