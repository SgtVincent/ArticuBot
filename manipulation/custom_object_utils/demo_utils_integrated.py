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

from manipulation.custom_object_utils.demo_utils import get_graspgen_host_root_for_asset


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
    use_viser: bool = False,
    viser_port: int = 8080,
    attempt_number: int = 0,
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
        object_z_offset: Z-offset to elevate object above ground.
        use_viser: Whether to enable interactive viser visualization.
        viser_port: Port number for viser server.
        attempt_number: Current attempt number for tracking.
    """
    cprint(f"[INTEGRATED] Parsing config: {config_path}", "cyan")
    
    # Parse config
    config = yaml.safe_load(open(config_path, "r"))
    object_name = None
    base_euler = None
    target_position = None
    predicted_grasp_pos = None
    predicted_grasp_orn = None
    object_scale = 1.0
    urdf_path_for_graspgen = None
    urdf_orientation_quat = None
    grasp_candidates_obj = None  # Optional list[(pos_obj_scaled, quat_obj_xyzw, confidence)]
    
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'euler' in config_dict:
            base_euler = parse_center(config_dict['euler'])
        if 'orientation' in config_dict:
            # Quaternion is stored in xyzw ordering across the codebase.
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
        cprint("[INTEGRATED] ERROR: object_name is None in config", "red")
        q.put(False)
        return False
        
    if target_position is None:
        target_position = np.array([0.4, 0.0, 0.0])
        cprint(f"[INTEGRATED] Using default target_position: {target_position}", "yellow")
        
    if base_euler is None:
        base_euler = np.array([0.0, 0.0, 0.0])
        
    cprint(f"[INTEGRATED] Object: {object_name}, scale: {object_scale}", "cyan")
    cprint(f"[INTEGRATED] Target position: {target_position}", "cyan")
    cprint(f"[INTEGRATED] Grasp in config: pos={predicted_grasp_pos}, orn={predicted_grasp_orn}", "cyan")
    cprint(f"[INTEGRATED] Object Z offset: {object_z_offset}", "cyan")

    # Create environment first (needed for on-demand GraspGen)
    env, _ = build_up_env_gen(config_path, env_name, render=render)
    env.reset()

    # Get object ID and current state
    object_id = env.urdf_ids[object_name]
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)

    # Use the *runtime* orientation/scale as ground truth.
    # - Orientation: env loads as config orientation (and may apply additional yaw if randomized).
    # - Scale: env may correct scaling based on AABB and stores it in env.simulator_sizes.
    try:
        base_euler = R.from_quat(np.array(init_orient, dtype=float)).as_euler('xyz')
    except Exception:
        pass
    try:
        effective_scale = float(getattr(env, "simulator_sizes", {}).get(object_name, object_scale))
        if abs(effective_scale - float(object_scale)) > 1e-6:
            cprint(
                f"[INTEGRATED] Using env effective scale {effective_scale:.6f} (yaml size {float(object_scale):.6f})",
                "yellow",
            )
        object_scale = effective_scale
    except Exception:
        pass

    # If config provides an orientation quaternion, prefer it as a sanity check.
    if urdf_orientation_quat is not None:
        try:
            cfg_euler = R.from_quat(np.array(urdf_orientation_quat, dtype=float)).as_euler('xyz')
            # Warn if config 'euler' disagrees with 'orientation' (common source of mismatch).
            if base_euler is not None and np.linalg.norm((cfg_euler - np.array(base_euler, dtype=float))) > 1e-3:
                cprint(
                    "[INTEGRATED] Note: config 'euler' differs from 'orientation'; using runtime quaternion-derived euler",
                    "yellow",
                )

            # Warn if runtime base orientation differs from config orientation.
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

    # If no predicted grasp, try on-demand GraspGen
    if predicted_grasp_pos is None or predicted_grasp_orn is None:
        print("No predicted grasp in config, attempting on-demand GraspGen...")
        
        # Determine URDF path
        urdf_path = urdf_path_for_graspgen
        if urdf_path is None:
            cfg_parent = pathlib.Path(config_path).resolve().parent.parent
            candidates = list(pathlib.Path(cfg_parent).glob('*.urdf'))
            urdf_path = str(candidates[0]) if candidates else None
        
        if urdf_path is None:
            print("No URDF path available for on-demand GraspGen.")
            env.close()
            q.put(False)
            return False
        
        # Collect current joint states
        joint_states = {}
        num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
        for ji in range(num_joints):
            jinfo = p.getJointInfo(object_id, ji, physicsClientId=env.id)
            jname = jinfo[1].decode('utf-8')
            jtype = jinfo[2]
            if jtype == p.JOINT_FIXED:
                continue
            jstate = p.getJointState(object_id, ji, physicsClientId=env.id)[0]
            joint_states[jname] = float(jstate)
        
        # Prepare temporary URDF folder in a location that the GraspGen container can see.
        # In the default setup, GraspGenModels is mounted to /models, not the ArticuBot repo.
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
                env.close()
                q.put(False)
                return False
            
            order = np.argsort(-np.asarray(confidences))
            idx = int(order[0])
            sel = grasps[idx]
            predicted_grasp_pos = sel[:3, 3].tolist()
            rot = R.from_matrix(sel[:3, :3])
            predicted_grasp_orn = rot.as_quat().tolist()

            # Keep top-k candidates for visualization.
            top_k = int(min(10, len(order)))
            grasp_candidates_obj = []
            for j in range(top_k):
                gi = int(order[j])
                g = grasps[gi]
                g_pos = np.asarray(g[:3, 3], dtype=float)
                g_quat = R.from_matrix(g[:3, :3]).as_quat().astype(float)  # xyzw
                g_conf = float(confidences[gi])
                grasp_candidates_obj.append((g_pos, g_quat, g_conf))

            print(f"GraspGen found {len(grasps)} grasps, using best one (top_k={top_k} for viser)")
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"On-demand GraspGen failed: {e}")
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
            env.close()
            q.put(False)
            return False

    # Create integrated inverse map sampler
    sampler = None
    if rm4d_map_path is not None:
        sampler = create_sampler_from_map(
            rm4d_map_path,
            xy_resolution=xy_resolution,
            theta_bins=theta_bins,
            aggregation_method=aggregation_method,
            coverage_threshold=coverage_threshold,
        )

    # Predicted grasp positions stored in configs are already in the *scaled* object frame
    # (see create_variant_config in demo_utils.py). Keep that convention for sampling,
    # but convert to an unscaled grasp pose when generating trajectories (which apply
    # scaling internally via `scale=`).
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
            print(f"Computed trajectory with {len(trajectory_obj)} waypoints")
        except Exception as e:
            print(f"Failed to compute trajectory: {e}")
            trajectory_obj = None

    if trajectory_obj is None:
        # Fall back to straight-line approach
        print("Falling back to straight-line approach trajectory")
        grasp_p = grasp_pos_obj_scaled
        grasp_R = R.from_quat(grasp_quat).as_matrix()
        approach_dir = grasp_R[:, 2]
        trajectory_obj = []
        for i in range(20):
            t = i / 19.0
            offset = approach_distance * (1 - t)
            p_new = grasp_p - approach_dir * offset
            trajectory_obj.append((grasp_R.copy(), p_new.copy()))

    # Get object's z-coordinate after placement (already have env from above)
    # object_id and init_pos already obtained earlier
    
    # Create sampling config
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

    # Create viser visualizer if requested
    viser_visualizer = None
    if use_viser:
        from manipulation.custom_object_utils.viser_visualization import IntegratedInverseMapVisualizer
        viser_visualizer = IntegratedInverseMapVisualizer(port=viser_port)
        cprint(f"[INTEGRATED] Viser server started on port {viser_port}", "cyan")

    # Run integrated sampling
    # - grasp_pos_obj_scaled: object-frame position matching the scaled object in PyBullet.
    # - grasp_quat: xyzw quaternion in object frame.
    grasp_pos_scaled = grasp_pos_obj_scaled
    
    cprint(f"[INTEGRATED] Starting integrated sampling with target_position={target_position}", "cyan")
    cprint(f"[INTEGRATED] Grasp pos (scaled): {grasp_pos_scaled}, quat: {grasp_quat}", "cyan")
    cprint(f"[INTEGRATED] Trajectory has {len(trajectory_obj)} waypoints", "cyan")
    cprint(f"[INTEGRATED] Sampler available: {sampler is not None}", "cyan")
    
    # Get object URDF path for viser visualization
    object_urdf_path = urdf_path_for_graspgen
    if object_urdf_path is None and asset_dir is not None:
        candidates = list(pathlib.Path(asset_dir).glob("*.urdf"))
        object_urdf_path = str(candidates[0]) if candidates else None
    
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
        viser_visualizer=viser_visualizer,
        attempt_number=attempt_number,
        object_urdf_path=object_urdf_path,
        object_scale=float(object_scale),
        grasp_candidates_obj=grasp_candidates_obj,
    )

    if not success:
        cprint("[INTEGRATED] Failed to find valid initial state", "red")
        env.close()
        q.put(False)
        return False
    
    cprint(f"[INTEGRATED] SUCCESS: obj_pos={obj_pos}, joint_angles={joint_angles[:3]}...", "green")

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
            # Save grasp pose for execution phase
            config_dict['predicted_grasp_position'] = str(tuple(grasp_pos_scaled.tolist()))
            config_dict['predicted_grasp_orientation'] = str(tuple(grasp_quat.tolist()))
            # Save z-offset for execution phase to elevate object correctly
            config_dict['object_z_offset'] = float(object_z_offset)
             
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)
        
    env.close()
    q.put(True)
    return True


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
    use_viser: bool = False,
    viser_port: int = 8080,
    attempt_number: int = 0,
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
            use_viser,
            viser_port,
            attempt_number,
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
