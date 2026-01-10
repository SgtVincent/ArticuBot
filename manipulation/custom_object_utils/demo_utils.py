"""Utilities for generating demonstrations for custom objects."""
from __future__ import annotations

import multiprocessing as mp
import pathlib
import time
import yaml
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional

from manipulation.rm4d_filtering import (
    TrajectoryReachabilityFilter,
    load_or_create_rm4d_map,
    load_object_trajectory,
    score_trajectory_reachability,
    filter_object_pose_by_trajectory,
)

from manipulation.utils import build_up_env_gen
from manipulation.utils import parse_center
from manipulation.custom_object_utils.object_utils import (
    detect_asset_version,
    determine_reward_asset_path,
    serialize_vector,
)
import manipulation.envs.articulated as articulated_env_module
from manipulation.gen_demo import _execute as base_inner_execute
import manipulation.custom_object_utils.primitive_api as custom_api
# Optional on-demand GraspGen integration
from manipulation.custom_object_utils.graspgen_client import (
    GraspGenConfig,
    predict_grasps_for_urdf_folder,
    prepare_urdf_with_joint_state,
    load_predicted_grasps_yaml,
)


def get_graspgen_host_root_for_asset(asset_dir: pathlib.Path) -> str:
    """
    #TODO: Call the dockerized graspgen client to do path conversion instead of hardcoding logic here.
    #TODO: Maybe mount docker volume to ArticuBot repo root instead of just GraspGenModels for easier path mapping.
    Compute the GraspGen host root path for path conversion.
    
    For the Docker container to access assets, they must be under the 
    GraspGenModels directory (or a symlink to it). This function returns
    the appropriate host_graspgen_root based on asset layout.
    
    Expected Docker mount: /code/GraspGenModels -> GraspGen/GraspGenModels
    With symlink: GraspGenModels/custom_objects_v2 -> ArticuBot/data/custom_objects_v2
    
    Args:
        asset_dir: Path to the asset directory.
        
    Returns:
        The host path that maps to /code/GraspGenModels in the container.
    """
    asset_dir = pathlib.Path(asset_dir).resolve()
    version = detect_asset_version(asset_dir)
    
    # Look for GraspGenModels in the path hierarchy
    # This handles both direct assets and symlinked assets
    for parent in [asset_dir] + list(asset_dir.parents):
        if parent.name == "GraspGenModels":
            # Found GraspGenModels, return its parent + GraspGenModels
            return str(parent)
        # Also check for custom_objects_v2/sim/<object> pattern
        if parent.name == "custom_objects_v2":
            # Return path to GraspGenModels which should contain custom_objects_v2 symlink
            # Typical structure: GraspGen/GraspGenModels/custom_objects_v2 (symlink)
            graspgen_root = pathlib.Path("/home/junting/repo/articulated_objects/GraspGen/GraspGenModels")
            if graspgen_root.exists():
                return str(graspgen_root)
    
    # Fallback: for v1 assets, asset is typically in custom_objects under GraspGenModels
    # v1: GraspGenModels/custom_objects/sim_microwave_good
    # v2: GraspGenModels/custom_objects_v2/sim/microwave_7167
    if version == 2:
        # For v2, asset_dir is like: .../custom_objects_v2/sim/microwave_7167
        # We need to find the root that contains custom_objects_v2
        # Return the parent of custom_objects_v2 as the graspgen root
        for parent in asset_dir.parents:
            if parent.name == "data":
                # This is ArticuBot/data, but GraspGen mounts GraspGenModels
                # Return the GraspGenModels path that has symlink to this
                return "/home/junting/repo/articulated_objects/GraspGen/GraspGenModels"
    else:
        # v1: go up 3 levels from URDF parent
        # URDF at asset_dir/*.urdf, parent.parent.parent is GraspGenModels
        return str(asset_dir.parent.parent)


def _resolve_relative_path(path_value: pathlib.Path | str, base_dir: pathlib.Path) -> pathlib.Path:
    """Return an absolute path, interpreting relative inputs against ``base_dir``."""
    candidate = pathlib.Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve(strict=False)

# Export a public name expected by callers
def resolve_relative_path(path_value: pathlib.Path | str, base_dir: pathlib.Path) -> pathlib.Path:
    return _resolve_relative_path(path_value, base_dir)


def parse_config_metadata(config_path: str) -> Tuple[str, str, str, str]:
    """Parse object name, handle name, annotation path, and solution path from config.
    
    Returns:
        Tuple of (object_name, handle_name, annotation_path, solution_path)
    """
    config = yaml.safe_load(open(config_path, "r"))
    object_name = None
    handle_name = None
    solution_path = None
    annotation_path = None
    urdf_path = None
    for block in config:
        if "name" in block:
            object_name = block["name"].lower()
            handle_name = block.get("handle_name", "handle").lower()
            annotation_path = block.get("annotation_path")
            urdf_path = block.get("urdf_path")
        if "solution_path" in block:
            solution_path = block["solution_path"]
    if object_name is None or solution_path is None:
        raise ValueError(f"Malformed config: {config_path}")
    annotation_str = str(annotation_path) if annotation_path else ""
    return str(object_name), str(handle_name), annotation_str, str(solution_path)


def create_variant_config(
    base_config: str,
    asset_dir: pathlib.Path,
    variant_idx: int,
    handle_name: str,
    annotation_path: pathlib.Path,
    center_jitter: float,
    size_range: Tuple[float, float],
    urdf_path: Optional[pathlib.Path] = None,
    predicted_grasps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    target_position: Optional[Tuple[float, float, float]] = None,
) -> pathlib.Path:
    """Create a variant config with randomized parameters.
    
    Args:
        target_position: Target world xy position for object placement.
            Defaults to (0.4, 0.0, 0.0) which is in front of the robot.
            Note: 'center' in config is the URDF's geometric center,
            while target_position is the desired world placement.
    """
    config = yaml.safe_load(open(base_config, "r"))
    new_config = []
    reward_asset_path = determine_reward_asset_path(asset_dir)
    
    # Unpack predicted grasps if available
    grasps, confidences = predicted_grasps if predicted_grasps is not None else (None, None)
    
    # Default target position: in front of robot, within arm workspace
    if target_position is None:
        target_position = (0.4, 0.0, 0.0)
    
    for block in config:
        block = dict(block)
        scale = 1.0
        if ("name" in block or "type" in block) and urdf_path is not None:
            block["urdf_path"] = str(urdf_path)

        if "size" in block:
            # Scale randomization based on the base size
            base_size = float(block["size"])
            scale = np.random.uniform(size_range[0], size_range[1])
            block["size"] = base_size * scale
            
        if "center" in block:
            # Note: 'center' here is kept as URDF geometric center for initial loading
            # The actual world placement is controlled by target_position
            center_value = block["center"]
            if isinstance(center_value, str):
                center = parse_center(center_value)
            else:
                center = np.array(center_value, dtype=float)
            center[0] += np.random.uniform(-center_jitter, center_jitter)
            center[1] += np.random.uniform(-center_jitter, center_jitter)
            block["center"] = serialize_vector(center.tolist())
            
            # Add target_position for _custom_gen_init_state to use
            block["target_position"] = serialize_vector(list(target_position))
            
        if "annotation_path" in block:
            block["annotation_path"] = str(annotation_path)
            
        # Inject computed handle pts path for runtime use by env
        parts_render = asset_dir / "parts_render"
        pt_files = list(parts_render.glob(f"*{handle_name}_points.npy")) if parts_render.exists() else []
        if pt_files:
            block["handle_pts_path"] = str(pt_files[0])
            
        if "handle_name" in block:
            block["handle_name"] = handle_name
            block["link_name"] = handle_name
            
        if "solution_path" in block:
            block["solution_path"] = str(asset_dir)

        if "reward_asset_path" in block:
            block["reward_asset_path"] = reward_asset_path
            
        # Inject predicted grasp if available
        if grasps is not None and confidences is not None and len(grasps) > 0:
            # Select a grasp (e.g., based on confidence or random)
            # For now, let's pick one randomly weighted by confidence
            probs = confidences / np.sum(confidences)
            idx = np.random.choice(len(grasps), p=probs)
            selected_grasp = grasps[idx]
            
            # Store the selected grasp in the config
            # We need to serialize it. Let's store position and quaternion.
            # Scale the position because the object size is scaled
            pos = selected_grasp[:3, 3] * scale
            rot = R.from_matrix(selected_grasp[:3, :3])
            quat = rot.as_quat() # x, y, z, w
            
            block["predicted_grasp_position"] = serialize_vector(pos.tolist())
            block["predicted_grasp_orientation"] = serialize_vector(quat.tolist())
            
        new_config.append(block)
        
    configs_dir = asset_dir / "configs"
    configs_dir.mkdir(exist_ok=True)
    variant_path = configs_dir / f"config_custom_{variant_idx:04d}.yaml"
    with variant_path.open("w", encoding="utf-8") as fp:
        yaml.dump(new_config, fp, sort_keys=False)
    return variant_path


def _custom_gen_init_state(
    q,
    config_path,
    env_name,
    render,
    far_distance=0.7,
    near_distance=0.3,
    rm4d_map_path: Optional[str] = None,
    object_traj_path: Optional[str] = None,
    coverage_threshold: float = 0.8,
    reachability_threshold: float = 0.5,
):
    """Generate initial state for custom object demo.
    
    For custom objects, the 'center' in config is the URDF's geometric center offset,
    NOT the target world placement position. We use 'target_position' for world 
    placement (defaults to [0.4, 0.0, 0.0] which is in front of the robot).
    Optional RM4D map and object trajectory allow rejecting sampled object
    placements that fall outside the robot workspace using inverse reachability
    filtering before attempting motion planning.
    """
    config = yaml.safe_load(open(config_path, "r"))
    object_name = None
    base_euler = None
    target_position = None  # Target world placement position
    predicted_grasp_pos = None
    predicted_grasp_orn = None
    
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'euler' in config_dict:
            base_euler = parse_center(config_dict['euler'])
        # 'target_position' is the desired world placement xy position
        # This is different from 'center' which is the URDF geometric center
        if 'target_position' in config_dict:
            target_position = parse_center(config_dict['target_position'])
        if 'predicted_grasp_position' in config_dict:
            predicted_grasp_pos = parse_center(config_dict['predicted_grasp_position'])
            predicted_grasp_orn = parse_center(config_dict['predicted_grasp_orientation'])
        # capture urdf path if present (for on-demand GraspGen)
        if 'urdf_path' in config_dict and 'urdf_path' not in locals():
            urdf_path_for_graspgen = config_dict['urdf_path']
            
    if object_name is None:
        q.put(False)
        return False
        
    # Default target placement position: in front of the robot, reachable by the arm
    if target_position is None:
        target_position = np.array([0.4, 0.0, 0.0])
        
    if base_euler is None:
        base_euler = np.array([0.0, 0.0, 0.0])

    # Load RM4D reachability map and object trajectory for filtering
    rm4d_map = None
    traj_obj = None
    traj_filter = None
    if rm4d_map_path is not None:
        try:
            rm4d_map = load_or_create_rm4d_map(rm4d_map_path)
        except Exception:
            rm4d_map = None
    if object_traj_path is not None:
        try:
            traj_obj = load_object_trajectory(object_traj_path)
        except Exception:
            traj_obj = None
    
    # Create trajectory filter if both map and trajectory are available
    if rm4d_map is not None and traj_obj is not None:
        traj_filter = TrajectoryReachabilityFilter(
            rmap=rm4d_map,
            coverage_threshold=coverage_threshold,
            intersection_threshold=reachability_threshold,
        )

    # create env
    env, _ = build_up_env_gen(config_path, env_name, render=render)
    env.reset()

    # get the joint limits for the robot arm, using a smaller range
    initial_joint_angles = [0 for _ in range(7)]
    low = [-2.9, -1.8, -2.9, -3.1, -2.9, -0.0, -2.9]
    high = [2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9]
    for i in range(7):
        joint_range = high[i] - low[i]
        low[i] += joint_range * 0.2
        high[i] -= joint_range * 0.2    

    # Get the initial position and orientation of the object after env.reset()
    # Note: init_pos[2] is already adjusted by adjust_object_positions to sit on ground
    object_id = env.urdf_ids[object_name]   
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    init_euler = p.getEulerFromQuaternion(init_orient)
    
    # Get the object's height above ground (z coordinate after placement)
    object_z = init_pos[2]  # This is the proper z after adjust_object_positions

    # Interpret target_position[2] as an additional Z offset (meters).
    # Default target_position is [0.4, 0.0, 0.0], so behavior stays unchanged.
    try:
        target_z_offset = float(target_position[2])
    except Exception:
        target_z_offset = 0.0
    desired_object_z = object_z + target_z_offset

    good_init_pos = False
    new_pos = None
    new_orient = None
    
    start_time = time.time()
    while not good_init_pos:
        if time.time() - start_time > 60:  # Internal timeout
            break

        # Randomly sample the position around target_position.
        # Keep the object's z coordinate (already properly adjusted to sit on ground),
        # optionally adding a user-provided z offset.
        new_pos = np.array([
            target_position[0] + np.random.uniform(-0.1, 0.1),
            target_position[1] + np.random.uniform(-0.1, 0.1),
            desired_object_z
        ])

        # Random orientation around z-axis
        new_orient = p.getQuaternionFromEuler([
            init_euler[0], 
            init_euler[1], 
            base_euler[2] + np.random.uniform(-np.pi / 6, np.pi / 6)
        ])
        p.resetBasePositionAndOrientation(object_id, new_pos, new_orient, physicsClientId=env.id)

        # Use RM4D inverse reachability filtering if available
        if traj_filter is not None:
            R_bo = R.from_quat(new_orient).as_matrix()
            accepted, coverage, intersection = traj_filter.filter_pose(
                (R_bo, np.array(new_pos)), traj_obj
            )
            if not accepted:
                # Reject poses that fall outside the robot workspace
                continue

        if predicted_grasp_pos is not None:
            # Use predicted grasp
            grasp_pos_world, _ = p.multiplyTransforms(new_pos, new_orient, predicted_grasp_pos, predicted_grasp_orn)
            
            # Check robot collision and reachability
            for test_time in range(100): 
                for i in range(7):
                    initial_joint_angles[i] = np.random.uniform(low[i], high[i])   
                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, initial_joint_angles)
                for _ in range(5): p.stepSimulation()
                
                # Collision check
                contact_points = p.getContactPoints(env.robot.body, object_id, physicsClientId=env.id)
                closest_points = p.getClosestPoints(env.robot.body, object_id, distance=0.01, physicsClientId=env.id)
                if len(contact_points) > 0 or len(closest_points) > 0: 
                    continue

                link_contact = False
                num_links = p.getNumJoints(env.robot.body, physicsClientId=env.id)
                for link_idx in range(1, num_links):
                    contact_points = p.getClosestPoints(bodyA=env.robot.body, linkIndexA=link_idx, bodyB=env.urdf_ids['plane'], distance=0.01, physicsClientId=env.id)
                    if len(contact_points) > 0:
                        link_contact = True
                        break
                if link_contact: 
                    continue
                
                # Check distance from EEF to grasp_pos_world
                robot_eef_pos, _ = env.robot.get_pos_orient(env.robot.right_end_effector)
                distance = np.linalg.norm(np.array(grasp_pos_world) - np.array(robot_eef_pos))
                
                if distance < far_distance and distance > near_distance:
                    good_init_pos = True
                    break
        else:
            # No predicted grasp in config: try on-demand GraspGen if available
            # Determine URDF path
            urdf_path = locals().get('urdf_path_for_graspgen', None)
            if urdf_path is None:
                # try to find a urdf in the same folder as config
                cfg_parent = pathlib.Path(config_path).resolve().parent.parent
                candidates = list(pathlib.Path(cfg_parent).glob('*.urdf'))
                urdf_path = str(candidates[0]) if candidates else None

            if urdf_path is None:
                print("No URDF path available for on-demand GraspGen. Skipping.")
                break

            # Collect current joint states of the object
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

            # Prepare temporary URDF folder with joint states recorded
            # Place intermediate data inside the object's artifact folder so
            # the container mount can access it.
            import shutil, uuid
            urdf_parent = pathlib.Path(urdf_path).resolve().parent
            tmp_dir = str(urdf_parent / f"_tmp_{uuid.uuid4().hex[:8]}")
            pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            try:
                prepared = prepare_urdf_with_joint_state(urdf_path, joint_states, tmp_dir, scale=1.0)
                # Configure GraspGenConfig so host<->container path mapping matches runtime mounts
                # Detect asset version and compute correct GraspGen root path
                # For v2 assets: URDF is in urdf/ subfolder, asset_dir is 1 level up
                urdf_path_obj = pathlib.Path(urdf_path).resolve()
                version = detect_asset_version(urdf_path_obj.parent.parent) if urdf_path_obj.parent.name == "urdf" else 1
                
                if version == 2:
                    # v2: URDF at asset_dir/urdf/*.urdf, asset_dir is urdf_parent.parent
                    asset_dir = urdf_path_obj.parent.parent
                else:
                    # v1: URDF at asset_dir/*.urdf
                    asset_dir = urdf_path_obj.parent
                
                host_root = get_graspgen_host_root_for_asset(asset_dir)
                gg_cfg = GraspGenConfig(
                    host_graspgen_root=host_root,
                    container_graspgen_root="/code/GraspGenModels",
                )
                grasps, confidences = predict_grasps_for_urdf_folder(prepared, gg_cfg)
                if len(confidences) == 0:
                    shutil.rmtree(tmp_dir)
                    break

                # pick best grasp by confidence
                idx = int(np.argmax(confidences))
                sel = grasps[idx]
                pos = sel[:3, 3]
                rot = R.from_matrix(sel[:3, :3])
                quat = rot.as_quat()  # x,y,z,w
                predicted_grasp_pos = pos.tolist()
                predicted_grasp_orn = quat.tolist()
                shutil.rmtree(tmp_dir)
            except Exception as e:
                print(f"[GraspGen] Exception during on-demand grasp generation: {e}")
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
                break
            
    if not good_init_pos:
        q.put(False)
        return False

    # Save state
    initial_finger_angle = np.random.uniform(env.robot.finger_fully_close_joint_angle, env.robot.finger_fully_open_joint_angle)
    env.robot.set_gripper_open_position(env.robot.right_gripper_indices, [initial_finger_angle, initial_finger_angle], set_instantly=True)
    p.stepSimulation()

    config = yaml.safe_load(open(config_path, "r"))
    for config_dict in config:
        if 'center' in config_dict:
            # Save the actual world position (z=0 is ground level)
            saved_pos = [new_pos[0], new_pos[1], 0.0]
            config_dict['center'] = str(tuple(saved_pos))
            config_dict['orientation'] = str(tuple(new_orient))
            # Keep euler consistent with quaternion (many utilities read 'euler').
            try:
                config_dict['euler'] = str(tuple(R.from_quat(np.array(new_orient, dtype=float)).as_euler('xyz').tolist()))
            except Exception:
                pass
            config_dict['is_crop_size'] = False
            if 'initial_joint_angles' not in config_dict:
                config_dict['initial_joint_angles'] = str(tuple(initial_joint_angles))
            if 'initial_finger_angle' not in config_dict:
                config_dict['initial_finger_angle'] = initial_finger_angle
             
    with open(config_path, 'w') as f:
        yaml.dump(config, f, indent=4)
        
    env.close()
    q.put(True)
    return True


def custom_gen_init_state(
    config_path,
    env_name,
    render,
    far_distance=0.7,
    near_distance=0.3,
    timeout: float = 60.0,
    rm4d_map_path: Optional[str] = None,
    object_traj_path: Optional[str] = None,
    coverage_threshold: float = 0.8,
    reachability_threshold: float = 0.5,
):
    q = mp.Queue()
    proc = mp.Process(
        target=_custom_gen_init_state,
        args=(
            q,
            config_path,
            env_name,
            render,
            far_distance,
            near_distance,
            rm4d_map_path,
            object_traj_path,
            coverage_threshold,
            reachability_threshold,
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


def _custom_execute_wrapper(q, config_path, env_name, solution_path, experiment_path, time_string=None):
    # Monkeypatch approach_object_link_parallel in the subprocess
    articulated_env_module.approach_object_link_parallel = custom_api.approach_object_link_parallel
    
    # Call the original _execute
    base_inner_execute(q, config_path, env_name, solution_path, experiment_path, time_string)


def custom_execute(config_path, env_name, solution_path, experiment_path, timeout: float = 300.0):
    q = mp.Queue()
    proc = mp.Process(target=_custom_execute_wrapper, args=(q, config_path, env_name, solution_path, experiment_path, None))
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
