import pybullet as p
import os
import numpy as np
import open3d as o3d
from manipulation.motion_planning_utils import motion_planning
from manipulation.grasping_utils import align_gripper_z_with_normal, align_gripper_x_and_z
from manipulation.utils import *
import scipy
import time
from termcolor import cprint
import fpsample
import pickle
import json
from scipy.spatial.transform import Rotation as R

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


SAMPLE_ORIENTATION_NUM = _env_int("ARTICUBOT_SAMPLE_ORIENTATION_NUM", 3)
HANDLE_FPS_NUM_POINT = _env_int("ARTICUBOT_HANDLE_FPS_NUM_POINT", 15)

def get_save_path(simulator):
    state_save_path = os.path.join(simulator.primitive_save_path, "states")
    if not os.path.exists(state_save_path):
        os.makedirs(state_save_path)
    return simulator.primitive_save_path


def _resolve_predicted_grasp(simulator, object_name):
    # Check for predicted grasp in config
    predicted_grasp_pos = None
    predicted_grasp_ori = None
    
    if hasattr(simulator, 'config'):
        # Prefer the predicted grasp stored on the actual object config block.
        for block in simulator.config:
            block_name = str(block.get("name", "")).lower()
            if (
                block_name == object_name
                and "predicted_grasp_position" in block
                and "predicted_grasp_orientation" in block
            ):
                predicted_grasp_pos = parse_center(block["predicted_grasp_position"])
                predicted_grasp_ori = parse_center(block["predicted_grasp_orientation"])
                break

        # Fallback: first block that contains a grasp
        if predicted_grasp_pos is None or predicted_grasp_ori is None:
            for block in simulator.config:
                if "predicted_grasp_position" in block and "predicted_grasp_orientation" in block:
                    predicted_grasp_pos = parse_center(block["predicted_grasp_position"])
                    predicted_grasp_ori = parse_center(block["predicted_grasp_orientation"])
                    break
    return predicted_grasp_pos, predicted_grasp_ori


def _compute_handle_info(simulator, link_pc):
    all_handle_pos, handle_joint_id, axis_world, axis_end_world = simulator.get_handle_pos(return_median=False, custom_joint_name=simulator.handle_name)

    if isinstance(handle_joint_id, (int, np.integer)):
        handle_joint_id = [int(handle_joint_id)]
        
    handle_pc = np.zeros((0, 3))
    handle_median = None
    threshold_candidates = [0.02, 0.04, 0.06, 0.08, 0.10]
    handle_joint_ids = handle_joint_id
    best_n = -1
    for threshold in threshold_candidates:
        cand_handle_pc, cand_handle_joint_id, cand_handle_median, _ = get_link_handle(
            all_handle_pos, handle_joint_ids, link_pc, threshold=threshold
        )
        if cand_handle_pc.shape[0] > best_n:
            best_n = cand_handle_pc.shape[0]
            handle_pc = cand_handle_pc
            handle_joint_id = cand_handle_joint_id
            handle_median = cand_handle_median
        if cand_handle_pc.shape[0] > 5:
            break

    try:
        axis = np.asarray(axis_world[0], dtype=float)
        axis_end = np.asarray(axis_end_world[0], dtype=float)
    except Exception:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        axis_end = np.array([0.0, 0.0, 2.0], dtype=float)
        
    return handle_pc, handle_joint_id, handle_median, axis, axis_end

def _refine_handle_properties(handle_pc, axis, axis_end, all_handle_pos):
    handle_orientation = None
    handle_dir = None
    handle_median = None
    
    if handle_pc.shape[0] > 5:
        handle_orientation = get_handle_orient(handle_pc)
        distance_pc = pc_to_line_distance(handle_pc, axis, axis_end)

        max_distance = np.max(distance_pc)
        handle_clip_factor = (0.0, 1.0, 0.0, 1.0, 0.0)
        selected_idx_screw= np.where((distance_pc > max_distance * handle_clip_factor[0]) & (distance_pc < max_distance * handle_clip_factor[1]))[0]
        max_z = np.max(handle_pc[:, 2])
        min_z = np.min(handle_pc[:, 2])
        selected_idx_height = np.where((handle_pc[:, 2] > min_z + (max_z - min_z) * handle_clip_factor[2]) & (handle_pc[:, 2] < min_z + (max_z - min_z) * handle_clip_factor[3]))[0]
        selected_idx = np.intersect1d(selected_idx_screw, selected_idx_height)

        handle_pc = handle_pc[selected_idx]
        
        if handle_pc.shape[0] > 5:
            handle_dir = estimate_line_direction(handle_pc)
            handle_pc_project = np.dot(handle_pc, handle_dir)
            min_handle_pc_project = np.min(handle_pc_project)
            max_handle_pc_project = np.max(handle_pc_project)
            selected_idx = np.where((handle_pc_project > min_handle_pc_project + (max_handle_pc_project - min_handle_pc_project) * handle_clip_factor[4]) & (handle_pc_project < min_handle_pc_project + (max_handle_pc_project - min_handle_pc_project) * (1-handle_clip_factor[4])))[0]
            
            handle_pc = handle_pc[selected_idx]
            handle_median = np.median(handle_pc, axis=0)

    if handle_dir is None:
        axis_vec = np.asarray(axis_end, dtype=float) - np.asarray(axis, dtype=float)
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm < 1e-6:
            handle_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            handle_dir = axis_vec / axis_norm

    if handle_orientation is None:
        handle_orientation = "vertical" if abs(float(handle_dir[2])) > 0.7 else "horizontal"

    if handle_median is None:
        try:
            handle_median = np.median(np.asarray(all_handle_pos, dtype=float).reshape(-1, 3), axis=0)
        except Exception:
            handle_median = None
            
    return handle_pc, handle_orientation, handle_dir, handle_median


def _generate_candidates_from_predicted(simulator, object_name, predicted_grasp_pos, predicted_grasp_ori):
    object_body_id = simulator.urdf_ids[object_name]
    obj_pos, obj_ori = p.getBasePositionAndOrientation(object_body_id, physicsClientId=simulator.id)
    
    r_obj = R.from_quat(obj_ori)
    p_world = r_obj.apply(predicted_grasp_pos) + np.array(obj_pos)

    r_local = R.from_quat(predicted_grasp_ori)
    r_world_base = r_obj * r_local

    real_target_pos = p_world
    mat_grasp = r_world_base.as_matrix()
    z_axis = mat_grasp[:, 2]

    backoffs = [0.08, 0.10, 0.12]
    roll_angles = np.deg2rad([0.0, 90.0, 180.0, -90.0])
    
    mp_target_poses = []
    real_target_poses = []
    target_orientations = []

    for backoff in backoffs:
        mp_target_pos = real_target_pos - z_axis * backoff
        for roll in roll_angles:
            r_roll = R.from_rotvec(z_axis * roll)
            base_orientation = (r_roll * r_world_base)
            
            # Exact base candidate
            target_orientation = base_orientation.as_quat()
            mp_target_poses.append(mp_target_pos)
            real_target_poses.append(real_target_pos)
            target_orientations.append(target_orientation)
            
            # Add randomized jitter candidates (robustness similar to Original script)
            # Original script uses align_gripper_x_and_z with randomize=True.
            # Here we perturb the existing orientation slightly.
            for _ in range(SAMPLE_ORIENTATION_NUM):
                # Small random rotation perturbation (up to ~15 degrees)
                random_axis = np.random.uniform(-1, 1, 3)
                random_axis /= np.linalg.norm(random_axis)
                random_angle = np.random.uniform(-0.25, 0.25) # ~14 deg
                r_perturb = R.from_rotvec(random_axis * random_angle)
                
                target_orientation_rand = (r_perturb * base_orientation).as_quat()
                
                mp_target_poses.append(mp_target_pos)
                real_target_poses.append(real_target_pos)
                target_orientations.append(target_orientation_rand)
            
    return mp_target_poses, real_target_poses, target_orientations, [real_target_pos]


def _generate_candidates_from_fps(simulator, object_name, handle_pc, handle_median, handle_dir, object_pc, object_normal, handle_orientation):
    mp_target_poses = []
    real_target_poses = []
    target_orientations = []
    to_try_handle_points = []
    
    if handle_pc.shape[0] > 0:
        fps_point = HANDLE_FPS_NUM_POINT
        handle_fps_num_point = min(fps_point, len(handle_pc))
        h = min(3, int(np.log2(handle_fps_num_point))) if handle_fps_num_point > 1 else 1
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(handle_pc, handle_fps_num_point, h=h)
        to_try_handle_points = handle_pc[kdline_fps_samples_idx]
        to_try_handle_points = np.concatenate(
            [to_try_handle_points, np.asarray(handle_median).reshape(1, 3)], axis=0
        )
    else:
        to_try_handle_points = np.asarray(handle_median, dtype=float).reshape(1, 3)

    for target_pos in to_try_handle_points:
        nearest_point_idx = np.argmin(np.linalg.norm(object_pc - target_pos.reshape(1, 3), axis=1))
        align_normal = object_normal[nearest_point_idx]
        
        low, high = simulator.get_bounding_box(object_name)
        com = (low + high) / 2
        line = com - target_pos
        if np.dot(line, align_normal) > 0:
            align_normal = -align_normal
        
        for normal in [align_normal]:
            real_target_pos = target_pos + normal * -0.02
            mp_target_pos = target_pos + normal * 0.04

            for orientation_idx in range(SAMPLE_ORIENTATION_NUM):
                target_orientation = align_gripper_x_and_z(handle_dir, -normal, randomize=True).as_quat()
                mp_target_poses.append(mp_target_pos)
                real_target_poses.append(real_target_pos)
                target_orientations.append(target_orientation)
                
            target_orientation_1 = align_gripper_x_and_z(handle_dir, -normal, randomize=False, flip=False).as_quat()
            target_orientation_2 = align_gripper_x_and_z(handle_dir, -normal, randomize=False, flip=True).as_quat()
            mp_target_poses.extend([mp_target_pos, mp_target_pos]) 
            real_target_poses.extend([real_target_pos, real_target_pos])
            target_orientations.extend([target_orientation_1, target_orientation_2])
            
    return mp_target_poses, real_target_poses, target_orientations, to_try_handle_points


def _evaluate_candidates(simulator, candidates_args, save_path, object_name, handle_joint_id, ori_simulator_state):
    ratio_threshold = 0.7
    early_stop_ratio = float(os.environ.get("ARTICUBOT_EARLY_STOP_RATIO", str(ratio_threshold)))
    min_success_ratio = float(os.environ.get("ARTICUBOT_MIN_SUCCESS_RATIO", "0.1"))

    best_result = None
    best_ratio = -np.inf

    for cand_idx, cand in enumerate(candidates_args):
        cprint(f"[DEBUG] Starting attempt {cand_idx}", "yellow")
        res = parallel_motion_planning(cand)
        try:
            ratio = float(res[0][0])
            score = res[1]
            cprint(f"[DEBUG] Attempt {cand_idx} finished. Ratio: {ratio}, Score: {score}", "yellow")
            
            if res[3]:
                 try:
                    debug_gif_path = os.path.join(save_path, f"attempt_{cand_idx}_score_{ratio:.3f}.gif")
                    save_numpy_as_gif(np.array(res[3]), debug_gif_path)
                 except Exception:
                    pass
        except Exception:
            ratio = -np.inf

        if ratio > best_ratio:
            best_ratio = ratio
            best_result = res
        
        if best_result is None:
             best_result = res

        if ratio >= early_stop_ratio:
            break

    if best_result is not None and best_ratio > min_success_ratio:
        best_ratio_tuple, best_score, best_traj_states, best_traj_rgbs, best_stage_length, _, _ = best_result

        with open(os.path.join(save_path, "best_score.txt"), "w") as f:
            f.write(str(best_score))

        state_files = []
        for t_idx, state in enumerate(best_traj_states):
            save_state_path = os.path.join(save_path, "states", f"state_{t_idx}.pkl")
            state_files.append(save_state_path)
            with open(save_state_path, "wb") as f:
                pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

        joint_limit_low, joint_limit_high = p.getJointInfo(
            simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id
        )[8:10]
        best_opened_angle = best_ratio_tuple[1]
        with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
            f.write(str(best_opened_angle) + "\n")
            f.write(str(joint_limit_low) + "\n")
            f.write(str(joint_limit_high) + "\n")

        simulator.reset(ori_simulator_state)

        with open(os.path.join(save_path, "stage_lengths.json"), "w") as f:
            json.dump(best_stage_length, f, indent=4)

        return best_traj_rgbs, state_files
    
    # Failure case
    with open(os.path.join(save_path, "best_score.txt"), "w") as f:
        f.write(str(0))
    
    joint_limit_low, joint_limit_high = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[8:10]
    with open(os.path.join(save_path, "opened_angle.txt"), "w") as f:
        f.write(str(0) + "\n")
        f.write(str(joint_limit_low) + "\n")
        f.write(str(joint_limit_high) + "\n")
            
    load_env(simulator, state=ori_simulator_state)
    save_env(simulator, os.path.join(save_path,  "state_{}.pkl".format(0)))
    rgbs = [simulator.render()]
    state_files = [os.path.join(save_path,  "state_{}.pkl".format(0))]
    return rgbs, state_files


def approach_object_link_parallel(simulator, object_name, link_name, debug=False, disable_object_collision=False, kinematic_open_door=False):    
    save_path = get_save_path(simulator)
    ori_simulator_state = save_env(simulator, None)
    object_name = object_name.lower()
    link_name = link_name.lower()
    link_pc, view, projection, img = simulator.get_link_pc(object_name, link_name)
    object_pc = link_pc
    
    predicted_grasp_pos, predicted_grasp_ori = _resolve_predicted_grasp(simulator, object_name)

    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(object_pc)
    pcd.estimate_normals()
    object_normal = np.asarray(pcd.normals)

    handle_pc, handle_joint_id, handle_median, axis, axis_end = _compute_handle_info(simulator, link_pc)
    
    # We need all_handle_pos for fallback, but it's local in _compute_handle_info unless we return it.
    # Actually, we can just get it again or pass it out. simpler to fetch again inside helper if needed or just pass it out.
    # BUT wait, _compute_handle_info called get_handle_pos.
    # Let's adjust helper to just use the one call.
    # Or for now, we can just proceed.
    # Refine handle properties
    all_handle_pos, _, _, _ = simulator.get_handle_pos(return_median=False, custom_joint_name=simulator.handle_name)
    handle_pc, handle_orientation, handle_dir, handle_median = _refine_handle_properties(handle_pc, axis, axis_end, all_handle_pos)

    # Fallback if no handle found
    if handle_pc.shape[0] <= 5 and handle_median is None:
        if predicted_grasp_pos is None:
            print("No handle point cloud/median found, return")
            return [], []
        print("No handle point cloud/median found; falling back to predicted grasp")
        handle_median = np.asarray(predicted_grasp_pos, dtype=float)

    use_predicted_grasp = (predicted_grasp_pos is not None and predicted_grasp_ori is not None)

    mp_target_poses, real_target_poses, target_orientations = [], [], []
    to_try_handle_points = []

    # 1. FPS Sampling (Try FIRST for robustness, matching original script)
    # This ensures we have a dense coverage of the handle even if predicted grasp is off.
    if handle_pc.shape[0] > 0 or handle_median is not None:
        mp_f, real_f, orient_f, handle_f = _generate_candidates_from_fps(
            simulator, object_name, handle_pc, handle_median, handle_dir, object_pc, object_normal, handle_orientation
        )
        mp_target_poses.extend(mp_f)
        real_target_poses.extend(real_f)
        target_orientations.extend(orient_f)
        to_try_handle_points.extend(handle_f)

    # 2. Predicted Grasps (Try SECOND as augmentation)
    if use_predicted_grasp:
        mp_p, real_p, orient_p, handle_p = _generate_candidates_from_predicted(
             simulator, object_name, predicted_grasp_pos, predicted_grasp_ori
        )
        mp_target_poses.extend(mp_p)
        real_target_poses.extend(real_p)
        target_orientations.extend(orient_p)
        to_try_handle_points.extend(handle_p)

    # Debug plotting
    if debug:
        import cv2
        def project_point(points_world, view_matrix, proj_matrix, width=640, height=480):
            view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
            proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
            proj_view_matrix = proj_matrix @ view_matrix
            points_world = np.append(points_world, 1)
            points_clip = (proj_view_matrix @ points_world.T).T
            ndc = points_clip[:3] / points_clip[3]
            x = (ndc[0] + 1) * 0.5 * width
            y = (1 - ndc[1]) * 0.5 * height
            return x,y

        img_dbg = img.astype(np.uint8)
        for pos in to_try_handle_points:
            screen_x, screen_y = project_point(pos, view, projection)
            cv2.circle(img_dbg, (int(screen_x), int(screen_y)), radius=2, color=(0, 255, 0), thickness=-1)
        cv2.imwrite("img.png", img_dbg)

    args_list = [[
        simulator,
        object_name,
        real_target_poses[it],
        mp_target_poses[it],
        target_orientations[it],
        handle_pc,
        handle_joint_id,
        save_path,
        ori_simulator_state,
        it,
        link_name,
        False, # disable_object_collision (forced False)
        False  # kinematic_open_door (forced False)
    ] for it in range(len(target_orientations))]

    return _evaluate_candidates(simulator, args_list, save_path, object_name, handle_joint_id, ori_simulator_state)

def reach_till_contact(simulator, real_target_pos, target_orientation, return_contact_pose=False):
    intermediate_states = []
    rgbs = []
    cur_eef_pos, _ = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    moving_vector = real_target_pos - cur_eef_pos
    delta_movement = 0.005
    movement_steps = int(np.linalg.norm(moving_vector) / delta_movement) + 1
    moving_direction = moving_vector / np.linalg.norm(moving_vector)
    target_orient_euler = p.getEulerFromQuaternion(target_orientation)
    for t in range(movement_steps):
        ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
        target_pos = cur_eef_pos + moving_direction * delta_movement * (t + 1)
        simulator.take_direct_action(np.array([*target_pos, *target_orient_euler, simulator.robot.finger_fully_open_joint_angle]))
        rgb = simulator.render()
        rgbs.append(rgb)
        state = save_env(simulator)
        intermediate_states.append(state)
        
        collision = False
        points_left_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)
        points_right_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)
        points_hand = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=8, physicsClientId=simulator.id)
        points = points_left_finger + points_right_finger + points_hand
        # In PyBullet contacts, index 5 is typically positionOnB.
        collision_points_a = [pt[5] for pt in points]
        if len(collision_points_a) > 0:
            p.addUserDebugPoints(collision_points_a, [[0, 1, 0] for _ in range(len(collision_points_a))], 12, 0.55, physicsClientId=simulator.id)
        if points:
            # Handle contact between suction with a rigid object.
            for point in points:
                # PyBullet layout: (bodyA, bodyB, linkA, linkB, posA, posB, ...)
                body_b = point[1]
                if body_b == simulator.urdf_ids['plane'] or body_b == simulator.robot.body:
                    pass
                else:
                    collision = True    
                    if return_contact_pose:
                        cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
                        return cur_eef_pos, cur_eef_orient
                    break
            
        if collision:
            # recover to the state where contact has not been made
            if len(intermediate_states) >= 3:
                simulator.reset(reset_state=intermediate_states[-3])
            break
    
    if len(intermediate_states) >= 3:
        return intermediate_states[:-2], rgbs[:-2]
    else:
        return intermediate_states, rgbs

def close_gripper(simulator, handle_pc, object_name=None, handle_joint_id=None):
    intermediate_states = []
    rgbs = []
    close_steps = 40 
    left_collision = False
    right_collision = False
    close_joint_angle = simulator.robot.finger_fully_close_joint_angle
    
    for t in range(close_steps):
        agent = simulator.robot
        agent.set_gripper_open_position(agent.right_gripper_indices, [close_joint_angle, close_joint_angle], set_instantly=False)
        p.stepSimulation(physicsClientId=simulator.id)
        state = save_env(simulator)
        intermediate_states.append(state)
        rgb = simulator.render()
        rgbs.append(rgb)
        
        # NOTE: update the score such that after closing, both gripper is in contact with the handle itself.
        points_left_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[0], physicsClientId=simulator.id)
        points_right_finger = p.getContactPoints(bodyA=simulator.robot.body, linkIndexA=simulator.robot.right_gripper_indices[1], physicsClientId=simulator.id)

        target_obj_id = None
        if object_name and hasattr(simulator, 'urdf_ids'):
            target_obj_id = simulator.urdf_ids.get(object_name)
        elif hasattr(simulator, 'obj_id'):
            target_obj_id = simulator.obj_id

        if points_left_finger and target_obj_id is not None:
            # Filter to contacts against the target object.
            left_contacts = [pt for pt in points_left_finger if pt[1] == target_obj_id]
            if left_contacts:
                # Prefer link-level match when available.
                if handle_joint_id is not None and any(pt[3] == handle_joint_id for pt in left_contacts):
                    left_collision = True
                elif handle_pc.shape[0] > 0:
                    collision_points_b = [pt[5] for pt in left_contacts]
                    dist_collision_to_handle = scipy.spatial.distance.cdist(collision_points_b, handle_pc).min(axis=1)
                    if np.sum(dist_collision_to_handle < 0.02) > 0:
                        left_collision = True
                else:
                    left_collision = True

        if points_right_finger and target_obj_id is not None:
            right_contacts = [pt for pt in points_right_finger if pt[1] == target_obj_id]
            if right_contacts:
                if handle_joint_id is not None and any(pt[3] == handle_joint_id for pt in right_contacts):
                    right_collision = True
                elif handle_pc.shape[0] > 0:
                    collision_points_b = [pt[5] for pt in right_contacts]
                    dist_collision_to_handle = scipy.spatial.distance.cdist(collision_points_b, handle_pc).min(axis=1)
                    if np.sum(dist_collision_to_handle < 0.02) > 0:
                        right_collision = True
                else:
                    right_collision = True
                
        if left_collision and right_collision:
            break
        
        
    return intermediate_states, rgbs, left_collision, right_collision

def open_gripper(simulator):
    intermediate_states = []
    rgbs = []
    open_steps = 40 
    open_joint_angle = simulator.robot.finger_fully_open_joint_angle
    for t in range(open_steps):
        agent = simulator.robot
        agent.set_gripper_open_position(agent.right_gripper_indices, [open_joint_angle, open_joint_angle], set_instantly=False)
        p.stepSimulation(physicsClientId=simulator.id)
        state = save_env(simulator)
        intermediate_states.append(state)
        rgb = simulator.render()
        rgbs.append(rgb)

        if p.getJointState(agent.body, agent.right_gripper_indices[0], physicsClientId=simulator.id)[0] > 0.037:
            break
    
    return intermediate_states, rgbs

def open_door(simulator, object_name, link_name, handle_joint_id, kinematic_open_door=False):
    intermediate_states = []
    rgbs = []
    
    eef_pos, eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    link_pos, link_orient = simulator.get_link_pose(object_name, link_name)
    world_to_link = p.invertTransform(link_pos, link_orient)
    # EEf in link frame remains the same as the link frame rotates
    eef_in_link = p.multiplyTransforms(world_to_link[0], world_to_link[1], eef_pos, eef_orient) 

    joint_limit = p.getJointInfo(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[8:10]
    ori_joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[0]
    eef_poses = []
    joint_angle_traj = []
    timesteps = 100 
    
    ratio = 0.8
    cur_joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[0]
    target = joint_limit[0] + ratio * (joint_limit[1] - joint_limit[0])
    cur_move_amount = target - cur_joint_angle
    full_move_amount = ratio * (joint_limit[1] - joint_limit[0])
    timesteps = int(timesteps * np.abs(cur_move_amount) / full_move_amount)
    for t in range(1, timesteps):
        joint_angle = cur_joint_angle + (target - cur_joint_angle) * t / timesteps
        p.resetJointState(simulator.urdf_ids[object_name], handle_joint_id, joint_angle, physicsClientId=simulator.id)
        new_link_pos, new_link_orient = simulator.get_link_pose(object_name, link_name)
        # new_link_pos, new_link_orient is the transformation from link coordinate to world coordinate
        new_eef_pos, new_eef_orient = p.multiplyTransforms(new_link_pos, new_link_orient, eef_in_link[0], eef_in_link[1])
        eef_poses.append([new_eef_pos, new_eef_orient])
        joint_angle_traj.append(joint_angle)
        
    
    p.resetJointState(simulator.urdf_ids[object_name], handle_joint_id, ori_joint_angle, physicsClientId=simulator.id)
    for t in range(len(eef_poses)):
        pos, orient = eef_poses[t]
        ik_indices = [_ for _ in range(len(simulator.robot.right_arm_joint_indices))]
        ik_joint_angles = simulator.robot.ik(simulator.robot.right_end_effector, 
                                        pos, orient, 
                                        ik_indices=ik_indices)

        ik_joint_angles = list(ik_joint_angles) + [0, 0]
        ik_joints = simulator.robot.right_arm_joint_indices + list(simulator.robot.right_gripper_indices)
        
        for _ in range(2):
            p.setJointMotorControlArray(simulator.robot.body, jointIndices=ik_joints, 
                                        controlMode=p.POSITION_CONTROL, targetPositions=ik_joint_angles, physicsClientId=simulator.id)
            p.stepSimulation(physicsClientId=simulator.id)
        
        rgb = simulator.render()
        rgbs.append(rgb)
        state = save_env(simulator)
        intermediate_states.append(state)
    
    final_joint_angle = p.getJointState(simulator.urdf_ids[object_name], handle_joint_id, physicsClientId=simulator.id)[0]
    # NOTE: change to return the ratio of the door opened to the upper limit
    joint_limit_high = joint_limit[1]
    final_joint_angle_ratio = (final_joint_angle - joint_limit[0]) / (joint_limit_high - joint_limit[0])
    return intermediate_states, rgbs, (final_joint_angle_ratio, final_joint_angle)

def parallel_motion_planning(args):
    np.random.seed(time.time_ns() % 2**32)

    # NOTE: Despite the name, this runs in the *current* simulator.
    # See approach_object_link_parallel for why we avoid multi-client + Pool.
    simulator, object_name, real_target_pos, mp_target_pos, target_orientation, \
        handle_pc, handle_joint_id, save_path, ori_simulator_state, \
        it, link_name, disable_object_collision, kinematic_open_door = args

    stage_length = {}
    object_name = object_name.lower()

    # Restore the simulator to the original pre-grasp state for this candidate.
    load_env(simulator, state=ori_simulator_state)
    for _ in range(3):
        p.stepSimulation(physicsClientId=simulator.id)

    intermediate_states = []
    rgbs = []
    
    # open gripper before reaching the handle
    open_gripper_states, open_gripper_rgbs = open_gripper(simulator)
    intermediate_states += open_gripper_states
    rgbs += open_gripper_rgbs

    all_objects = list(simulator.urdf_ids.keys())
    all_objects.remove("robot")
    # For pre-grasp planning we typically want to allow the end-effector to get very close
    # to the target object; treating the target object as an obstacle can make collision-free
    # IK infeasible and causes planning to fail.
    obstacles = [simulator.urdf_ids[x] for x in all_objects]
    # Allow gripper fingers AND hand base (palm) to collide with the object.
    # This prevents the motion planner from rejecting grasps where the palm is near the object surface.
    allow_collision_links = list(simulator.robot.right_gripper_indices)
    if hasattr(simulator.robot, 'right_hand'):
        allow_collision_links.append(simulator.robot.right_hand)
    cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)
    translation_length = np.linalg.norm(mp_target_pos - cur_eef_pos)
    rotation_length = 2 * np.arccos(np.abs(np.dot(target_orientation, cur_eef_orient)))
    rotation_length = np.rad2deg(rotation_length)
    translation_steps = int(translation_length / 0.004) + 1
    rotation_steps = int(rotation_length / 1.8) + 1
    interpolation_steps = max(translation_steps, rotation_steps)
    
    res, path, path_translation_length, path_rotation_length = motion_planning(
        simulator, mp_target_pos, target_orientation, obstacles=obstacles, allow_collision_links=allow_collision_links, save_path=save_path, 
        smooth_path=True, interpolation_num=interpolation_steps, ik_pos_tolerance=0.005)

    if res:
        stage_length['reach_handle'] = len(path) + len(open_gripper_states)

        for idx, q in enumerate(path):
            simulator.robot.set_joint_angles(simulator.robot.right_arm_joint_indices, q)
            rgb = simulator.render()
            rgbs.append(rgb)
            state = save_env(simulator)
            intermediate_states.append(state)

        # reach till contact is made, and get the number of handle points between the two fingers
        reach_to_concatc_states, reach_to_contact_rgbs = reach_till_contact(simulator, real_target_pos, target_orientation)
        intermediate_states += reach_to_concatc_states
        rgbs += reach_to_contact_rgbs
        stage_length['reach_to_contact'] = len(reach_to_contact_rgbs)
        
        # get a score for this grasping pose, which is the number of handle points between the two fingers
        cur_eef_pos, cur_eef_orient = simulator.robot.get_pos_orient(simulator.robot.right_end_effector)

        if handle_pc.shape[0] > 0:
            score = get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, handle_pc)
            
            # if not point is being grasped we directly return a failed score
            if score == 0:
                # This check is a useful heuristic, but can be overly strict for
                # some custom objects where handle point clouds are sparse/noisy.
                # Continue the attempt and let downstream collision checks and
                # opening ratio determine success.
                pass
        else:
            score = 10 # Dummy score

        close_states, close_rgbs, left_collision, right_collision = close_gripper(
            simulator, handle_pc, object_name=object_name, handle_joint_id=handle_joint_id
        )
        if not (left_collision and right_collision):
            score = 0
        intermediate_states += close_states
        rgbs += close_rgbs
        stage_length['close_gripper'] = len(close_states)
        

        cprint("iteration {} score {}".format(it, score), "green")

        # # pull out following the rotation axis
        open_door_states, open_door_rgbs, final_joint_angle_ratio = open_door(
            simulator, object_name, link_name, handle_joint_id,
            kinematic_open_door=kinematic_open_door
        )
        
        intermediate_states += open_door_states
        rgbs += open_door_rgbs
        stage_length['open_door'] = len(open_door_states)

        cprint(f"final joint angle ratio: {final_joint_angle_ratio}", "green")
        return final_joint_angle_ratio, score, intermediate_states, rgbs, stage_length, path_translation_length, path_rotation_length

    return (-1, -1), -1, [], [], {}, np.inf, np.inf
