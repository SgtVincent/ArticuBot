import numpy as np
import pybullet_ompl.pb_ompl as pb_ompl
import pybullet as p
import copy
import os, pickle
import time
from termcolor import cprint

PLANNER = "RRTConnect" # BITstar

# add argument 'object_id' to indicate whether you need to plan the motion while the robot holding an object
def motion_planning(env, target_pos, target_orientation, planner=None, 
                obstacles=[], allow_collision_links=[], object_id=None, save_path=None, 
                robot_target_joint_angle=None, target_link=None, max_sampling_it=80, smooth_path=True,
                try_times=3, interpolation_num=None, ik_pos_tolerance: float = 0.001):
    np.random.seed(time.time_ns() % 2**32)
    if target_link is None:
        target_link = env.robot.right_end_effector

    
    current_joint_angles = copy.deepcopy(env.robot.get_joint_angles(indices=env.robot.right_arm_joint_indices))
    ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body, control_joint_idx=env.robot.right_arm_joint_indices, object_id=object_id, env=env)
    ompl_robot.set_state(current_joint_angles)

    # Clamp to OMPL robot bounds (these are the true planning bounds).
    joint_bounds = np.asarray(ompl_robot.joint_bounds, dtype=float)
    joint_lower = joint_bounds[:, 0]
    joint_upper = joint_bounds[:, 1]

    allow_collision_robot_link_pairs = []
    pb_ompl_interface = pb_ompl.PbOMPL(ompl_robot, obstacles, allow_collision_links, 
                                       allow_collision_robot_link_pairs=allow_collision_robot_link_pairs,
                                       object_id=object_id, interpolation_num=interpolation_num)

    paths = []
    path_translation_lengths, path_rotation_lengths = [], []
    for try_idx in range(try_times):
        ompl_robot.set_state(current_joint_angles)
        # first need to compute a collision-free IK solution
        ik_lower_limits = np.asarray(env.robot.ik_lower_limits, dtype=float)
        ik_upper_limits = np.asarray(env.robot.ik_upper_limits, dtype=float)
        ik_joint_ranges = ik_upper_limits - ik_lower_limits

        ik_success = False
        if robot_target_joint_angle is None:
            it = 0
            solutions = []
            while True:
                target_joint_solutions = []
                
                # Use current motor joints as a rest pose, then randomize arm joints.
                # This biases IK towards feasible, in-bounds solutions.
                ik_start_pose = np.asarray(env.robot.get_motor_joint_states()[1], dtype=float)
                arm_joint_indices = env.robot.right_arm_joint_indices
                ik_start_pose[arm_joint_indices] = np.random.uniform(
                    ik_lower_limits[arm_joint_indices],
                    ik_upper_limits[arm_joint_indices],
                    size=len(arm_joint_indices),
                )

                ompl_robot.set_state(ik_start_pose[arm_joint_indices])

                target_joint_angle = np.array(
                    p.calculateInverseKinematics(
                        env.robot.body,
                        target_link,
                        targetPosition=target_pos,
                        targetOrientation=target_orientation,
                        lowerLimits=ik_lower_limits.tolist(),
                        upperLimits=ik_upper_limits.tolist(),
                        jointRanges=ik_joint_ranges.tolist(),
                        restPoses=ik_start_pose.tolist(),
                        maxNumIterations=10000,
                        residualThreshold=1e-4,
                        physicsClientId=env.id,
                    ),
                    dtype=float,
                )
                target_joint_solutions.append(target_joint_angle)

                for target_joint_solution in target_joint_solutions:
                    target_joint_angle = np.array(target_joint_solution, dtype=float)[:len(arm_joint_indices)]
                    target_joint_angle = np.clip(target_joint_angle, joint_lower, joint_upper)
                    ompl_robot.set_state(target_joint_angle)

                    eef_pos, _ = env.robot.get_pos_orient(target_link)
                    ik_error = np.linalg.norm(np.asarray(eef_pos, dtype=float) - target_pos)

                    threshold = float(ik_pos_tolerance)
                    eps = 1e-3
                    cond1 = np.all(target_joint_angle >= (joint_lower - eps))
                    cond2 = np.all(target_joint_angle <= (joint_upper + eps))
                    cond3 = pb_ompl_interface.is_state_valid(target_joint_angle)
                    cond4 = ik_error < threshold
                    if cond1 and cond2 and cond3 and cond4:
                        ik_success = True
                        solutions.append(target_joint_angle)

                it += 1

                if it > max_sampling_it:
                    ompl_robot.set_state(current_joint_angles)
                    # ik_success = False
                    break
            
            if len(solutions) > 0:
                solutions = np.array(solutions)
                distance = np.linalg.norm(solutions - current_joint_angles, axis=1)
                best_idx = np.argmin(distance)
                target_joint_angle = solutions[best_idx]
                ik_success = True
        else:
            target_joint_angle = robot_target_joint_angle
            ik_success = True
            
            
        if not ik_success:
            continue
        else:
            cprint(f"[MP] try_idx: {try_idx}, IK succeeded with {len(solutions) if robot_target_joint_angle is None else 1} solutions", "cyan")
        
        for planner in ["RRTstar", "BITstar", "ABITstar"]:
            pb_ompl_interface.set_planner(planner)
            # then plan using ompl
            assert len(target_joint_angle) == ompl_robot.num_dim
            assert pb_ompl_interface.is_state_valid(target_joint_angle)

            ompl_robot.set_state(current_joint_angles)
            res, path = pb_ompl_interface.plan(target_joint_angle, smooth_path=smooth_path)
            ompl_robot.set_state(current_joint_angles)

            if not res:
                pass  # Motion planning failed to find a path
            else:
                paths.append(path)
                translation_length, rotation_length = get_path_length(env, path)
                path_translation_lengths.append(translation_length)
                path_rotation_lengths.append(rotation_length)
                ompl_robot.set_state(current_joint_angles)
                # cprint(f"try_idx: {try_idx}, planner: {planner}, translation length: {translation_length}, rotation length: {rotation_length}", "red")
    
    if len(paths) == 0:
        return None, None, None, None
    
    ompl_robot.set_state(current_joint_angles)    
    rank_translation = np.argsort(path_translation_lengths)
    rank_rotation = np.argsort(path_rotation_lengths)
    total_rank = rank_translation + rank_rotation
    best_idx = np.argmin(total_rank)

    best_idx = np.argmin(path_translation_lengths)
    path = paths[best_idx]
        
    if save_path is not None:
        with open(os.path.join(save_path, "target_joint_angle.pkl"), "wb") as f:
            pickle.dump(target_joint_angle, f)
        with open(os.path.join(save_path, "current_joint_angle.pkl"), "wb") as f:
            pickle.dump(current_joint_angles, f)

    return True, path, path_translation_lengths[best_idx], path_rotation_lengths[best_idx]

def get_path_length(env, path):
    cur_pos, cur_orient = env.robot.get_pos_orient(env.robot.right_end_effector)
    length_pos = 0
    length_orient = 0
    for idx, q in enumerate(path):
        env.robot.set_joint_angles(env.robot.right_arm_joint_indices, q)
        pos, orient = env.robot.get_pos_orient(env.robot.right_end_effector)
        length_pos += np.linalg.norm(pos - cur_pos)
        
        temp = 2 * np.dot(orient, cur_orient)**2 - 1
        temp = np.clip(temp, -1, 1)
        length_orient += np.arccos(temp)
        cur_pos, cur_orient = pos, orient
    return length_pos, length_orient
    

def motion_planning_joint_angle(env, target_joint_angle, planner="BITstar", obstacles=[], allow_collision_links=[]):
    current_joint_angles = copy.deepcopy(env.robot.get_joint_angles(indices=env.robot.right_arm_joint_indices))
    ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body, control_joint_idx=env.robot.right_arm_joint_indices)
    # ompl_robot = pb_ompl.PbOMPLRobot(env.robot.body)
    ompl_robot.set_state(current_joint_angles)
    pb_ompl_interface = pb_ompl.PbOMPL(ompl_robot, obstacles, allow_collision_links)
    pb_ompl_interface.set_planner(planner)
        
    #  plan using ompl
    assert len(target_joint_angle) == ompl_robot.num_dim
    for idx in range(ompl_robot.num_dim):
        print("joint: ", idx, " lower limit: ", ompl_robot.joint_bounds[idx][0], " upper limit: ", ompl_robot.joint_bounds[idx][1], " target: ", target_joint_angle[idx])
        assert (ompl_robot.joint_bounds[idx][0] <= target_joint_angle[idx]) & (target_joint_angle[idx] <= ompl_robot.joint_bounds[idx][1])

    res, path = pb_ompl_interface.plan(target_joint_angle, smooth_path=True)
    
    if not res:
        print("motion planning failed to find a path")

    return res, path