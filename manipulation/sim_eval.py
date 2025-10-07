import numpy as np
import pybullet as p
import gym
from gym.utils import seeding
from gym import spaces
import pickle
import yaml
import os.path as osp
from collections import defaultdict
import matplotlib.pyplot as plt
import open3d 
from termcolor import cprint
import time
import scipy
import os
from manipulation.gpt_primitive_api import get_pc_num_within_gripper
from manipulation.utils import parse_config, load_env, save_env, radial_shift
from manipulation.gpt_reward_api import get_joint_id_from_name, get_handle_pos, get_link_pc
from manipulation.gpt_primitive_api import get_link_handle
from scipy.spatial.transform import Rotation as R
from manipulation.panda import Panda
from typing import List, Optional
from manipulation.sim_base import SimpleEnvBase

class SimpleEnvEval(SimpleEnvBase):
    def __init__(self, 
                    dt=1/240, 
                    config_path=None, 
                    gui=False, 
                    control_step=2, 
                    horizon=250, 
                    restore_state_file=None, 
                    vhacd=True, # if to perform vhacd on the object for better collision detection for pybullet
                    randomize=0, # if to randomize the scene
                    obj_id=0, # which object to choose to use from the candidates
                    mobile=False,
                    task_name=None,
                    open_gripper_at_reset=True,
                    ik_limit=True, 
                ):
        # Call parent constructor with all parameters
        super().__init__(
            dt=dt,
            config_path=config_path,
            gui=gui,
            control_step=control_step,
            horizon=horizon,
            restore_state_file=restore_state_file,
            vhacd=vhacd,
            randomize=randomize,
            obj_id=obj_id,
            task_name=task_name,
            open_gripper_at_reset=open_gripper_at_reset,
            ik_limit=ik_limit,
            mobile=mobile
        )
    
    def after_robot_initialized(self):
        """Set suction_id after robot is initialized."""
        self.suction_id = self.robot.right_gripper_indices[0]
        
    def _parse_config(self):
        """Override to use different parse_config parameters."""
        return parse_config(
            self.config,
            obj_id=self.obj_id,
            use_vhacd=True
        )
    
    def apply_articulated_joint_angles(self, articulated_init_joint_angles):
        """Override to handle GPT joint angles for evaluation."""
        self.handle_gpt_joint_angle(articulated_init_joint_angles)
    
    def handle_gpt_joint_angle(self, articulated_init_joint_angles):
        for name in articulated_init_joint_angles:
            obj_id = self.urdf_ids[name.lower()]

            if "set_joint_angle_joint_id" not in articulated_init_joint_angles[name].keys():
                for joint_name, joint_angle in articulated_init_joint_angles[name].items():
                    joint_idx = get_joint_id_from_name(self, name.lower(), joint_name)
                    joint_limit_low, joint_limit_high = p.getJointInfo(obj_id, joint_idx, physicsClientId=self.id)[8:10]
                    if joint_limit_low > joint_limit_high:
                        joint_limit_low, joint_limit_high = joint_limit_high, joint_limit_low
                    if 'random' not in joint_angle:
                        joint_angle = joint_limit_low
                    else:
                        joint_angle = self.np_random.uniform(joint_limit_low, joint_limit_high)
                    p.resetJointState(obj_id, joint_idx, joint_angle, physicsClientId=self.id)
            else:
                # TODO: account for cases when there are multiple joints to be set.
                p.resetJointState(obj_id, articulated_init_joint_angles[name]["set_joint_angle_joint_id"], 
                              articulated_init_joint_angles[name]['set_joint_angle_joint_angle'], physicsClientId=self.id)

    def take_direct_action(self, actions, gains=None, forces=None, save_img_interval=0, ik_try_times=25, far_target=False):
        if gains is None:
            gains = [a.motor_gains for a in self.agents]
        elif type(gains) not in (list, tuple): 
            gains = [gains]*len(self.agents)
        if forces is None:
            forces = [a.motor_forces for a in self.agents]
        elif type(forces) not in (list, tuple):
            forces = [forces]*len(self.agents)

        np.random.seed(time.time_ns() % 2**32)

        self.control_rgbs = []
        action_index = 0
        for i, agent in enumerate(self.agents):
            agent_action_len = self.base_action_space.shape[0] 
            action = np.copy(actions[action_index:action_index+agent_action_len])
            action_index += agent_action_len
            translation = action[:3] ### target position
            rotation = action[3:6] ### target orientation in euler angle
            finger_joint_angle = action[6] ### gripper joint angle
            original_joint_angles = agent.get_joint_angles(agent.all_joint_indices)

            ik_indices = [_ for _ in range(len(agent.right_arm_ik_indices))]
            
            pos = translation
            orient = p.getQuaternionFromEuler(rotation)

            ### try to get a ik solution using tracIK
            tracIK_solutions = agent.ik_tracik_franka(pos, orient, ik_indices)
            
            ### try to get a list of IK solutions using the bullet's default ik solver
            bullet_solutions = []
            old_state = save_env(self)
            ik_indices = [_ for _ in range(len(self.robot.right_arm_joint_indices))]
            for try_idx in range(ik_try_times):
                if try_idx > 0: 
                    new_joint_angles = original_joint_angles[ik_indices] + np.random.uniform(-0.3, 0.3, size=len(ik_indices))
                    self.robot.set_joint_angles(ik_indices, new_joint_angles)

                ik_joint_angles = self.robot.ik(self.robot.right_end_effector, pos, orient, ik_indices=ik_indices, max_iterations=10000, residualThreshold=1e-4)
                if np.all(ik_joint_angles >= self.robot.ik_lower_limits[ik_indices]) and np.all(ik_joint_angles <= self.robot.ik_upper_limits[ik_indices]):
                    bullet_solutions.append(ik_joint_angles)

            ### choose ik solutions that are close to current joint angles
            load_env(self, state = old_state)
            all_possible_solutions = tracIK_solutions + bullet_solutions
            if len(all_possible_solutions) > 0:
                all_possible_solutions = np.array(all_possible_solutions).reshape(-1, len(ik_indices))
                distance_to_cur_angle = np.linalg.norm(all_possible_solutions - original_joint_angles[agent.controllable_joint_indices].reshape(1, -1), axis=1)
                min_idx = np.argmin(distance_to_cur_angle)
                min_joint_distance = distance_to_cur_angle[min_idx]
                best_joint_angles = all_possible_solutions[min_idx]
                agent_joint_angles = best_joint_angles
                ik_success = min_joint_distance < 0.3
            else:
                ik_success = False

            self.ik_failure = (not ik_success) or self.ik_failure
            
            it = 0
            if ik_success:
                control_total = 50
                ### control gripper
                for _ in range(2):
                    agent.set_gripper_open_position(agent.right_gripper_indices, [finger_joint_angle, finger_joint_angle], set_instantly=False)
                    p.stepSimulation(physicsClientId=self.id) 
                    if save_img_interval > 0:
                        rgb = self.render()
                        self.control_rgbs.append(rgb)

                cur_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)

                old_err = 1e10
                ### control until the joint angles are close enough to the target joint angles
                while True:
                    agent.control(agent.controllable_joint_indices, agent_joint_angles)
                    cur_joint_angles = agent.get_joint_angles(agent.controllable_joint_indices)
                    err = np.linalg.norm(cur_joint_angles - agent_joint_angles) 
                    if err < 1e-4:
                        break

                    old_err = err
                    if save_img_interval > 0 and it % save_img_interval == 0:
                        rgb = self.render()
                        self.control_rgbs.append(rgb)

                    if it > control_total:
                        break
                    
                    it += 1
                    p.stepSimulation(physicsClientId=self.id) 
                    
                
                end = time.time()
            else:
                pass

    def _get_info(self):
        ### we focus on storagefurniture so it is hardcoded here
        object_name = 'storagefurniture'
        if self.handle_joint is None:
            all_handle_pos, all_handle_joint_id, handle_pts_obj_frame, mobility_info = get_handle_pos(self, object_name, return_median=False, return_info=True)
            self.handle_pts_obj_frame = handle_pts_obj_frame
            self.mobility_info = mobility_info
            link_name = "link_0"
            link_pc = get_link_pc(self, object_name, link_name)
            _, link_handle_joint_id, link_handle_median, min_link_idx = get_link_handle(all_handle_pos, all_handle_joint_id, link_pc)
            self.handle_joint = link_handle_joint_id
            self.handle_pos = link_handle_median
            self.min_link_idx = min_link_idx
            self.all_handle_points = all_handle_pos[min_link_idx]
        else:
            all_handle_pos, _ = get_handle_pos(self, object_name, return_median=False, handle_pts_obj_frame=self.handle_pts_obj_frame, mobility_info=self.mobility_info)
            handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
            self.handle_pos = handle_median_points[self.min_link_idx]
            self.all_handle_points = all_handle_pos[self.min_link_idx]
            
        opened_joint_angle = p.getJointState(self.urdf_ids[object_name], self.handle_joint, physicsClientId=self.id)[0]
        if self.init_joint_angle is None:
            self.init_joint_angle = opened_joint_angle
            
        cur_eef_pos, cur_eef_orient = self.robot.get_pos_orient(self.robot.right_end_effector)
        handle_points = self.all_handle_points
        num_handle_points_within_gripper = get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, handle_points)
        distance_eef_to_handle = np.linalg.norm(self.handle_pos.flatten() - cur_eef_pos.flatten())
        if num_handle_points_within_gripper > 0:
            points_left_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[0], physicsClientId=self.id)
            points_right_finger = p.getContactPoints(bodyA=self.robot.body, linkIndexA=self.robot.right_gripper_indices[1], physicsClientId=self.id)
            if len(points_left_finger) > 0 and len(points_right_finger) > 0:
                contact_points_left = np.array([point[6] for point in points_left_finger])
                contact_points_right = np.array([point[6] for point in points_right_finger])
                left_distance = scipy.spatial.distance.cdist(handle_points, contact_points_left)
                right_distance = scipy.spatial.distance.cdist(handle_points, contact_points_right)
                min_distance_left = np.min(left_distance)
                min_distance_right = np.min(right_distance)
                if min_distance_left < 0.02 or min_distance_right < 0.02:
                    grasped_handle = True
                    self.grasped_handle = self.grasped_handle or grasped_handle
        
        right_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[0])
        left_finger_pos, _ = self.robot.get_pos_orient(self.robot.right_gripper_indices[1])
        finger_distance = np.linalg.norm(right_finger_pos - left_finger_pos)
        
        return {
            "opened_joint_angle": opened_joint_angle,
            "improved_joint_angle": opened_joint_angle - self.init_joint_angle,
            "handle_pos": self.handle_pos, 
            "initial_joint_angle": self.init_joint_angle,
            "ik_failure": self.ik_failure,
            "grasped_handle": self.grasped_handle,
            "finger_distance": finger_distance, 
        }
    
    
if __name__ == "__main__":
    from manipulation.utils import build_up_env
    env = SimpleEnv(
        config_path="data/temp/open_the_door_of_the_storagefurniture_by_its_handle_StorageFurniture_46462_2024-03-27-23-35-10/task_open_the_door_of_the_storagefurniture_by_its_handle/experiment/0511-vary-obj-4-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first/2024-05-11-00-58-12/task_config.yaml",
        gui=True,
        mobile=True,
        control_step=10,
        max_translation=0.05,
    )
    
    env.reset()
    
    for t in range(10000):
        p.stepSimulation(physicsClientId=env.id)
