"""Custom RobogenPointCloudWrapper for custom articulated objects.

This module provides a wrapper that inherits from RobogenPointCloudWrapper
and overrides methods that need special handling for custom objects with
different link naming conventions (e.g., 'link0' vs 'link_0').
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
import pybullet as p
from gym import spaces
from scipy.spatial.distance import cdist

from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from manipulation.utils import get_pc, rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
from manipulation.custom_object_utils.env_utils import get_handle_pos_custom


def get_link_pc_custom(simulator, object_name: str, link_name: str):
    """Get point cloud for a link, trying different naming conventions.
    
    This function attempts to find the link using various naming patterns
    common in custom objects (with and without underscores).
    
    Args:
        simulator: The simulation environment
        object_name: Name of the object
        link_name: Name of the link to get point cloud for
        
    Returns:
        Point cloud array for the link, or None if not found
    """
    from manipulation.utils.env_utils import get_link_pc, get_link_id_from_name
    
    object_name = object_name.lower()
    
    # Try the provided link name first
    link_candidates = [link_name]
    
    # Add alternative naming conventions
    if '_' in link_name:
        # Try without underscore: link_0 -> link0
        link_candidates.append(link_name.replace('_', ''))
    else:
        # Try with underscore: link0 -> link_0
        # Find where letters end and numbers begin
        for i, c in enumerate(link_name):
            if c.isdigit():
                link_candidates.append(link_name[:i] + '_' + link_name[i:])
                break
    
    for candidate in link_candidates:
        try:
            link_id = get_link_id_from_name(simulator, object_name, candidate)
            if link_id is not None:
                link_pc = get_link_pc(simulator, object_name, candidate)
                if link_pc is not None:
                    return link_pc
        except (TypeError, KeyError, IndexError):
            continue
    
    return None


class RobogenPointCloudWrapperCustom(RobogenPointCloudWrapper):
    """Custom wrapper for articulated objects with non-standard link names.
    
    This class extends RobogenPointCloudWrapper to handle custom objects
    that may have different link naming conventions (e.g., 'link0' instead
    of 'link_0') and uses custom utilities for handle position detection.
    
    It also overrides __init__ to avoid the parent class's attempt to load
    stage_lengths.json from the solution path (which doesn't exist for 
    custom objects during extraction).
    """
    
    def __init__(self, 
                 env, 
                 object_name,
                 handle_name: str = None,
                 asset_dir_hint: str = None,
                 rpy_mean_list=[[0, 0, -45], [0, 0, -135]], 
                 seed=None, 
                 num_points=4500,
                 horizon=400,
                 observation_mode=None,
                 camera_height=480,
                 camera_width=640,
                 camera_elevation=30,
                 only_object=True,
                 noise_real_world_pcd=False,
                 real_world_camera=False,
                 skip_goal_loading=True,
            ):
        """Initialize the custom wrapper.
        
        Args:
            env: The simulation environment
            object_name: Name of the articulated object
            handle_name: Name of the handle link (e.g., 'link1'). If None,
                        will try to detect from config or use 'handle'.
            asset_dir_hint: Optional path to the asset directory for loading
                           mobility info and handle points.
            skip_goal_loading: If True, skip loading goal states from stage_lengths.json
                              (used during extraction when goals are not available)
            **kwargs: Additional arguments passed to parent class
        """
        # Store custom object specific attributes before calling parent init
        self._handle_name = handle_name
        self._asset_dir_hint = asset_dir_hint
        self._skip_goal_loading = skip_goal_loading
        
        # Initialize without calling parent __init__ to avoid stage_lengths loading
        # We replicate the parent initialization logic here but skip the goal loading
        np.random.seed(time.time_ns() % 2**32)
        if seed is not None:
            np.random.seed(seed)
            
        self.noise_real_world_pcd = noise_real_world_pcd
        self.real_world_camera = real_world_camera

        self._env = env
        self._object_name = object_name
        self.horizon = horizon
        
        self.num_points = num_points
        self.observation_mode = observation_mode or 'act3d_goal_displacement_gripper_to_object'
        self.camera_elevation = camera_elevation
        self.camera_width = camera_width
        self.camera_height = camera_height

        ### setup observation space
        self.action_low = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        self.action_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.observation_space = spaces.Dict({
            'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
            'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32),
            'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32),
        })

        if 'goal' in self.observation_mode:
            self.observation_space['goal_gripper_pcd'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32
            )
        if 'displacement_gripper_to_object' in self.observation_mode:
            self.observation_space['displacement_gripper_to_object'] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32
            )

        if 'dp3' in self.observation_mode:
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
                'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32),
            })
        elif 'act3d' in self.observation_mode:
            self.observation_space = spaces.Dict({
                'point_cloud': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 1280, 3), dtype=np.float32),
                'agent_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 10), dtype=np.float32),
                'gripper_pcd': spaces.Box(low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32),
            })
            if 'goal' in self.observation_mode:
                self.observation_space['goal_gripper_pcd'] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32
                )
            if 'displacement_gripper_to_object' in self.observation_mode:
                self.observation_space['displacement_gripper_to_object'] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, 4, 3), dtype=np.float32
                )

        ### setup cameras
        min_aabb = np.array([0, 0, 0])  # Default values
        max_aabb = np.array([1, 1, 1])
        self.mean_camera_target = np.array([0.5, 0, 0.5])
        
        for name in self._env.urdf_ids:
            if name in ['robot', 'plane', 'init_table']:
                continue
            obj_id = self._env.urdf_ids[name]
            min_aabb, max_aabb = self._env.get_aabb(obj_id)
            center = (min_aabb + max_aabb) / 2
            self.mean_camera_target = center 
            break
        
        self.rpy_mean_list = [[-10, 0, -45], [-10, 0, -135]]
        self.mean_distance = np.linalg.norm(max_aabb - min_aabb) * 0.9
        self.camera_height = 256
        self.camera_width = 256

        self.depth_near = 0.01
        self.depth_far = 100
        self.view_matrices = []
        self.project_matrices = []
        for rpy_mean in self.rpy_mean_list:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.mean_camera_target, 
                distance=self.mean_distance, 
                yaw=rpy_mean[2], 
                pitch=rpy_mean[0], 
                roll=rpy_mean[1], 
                upAxisIndex=2, 
                physicsClientId=self._env.id
            )
            project_matrix = p.computeProjectionMatrixFOV(
                fov=60, 
                aspect=640/480, 
                nearVal=self.depth_near, 
                farVal=self.depth_far, 
                physicsClientId=self._env.id
            )
            self.view_matrices.append(view_matrix)
            self.project_matrices.append(project_matrix)
        
        self._env.view_matrix = self.view_matrices[0]
        self._env.projection_matrix = self.project_matrices[0]
        
        # Initialize time_step counter (used in step() method)
        self.time_step = 0
        
        # Skip goal loading during extraction - goals will be computed from trajectory
        if "act3d_goal" in self.observation_mode and not skip_goal_loading:
            self._load_goal_states()
        else:
            # Set default goal state placeholders
            self.grasping_goal = None
            self.final_goal = None
            self.grasped_handle = False
            self.goal_gripper_pcd = None

        self.only_object = only_object
        
        # Try to extract handle_name from environment config if not provided
        if self._handle_name is None:
            self._handle_name = self._detect_handle_name()
    
    def _load_goal_states(self):
        """Load goal states from stage_lengths.json (optional)."""
        config_path = self._env.config_path
        task_name = getattr(self._env, 'task_name', 'primitive')
        parent_path = os.path.dirname(config_path)
        
        state_path = os.path.join(parent_path, "{}_primitive".format(task_name), "states")
        stage_lengths_json_file = os.path.join(parent_path, "{}_primitive".format(task_name), 'stage_lengths.json')
        if not os.path.exists(stage_lengths_json_file):
            state_path = os.path.join(parent_path, "states")
            stage_lengths_json_file = os.path.join(parent_path, 'stage_lengths.json')
        
        if not os.path.exists(stage_lengths_json_file):
            # Can't load goals, use defaults
            self.grasping_goal = None
            self.final_goal = None
            self.grasped_handle = False
            self.goal_gripper_pcd = None
            return
        
        with open(stage_lengths_json_file, 'r') as f:
            stage_lengths = json.load(f)
        
        open_begin_t_idx = (
            stage_lengths.get('reach_handle', 0) + 
            stage_lengths.get('reach_to_contact', 0) + 
            stage_lengths.get('close_gripper', 0)
        )
        all_time_steps = (
            open_begin_t_idx + stage_lengths.get('open_door', 0)
        )

        goal_1_state = os.path.join(state_path, "state_{}.pkl".format(open_begin_t_idx))
        goal_2_state = os.path.join(state_path, "state_{}.pkl".format(all_time_steps - 1))
        
        if not os.path.exists(goal_1_state) or not os.path.exists(goal_2_state):
            self.grasping_goal = None
            self.final_goal = None
            self.grasped_handle = False
            self.goal_gripper_pcd = None
            return
        
        with open(goal_1_state, 'rb') as f:
            goal_1_state_data = pickle.load(f)
        with open(goal_2_state, 'rb') as f:
            goal_2_state_data = pickle.load(f)
        
        self._env.reset(reset_state=goal_1_state_data)
        grasping_eef_pc = self.get_gripper_pc()

        self._env.reset(reset_state=goal_2_state_data)
        final_eef_pc = self.get_gripper_pc()
        
        self.grasping_goal = grasping_eef_pc
        self.final_goal = final_eef_pc
        self.grasped_handle = False
        self.goal_gripper_pcd = None
    
    def _detect_handle_name(self) -> str:
        """Detect handle name from environment config or return default."""
        # Try to get from environment's config
        if hasattr(self._env, 'config'):
            config = self._env.config
            if isinstance(config, list):
                for block in config:
                    if isinstance(block, dict) and 'handle_name' in block:
                        return block['handle_name'].lower()
        
        # Try to detect from mobility_v2.json via the environment
        if hasattr(self._env, 'urdf_paths'):
            urdf_path = self._env.urdf_paths.get(self._object_name.lower())
            if urdf_path:
                from pathlib import Path
                import json
                asset_root = Path(urdf_path).parent
                mobility_path = asset_root / "mobility_v2.json"
                if mobility_path.exists():
                    with mobility_path.open('r') as f:
                        mobility_info = json.load(f)
                    # Find the first joint that's not the base
                    for joint_info in mobility_info:
                        if joint_info.get('parent', -1) >= 0:
                            parts = joint_info.get('parts', [])
                            if parts:
                                return parts[0].get('name', 'handle').lower()
        
        return 'handle'
    
    def _get_handle_pc(self):
        """Get handle point cloud using custom object utilities.
        
        Returns:
            Tuple of (handle_pc, handle_joint_id) or raises if not found
        """
        # Use the custom get_handle_pos function that works with custom asset layouts
        try:
            result = get_handle_pos_custom(
                self._env,
                self._object_name,
                asset_dir_hint=self._asset_dir_hint,
                return_median=False,
                target_object=self._handle_name or 'handle',
            )
            all_handle_pos, handle_joint_id = result[0], result[1]
            
            if all_handle_pos and len(all_handle_pos) > 0:
                handle_pc = np.concatenate(all_handle_pos, axis=0)
                return handle_pc, handle_joint_id
        except Exception as e:
            print(f"Warning: get_handle_pos_custom failed: {e}")
        
        # Fallback to environment's get_handle_pos if available
        if hasattr(self._env, 'get_handle_pos'):
            try:
                all_handle_pos, handle_joint_id = self._env.get_handle_pos(return_median=False)
                if all_handle_pos and len(all_handle_pos) > 0:
                    handle_pc = np.concatenate(all_handle_pos, axis=0)
                    return handle_pc, handle_joint_id
            except Exception as e:
                print(f"Warning: env.get_handle_pos failed: {e}")
        
        raise ValueError(f"Could not get handle point cloud for {self._object_name}")
    
    def _get_link_pc(self, link_name: str = 'link_0'):
        """Get link point cloud, handling different naming conventions.
        
        Args:
            link_name: Name of the link (will try alternatives if not found)
            
        Returns:
            Point cloud array for the link
        """
        return get_link_pc_custom(self._env, self._object_name, link_name)

    def reset_random_cameras(self):
        """Reset cameras to random positions that observe the handle.
        
        Overrides parent method to use custom handle position detection.
        """
        try_times = 0
        
        # Get handle point cloud using custom method
        try:
            handle_pc, _ = self._get_handle_pc()
        except ValueError as e:
            print(f"Warning: Could not get handle PC for camera reset: {e}")
            # Use a fallback - just use the object center
            obj_id = self._env.urdf_ids[self._object_name.lower()]
            min_aabb, max_aabb = self._env.get_aabb(obj_id)
            handle_pc = np.array([(min_aabb + max_aabb) / 2])
        
        while try_times < 5000:
            view_matrices = []
            project_matrices = []
            try_times += 1
            distance = np.random.uniform(0.8, 1.2) * self.mean_distance + np.random.normal(0, 0.05, 1)
            camera_center = self.mean_camera_target + np.random.normal(0, 0.05, 3)
            for _ in range(2):
                rpy = np.zeros(3)
                rpy[0] = np.random.uniform(-20, 20)
                rpy[1] = np.random.uniform(-40, 0)
                if np.random.uniform() > 0.5:
                    rpy[2] = np.random.uniform(-110, -160)
                else:
                    rpy[2] = np.random.uniform(-20, -70)
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=camera_center, 
                    distance=distance, 
                    yaw=rpy[2], 
                    pitch=rpy[0], 
                    roll=rpy[1], 
                    upAxisIndex=2, 
                    physicsClientId=self._env.id
                )
                project_matrix = p.computeProjectionMatrixFOV(
                    fov=60, 
                    aspect=640/480, 
                    nearVal=self.depth_near, 
                    farVal=self.depth_far, 
                    physicsClientId=self._env.id
                )
                view_matrices.append(view_matrix)
                project_matrices.append(project_matrix)
            self.view_matrices = view_matrices
            self.project_matrices = project_matrices
            if self.check_handle_observed_in_pc(handle_pc=handle_pc) > 5:
                self._env.projection_matrix = project_matrices[0]
                self._env.view_matrix = view_matrices[0]
                break
        if try_times >= 5000:
            raise ValueError("Cannot find a camera view that has handle points in the point cloud")
    
    def check_handle_observed_in_pc(self, handle_pc=None):
        """Check if handle is visible in the current camera view.
        
        Overrides parent method to use custom handle position detection.
        
        Args:
            handle_pc: Pre-computed handle point cloud (optional)
            
        Returns:
            Number of points close to handle in the observed point cloud
        """
        if handle_pc is None:
            try:
                handle_pc, _ = self._get_handle_pc()
            except ValueError as e:
                print(f"Warning: Could not get handle PC: {e}")
                return 0
        
        pcs = []
        rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices = self.take_images_around_object(
            self._env, 
            self._object_name.lower(), 
            camera_elevation=self.camera_elevation,
            return_camera_matrices=True, 
            camera_height=self.camera_height, 
            camera_width=self.camera_width,
        )
        
        for rgb, depth, segmask, view_matrix, project_matrix in zip(
            rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices
        ):
            pc = get_pc(
                proj_matrix=project_matrix, 
                view_matrix=view_matrix, 
                depth=depth, 
                width=self.camera_width, 
                height=self.camera_height, 
                mask_infinite=False
            )
            
            segmask_obj_id = segmask & ((1 << 24) - 1)
            object_mask = np.zeros_like(depth).astype(np.float32)
            object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name.lower()]] = 1
            object_mask_ = np.flatnonzero(object_mask.flatten())
            pcs.append(pc[object_mask_])
        
        if not pcs or all(len(pc) == 0 for pc in pcs):
            return 0
            
        pcs = np.concatenate(pcs, axis=0)
        
        if len(pcs) == 0 or len(handle_pc) == 0:
            return 0
            
        pcd_distance = cdist(pcs, handle_pc)  # Shape: (M, N)

        # Find the minimum distance for each point in pcs
        min_distance = np.min(pcd_distance, axis=1)  # Shape: (M,)

        # Filter the distances that are less than 0.02
        min_distance = min_distance[min_distance < 0.02]
        
        return len(min_distance)

    def _get_act3d_observation(self, rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices, using_torch=False, only_object=True):
        """Get act3d observation with custom handling for goal states.
        
        This overrides the parent method to handle the case when goal states
        are not loaded (skip_goal_loading=True during extraction).
        """
        import scipy.spatial.distance
        import fpsample
        
        obs_dict_input = {}
        
        # Get robot state: eef pos, orient (6D), and finger joint angle
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        rotate_matrix = p.getMatrixFromQuaternion(orient)
        orient = rotation_transfer_matrix_to_6D(rotate_matrix)
        cur_joint_angle = p.getJointState(
            self._env.robot.body, 
            self._env.robot.right_gripper_indices[0], 
            physicsClientId=self._env.id
        )
        pos_ori = pos.tolist() + orient.tolist() + [cur_joint_angle[0]]
        
        pcs = []
        gripper_pcd = []
        goal_gripper_pcd = None
        
        for rgb, depth, segmask, view_matrix, project_matrix in zip(
            rgbs, depths, segmasks, view_camera_matrices, project_camera_matrices
        ):
            # Get the object pcd
            pc = get_pc(
                proj_matrix=project_matrix, 
                view_matrix=view_matrix, 
                depth=depth, 
                width=self.camera_width, 
                height=self.camera_height, 
                mask_infinite=False
            )
            segmask_obj_id = segmask & ((1 << 24) - 1)
            object_mask = np.zeros_like(depth).astype(np.float32)
            object_mask[segmask_obj_id == self._env.urdf_ids[self._object_name.lower()]] = 1
            object_mask_ = np.flatnonzero(object_mask.flatten())
            pcs.append(pc[object_mask_])
            
            # Get the gripper 4 points
            gripper_pc = self.get_gripper_pc()
            gripper_pcd.append(gripper_pc)
            
            # Handle goal - use current gripper as placeholder if goals not loaded
            if 'goal' in self.observation_mode:
                if self._skip_goal_loading or self.grasping_goal is None:
                    # Use current gripper position as placeholder
                    goal_gripper_pcd = gripper_pc.copy()
                else:
                    grasped = getattr(self._env, 'grasped_handle', False) or getattr(self, 'grasped_handle', False)
                    if grasped:
                        goal_gripper_pcd = self.final_goal
                    else:
                        goal_gripper_pcd = self.grasping_goal
                self.goal_gripper_pcd = goal_gripper_pcd
        
        point_cloud = np.concatenate(pcs, axis=0) if pcs else np.zeros((0, 3))
        
        num_points = self.num_points
        
        # Handle case when point cloud is empty or too small
        if point_cloud.shape[0] == 0:
            point_cloud = np.zeros((num_points, 3))
        elif point_cloud.shape[0] < num_points:
            to_add_points_num = num_points - point_cloud.shape[0]
            random_sampled_points = np.random.choice(point_cloud.shape[0], to_add_points_num, replace=True)
            point_cloud = np.concatenate([point_cloud, point_cloud[random_sampled_points]], axis=0)
        
        # Perform FPS on the point clouds
        try:
            h = min(9, np.log2(num_points))
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(point_cloud[:, :3], num_points, h=h)
        except:
            kdline_fps_samples_idx = fpsample.fps_npdu_kdtree_sampling(point_cloud[:, :3], num_points)
        kdline_fps_samples_idx = np.array(sorted(kdline_fps_samples_idx))
        point_cloud = point_cloud[kdline_fps_samples_idx]
        
        point_cloud = point_cloud.tolist()
        
        # Build the observation dictionary
        obs_dict_input['point_cloud'] = np.array(point_cloud).astype(np.float32)
        obs_dict_input['agent_pos'] = np.array(pos_ori).astype(np.float32)
        obs_dict_input['gripper_pcd'] = gripper_pcd[0].astype(np.float32) if gripper_pcd else np.zeros((4, 3), dtype=np.float32)
        
        if 'goal' in self.observation_mode:
            if goal_gripper_pcd is not None:
                obs_dict_input['goal_gripper_pcd'] = np.array(goal_gripper_pcd).astype(np.float32)
            else:
                obs_dict_input['goal_gripper_pcd'] = obs_dict_input['gripper_pcd'].copy()
        
        if 'displacement_gripper_to_object' in self.observation_mode:
            gripper_pcd_arr = obs_dict_input['gripper_pcd']
            object_pcd = obs_dict_input['point_cloud']
            if object_pcd.shape[0] > 0:
                distance = scipy.spatial.distance.cdist(gripper_pcd_arr, object_pcd)
                min_distance_obj_idx = np.argmin(distance, axis=1)
                closest_point = object_pcd[min_distance_obj_idx]
                displacement = closest_point - gripper_pcd_arr
            else:
                displacement = np.zeros_like(gripper_pcd_arr)
            obs_dict_input['displacement_gripper_to_object'] = displacement.astype(np.float32)
        
        return obs_dict_input


__all__ = [
    'RobogenPointCloudWrapperCustom',
    'get_link_pc_custom',
]
