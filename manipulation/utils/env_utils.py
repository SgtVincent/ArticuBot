import copy
import importlib
import json
import os
import os.path as osp
import pickle
from pathlib import Path

from typing import List, Optional

import numpy as np
import pybullet as p
import yaml
from objaverse_utils.utils import partnet_mobility_dict
from scipy import ndimage

from .defaults import default_config, data_dir
from .geometry_utils import get_pc, in_bbox, sample_point_inside_triangle
from .handle_utiils import find_nearest_point_on_line, load_obj, rotate_point_around_axis
from .object_utils import down_load_single_object, preprocess_urdf


def build_up_env_eval(task_config=None, solution_path=None, task_name=None, restore_state_file=None, return_env_class=False, 
                    render=False, randomize=False, 
                    obj_id=0,  **kwargs,
                ):
    """
    Build and initialize an evaluation environment for manipulation tasks.
    
    This function creates an environment instance by dynamically importing
    the environment class from the solution path and initializing it with
    the provided configuration.
    
    Args:
        task_config (str, optional): Path to task configuration YAML file.
        solution_path (str, optional): Python module path to the environment class.
        task_name (str, optional): Name of the environment class to instantiate.
        restore_state_file (str, optional): Path to saved state file for restoration.
        return_env_class (bool, optional): If True, return both environment and class.
            Defaults to False.
        render (bool, optional): Enable GUI rendering. Defaults to False.
        randomize (bool, optional): Enable randomization. Defaults to False.
        obj_id (int, optional): Object ID for multi-object scenarios. Defaults to 0.
        **kwargs: Additional configuration parameters.
    
    Returns:
        Environment or tuple: If return_env_class is False, returns the environment
            instance. If True, returns (environment, environment_class).
    
    Example:
        >>> env = build_up_env_eval(
        ...     task_config='config.yaml',
        ...     solution_path='manipulation.envs.articulated',
        ...     task_name='ArticulatedEnv',
        ...     render=True
        ... )
    """
    
    save_config = copy.deepcopy(default_config)
    save_config['config_path'] = task_config
    save_config['task_name'] = task_name
    save_config['restore_state_file'] = restore_state_file
    save_config['gui'] = render
    save_config['randomize'] = randomize
    save_config['obj_id'] = obj_id
    save_config['task_name'] = task_name
    for key, value in kwargs.items():
        save_config[key] = value

    ### you might want to restore to a specific state
    module = importlib.import_module("{}.{}".format(solution_path.replace("/", "."), task_name))
    env_class = getattr(module, task_name)
    env = env_class(**save_config)

    if not return_env_class:
        return env, save_config
    else:
        return env, save_config, env_class


def build_up_env_gen(task_config=None, env_name=None, task_name=None, restore_state_file=None, return_env_class=False, 
                    action_space='delta-translation', render=False, randomize=False, 
                    obj_id=0, random_object_translation: Optional[List]=None, **kwargs,
                ):
    """
    Build and initialize an environment for demonstration generation.
    
    This function creates an environment for generating manipulation demonstrations,
    parsing the task configuration and initializing the environment with object
    and robot parameters.
    
    Args:
        task_config (str, optional): Path to task configuration YAML file containing
            object properties (name, link_name, init_angle, etc.).
        env_name (str, optional): Name of environment module to import from
            manipulation.envs.
        task_name (str, optional): Task identifier.
        restore_state_file (str, optional): Path to saved state for restoration.
        return_env_class (bool, optional): Return both env and class if True.
            Defaults to False.
        action_space (str, optional): Action space type. Defaults to 'delta-translation'.
        render (bool, optional): Enable GUI rendering. Defaults to False.
        randomize (bool, optional): Enable randomization. Defaults to False.
        obj_id (int, optional): Object ID. Defaults to 0.
        random_object_translation (list, optional): Random translation offset for object.
        **kwargs: Additional configuration parameters.
    
    Returns:
        Environment or tuple: Environment instance, or (env, env_class) if
            return_env_class is True.
    
    Example:
        >>> env, _ = build_up_env_gen(
        ...     task_config='data/color_0025_0_v2/configs/config_500.yaml',
        ...     env_name='articulated',
        ...     render=False
        ... )
    """
    config = yaml.safe_load(open(task_config, "r"))
    link_name = 'link_0'
    init_angle = None
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
        if 'init_angle' in config_dict:
            init_angle = config_dict['init_angle']

    save_config = copy.deepcopy(default_config)
    save_config['config_path'] = task_config
    save_config['task_name'] = task_name
    save_config['object_name'] = object_name
    save_config['link_name'] = link_name
    save_config['init_angle'] = init_angle
    save_config['restore_state_file'] = restore_state_file
    save_config['translation_mode'] = action_space
    save_config['gui'] = render
    save_config['randomize'] = randomize
    save_config['obj_id'] = obj_id
    save_config['random_object_translation'] = random_object_translation
    for key, value in kwargs.items():
        save_config[key] = value

    ### you might want to restore to a specific state
    # module = importlib.import_module("{}.{}".format(solution_path.replace("/", "."), task_name))
    # env_class = getattr(module, task_name)
    module = importlib.import_module("manipulation.envs.{}".format(env_name))
    env_class = getattr(module, env_name)
    env = env_class(**save_config)


    if not return_env_class:
        return env, save_config
    else:
        return env, save_config, env_class


def build_up_env_random(task_config=None, env_name='articulated', render=False, 
                        randomize_object_pose=True, randomize_robot_joints=True,
                        randomize_initial_joint_angle=True, horizon=600, 
                        max_attempts=100, far_distance=0.7, near_distance=0.3, **kwargs):
    """
    Build and initialize an environment with random collision-free initialization for evaluation.
    
    This function creates an environment and performs intelligent randomization similar to 
    _gen_init_state in gen_demo.py. It ensures:
    1. No collision between robot and object
    2. No collision between robot and ground plane
    3. Object handle/joint is within reachable distance from end effector
    4. Object faces the robot arm for manipulation
    
    Args:
        task_config (str, optional): Path to task configuration YAML file.
        env_name (str, optional): Name of environment module (default: 'articulated').
        render (bool, optional): Enable GUI rendering. Defaults to False.
        randomize_object_pose (bool, optional): Randomize object position/orientation.
            Defaults to True.
        randomize_robot_joints (bool, optional): Randomize robot initial joint angles.
            Defaults to True.
        randomize_initial_joint_angle (bool, optional): Randomize articulated joint angle.
            Defaults to True.
        horizon (int, optional): Episode horizon. Defaults to 600.
        max_attempts (int, optional): Maximum attempts to find collision-free state. 
            Defaults to 100.
        far_distance (float, optional): Maximum distance from end effector to handle.
            Defaults to 0.7.
        near_distance (float, optional): Minimum distance from end effector to handle.
            Defaults to 0.3.
        **kwargs: Additional configuration parameters passed to environment.
    
    Returns:
        tuple: (env, save_config) - The initialized environment and its config dict.
    
    Example:
        >>> env, _ = build_up_env_random(
        ...     task_config='data/custom_objects/color_0025_0_v2/configs/config_500.yaml',
        ...     render=False
        ... )
    """
    import time
    
    # Parse config to get object properties
    config = yaml.safe_load(open(task_config, "r"))
    link_name = 'link_0'
    object_name = None
    base_pos = None
    base_euler = None
    
    for config_dict in config:
        if 'name' in config_dict:
            object_name = config_dict['name'].lower()
        if 'link_name' in config_dict:
            link_name = config_dict['link_name']
        if 'center' in config_dict:
            center_str = config_dict['center']
            if isinstance(center_str, str):
                base_pos = parse_center(center_str)
            else:
                base_pos = np.array(center_str)
        if 'euler' in config_dict:
            euler_str = config_dict['euler']
            if isinstance(euler_str, str):
                base_euler = parse_center(euler_str)
            else:
                base_euler = np.array(euler_str)
    
    if base_pos is None:
        base_pos = np.array([0.3, 0.0, 0.0])
    if base_euler is None:
        base_euler = np.array([0.0, 0.0, 0.0])
    
    # Build environment without restore_state_file
    save_config = copy.deepcopy(default_config)
    save_config['config_path'] = task_config
    save_config['task_name'] = None  # Will be set by environment
    save_config['object_name'] = object_name
    save_config['link_name'] = link_name
    save_config['init_angle'] = None  # Will randomize later
    save_config['restore_state_file'] = None
    save_config['gui'] = render
    save_config['randomize'] = False  # We handle randomization manually
    save_config['horizon'] = horizon
    
    for key, value in kwargs.items():
        save_config[key] = value
    
    # Import and create environment
    module = importlib.import_module("manipulation.envs.{}".format(env_name))
    env_class = getattr(module, env_name)
    env = env_class(**save_config)
    
    # Reset to get initial state
    env.reset()
    
    # Get object ID and initial pose
    object_id = env.urdf_ids[object_name]
    init_pos, init_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
    init_euler = p.getEulerFromQuaternion(init_orient)
    
    # Get robot base position for distance calculations
    robot_base_pos = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]
    
    # Compute object bounding box to understand its size
    # Get AABB for all links to find complete object extent
    num_links = p.getNumJoints(object_id, physicsClientId=env.id)
    all_aabbs = []
    
    # Base link AABB
    base_aabb = p.getAABB(object_id, -1, physicsClientId=env.id)
    all_aabbs.append(base_aabb)
    
    # All other links
    for link_idx in range(num_links):
        link_aabb = p.getAABB(object_id, link_idx, physicsClientId=env.id)
        all_aabbs.append(link_aabb)
    
    # Compute overall bounding box
    all_mins = [aabb[0] for aabb in all_aabbs]
    all_maxs = [aabb[1] for aabb in all_aabbs]
    obj_min = np.min(all_mins, axis=0)
    obj_max = np.max(all_maxs, axis=0)
    obj_size = obj_max - obj_min
    obj_center = (obj_min + obj_max) / 2.0
    
    # Calculate minimum safe distance based on object size
    obj_diagonal = np.linalg.norm(obj_size)
    min_safe_distance = max(0.15, obj_diagonal * 0.5)  # At least 15cm or half diagonal
    
    if render:
        print(f"Object bounding box size: {obj_size}")
        print(f"Object diagonal: {obj_diagonal:.3f}m")
        print(f"Minimum safe distance: {min_safe_distance:.3f}m")
    
    # Get robot joint limits with safety margin (20%)
    low = np.array([-2.9, -1.8, -2.9, -3.1, -2.9, -0.0, -2.9])
    high = np.array([2.9, 1.8, 2.9, 0.0, 2.9, 3.8, 2.9])
    joint_range = high - low
    low = low + joint_range * 0.2
    high = high - joint_range * 0.2
    
    # Find handle joint for articulated objects
    handle_joint_id = None
    if randomize_initial_joint_angle:
        try:
            handle_joint_id = env.get_handle_joint_id()
        except (FileNotFoundError, KeyError, AttributeError):
            # Fallback: find first revolute joint
            num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
            for i in range(num_joints):
                joint_info = p.getJointInfo(object_id, i, physicsClientId=env.id)
                joint_type = joint_info[2]
                if joint_type == 0:  # JOINT_REVOLUTE
                    handle_joint_id = i
                    break
            
            if handle_joint_id is None:
                print(f"Warning: No revolute joint found for {object_name}")
    
    # Don't store handle_joint_id - let environment initialize it properly
    # The environment's _get_info() will discover and set it on first call
    
    # Attempt to find a good collision-free initialization
    good_init_found = False
    attempt = 0
    
    while not good_init_found and attempt < max_attempts:
        attempt += 1
        
        # 1. Randomize object pose with bounding box awareness
        if randomize_object_pose:
            new_pos = np.array(init_pos).copy()
            # Randomize position ensuring minimum safe distance from robot base
            # Sample positions that keep object in reachable zone but avoid collision
            max_offset = 0.15  # Maximum 15cm offset from base position
            new_pos[0] = base_pos[0] + np.random.uniform(-max_offset, max_offset)
            new_pos[1] = base_pos[1] + np.random.uniform(-max_offset, max_offset)
            new_pos[2] = init_pos[2]
            
            # Check if new position maintains minimum safe distance from robot base
            dist_to_robot = np.linalg.norm(new_pos[:2] - np.array(robot_base_pos[:2]))
            if dist_to_robot < min_safe_distance:
                # Adjust position to maintain minimum distance
                direction = (new_pos[:2] - np.array(robot_base_pos[:2]))
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    new_pos[:2] = np.array(robot_base_pos[:2]) + direction * min_safe_distance
            
            new_euler = np.array(init_euler).copy()
            new_euler[2] = base_euler[2] + np.random.uniform(-np.pi / 6, np.pi / 6)
            new_orient = p.getQuaternionFromEuler(new_euler)
            
            p.resetBasePositionAndOrientation(object_id, new_pos, new_orient, physicsClientId=env.id)
        
        # 2. Randomize articulated joint angle
        if randomize_initial_joint_angle and handle_joint_id is not None:
            joint_limit_low, joint_limit_high = p.getJointInfo(
                object_id, handle_joint_id, physicsClientId=env.id
            )[8:10]
            # Ensure we stay within valid limits
            max_init_angle = joint_limit_low + 0.2 * (joint_limit_high - joint_limit_low)
            # Clamp to ensure we don't exceed the actual limits
            max_init_angle = min(max_init_angle, joint_limit_high)
            random_joint = np.random.uniform(joint_limit_low, max_init_angle)
            p.resetJointState(object_id, handle_joint_id, random_joint, physicsClientId=env.id)
        
        # Let object settle
        for _ in range(5):
            p.stepSimulation(physicsClientId=env.id)
        
        # Verify joint state is stable
        if randomize_initial_joint_angle and handle_joint_id is not None:
            current_joint = p.getJointState(object_id, handle_joint_id, physicsClientId=env.id)[0]
            if abs(current_joint - random_joint) > 1e-3:
                continue  # Object moved, try again
        
        # 3. Randomize robot joint angles and check for collisions
        robot_attempt = 0
        robot_good = False
        
        while robot_attempt < 100 and not robot_good:
            robot_attempt += 1
            
            # Sample random joint angles
            if randomize_robot_joints:
                random_joints = np.random.uniform(low, high)
                env.robot.set_joint_angles(env.robot.right_arm_joint_indices, random_joints)
            
            # Reset object joint after robot movement
            if randomize_initial_joint_angle and handle_joint_id is not None:
                p.resetJointState(object_id, handle_joint_id, random_joint, physicsClientId=env.id)
            
            # Let robot settle
            for _ in range(5):
                p.stepSimulation(physicsClientId=env.id)
            
            # Verify joint state is still stable
            if randomize_initial_joint_angle and handle_joint_id is not None:
                current_joint = p.getJointState(object_id, handle_joint_id, physicsClientId=env.id)[0]
                if abs(current_joint - random_joint) > 1e-3:
                    continue
            
            # Check collision between robot and object
            contact_points = p.getContactPoints(env.robot.body, object_id, physicsClientId=env.id)
            closest_points = p.getClosestPoints(env.robot.body, object_id, distance=0.01, physicsClientId=env.id)
            if len(contact_points) > 0 or len(closest_points) > 0:
                continue
            
            # Additional check: verify object bounding box doesn't overlap with robot base area
            # Update object AABB after randomization
            obj_aabb_min, obj_aabb_max = p.getAABB(object_id, -1, physicsClientId=env.id)
            robot_base_pos_current = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)[0]
            
            # Check if object is too close to robot base (XY plane)
            robot_radius = 0.1  # Approximate robot base radius
            closest_point_2d = np.array([
                max(obj_aabb_min[0], min(robot_base_pos_current[0], obj_aabb_max[0])),
                max(obj_aabb_min[1], min(robot_base_pos_current[1], obj_aabb_max[1]))
            ])
            dist_to_base_2d = np.linalg.norm(closest_point_2d - np.array(robot_base_pos_current[:2]))
            
            if dist_to_base_2d < robot_radius:
                continue  # Object too close to robot base
            
            # Check collision between robot links and ground plane
            link_contact = False
            num_links = p.getNumJoints(env.robot.body, physicsClientId=env.id)
            for link_idx in range(1, num_links):
                contact_points = p.getClosestPoints(
                    bodyA=env.robot.body, linkIndexA=link_idx, 
                    bodyB=env.urdf_ids['plane'], distance=0.005, 
                    physicsClientId=env.id
                )
                if len(contact_points) > 0:
                    link_contact = True
                    break
            
            if link_contact:
                continue
            
            # Check if end effector is in good position to grasp handle
            robot_eef_pos, _ = env.robot.get_pos_orient(env.robot.right_end_effector)
            
            # Get handle position directly from PyBullet
            if handle_joint_id is not None:
                try:
                    # Get the link state for the handle/joint
                    link_state = p.getLinkState(object_id, handle_joint_id, physicsClientId=env.id)
                    handle_pos = np.array(link_state[0])  # World position of link frame
                    
                    distance = np.linalg.norm(handle_pos - robot_eef_pos)
                    if near_distance < distance < far_distance:
                        robot_good = True
                        good_init_found = True
                        break
                except Exception:
                    # If we can't get handle position, just accept the configuration
                    robot_good = True
                    break
            else:
                # No handle joint, just accept the configuration
                robot_good = True
                break
        
        if good_init_found:
            break
    
    if not good_init_found:
        print(f"Warning: Could not find collision-free state after {max_attempts} attempts. Using last configuration.")
    
    # 4. Randomize gripper opening
    if hasattr(env.robot, 'finger_fully_close_joint_angle') and hasattr(env.robot, 'finger_fully_open_joint_angle'):
        initial_finger_angle = np.random.uniform(
            env.robot.finger_fully_close_joint_angle, 
            env.robot.finger_fully_open_joint_angle
        )
        env.robot.set_gripper_open_position(
            env.robot.right_gripper_indices, 
            [initial_finger_angle, initial_finger_angle], 
            set_instantly=True
        )
    
    # Final settlement
    for _ in range(10):
        p.stepSimulation(physicsClientId=env.id)
    
    return env, save_config

def parse_center(center):
    """
    Parse a center coordinate string into a numpy array.
    
    This function converts string representations of coordinates
    (e.g., "(0.1, 0.2, 0.3)" or "0.1, 0.2, 0.3") into numpy arrays.
    
    Args:
        center (str): String representation of coordinates. Can be:
            - Bracketed: "(x, y, z)" or "[x, y, z]"
            - Unbracketed: "x, y, z"
    
    Returns:
        np.ndarray: Array of float coordinates with shape (3,).
    
    Example:
        >>> center = parse_center("(0.5, 0.2, 0.1)")
        >>> print(center)  # [0.5 0.2 0.1]
        >>> center = parse_center("0.5, 0.2, 0.1")
        >>> print(center)  # [0.5 0.2 0.1]
    """
    if center.startswith("(") or center.startswith("["):
        center = center[1:-1]

    center = center.split(",")
    center = [float(x) for x in center]
    return np.array(center)

def parse_config(config, obj_id=None, use_vhacd=True, default_initial_joint_angle=0):
    """
    Parse configuration dictionary to extract object URDF paths and properties.
    
    Processes a config dict (typically from base_config.yaml) to extract paths,
    sizes, positions, orientations, and joint angles for scene objects.
    
    Args:
        config (dict): Configuration dictionary with keys like 'center', 'euler',
            'size', 'urdf_path', 'rotation', 'joint_state', etc.
        obj_id (int, optional): Specific object ID to parse. If None, uses config data.
            Defaults to None.
        use_vhacd (bool, optional): Whether to use V-HACD processed URDFs. Defaults to True.
        default_initial_joint_angle (float, optional): Default joint angle if not
            specified in config. Defaults to 0.
    
    Returns:
        tuple: (urdf_paths, urdf_sizes, urdf_locations, urdf_orientations, urdf_names,
                urdf_init_joint_angles, urdf_axis, task_objects, task_links)
            - urdf_paths (list): Paths to URDF files
            - urdf_sizes (list): Object sizes
            - urdf_locations (list): Object positions [x, y, z]
            - urdf_orientations (list): Object orientations (quaternions)
            - urdf_names (list): Object names
            - urdf_init_joint_angles (list): Initial joint angles
            - urdf_axis (list): Joint axes
            - task_objects (list): Task-relevant object names
            - task_links (list): Task-relevant link names
    
    Example:
        >>> config = yaml.safe_load(open('base_config.yaml'))
        >>> paths, sizes, locs, *rest = parse_config(config, use_vhacd=True)
    """
    urdf_paths = []
    urdf_sizes = []
    urdf_locations = []
    urdf_orientations = []
    urdf_names = []
    urdf_types = []
    urdf_on_tables = []
    urdf_movables = []
    urdf_crop_sizes = []
    use_table = False
    articulated_joint_angles = {}
    spatial_relationships = []
    distractor_config_path = None

    robot_initial_joint_angles = [0.0, 0.0, 0.0, -0.4, 0.0, 0.4, 0.0]
    initial_finger_angle = default_initial_joint_angle

    for obj in config:
        # print(obj)

        if "use_table" in obj.keys():
            use_table = obj['use_table']

        if "set_joint_angle_object_name" in obj.keys():
            new_obj = copy.deepcopy(obj)
            new_obj.pop('set_joint_angle_object_name')
            articulated_joint_angles[obj['set_joint_angle_object_name']] = new_obj

        if "spatial_relationships" in obj.keys():
            spatial_relationships = obj['spatial_relationships']

        if 'task_name' in obj.keys() or 'task_description' in obj.keys():
            continue

        if "distractor_config_path" in obj.keys():
            distractor_config_path = obj['distractor_config_path']

        if 'initial_joint_angles' in obj.keys():
            initial_joint_angles = obj['initial_joint_angles']
            initial_joint_angles = parse_center(initial_joint_angles)
            robot_initial_joint_angles = initial_joint_angles

        if 'initial_finger_angle' in obj.keys():
            initial_finger_angle = obj['initial_finger_angle']
            initial_finger_angle = float(initial_finger_angle)
            

        if "type" not in obj.keys():
            if "urdf_path" in obj.keys():
                obj['type'] = 'urdf'
            else:
                continue
        
        if obj['type'] == 'mesh':
            if 'uid' not in obj.keys():
                continue
            if obj_id is None:
                uid = obj['uid'][np.random.randint(len(obj['uid']))]
            else:
                uid = obj['uid'][obj_id]
                
            urdf_file_path = osp.join("objaverse_utils/data/obj", "{}".format(uid), "material.urdf")
            if not os.path.exists(urdf_file_path):
                down_load_single_object(name=obj['lang'], uids=[uid])
            

            if not use_vhacd:
                new_urdf_file_path = urdf_file_path.replace("material.urdf", "material_non_vhacd.urdf")
                new_urdf_lines = []
                with open(urdf_file_path, 'r') as f:
                    urdf_lines = f.readlines()
                for line in urdf_lines:
                    if 'vhacd' in line:
                        new_line = line.replace("_vhacd", "")
                        new_urdf_lines.append(new_line)
                    else:
                        new_urdf_lines.append(line)
                with open(new_urdf_file_path, 'w') as f:
                    f.writelines(new_urdf_lines)
                urdf_file_path = new_urdf_file_path
                
            urdf_paths.append(urdf_file_path)
            urdf_types.append('mesh')
            urdf_movables.append(True) # all mesh objects are movable
           
        elif obj['type'] == 'urdf':
            if 'urdf_path' in obj.keys():
                urdf_paths.append(obj['urdf_path'])
            else:
                if 'reward_asset_path' in obj.keys():
                    obj_path = obj['reward_asset_path']
                else:
                    try:
                        category = obj['lang']
                        possible_obj_path = partnet_mobility_dict[category]
                    except:
                        category = obj['name']
                        if category == 'Computer display':
                            category = 'Display'
                        possible_obj_path = partnet_mobility_dict[category]
                    
                    obj_path = np.random.choice(possible_obj_path)
                    if category == 'Toaster':
                        obj_path = str(103486)
                    if category == 'Microwave':
                        obj_path = str(7310)
                    if category == "Oven":
                        obj_path = str(101808)
                    if category == 'Refrigerator':
                        obj_path = str(10638)
                
                if 'custom_objects' in obj_path or osp.isabs(obj_path):
                    if osp.isabs(obj_path):
                        urdf_file_path = osp.join(obj_path, "mobility.urdf")
                    else:
                        urdf_file_path = osp.join(f"{data_dir}", obj_path, "mobility.urdf")
                    urdf_paths.append(urdf_file_path)
                else:
                    urdf_file_path = osp.join(f"{data_dir}/dataset", obj_path, "mobility.urdf")
                    if use_vhacd:
                        new_urdf_file_path = urdf_file_path.replace("mobility.urdf", "mobility_vhacd.urdf")
                        if not osp.exists(new_urdf_file_path):
                            new_urdf_file_path = preprocess_urdf(urdf_file_path)
                        urdf_paths.append(new_urdf_file_path)
                    else:
                        urdf_paths.append(urdf_file_path)

            urdf_types.append('urdf')
            urdf_movables.append(obj.get('movable', False)) # by default, urdf objects are not movable, unless specified

        urdf_sizes.append(obj['size'])
        urdf_locations.append(parse_center(obj['center']))
        ori = obj.get('orientation', [0, 0, 0, 1])
        if type(ori) == str:
            ori = parse_center(ori)
        urdf_orientations.append(ori)
        urdf_names.append(obj['name'])
        urdf_on_tables.append(obj.get('on_table', False))
        urdf_crop_sizes.append(obj.get('is_crop_size', True))
    return urdf_paths, urdf_sizes, urdf_locations, urdf_orientations, urdf_names, urdf_types, urdf_on_tables, use_table, urdf_crop_sizes, \
        articulated_joint_angles, spatial_relationships, distractor_config_path, urdf_movables, robot_initial_joint_angles, initial_finger_angle
             
def save_env(env, save_path=None, simplified=False):
    """
    Save the complete state of a PyBullet simulation environment.
    
    Captures all object joint angles, positions, orientations, and environment metadata
    for later restoration.
    
    Args:
        env: Simulation environment object with urdf_ids, id, and other attributes.
        save_path (str, optional): Path to save the state pickle file. If None,
            only returns the state dict. Defaults to None.
        simplified (bool, optional): If True, saves a simplified state without
            robot joint angles. Defaults to False.
    
    Returns:
        dict: State dictionary containing:
            - object_joint_angle_dicts: Joint angles for all objects
            - object_joint_name_dicts: Joint names for all objects
            - object_link_name_dicts: Link names for all objects
            - object_base_position: Base positions for all objects
            - object_base_orientation: Base orientations for all objects
            - robot_joint_angles: Robot joint angles (if not simplified)
            - urdf_paths, object_sizes, robot_name, grasped_handle (if available)
    
    Side Effects:
        - If save_path is provided, writes state to pickle file
    
    Example:
        >>> state = save_env(env, save_path='checkpoint.pkl')
        >>> # Later: load_env(env, load_path='checkpoint.pkl')
    """
    object_joint_angle_dicts = {}
    object_joint_name_dicts = {}
    object_link_name_dicts = {}
    for obj_name, obj_id in env.urdf_ids.items():
        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        object_joint_angle_dicts[obj_name] = []
        object_joint_name_dicts[obj_name] = []
        object_link_name_dicts[obj_name] = []
        for link_idx in range(0, num_links):
            joint_angle = p.getJointState(obj_id, link_idx, physicsClientId=env.id)[0]
            object_joint_angle_dicts[obj_name].append(joint_angle)
            joint_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[1].decode('utf-8')
            object_joint_name_dicts[obj_name].append(joint_name)
            link_name = p.getJointInfo(obj_id, link_idx, physicsClientId=env.id)[12].decode('utf-8')
            object_link_name_dicts[obj_name].append(link_name)

    object_base_position = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_position[obj_name] = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)[0]

    object_base_orientation = {}
    for obj_name, obj_id in env.urdf_ids.items():
        object_base_orientation[obj_name] = p.getBasePositionAndOrientation(obj_id, physicsClientId=env.id)[1]

    state = {
        'object_joint_angle_dicts': object_joint_angle_dicts,
        'object_joint_name_dicts': object_joint_name_dicts,
        'object_link_name_dicts': object_link_name_dicts,
        'object_base_position': object_base_position,
        'object_base_orientation': object_base_orientation,     
    }

    
    if not simplified:
        state['urdf_paths'] = copy.deepcopy(env.urdf_paths)
        state['object_sizes'] = env.simulator_sizes
        state['robot_name'] = env.robot_name
        state['grasped_handle'] = env.grasped_handle

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(state, f, pickle.HIGHEST_PROTOCOL)

    return state

def load_env(env, load_path=None, state=None, simplified=False):

    if load_path is not None:
        with open(load_path, 'rb') as f:
            state = pickle.load(f)
        
    ### set env to stored object position and orientation
    for obj_name, obj_id in env.urdf_ids.items():
        if obj_name not in state['object_base_position'].keys():
            continue
        p.resetBasePositionAndOrientation(obj_id, state['object_base_position'][obj_name], state['object_base_orientation'][obj_name], physicsClientId=env.id)

    ### set env to stored object joint angles
    for obj_name, obj_id in env.urdf_ids.items():
        if obj_name not in state['object_joint_angle_dicts'].keys():
            continue
        
        num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
        for link_idx in range(0, num_links):
            joint_angle = state['object_joint_angle_dicts'][obj_name][link_idx]
            p.resetJointState(obj_id, link_idx, joint_angle, physicsClientId=env.id)

    ### recover suction
    if not simplified:
        if "urdf_paths" in state:
            env.urdf_paths = state["urdf_paths"]

        if "object_sizes" in state:
            env.simulator_sizes = state["object_sizes"]

        if "robot_name" in state:
            env.robot_name = state["robot_name"]

        if "grasped_handle" in state:
            env.grasped_handle = state["grasped_handle"]

    return state

def take_round_images(env, center, distance, elevation=30, azimuth_interval=30, camera_width=640, camera_height=480,
                        return_camera_matrices=False, z_near=0.01, z_far=10, save_path=None):
    camera_target = center
    delta_z = distance * np.sin(np.deg2rad(elevation))
    xy_distance = distance * np.cos(np.deg2rad(elevation))

    env_prev_view_matrix, env_prev_projection_matrix = env.view_matrix, env.projection_matrix

    rgbs = []
    depths = []
    view_camera_matrices = []
    project_camera_matrices = []
    for azimuth in range(0, 360, azimuth_interval):
        delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
        delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
        camera_position = [camera_target[0] + delta_x, camera_target[1] + delta_y, camera_target[2] + delta_z]
        env.setup_camera(camera_position, camera_target, 
                            camera_width=camera_width, camera_height=camera_height)

        rgb, depth = env.render(return_depth=True)
        rgbs.append(rgb)
        depths.append(depth)
        view_camera_matrices.append(env.view_matrix)
        project_camera_matrices.append(env.projection_matrix)
    
    env.view_matrix, env.projection_matrix = env_prev_view_matrix, env_prev_projection_matrix

    if not return_camera_matrices:
        return rgbs, depths
    else:
        return rgbs, depths, view_camera_matrices, project_camera_matrices
    
def take_round_images_around_object(env, object_name, distance=None, save_path=None, azimuth_interval=30, 
                                    elevation=30, return_camera_matrices=False, camera_width=640, camera_height=480, 
                                    only_object=False):
    if only_object:
        ### make all other objects invisiable
        prev_rgbas = []
        object_id = env.urdf_ids[object_name]
        for obj_name, obj_id in env.urdf_ids.items():
            if obj_name != object_name:
                num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                for link_idx in range(-1, num_links):
                    prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=env.id)[0][14:18]
                    prev_rgbas.append(prev_rgba)
                    p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=env.id)
    else:
        prev_rgbas = []

                                    
    obj_id = env.urdf_ids[object_name]
    min_aabb, max_aabb = env.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    if distance is None:
        distance = np.linalg.norm(max_aabb - min_aabb) * 1.1

    res = take_round_images(env, camera_target, distance, elevation=elevation, 
                             azimuth_interval=azimuth_interval, camera_width=camera_width, camera_height=camera_height, 
                             save_path=save_path, return_camera_matrices=return_camera_matrices)

    if only_object:
        cnt = 0
        object_id = env.urdf_ids[object_name]
        for obj_name, obj_id in env.urdf_ids.items():
            if obj_name != object_name:
                num_links = p.getNumJoints(obj_id, physicsClientId=env.id)
                for link_idx in range(-1, num_links):
                    p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=env.id)
                    cnt += 1
                    
    return res

def center_camera_at_object(env, object_name, distance=None, elevation=30, azimuth=0, camera_width=640, camera_height=480):
    obj_id = env.urdf_ids[object_name]
    min_aabb, max_aabb = env.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    if distance is None:
        distance = np.linalg.norm(max_aabb - min_aabb) * 1.1

    delta_z = distance * np.sin(np.deg2rad(elevation))
    xy_distance = distance * np.cos(np.deg2rad(elevation))

    delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
    delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
    camera_position = [camera_target[0] + delta_x, camera_target[1] + delta_y, camera_target[2] + delta_z]
    env.setup_camera(camera_position, camera_target, 
                        camera_width=camera_width, camera_height=camera_height)

def get_joint_id_from_name(simulator, object_name, joint_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[1].decode("utf-8") == joint_name:
            joint_index = i
            break

    return joint_index

def get_handle_pos(simulator, obj_name, return_median=True, handle_pts_obj_frame=None, mobility_info=None, return_info=False, target_object='handle'):
    obj_name = obj_name.lower()
    scaling = simulator.simulator_sizes[obj_name]

    # get the parent frame of the revolute joint.
    obj_id = simulator.urdf_ids[obj_name] 

    # axis in parent frame, transform everything to world frame
    if mobility_info is None:
        urdf_path = simulator.urdf_paths[obj_name]
        parent_dir = os.path.dirname(urdf_path)
        start_idx = parent_dir.find("data/dataset")
        if start_idx != -1:
            parent_dir = parent_dir[start_idx:]
            parent_dir = os.path.join(os.environ["PROJECT_DIR"], parent_dir)
        mobility_info = json.load(open(f"{parent_dir}/mobility_v2.json", "r"))
    else:
        # If mobility_info is provided, we still need parent_dir for loading meshes
        # Try to derive it from urdf_path if possible, or assume it's passed in some other way
        # For now, let's try to derive it same as above
        urdf_path = simulator.urdf_paths[obj_name]
        parent_dir = os.path.dirname(urdf_path)
        start_idx = parent_dir.find("data/dataset")
        if start_idx != -1:
            parent_dir = parent_dir[start_idx:]
            parent_dir = os.path.join(os.environ["PROJECT_DIR"], parent_dir)
    
    # return a list of handle points in world frame
    ret_handle_pt_list = []
    ret_joint_idx_list = []

    joint_name = None
    parent_joint_name = None
    handle_idx = 0
    all_handle_pts_object_frame = []
    for idx, joint_info in enumerate(mobility_info):
        all_parts = [part["name"] for part in joint_info["parts"]]
        if target_object in all_parts:
            all_ids = [part["id"] for part in joint_info["parts"]]
            index = all_parts.index(target_object)
            handle_id = all_ids[index]
            joint_name = "joint_{}".format(joint_info["id"])
            parent_joint_name = "joint_{}".format(joint_info["parent"])
            joint_data = joint_info['jointData']
            axis_body = np.array(joint_data["axis"]["origin"]) * scaling
            axis_dir_body = np.array(joint_data["axis"]["direction"])
            joint_limit = joint_data["limit"]
            if joint_limit['a'] > joint_limit['b']:
                axis_dir_body = -axis_dir_body

            joint_idx = get_joint_id_from_name(simulator, obj_name, joint_name) # this is the joint id in pybullet
            parent_joint_idx = get_joint_id_from_name(simulator, obj_name, parent_joint_name) # this is the joint id in pybullet
            
            parent_link_state = p.getLinkState(obj_id, parent_joint_idx, physicsClientId=simulator.id) # NOTE: the handle link id should be dependent on the object urdf.
            # parent_link_state = p.getLinkState(obj_id, joint_idx, physicsClientId=simulator.id) # NOTE: the handle link id should be dependent on the object urdf.
            link_urdf_world_pos, link_urdf_world_orn = parent_link_state[0], parent_link_state[1]
            # this is the transformation from the parent frame to the world frame. 
            T_body_to_world = np.eye(4) # transformation from the parent body frame to the world frame
            T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(link_urdf_world_orn)).reshape(3, 3)
            T_body_to_world[:3, 3] = link_urdf_world_pos
            
            axis_world = T_body_to_world[:3, :3] @ axis_body + T_body_to_world[:3, 3]   
            axis_pt2_body = np.array(axis_body) + axis_dir_body
            axis_end_world = T_body_to_world[:3, :3] @ axis_pt2_body + T_body_to_world[:3, 3]
            axis_dir_world = axis_end_world - axis_world

            # get the handle points in world frame
            if handle_pts_obj_frame is None:
                handle_obj_path = f"{parent_dir}/parts_render/{handle_id}{target_object}.obj" # NOTE: this path should be dependent on the object. 
                handle_pts, handle_faces = load_obj(handle_obj_path) # this is in object frame

                handle_pts = handle_pts * scaling
                # add more dense points around handle
                added_points = []
                for f in handle_faces:
                    v1,v2,v3 = f
                    v1 = handle_pts[v1-1]
                    v2 = handle_pts[v2-1]
                    v3 = handle_pts[v3-1]
                    a = np.linalg.norm(v1-v2)
                    b = np.linalg.norm(v2-v3)
                    c = np.linalg.norm(v3-v1)
                    s = (a+b+c) / 2
                    temp = max(0, s*(s-a)*(s-b)*(s-c))
                    surface = np.sqrt(temp)
                    num_points = surface * 1e6
                    num_points = int(num_points)
                    num_points = np.clip(num_points, 0, 5)
                    added_points.extend([sample_point_inside_triangle(v1,v2,v3) for _ in range(num_points)])

                if added_points != []:
                    added_points = np.array(added_points)
                    handle_pts = np.concatenate((handle_pts, added_points), axis=0)
                    
                all_handle_pts_object_frame.append(handle_pts)
                    
            else:
                handle_pts = handle_pts_obj_frame[handle_idx]
            
            
            # transform this to the world frame using the object *base*'s position and orientation
            handle_points_world = T_body_to_world[:3, :3] @ handle_pts.T + T_body_to_world[:3, 3].reshape(3, 1) # 3 x N
            if return_median:
                handle_point_median = np.median(handle_points_world, axis=1)
            else:
                handle_point_median = handle_points_world.T

            # find the projection of the handle point to the rotation axis, in world frame. 
            project_on_rotation_axis = find_nearest_point_on_line(axis_world, axis_end_world, handle_point_median)
            # p.addUserDebugLine(project_on_rotation_axis, handle_point_median, [1, 0, 0], 25, 0)

            # TODO: GPT can parse the mobility.json to get the joint name. 
            joint_info = p.getJointInfo(obj_id, joint_idx, physicsClientId=simulator.id)
            joint_type = joint_info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                rotation_angle = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0] # NOTE: this joint id should be dependent on the object urdf.
                rotated_handle_pt_local = rotate_point_around_axis(handle_point_median - project_on_rotation_axis, axis_dir_world, rotation_angle)
                rotated_handle_pt = project_on_rotation_axis + rotated_handle_pt_local
            elif joint_type == p.JOINT_PRISMATIC:
                translation = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0]
                rotated_handle_pt = handle_point_median + axis_dir_world * translation
            else:
                rotated_handle_pt = handle_point_median
                
            # import pdb; pdb.set_trace()
            # rotated_handle_pt = handle_points_world.T

            if return_median:
                ret_handle_pt_list.append(rotated_handle_pt.flatten())
            else:
                ret_handle_pt_list.append(rotated_handle_pt)
            ret_joint_idx_list.append(joint_idx)
            
            handle_idx += 1
            
    if return_info:
        return ret_handle_pt_list, ret_joint_idx_list, all_handle_pts_object_frame, mobility_info
    
    return ret_handle_pt_list, ret_joint_idx_list


def get_link_pc(simulator, object_name, custom_link_name):
    object_name = object_name.lower()
    urdf_link_name = custom_link_name 
    link_com, all_pc = render_to_get_link_com(simulator, object_name, urdf_link_name)

    return all_pc

def render_to_get_link_com(simulator, object_name, urdf_link_name):    
    ### make all other objects invisiable
    prev_rgbas = []
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                prev_rgba = p.getVisualShapeData(obj_id, link_idx, physicsClientId=simulator.id)[0][14:18]
                prev_rgbas.append(prev_rgba)
                p.changeVisualShape(obj_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### center camera to the target object
    env_prev_view_matrix, env_prev_projection_matrix = simulator.view_matrix, simulator.projection_matrix
    camera_width = 640
    camera_height = 480
    obj_id = object_id
    min_aabb, max_aabb = simulator.get_aabb(obj_id)
    camera_target = (max_aabb + min_aabb) / 2
    distance = np.linalg.norm(max_aabb - min_aabb) * 1.2
    elevation = 30

    ### get a round of images of the target object
    rgbs, depths, view_matrices, projection_matrices = take_round_images(
        simulator, camera_target, distance, elevation, 
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10,
        return_camera_matrices=True)

    ### make the target link invisiable
    link_id = get_link_id_from_name(simulator, object_name, urdf_link_name)
    # import pdb; pdb.set_trace()
    prev_link_rgba = p.getVisualShapeData(obj_id, link_id, physicsClientId=simulator.id)[0][14:18]
    p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0], physicsClientId=simulator.id)

    ### get a round of images of the target object with link invisiable
    rgbs_link_invisiable, depths_link_invisible, _, _ = take_round_images(
        simulator, camera_target, distance, elevation,
        camera_width=camera_width, camera_height=camera_height, 
        z_near=0.01, z_far=10, return_camera_matrices=True
    )

    ### use subtraction to get the link mask
    max_num_diff_pixels = 0
    best_idx = 0
    for idx, (depth, depth_) in enumerate(zip(depths, depths_link_invisible)):
        diff_image = np.abs(depth - depth_)
        diff_pixels = np.sum(diff_image > 0)
        if diff_pixels > max_num_diff_pixels:
            max_num_diff_pixels = diff_pixels
            best_idx = idx
    best_mask = np.abs(depths[best_idx] - depths_link_invisible[best_idx]) > 0
    # best_mask = np.any(best_mask)


    ### get the link mask center
    center = ndimage.measurements.center_of_mass(best_mask)
    center = [int(center[0]), int(center[1])]

    ### back project the link mask center to get the link com in 3d coordinate
    best_pc = get_pc(projection_matrices[best_idx], view_matrices[best_idx], depths[best_idx], camera_width, camera_height)
    
    pt_idx = center[0] * camera_width + center[1]
    link_com = best_pc[pt_idx]
    best_pc = best_pc.reshape((camera_height, camera_width, 3))
    all_pc = best_pc[best_mask]


    ### reset the object and link rgba to previous values, and the simulator view matrix and projection matrix
    p.changeVisualShape(obj_id, link_id, rgbaColor=prev_link_rgba, physicsClientId=simulator.id)

    cnt = 0
    object_id = simulator.urdf_ids[object_name]
    for obj_name, obj_id in simulator.urdf_ids.items():
        if obj_name != object_name:
            num_links = p.getNumJoints(obj_id, physicsClientId=simulator.id)
            for link_idx in range(-1, num_links):
                p.changeVisualShape(obj_id, link_idx, rgbaColor=prev_rgbas[cnt], physicsClientId=simulator.id)
                cnt += 1

    simulator.view_matrix, simulator.projection_matrix = env_prev_view_matrix, env_prev_projection_matrix

    ### add a safety check here in case the rendering fails
    bounding_box = get_bounding_box_link(simulator, object_name, urdf_link_name)
    if not in_bbox(link_com, bounding_box[0], bounding_box[1]):
        link_com = (bounding_box[0] + bounding_box[1]) / 2

    return link_com, all_pc

def get_bounding_box_link(simulator, object_name, link_name):
    object_name = object_name.lower()
    link_id = get_link_id_from_name(simulator, object_name, link_name)
    object_id = simulator.urdf_ids[object_name]
    return simulator.get_aabb_link(object_id, link_id)

def get_link_id_from_name(simulator, object_name, link_name):
    object_id = simulator.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=simulator.id)
    joint_index = None
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=simulator.id)
        if joint_info[12].decode("utf-8") == link_name:
            joint_index = i
            break

    return joint_index
