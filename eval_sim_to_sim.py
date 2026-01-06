#!/usr/bin/env python3
"""
Sim-to-Sim Evaluation Script for ArticuBot.

This script implements the sim-to-sim evaluation protocol where:
1. A policy is fine-tuned on demonstrations synthesized from digital twin models
   (e.g., data/custom_objects_v2/sim/microwave_7128/)
2. The fine-tuned policy is then evaluated on the corresponding original models
   (e.g., data/custom_objects_v2/sim/assets/7128/)

This evaluation protocol measures the transferability of policies trained on
simplified/reconstructed digital twins to the original high-fidelity models.

Dataset Structure:
    data/custom_objects_v2/sim/
    ├── microwave_7128/           # Digital twin model
    │   ├── urdf/
    │   │   └── microwave_7128.urdf
    │   └── color/
    ├── assets/
    │   └── 7128/                 # Original PartNet-Mobility model
    │       ├── mobility.urdf
    │       └── mobility_v2.json
    └── ...

Usage examples:
    # Evaluate a finetuned policy on original model
    python eval_sim_to_sim.py 
        --sim-dataset-dir data/custom_objects_v2/sim 
        --object-type microwave 
        --object-id 7128 
        --high-level-ckpt weighted_displacement_model/exps/microwave_7128_finetuned/best.pth 
        --num-trials 25

    # Evaluate multiple objects
    python eval_sim_to_sim.py 
        --sim-dataset-dir data/custom_objects_v2/sim 
        --object-list microwave_7128 microwave_7167 oven_7130 
        --high-level-ckpt path/to/checkpoint.pth 
        --num-trials 10

    # Evaluate on digital twin model (for comparison)
    python eval_sim_to_sim.py 
        --sim-dataset-dir data/custom_objects_v2/sim 
        --object-type microwave 
        --object-id 7128 
        --eval-on-twin 
        --high-level-ckpt path/to/checkpoint.pth 
        --num-trials 10
"""

import os
import sys
import pathlib
import json
import argparse
import tempfile
import numpy as np
import torch
import hydra
import yaml
import pybullet as p
from copy import deepcopy
from omegaconf import OmegaConf
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from termcolor import cprint

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
os.environ['PROJECT_DIR'] = str(PROJECT_ROOT)

from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from manipulation.custom_object_utils.robogen_wrapper_custom import RobogenPointCloudWrapperCustom
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env_eval, build_up_env_gen, save_numpy_as_gif


# =============================================================================
# Dataset Pairing Utilities
# =============================================================================

def parse_object_folder_name(folder_name: str) -> Tuple[str, str]:
    """Parse object folder name into object type and ID.
    
    Args:
        folder_name: Folder name like 'microwave_7128' or 'storagefurniture1_44781'
        
    Returns:
        Tuple of (object_type, object_id)
        
    Example:
        >>> parse_object_folder_name('microwave_7128')
        ('microwave', '7128')
        >>> parse_object_folder_name('storagefurniture1_44781')
        ('storagefurniture1', '44781')
    """
    # Find the last underscore followed by digits
    parts = folder_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    else:
        raise ValueError(f"Cannot parse object folder name: {folder_name}")


def get_digital_twin_path(sim_dataset_dir: str, object_type: str, object_id: str) -> str:
    """Get path to digital twin model.
    
    Args:
        sim_dataset_dir: Root directory of sim dataset (e.g., data/custom_objects_v2/sim)
        object_type: Object type (e.g., 'microwave')
        object_id: Object ID (e.g., '7128')
        
    Returns:
        Path to digital twin folder
    """
    folder_name = f"{object_type}_{object_id}"
    return os.path.join(sim_dataset_dir, folder_name)


def get_original_model_path(sim_dataset_dir: str, object_id: str) -> str:
    """Get path to original PartNet-Mobility model.
    
    Args:
        sim_dataset_dir: Root directory of sim dataset
        object_id: Object ID (e.g., '7128')
        
    Returns:
        Path to original model folder in assets/
    """
    return os.path.join(sim_dataset_dir, 'assets', object_id)


def find_all_paired_objects(sim_dataset_dir: str) -> List[Dict]:
    """Find all object pairs (digital twin + original model) in the dataset.
    
    Args:
        sim_dataset_dir: Root directory of sim dataset
        
    Returns:
        List of dicts with keys: object_type, object_id, twin_path, original_path
    """
    paired_objects = []
    
    # List all folders in sim_dataset_dir
    for entry in os.listdir(sim_dataset_dir):
        entry_path = os.path.join(sim_dataset_dir, entry)
        
        # Skip non-directories and special folders
        if not os.path.isdir(entry_path):
            continue
        if entry in ['assets', '__pycache__']:
            continue
            
        # Try to parse as object folder
        try:
            object_type, object_id = parse_object_folder_name(entry)
        except ValueError:
            continue
            
        # Check if both twin and original exist
        twin_path = entry_path
        original_path = get_original_model_path(sim_dataset_dir, object_id)
        
        # Verify paths exist and have required files
        twin_urdf_dir = os.path.join(twin_path, 'urdf')
        original_urdf = os.path.join(original_path, 'mobility.urdf')
        
        if os.path.exists(twin_urdf_dir) and os.path.exists(original_urdf):
            paired_objects.append({
                'object_type': object_type,
                'object_id': object_id,
                'twin_path': twin_path,
                'original_path': original_path,
                'folder_name': entry,
            })
    
    return sorted(paired_objects, key=lambda x: (x['object_type'], x['object_id']))


def create_original_model_config(
    original_path: str,
    object_type: str,
    output_dir: str,
    target_position: Tuple[float, float, float] = (0.4, 0.0, 0.0),
    scale: float = 1.0,
    init_joint_angle: float = 0.0,
) -> str:
    """Create a task config YAML for evaluating on original model.
    
    The original PartNet-Mobility models have different structure than digital twins,
    so we need to generate appropriate config files for them.
    
    Args:
        original_path: Path to original model folder
        object_type: Type of object (microwave, oven, etc.)
        output_dir: Directory to write config file
        target_position: Where to place the object
        scale: Scale factor for the object
        init_joint_angle: Initial joint angle
        
    Returns:
        Path to generated config file
    """
    # Read mobility_v2.json for joint information
    mobility_path = os.path.join(original_path, 'mobility_v2.json')
    if not os.path.exists(mobility_path):
        raise FileNotFoundError(f"mobility_v2.json not found: {mobility_path}")
    
    with open(mobility_path, 'r') as f:
        mobility_info = json.load(f)
    
    # Find the main articulated joint (first non-free joint with parent != -1)
    main_joint = None
    handle_link = None
    for joint_info in mobility_info:
        if joint_info.get('parent', -1) >= 0 and joint_info.get('joint') in ['hinge', 'slider']:
            main_joint = joint_info
            # Get the link name from parts
            parts = joint_info.get('parts', [])
            if parts:
                handle_link = f"link_{joint_info.get('id', 0)}"
            break
    
    if main_joint is None:
        raise ValueError(f"No articulated joint found in {mobility_path}")
    
    # Get joint limits
    joint_data = main_joint.get('jointData', {})
    limit_data = joint_data.get('limit', {})
    lower_limit = limit_data.get('a', 0)
    upper_limit = limit_data.get('b', 1.57)
    
    # For hinge joints, convert degrees to radians if needed
    if main_joint.get('joint') == 'hinge':
        if abs(upper_limit) > 3.2:  # Likely in degrees
            lower_limit = np.deg2rad(lower_limit)
            upper_limit = np.deg2rad(upper_limit)
    
    # Ensure limits are in correct order
    if lower_limit > upper_limit:
        lower_limit, upper_limit = upper_limit, lower_limit
    
    # Construct the URDF path
    urdf_path = os.path.join(original_path, 'mobility.urdf')
    
    # Determine object class name for environment
    object_class_map = {
        'microwave': 'Microwave',
        'oven': 'Oven',
        'dishwasher': 'Dishwasher',
        'washingmachine': 'WashingMachine',
        'safe': 'Safe',
        'box': 'Box',
        'toilet': 'Toilet',
        'trashcan': 'TrashCan',
        'storagefurniture1': 'StorageFurniture',
        'storagefurniture2': 'StorageFurniture',
        'storagefurniture3': 'StorageFurniture',
    }
    object_class = object_class_map.get(object_type.lower(), 'StorageFurniture')
    
    # NOTE: The simulator's config parser (`parse_config` in
    # manipulation/utils/env_utils.py) expects each URDF object entry to contain
    # at least: `name`, `type`, `urdf_path`, `size`, and `center`.
    # `size` is treated as the target diagonal length (meters) and is used to
    # scale the URDF via PyBullet's `globalScaling`.
    center_str = f"({target_position[0]}, {target_position[1]}, {target_position[2]})"
    euler_str = "(0.0, 0.0, 0.0)"

    # Heuristic default size similar to existing custom-object configs.
    # `scale` acts as a multiplier on this target diagonal length.
    target_size = float(scale) * 0.7

    config = [
        {'use_table': False},
        {
            'name': object_class,
            'type': 'urdf',
            'urdf_path': urdf_path,
            'reward_asset_path': original_path,
            'link_name': handle_link or 'link_0',
            'center': center_str,
            'euler': euler_str,
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'size': target_size,
            'on_table': False,
            'is_crop_size': False,
            'joint_lower_limit': float(lower_limit),
            'joint_upper_limit': float(upper_limit),
            'init_angle': float(init_joint_angle),
        },
    ]
    
    # Write config file
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'eval_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


# =============================================================================
# Policy Loading
# =============================================================================

def load_policies(low_level_exp_dir: str, low_level_ckpt_name: str,
                  high_level_ckpt_path: str, add_one_hot_encoding: bool = False):
    """Load pretrained or finetuned high-level and low-level policies.
    
    Args:
        low_level_exp_dir: Path to low-level policy experiment directory
        low_level_ckpt_name: Checkpoint filename within checkpoints/
        high_level_ckpt_path: Path to high-level policy checkpoint (.pth)
        add_one_hot_encoding: Whether to use one-hot encoding for modality
        
    Returns:
        Tuple of (cfg, low_level_policy, high_level_policy)
    """
    print("=" * 60)
    print("Loading Policies...")
    print("=" * 60)
    
    # Load low-level policy
    print(f"  Low-level: {low_level_exp_dir}/{low_level_ckpt_name}")
    with hydra.initialize(config_path='3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",
            overrides=OmegaConf.load(f"{low_level_exp_dir}/.hydra/overrides.yaml"),
        )
    
    cfg = recomposed_config
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = f"{low_level_exp_dir}/checkpoints/{low_level_ckpt_name}"
    workspace.load_checkpoint(path=checkpoint_dir)
    
    low_level_policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        low_level_policy = deepcopy(workspace.ema_model)
    
    low_level_policy.eval()
    low_level_policy.reset()
    low_level_policy = low_level_policy.to('cuda')
    cprint("  ✓ Low-level policy loaded", "green")
    
    # Load high-level policy
    print(f"  High-level: {high_level_ckpt_path}")
    num_class = 13
    input_channel = 5 if add_one_hot_encoding else 3
    
    from weighted_displacement_model.model_invariant import PointNet2_super
    high_level_policy = PointNet2_super(num_classes=num_class, input_channel=input_channel).to("cuda")
    high_level_policy.load_state_dict(torch.load(high_level_ckpt_path))
    high_level_policy.eval()
    cprint("  ✓ High-level policy loaded", "green")
    
    return cfg, low_level_policy, high_level_policy


# =============================================================================
# Environment Construction
# =============================================================================

def construct_env_for_original_model(
    cfg,
    original_path: str,
    object_type: str,
    temp_dir: str,
    num_points: int = 4500,
    observation_mode: str = "act3d_goal_displacement_gripper_to_object",
    render: bool = False,
    randomize_object_pose: bool = True,
    randomize_robot_joints: bool = True,
    randomize_joint_angle: bool = True,
):
    """Construct environment for evaluating on original PartNet-Mobility model.
    
    Args:
        cfg: Hydrated config with n_obs_steps, n_action_steps
        original_path: Path to original model folder
        object_type: Type of object
        temp_dir: Directory for temporary config files
        num_points: Number of points in point cloud
        observation_mode: Observation mode for wrapper
        render: Enable GUI rendering
        randomize_*: Randomization flags
        
    Returns:
        Tuple of (env, object_name, init_angle, joint_limits)
    """
    from manipulation.utils import build_up_env_random
    
    # Create config for original model
    config_path = create_original_model_config(
        original_path=original_path,
        object_type=object_type,
        output_dir=temp_dir,
    )
    
    # Read back the config to get object info
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    object_name = None
    link_name = 'link_0'
    for entry in config:
        if isinstance(entry, dict):
            if 'name' in entry:
                object_name = entry['name'].lower()
            if 'link_name' in entry:
                link_name = entry['link_name']
    
    # Build environment with random initialization
    # IMPORTANT: Use the custom-object environment for this project.
    # We treat the original PartNet asset as a "custom object" during eval.
    # Also disable joint-angle randomization here to avoid relying on legacy
    # handle/parts_render pipelines during initialization.
    env, save_config = build_up_env_random(
        task_config=config_path,
        env_name='articulated_custom_object',
        render=render,
        randomize_object_pose=randomize_object_pose,
        randomize_robot_joints=randomize_robot_joints,
        randomize_initial_joint_angle=False,
        horizon=600,
        max_attempts=100,
        far_distance=0.6,
        near_distance=0.25,
    )
    
    # Wrap with point cloud wrapper
    # NOTE: RobogenPointCloudWrapper tries to load goal states from
    # stage_lengths.json (saved demo trajectory). In sim-to-sim evaluation we
    # use random initialization (no saved states), so use the custom wrapper
    # with skip_goal_loading=True to avoid requiring those files.
    pointcloud_env = RobogenPointCloudWrapperCustom(
        env,
        object_name,
        asset_dir_hint=original_path,
        num_points=num_points,
        observation_mode=observation_mode,
        skip_goal_loading=True,
        skip_env_info=True,
    )
    
    # Wrap with multi-step wrapper
    wrapped_env = MultiStepWrapper(
        pointcloud_env,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=600,
        reward_agg_method='sum'
    )
    
    # Get initial joint angle and limits
    object_id = env.urdf_ids[object_name]
    init_angle = None
    joint_limits = None
    num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=env.id)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_state = p.getJointState(object_id, i, physicsClientId=env.id)
            init_angle = joint_state[0]
            joint_limits = (joint_info[8], joint_info[9])  # lower, upper
            break
    
    return wrapped_env, object_name, init_angle, joint_limits


def construct_env_for_digital_twin(
    cfg,
    twin_path: str,
    object_type: str,
    temp_dir: str,
    num_points: int = 4500,
    observation_mode: str = "act3d_goal_displacement_gripper_to_object",
    render: bool = False,
):
    """Construct environment for evaluating on digital twin model.
    
    Uses the custom wrapper since digital twins have different naming conventions.
    
    Args:
        cfg: Hydrated config
        twin_path: Path to digital twin folder
        object_type: Type of object
        temp_dir: Directory for temporary config files
        num_points: Number of points in point cloud
        observation_mode: Observation mode
        render: Enable GUI rendering
        
    Returns:
        Tuple of (env, object_name, init_angle, joint_limits)
    """
    from manipulation.utils import build_up_env_random
    
    # Find URDF file in twin
    urdf_dir = os.path.join(twin_path, 'urdf')
    urdf_files = [f for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
    if not urdf_files:
        raise FileNotFoundError(f"No URDF found in {urdf_dir}")
    
    urdf_path = os.path.join(urdf_dir, urdf_files[0])
    
    # Build minimal config
    object_class = object_type.capitalize()
    if 'storagefurniture' in object_type.lower():
        object_class = 'StorageFurniture'
    
    config = [
        {
            'name': object_class,
            'type': 'urdf',
            'urdf_path': urdf_path,
            'solution_path': twin_path,
            'reward_asset_path': twin_path,
            'link_name': 'link1',  # Digital twins typically use link0, link1
            'center': '(0.4, 0.0, 0.0)',
            'size': 0.7,
            'euler': '(0.0, 0.0, 0.0)',
            'orientation': [0.0, 0.0, 0.0, 1.0],
            'init_angle': 0.0,
        },
    ]
    
    config_path = os.path.join(temp_dir, 'eval_config_twin.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Build environment
    env, _ = build_up_env_random(
        task_config=config_path,
        env_name='articulated_custom_object',
        render=render,
        randomize_object_pose=True,
        randomize_robot_joints=True,
        randomize_initial_joint_angle=True,
        horizon=600,
        max_attempts=100,
    )
    
    object_name = object_class.lower()
    
    # Wrap with custom wrapper for digital twins
    pointcloud_env = RobogenPointCloudWrapperCustom(
        env,
        object_name,
        asset_dir_hint=twin_path,
        num_points=num_points,
        observation_mode=observation_mode,
        skip_goal_loading=True,
        skip_env_info=True,
    )
    
    wrapped_env = MultiStepWrapper(
        pointcloud_env,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=600,
        reward_agg_method='sum'
    )
    
    # Get initial joint state
    object_id = env.urdf_ids[object_name]
    init_angle = None
    joint_limits = None
    num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=env.id)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_state = p.getJointState(object_id, i, physicsClientId=env.id)[0]
            init_angle = joint_state
            joint_limits = (joint_info[8], joint_info[9])
            break
    
    return wrapped_env, object_name, init_angle, joint_limits


# =============================================================================
# Evaluation Logic
# =============================================================================

def high_level_policy_infer(obs_dict, high_level_policy,
                            output_obj_pcd_only: bool = True,
                            add_one_hot_encoding: bool = False):
    """Run high-level policy to predict goal end-effector points."""
    with torch.no_grad():
        pointcloud = obs_dict['point_cloud'][:, -1, :, :]
        gripper_pcd = obs_dict['gripper_pcd'][:, -1, :]
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
        
        if add_one_hot_encoding:
            batch_size = inputs.shape[0]
            num_points = inputs.shape[1]
            one_hot = torch.zeros(batch_size, num_points, 2).to(inputs.device)
            one_hot[:, :pointcloud.shape[1], 0] = 1
            one_hot[:, pointcloud.shape[1]:, 1] = 1
            inputs = torch.cat([inputs, one_hot], dim=-1)
        
        inputs = inputs.to('cuda')
        inputs_ = inputs.permute(0, 2, 1)
        outputs = high_level_policy(inputs_)
        
        weights = outputs[:, :, -1]
        outputs = outputs[:, :, :-1]
        
        if output_obj_pcd_only:
            outputs = outputs[:, :-4, :]
            weights = weights[:, :-4]
            inputs_for_residual = inputs[:, :-4, :3]
        else:
            inputs_for_residual = inputs[:, :, :3]
        
        B, N, _ = outputs.shape
        outputs = outputs.view(B, N, 4, 3)
        outputs = outputs + inputs_for_residual.unsqueeze(2)
        weights = torch.nn.functional.softmax(weights, dim=1)
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        outputs = outputs.unsqueeze(1)
    
    return outputs


def compute_normalized_opening(init_angle: float, final_angle: float,
                               joint_limits: Tuple[float, float]) -> float:
    """Compute normalized opening ratio.
    
    Args:
        init_angle: Initial joint angle
        final_angle: Final joint angle
        joint_limits: (lower_limit, upper_limit)
        
    Returns:
        Normalized opening ratio [0, 1], where 1 means fully opened
    """
    lower, upper = joint_limits
    range_size = abs(upper - lower)
    
    if range_size < 1e-6:
        return 0.0
    
    # Opening is measured relative to the closed state (lower limit)
    init_ratio = abs(init_angle - lower) / range_size
    final_ratio = abs(final_angle - lower) / range_size
    
    # Improvement in opening
    improvement = final_ratio - init_ratio
    
    return improvement


def run_single_trial(
    env,
    object_name: str,
    low_level_policy,
    high_level_policy,
    init_angle: float,
    joint_limits: Tuple[float, float],
    horizon: int = 70,
    update_goal_freq: int = 1,
    output_obj_pcd_only: bool = True,
    add_one_hot_encoding: bool = False,
    save_gif: bool = True,
):
    """Run a single evaluation trial.
    
    Returns:
        Tuple of (rgb_frames, metrics_dict)
    """
    from manipulation.utils import rotation_transfer_matrix_to_6D
    
    obs = env.reset()
    all_rgbs = []
    
    if save_gif:
        rgb = env.env.render()
        all_rgbs.append(rgb)
    
    last_goal = None
    steps_taken = 1
    
    for t in range(1, horizon):
        # Prepare input
        parallel_input_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to('cuda'))
        for key in obs:
            parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
        
        # Infer high-level goal
        if last_goal is None or t % update_goal_freq == 0:
            with torch.no_grad():
                goal_eef_points = high_level_policy_infer(
                    parallel_input_dict,
                    high_level_policy,
                    output_obj_pcd_only=output_obj_pcd_only,
                    add_one_hot_encoding=add_one_hot_encoding
                )
            last_goal = goal_eef_points
        else:
            goal_eef_points = last_goal
        
        # Add goal to input (repeat for all steps in observation history)
        T_obs = parallel_input_dict['point_cloud'].shape[1]
        goal_expanded = goal_eef_points.repeat(1, T_obs, 1, 1)
        np_goal = goal_eef_points.detach().cpu().numpy()
        
        parallel_input_dict['goal_eef_points'] = goal_expanded
        parallel_input_dict['goal_gripper_pcd'] = goal_expanded

        # Compute displacement history (required by Act3D encoders using displacement mode)
        B, T_hist, N_pts, _ = parallel_input_dict['point_cloud'].shape
        gripper_pcd_all = parallel_input_dict['gripper_pcd']
        point_cloud_all = parallel_input_dict['point_cloud']
        
        displacements_list = []
        import scipy.spatial.distance
        for b in range(B):
            step_disps = []
            for s in range(T_hist):
                g_pcd = gripper_pcd_all[b, s].cpu().numpy()
                o_pcd = point_cloud_all[b, s].cpu().numpy()
                dist_matrix = scipy.spatial.distance.cdist(g_pcd, o_pcd)
                min_idx = np.argmin(dist_matrix, axis=1)
                closest_pts = o_pcd[min_idx]
                step_disps.append(closest_pts - g_pcd)
            displacements_list.append(step_disps)
        parallel_input_dict['displacement_gripper_to_object'] = torch.from_numpy(np.array(displacements_list)).to('cuda').float()

        # Infer and execute action
        with torch.no_grad():
            action_dict = low_level_policy.predict_action(parallel_input_dict)
        action = action_dict['action'][0].detach().cpu().numpy()
        
        obs, reward, done, info = env.step(action)
        env.env.goal_gripper_pcd = np_goal.squeeze(0)[0]
        
        if save_gif:
            rgb = env.env.render()
            all_rgbs.append(rgb)
        
        steps_taken = t + 1
        if done:
            break
    
    # Get final joint angle
    base_env = env.env._env if hasattr(env.env, '_env') else env.env
    object_id = base_env.urdf_ids[object_name]
    
    final_angle = None
    num_joints = p.getNumJoints(object_id, physicsClientId=base_env.id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=base_env.id)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            final_angle = p.getJointState(object_id, i, physicsClientId=base_env.id)[0]
            break
    
    # Close environment
    base_env.close()
    
    # Compute metrics
    improved_angle = final_angle - init_angle if (init_angle is not None and final_angle is not None) else 0.0
    normalized_opening = compute_normalized_opening(init_angle, final_angle, joint_limits) if joint_limits else 0.0
    
    metrics = {
        'initial_angle': float(init_angle) if init_angle is not None else 0.0,
        'final_angle': float(final_angle) if final_angle is not None else 0.0,
        'improved_angle': float(improved_angle),
        'normalized_opening': float(normalized_opening),
        'steps_taken': steps_taken,
    }
    
    return all_rgbs, metrics


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def evaluate_object(
    cfg,
    low_level_policy,
    high_level_policy,
    object_info: Dict,
    eval_on_twin: bool,
    num_trials: int,
    save_dir: str,
    horizon: int = 70,
    render: bool = False,
    save_gifs: bool = True,
    add_one_hot_encoding: bool = False,
):
    """Evaluate policy on a single object.
    
    Args:
        cfg: Hydrated config
        low_level_policy: Low-level policy
        high_level_policy: High-level policy
        object_info: Dict with object_type, object_id, twin_path, original_path
        eval_on_twin: If True, evaluate on digital twin instead of original
        num_trials: Number of evaluation trials
        save_dir: Directory to save results
        horizon: Episode horizon
        render: Enable GUI
        save_gifs: Save GIF for each trial
        add_one_hot_encoding: Use one-hot encoding
        
    Returns:
        Dict with aggregated metrics
    """
    object_type = object_info['object_type']
    object_id = object_info['object_id']
    folder_name = object_info['folder_name']
    
    eval_target = "digital_twin" if eval_on_twin else "original_model"
    print(f"\n{'='*60}")
    print(f"Evaluating: {folder_name} on {eval_target}")
    print(f"{'='*60}")
    
    # Create output directory
    object_save_dir = os.path.join(save_dir, folder_name, eval_target)
    os.makedirs(object_save_dir, exist_ok=True)
    
    all_metrics = []
    temp_dir = tempfile.mkdtemp()
    
    for trial_idx in range(num_trials):
        print(f"\n  Trial {trial_idx + 1}/{num_trials}", end=" ")
        
        try:
            # Construct environment
            if eval_on_twin:
                env, object_name, init_angle, joint_limits = construct_env_for_digital_twin(
                    cfg,
                    twin_path=object_info['twin_path'],
                    object_type=object_type,
                    temp_dir=temp_dir,
                    render=render,
                )
            else:
                env, object_name, init_angle, joint_limits = construct_env_for_original_model(
                    cfg,
                    original_path=object_info['original_path'],
                    object_type=object_type,
                    temp_dir=temp_dir,
                    render=render,
                )
            
            # Run trial
            all_rgbs, metrics = run_single_trial(
                env,
                object_name,
                low_level_policy,
                high_level_policy,
                init_angle,
                joint_limits,
                horizon=horizon,
                add_one_hot_encoding=add_one_hot_encoding,
                save_gif=save_gifs,
            )
            
            metrics['trial'] = trial_idx
            metrics['success'] = True
            all_metrics.append(metrics)
            
            cprint(f"✓ Opening: {metrics['normalized_opening']:.3f}", "green")
            
            # Save GIF
            if save_gifs and len(all_rgbs) > 0:
                gif_path = os.path.join(
                    object_save_dir,
                    f"trial_{trial_idx:03d}_open_{metrics['normalized_opening']:.3f}.gif"
                )
                save_numpy_as_gif(np.array(all_rgbs), gif_path)
                
        except Exception as e:
            cprint(f"✗ Failed: {e}", "red")
            import traceback
            traceback.print_exc()
            all_metrics.append({
                'trial': trial_idx,
                'success': False,
                'error': str(e),
            })
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Compute summary statistics
    successful_trials = [m for m in all_metrics if m.get('success', False)]
    
    summary = {
        'object_type': object_type,
        'object_id': object_id,
        'eval_target': eval_target,
        'total_trials': num_trials,
        'successful_trials': len(successful_trials),
        'trials': all_metrics,
    }
    
    if successful_trials:
        openings = [m['normalized_opening'] for m in successful_trials]
        summary['mean_opening'] = float(np.mean(openings))
        summary['std_opening'] = float(np.std(openings))
        summary['max_opening'] = float(np.max(openings))
        summary['min_opening'] = float(np.min(openings))
        
        # Success rate (opening > 0.3)
        summary['success_rate_30'] = float(np.mean([o > 0.3 for o in openings]))
        summary['success_rate_50'] = float(np.mean([o > 0.5 for o in openings]))
    
    # Save metrics
    metrics_path = os.path.join(object_save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sim-to-Sim Evaluation for ArticuBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Dataset arguments
    parser.add_argument('--sim-dataset-dir', type=str, required=True,
                        help='Root directory of sim dataset (e.g., data/custom_objects_v2/sim)')
    parser.add_argument('--object-type', type=str, default=None,
                        help='Object type to evaluate (e.g., microwave)')
    parser.add_argument('--object-id', type=str, default=None,
                        help='Object ID to evaluate (e.g., 7128)')
    parser.add_argument('--object-list', type=str, nargs='+', default=None,
                        help='List of object folder names to evaluate')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate all paired objects in dataset')
    
    # Evaluation mode
    parser.add_argument('--eval-on-twin', action='store_true',
                        help='Evaluate on digital twin instead of original model')
    parser.add_argument('--eval-both', action='store_true',
                        help='Evaluate on both digital twin and original model')
    
    # Checkpoint paths
    parser.add_argument('--low-level-exp-dir', type=str,
                        default='data/low-level-ckpt',
                        help='Path to low-level policy experiment directory')
    parser.add_argument('--low-level-ckpt-name', type=str,
                        default='low-level.ckpt',
                        help='Low-level checkpoint filename')
    parser.add_argument('--high-level-ckpt', type=str,
                        default='data/high_level_200_obj_ckpt.pth',
                        help='Path to high-level checkpoint (.pth)')
    
    # Evaluation settings
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of evaluation trials per object')
    parser.add_argument('--horizon', type=int, default=70,
                        help='Steps per episode')
    
    # Output settings
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--no-gif', action='store_true',
                        help='Disable GIF saving')
    parser.add_argument('--render', action='store_true',
                        help='Enable GUI rendering')
    
    # Model settings
    parser.add_argument('--add-one-hot-encoding', type=int, default=0,
                        help='Add one-hot modality encoding')
    
    # Utility arguments
    parser.add_argument('--list-objects', action='store_true',
                        help='List all paired objects and exit')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List objects mode
    if args.list_objects:
        paired_objects = find_all_paired_objects(args.sim_dataset_dir)
        print(f"\nFound {len(paired_objects)} paired objects in {args.sim_dataset_dir}:\n")
        print(f"{'Folder Name':<35} {'Type':<20} {'ID':<10}")
        print("-" * 65)
        for obj in paired_objects:
            print(f"{obj['folder_name']:<35} {obj['object_type']:<20} {obj['object_id']:<10}")
        return
    
    # Determine which objects to evaluate
    paired_objects = find_all_paired_objects(args.sim_dataset_dir)
    
    if args.object_type and args.object_id:
        # Single object by type + id
        target_folder = f"{args.object_type}_{args.object_id}"
        objects_to_eval = [o for o in paired_objects if o['folder_name'] == target_folder]
        if not objects_to_eval:
            cprint(f"Error: Object {target_folder} not found in dataset", "red")
            sys.exit(1)
    elif args.object_list:
        # List of object folders
        objects_to_eval = [o for o in paired_objects if o['folder_name'] in args.object_list]
        if len(objects_to_eval) != len(args.object_list):
            found = [o['folder_name'] for o in objects_to_eval]
            missing = [n for n in args.object_list if n not in found]
            cprint(f"Warning: Objects not found: {missing}", "yellow")
    elif args.eval_all:
        objects_to_eval = paired_objects
    else:
        cprint("Error: Must specify --object-type/--object-id, --object-list, or --eval-all", "red")
        sys.exit(1)
    
    # Setup save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ckpt_name = os.path.splitext(os.path.basename(args.high_level_ckpt))[0]
        args.save_dir = os.path.join('outputs', 'sim_to_sim_eval', f"{timestamp}_{ckpt_name}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("Sim-to-Sim Evaluation")
    print("=" * 60)
    print(f"Dataset: {args.sim_dataset_dir}")
    print(f"Objects to evaluate: {len(objects_to_eval)}")
    print(f"Eval mode: {'Both' if args.eval_both else ('Digital Twin' if args.eval_on_twin else 'Original Model')}")
    print(f"Trials per object: {args.num_trials}")
    print(f"High-level ckpt: {args.high_level_ckpt}")
    print(f"Save directory: {args.save_dir}")
    print("=" * 60)
    
    # Load policies
    cfg, low_level_policy, high_level_policy = load_policies(
        args.low_level_exp_dir,
        args.low_level_ckpt_name,
        args.high_level_ckpt,
        bool(args.add_one_hot_encoding)
    )
    
    # Run evaluation
    all_summaries = []
    
    for obj_info in objects_to_eval:
        eval_modes = []
        if args.eval_both:
            eval_modes = [False, True]  # Original first, then twin
        else:
            eval_modes = [args.eval_on_twin]
        
        for eval_on_twin in eval_modes:
            summary = evaluate_object(
                cfg,
                low_level_policy,
                high_level_policy,
                obj_info,
                eval_on_twin=eval_on_twin,
                num_trials=args.num_trials,
                save_dir=args.save_dir,
                horizon=args.horizon,
                render=args.render,
                save_gifs=not args.no_gif,
                add_one_hot_encoding=bool(args.add_one_hot_encoding),
            )
            all_summaries.append(summary)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    
    # Group by eval target
    original_results = [s for s in all_summaries if s['eval_target'] == 'original_model']
    twin_results = [s for s in all_summaries if s['eval_target'] == 'digital_twin']
    
    for target_name, results in [('Original Model', original_results), ('Digital Twin', twin_results)]:
        if not results:
            continue
        
        print(f"\n{target_name}:")
        print(f"  {'Object':<30} {'Trials':<10} {'Mean Opening':<15} {'Success@50':<15}")
        print("  " + "-" * 70)
        
        for s in results:
            folder = f"{s['object_type']}_{s['object_id']}"
            trials = f"{s['successful_trials']}/{s['total_trials']}"
            mean_open = f"{s.get('mean_opening', 0):.3f} ± {s.get('std_opening', 0):.3f}"
            success = f"{s.get('success_rate_50', 0)*100:.1f}%"
            print(f"  {folder:<30} {trials:<10} {mean_open:<15} {success:<15}")
        
        # Aggregate statistics
        if results:
            all_openings = [s.get('mean_opening', 0) for s in results if 'mean_opening' in s]
            all_success = [s.get('success_rate_50', 0) for s in results if 'success_rate_50' in s]
            if all_openings:
                print("  " + "-" * 70)
                print(f"  {'AVERAGE':<30} {'':<10} {np.mean(all_openings):.3f} ± {np.std(all_openings):.3f}         {np.mean(all_success)*100:.1f}%")
    
    # Save overall summary
    summary_path = os.path.join(args.save_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': all_summaries,
        }, f, indent=2)
    
    print(f"\nResults saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
