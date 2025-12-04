#!/usr/bin/env python3
"""
Unified evaluation script for custom articulated objects.

This script evaluates both pre-trained and finetuned policies on custom URDF objects.
It uses the RobogenPointCloudWrapperCustom which handles non-standard link naming
conventions and does NOT require mobility_v2.json files with privileged information.

Usage examples:
    # Evaluate pre-trained checkpoints on custom object
    python eval_custom_object.py \
        --asset-dir data/custom_objects/sim_microwave_good \
        --experiment-name my_demos \
        --num-trials 10

    # Evaluate finetuned checkpoints
    python eval_custom_object.py \
        --asset-dir data/custom_objects/sim_microwave_good \
        --experiment-name my_demos \
        --high-level-ckpt weighted_displacement_model/exps/my_finetuned/best.pth \
        --num-trials 10

    # Use random initialization instead of saved states
    python eval_custom_object.py \
        --asset-dir data/custom_objects/sim_microwave_good \
        --use-random-init \
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

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
os.environ['PROJECT_DIR'] = str(PROJECT_ROOT)

from manipulation.custom_object_utils.robogen_wrapper_custom import RobogenPointCloudWrapperCustom
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env_gen, build_up_env_random, load_env, save_numpy_as_gif


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
    print(f"Loading low-level policy from: {low_level_exp_dir}")
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
    print("✓ Low-level policy loaded")
    
    # Load high-level policy
    print(f"Loading high-level policy from: {high_level_ckpt_path}")
    num_class = 13
    input_channel = 5 if add_one_hot_encoding else 3
    
    from weighted_displacement_model.model_invariant import PointNet2_super
    high_level_policy = PointNet2_super(num_classes=num_class, input_channel=input_channel).to("cuda")
    high_level_policy.load_state_dict(torch.load(high_level_ckpt_path))
    high_level_policy.eval()
    print("✓ High-level policy loaded")
    
    return cfg, low_level_policy, high_level_policy


def is_valid_training_demo(demo_path: str) -> bool:
    """Check if a demo folder has all required files for training.
    
    Valid demos must have:
    - all.gif (indicates successful execution)
    - stage_lengths.json (timing information)
    - states/ folder with state_*.pkl files
    
    Args:
        demo_path: Path to the demo folder
        
    Returns:
        True if the demo is valid for training
    """
    import glob
    
    required_files = [
        os.path.join(demo_path, "all.gif"),
        os.path.join(demo_path, "stage_lengths.json"),
        os.path.join(demo_path, "task_config.yaml"),
    ]
    
    # Check required files exist
    for req in required_files:
        if not os.path.exists(req):
            return False
    
    # Check states directory has files
    states_dir = os.path.join(demo_path, "states")
    if not os.path.exists(states_dir):
        return False
    
    state_files = glob.glob(os.path.join(states_dir, "state_*.pkl"))
    return len(state_files) > 0


def fix_config_paths(config: list, asset_dir: str) -> list:
    """Fix absolute paths in task config to point to the current asset directory.
    
    The task_config.yaml files may contain absolute paths from when demos were
    generated on a different machine. This function rewrites those paths to
    point to the correct local asset directory.
    
    Args:
        config: Parsed YAML config (list of dicts)
        asset_dir: Path to the current asset directory
        
    Returns:
        Config with fixed paths
    """
    fixed_config = []
    asset_dir = os.path.abspath(asset_dir)
    
    # Path keys that may need fixing
    path_keys = ['urdf_path', 'solution_path', 'reward_asset_path', 
                 'annotation_path', 'handle_pts_path']
    
    for entry in config:
        if isinstance(entry, dict):
            fixed_entry = entry.copy()
            for key in path_keys:
                if key in fixed_entry:
                    old_path = fixed_entry[key]
                    if old_path and os.path.isabs(old_path):
                        # Extract relative path from the old absolute path
                        # Look for common folder names
                        basename = os.path.basename(old_path)
                        
                        if key == 'urdf_path':
                            # Try to find the URDF in asset_dir
                            new_path = os.path.join(asset_dir, basename)
                            if os.path.exists(new_path):
                                fixed_entry[key] = new_path
                        elif key in ['solution_path', 'reward_asset_path']:
                            # These should point to asset_dir
                            fixed_entry[key] = asset_dir
                        elif key == 'annotation_path':
                            new_path = os.path.join(asset_dir, 'annotation.json')
                            if os.path.exists(new_path):
                                fixed_entry[key] = new_path
                        elif key == 'handle_pts_path':
                            # Try to find in parts_render
                            new_path = os.path.join(asset_dir, 'parts_render', basename)
                            if os.path.exists(new_path):
                                fixed_entry[key] = new_path
            
            fixed_config.append(fixed_entry)
        else:
            fixed_config.append(entry)
    
    return fixed_config


def construct_env_from_experiment(cfg, asset_dir: str, experiment_name: str, 
                                   trial_idx: int = 0, num_points: int = 4500,
                                   observation_mode: str = "act3d_goal_displacement_gripper_to_object",
                                   only_valid_demos: bool = True):
    """Construct environment from saved demonstration experiment states.
    
    Args:
        cfg: Hydrated config with n_obs_steps, n_action_steps
        asset_dir: Path to custom object asset directory
        experiment_name: Name of the experiment folder containing demos
        trial_idx: Which trial to load (0-indexed)
        num_points: Number of points in point cloud
        observation_mode: Observation mode for the wrapper
        only_valid_demos: If True, only use demos that meet training requirements
                         (have all.gif, stage_lengths.json, and states/*.pkl)
        
    Returns:
        Tuple of (env, object_name, init_angle)
    """
    experiment_dir = os.path.join(asset_dir, 'experiment', experiment_name)
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    # Find trial directories
    all_trial_dirs = sorted([
        d for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
    ])
    
    # Filter to only valid training demos if requested
    if only_valid_demos:
        trial_dirs = [
            d for d in all_trial_dirs
            if is_valid_training_demo(os.path.join(experiment_dir, d))
        ]
        print(f"Found {len(trial_dirs)} valid training demos out of {len(all_trial_dirs)} total")
    else:
        trial_dirs = all_trial_dirs
    
    if trial_idx >= len(trial_dirs):
        raise IndexError(f"Trial index {trial_idx} out of range (max: {len(trial_dirs)-1})")
    
    trial_dir = os.path.join(experiment_dir, trial_dirs[trial_idx])
    
    # Load task config
    task_config_path = os.path.join(trial_dir, 'task_config.yaml')
    if not os.path.exists(task_config_path):
        raise FileNotFoundError(f"Task config not found: {task_config_path}")
    
    with open(task_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix absolute paths in config to point to current asset directory
    config = fix_config_paths(config, asset_dir)
    
    # Write fixed config to a temporary file for env loading
    temp_config_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    )
    yaml.dump(config, temp_config_file)
    temp_config_file.close()
    temp_config_path = temp_config_file.name
    
    # Extract object name and solution path
    object_name = None
    solution_path = None
    for entry in config:
        if isinstance(entry, dict):
            if 'name' in entry:
                object_name = entry['name'].lower()
            if 'solution_path' in entry:
                solution_path = entry['solution_path']
    
    if object_name is None:
        object_name = 'storagefurniture'
    
    # Find substeps to get task name
    if solution_path:
        substeps_path = os.path.join(PROJECT_ROOT, solution_path, 'substeps.txt')
    else:
        substeps_path = os.path.join(asset_dir, 'substeps.txt')
    
    if os.path.exists(substeps_path):
        with open(substeps_path, 'r') as f:
            substeps = [line.strip() for line in f if line.strip()]
        task_name = substeps[0].replace(' ', '_') if substeps else 'open_the_door'
    else:
        task_name = 'open_the_door'
    
    # Find init state file - try multiple locations
    # The gen_demo_custom saves state_0.pkl directly in trial folder
    init_state_file = None
    candidate_paths = [
        os.path.join(trial_dir, 'state_0.pkl'),  # Direct in trial folder (gen_demo_custom format)
        os.path.join(trial_dir, 'states', 'state_0.pkl'),  # In states subfolder
        os.path.join(trial_dir, f"{task_name}_primitive", 'states', 'state_0.pkl'),  # With primitive suffix
    ]
    
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            init_state_file = candidate
            break
    
    if init_state_file is None:
        raise FileNotFoundError(
            f"Initial state not found. Tried: {candidate_paths}"
        )
    
    print(f"Loading trial: {trial_dirs[trial_idx]}")
    print(f"  Config: {task_config_path}")
    print(f"  Init state: {init_state_file}")
    
    # Extract handle_name from config for the wrapper
    handle_name = 'handle'
    for entry in config:
        if isinstance(entry, dict) and 'handle_name' in entry:
            handle_name = entry['handle_name'].lower()
            break
    
    # Build environment using build_up_env_gen for custom objects
    # Following the pattern from extract_pcd_from_states_custom.py:
    # 1. Build env without restore_state_file
    # 2. Wrap with RobogenPointCloudWrapperCustom
    # 3. Use load_env to restore the saved state
    try:
        env, _ = build_up_env_gen(
            task_config=temp_config_path,
            env_name='articulated_custom_object',
            task_name=task_name,
            restore_state_file=None,  # Don't restore here, do it after wrapping
            render=False,
            randomize=False,
        )
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
        raise e
    
    # Clean up temp config file now that env is built
    if os.path.exists(temp_config_path):
        os.unlink(temp_config_path)
    
    # Wrap with custom point cloud wrapper
    pointcloud_env = RobogenPointCloudWrapperCustom(
        env,
        object_name,
        handle_name=handle_name,
        asset_dir_hint=asset_dir,
        num_points=num_points,
        observation_mode=observation_mode,
        skip_goal_loading=True,  # We don't need goal states during eval
    )
    
    # Now restore the saved state using load_env
    load_env(env, load_path=init_state_file)
    
    # Initialize attributes that are normally set during reset()
    # These are needed by take_direct_action() when we skip reset
    env.oversized_joint_distance = False
    
    # Wrap with multi-step wrapper
    wrapped_env = MultiStepWrapper(
        pointcloud_env,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=600,
        reward_agg_method='sum'
    )
    
    # Get initial joint angle for metrics
    object_id = env.urdf_ids[object_name]
    init_angle = None
    num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=env.id)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            init_angle = p.getJointState(object_id, i, physicsClientId=env.id)[0]
            break
    
    return wrapped_env, object_name, init_angle


def construct_env_random(cfg, asset_dir: str, config_id: int = 0,
                         num_points: int = 4500,
                         observation_mode: str = "act3d_goal_displacement_gripper_to_object"):
    """Construct environment with random initialization (no saved states required).
    
    Args:
        cfg: Hydrated config
        asset_dir: Path to custom object asset directory
        config_id: Which config file to use
        num_points: Number of points in point cloud
        observation_mode: Observation mode for the wrapper
        
    Returns:
        Tuple of (env, object_name, init_angle)
    """
    # Find config file
    configs_dir = os.path.join(asset_dir, 'configs')
    if not os.path.exists(configs_dir):
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")
    
    config_files = sorted([
        f for f in os.listdir(configs_dir)
        if f.endswith('.yaml')
    ])
    
    if not config_files:
        raise FileNotFoundError(f"No config files found in {configs_dir}")
    
    # Select config file
    if config_id < len(config_files):
        config_file = os.path.join(configs_dir, config_files[config_id])
    else:
        config_file = os.path.join(configs_dir, config_files[0])
    
    print(f"Using config: {config_file}")
    
    # Build environment with random initialization
    env, task_config = build_up_env_random(
        task_config=config_file,
        env_name='articulated',
        render=False,
        randomize_object_pose=False,  # Keep object at configured position
        randomize_robot_joints=True,
        randomize_initial_joint_angle=True,
        horizon=600,
        max_attempts=100,
        far_distance=0.7,
        near_distance=0.3
    )
    
    object_name = task_config['object_name'].lower()
    
    # Wrap with custom point cloud wrapper
    pointcloud_env = RobogenPointCloudWrapperCustom(
        env,
        object_name,
        asset_dir_hint=asset_dir,
        num_points=num_points,
        observation_mode=observation_mode,
        skip_goal_loading=True,
    )
    
    # Wrap with multi-step wrapper
    wrapped_env = MultiStepWrapper(
        pointcloud_env,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=600,
        reward_agg_method='sum'
    )
    
    # Get initial joint angle
    object_id = env.urdf_ids[object_name]
    init_angle = None
    num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=env.id)
        if joint_info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            init_angle = p.getJointState(object_id, i, physicsClientId=env.id)[0]
            break
    
    return wrapped_env, object_name, init_angle


def high_level_policy_infer(obs_dict, high_level_policy, 
                             output_obj_pcd_only: bool = True,
                             add_one_hot_encoding: bool = False):
    """Run high-level policy to predict goal end-effector points.
    
    Args:
        obs_dict: Dictionary with 'point_cloud' and 'gripper_pcd'
        high_level_policy: Loaded high-level model
        output_obj_pcd_only: If True, only use object points (exclude gripper)
        add_one_hot_encoding: Whether to add modality one-hot encoding
        
    Returns:
        Predicted goal points tensor with shape (B, 1, 4, 3)
    """
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
        
        weights = outputs[:, :, -1]  # B, N
        outputs = outputs[:, :, :-1]  # B, N, 12
        
        if output_obj_pcd_only:
            # Only use object point cloud (exclude last 4 gripper points)
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


def get_joint_metrics(env, object_name: str):
    """Get articulated joint metrics from PyBullet directly.
    
    Args:
        env: Wrapped environment
        object_name: Name of the object
        
    Returns:
        Dictionary with joint information
    """
    # Navigate through wrappers to get base environment
    base_env = env.env._env if hasattr(env.env, '_env') else env.env
    
    object_id = base_env.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=base_env.id)
    
    joint_info = {}
    for i in range(num_joints):
        info = p.getJointInfo(object_id, i, physicsClientId=base_env.id)
        joint_type = info[2]
        joint_name = info[1].decode('utf-8')
        
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_state = p.getJointState(object_id, i, physicsClientId=base_env.id)
            joint_limits = info[8:10]
            
            joint_info[i] = {
                'name': joint_name,
                'type': 'revolute' if joint_type == p.JOINT_REVOLUTE else 'prismatic',
                'angle': joint_state[0],
                'limits': list(joint_limits)
            }
    
    return joint_info


def run_evaluation(env, object_name: str, low_level_policy, high_level_policy,
                   init_angle: float, horizon: int = 70,
                   update_goal_freq: int = 1,
                   output_obj_pcd_only: bool = True,
                   add_one_hot_encoding: bool = False,
                   save_gif: bool = True,
                   skip_reset: bool = False,
                   debug_save_obs: bool = False,
                   debug_save_dir: str = None):
    """Run a single evaluation episode.
    
    Args:
        env: Wrapped environment
        object_name: Name of the object
        low_level_policy: Low-level controller
        high_level_policy: High-level goal predictor
        init_angle: Initial joint angle for metrics
        horizon: Number of steps to run
        update_goal_freq: How often to update the goal
        output_obj_pcd_only: Whether to only use object points
        add_one_hot_encoding: Whether to use one-hot encoding
        save_gif: Whether to collect frames for GIF
        skip_reset: If True, skip env.reset() and get observation directly.
                   Use this when state has been pre-loaded via load_env().
        debug_save_obs: If True, save model input point clouds for debugging
        debug_save_dir: Directory to save debug observations
        
    Returns:
        Tuple of (rgb_frames, metrics_dict)
    """
    if skip_reset:
        # State was already loaded via load_env(), just get initial observation
        raw_obs = env.env._get_observation(only_object=True)
        # Need to handle multi-step wrapper's observation stacking
        # Stack the same observation n_obs_steps times
        n_obs_steps = env.n_obs_steps
        obs = {}
        for key, value in raw_obs.items():
            if isinstance(value, np.ndarray):
                obs[key] = np.stack([value] * n_obs_steps, axis=0)
            else:
                obs[key] = value
    else:
        obs = env.reset()
    
    all_rgbs = []
    if save_gif:
        rgb = env.env.render()
        all_rgbs.append(rgb)
    
    # Debug: store observations for saving
    debug_obs_list = []
    if debug_save_obs:
        debug_obs_list.append({
            'step': 0,
            'point_cloud': obs['point_cloud'].copy() if 'point_cloud' in obs else None,
            'gripper_pcd': obs['gripper_pcd'].copy() if 'gripper_pcd' in obs else None,
            'agent_pos': obs['agent_pos'].copy() if 'agent_pos' in obs else None,
        })
    
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
        
        # Add goal to input for low-level policy
        goal_expanded = goal_eef_points.repeat(1, 2, 1, 1)
        np_goal = goal_eef_points.detach().cpu().numpy()
        
        parallel_input_dict['goal_eef_points'] = goal_expanded
        parallel_input_dict['goal_gripper_pcd'] = goal_expanded
        
        # Infer low-level action
        with torch.no_grad():
            action_dict = low_level_policy.predict_action(parallel_input_dict)
        action = action_dict['action'][0].detach().cpu().numpy()
        
        # Execute action
        obs, reward, done, info = env.step(action)
        
        # Update goal visualization
        env.env.goal_gripper_pcd = np_goal.squeeze(0)[0]
        
        if save_gif:
            rgb = env.env.render()
            all_rgbs.append(rgb)
        
        # Debug: save observations
        if debug_save_obs:
            debug_obs_list.append({
                'step': t,
                'point_cloud': obs['point_cloud'].copy() if 'point_cloud' in obs else None,
                'gripper_pcd': obs['gripper_pcd'].copy() if 'gripper_pcd' in obs else None,
                'agent_pos': obs['agent_pos'].copy() if 'agent_pos' in obs else None,
                'goal_eef_points': np_goal.copy(),
                'action': action.copy(),
            })
        
        steps_taken = t + 1
        
        if done:
            break
    
    # Get final joint metrics
    final_joints = get_joint_metrics(env, object_name)
    
    # Find the main articulated joint (first revolute/prismatic)
    final_angle = None
    for joint_id, joint_data in final_joints.items():
        final_angle = joint_data['angle']
        break
    
    # Close environment
    base_env = env.env._env if hasattr(env.env, '_env') else env.env
    base_env.close()
    
    # Compute metrics
    if init_angle is not None and final_angle is not None:
        improved_angle = final_angle - init_angle
    else:
        improved_angle = 0.0
    
    metrics = {
        'initial_angle': float(init_angle) if init_angle is not None else 0.0,
        'final_angle': float(final_angle) if final_angle is not None else 0.0,
        'improved_angle': float(improved_angle),
        'horizon': horizon,
        'steps_taken': steps_taken,
    }
    
    # Save debug observations if enabled
    if debug_save_obs and debug_save_dir is not None:
        import pickle
        debug_obs_path = os.path.join(debug_save_dir, 'debug_observations.pkl')
        with open(debug_obs_path, 'wb') as f:
            pickle.dump(debug_obs_list, f)
        print(f"  Saved debug observations: {debug_obs_path}")
    
    return all_rgbs, metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate policies on custom articulated objects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate pre-trained checkpoints on custom object demos
  python eval_custom_object.py \\
      --asset-dir data/custom_objects/sim_microwave_good \\
      --experiment-name my_demos \\
      --num-trials 10

  # Evaluate finetuned high-level checkpoint
  python eval_custom_object.py \\
      --asset-dir data/custom_objects/sim_microwave_good \\
      --experiment-name my_demos \\
      --high-level-ckpt weighted_displacement_model/exps/finetuned/best.pth

  # Use random initialization (no saved demo states)
  python eval_custom_object.py \\
      --asset-dir data/custom_objects/sim_microwave_good \\
      --use-random-init \\
      --config-id 0
        """
    )
    
    # Required arguments
    parser.add_argument('--asset-dir', type=str, required=True,
                        help='Path to custom object asset directory')
    
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
    
    # Experiment settings
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment folder name containing demo states')
    parser.add_argument('--use-random-init', action='store_true',
                        help='Use random initialization instead of saved states')
    parser.add_argument('--config-id', type=int, default=0,
                        help='Config file ID for random initialization')
    
    # Evaluation settings
    parser.add_argument('--num-trials', type=int, default=10,
                        help='Number of evaluation trials')
    parser.add_argument('--horizon', type=int, default=70,
                        help='Steps per episode')
    parser.add_argument('--update-goal-freq', type=int, default=1,
                        help='Frequency to update high-level goal')
    
    # Output settings
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results (default: data/eval_custom/<asset_name>)')
    parser.add_argument('--no-gif', action='store_true',
                        help='Disable GIF saving')
    
    # Model settings
    parser.add_argument('--output-obj-pcd-only', type=int, default=1,
                        help='Only use object points for goal prediction')
    parser.add_argument('--add-one-hot-encoding', type=int, default=0,
                        help='Add one-hot modality encoding')
    parser.add_argument('--observation-mode', type=str,
                        default='act3d_goal_displacement_gripper_to_object',
                        help='Observation mode for wrapper')
    parser.add_argument('--num-points', type=int, default=4500,
                        help='Number of points in point cloud')
    
    # Demo filtering
    parser.add_argument('--all-demos', action='store_true',
                        help='Use all demos, not just valid training demos')
    parser.add_argument('--list-valid-demos', action='store_true',
                        help='List valid training demos and exit')
    
    # Debug options
    parser.add_argument('--debug-save-obs', action='store_true',
                        help='Save model input point clouds for debugging')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle --list-valid-demos flag
    if args.list_valid_demos:
        if args.experiment_name is None:
            print("Error: Must specify --experiment-name to list valid demos")
            sys.exit(1)
        
        experiment_dir = os.path.join(args.asset_dir, 'experiment', args.experiment_name)
        if not os.path.exists(experiment_dir):
            print(f"Error: Experiment directory not found: {experiment_dir}")
            sys.exit(1)
        
        all_dirs = sorted([
            d for d in os.listdir(experiment_dir)
            if os.path.isdir(os.path.join(experiment_dir, d))
        ])
        valid_dirs = [
            d for d in all_dirs
            if is_valid_training_demo(os.path.join(experiment_dir, d))
        ]
        
        print(f"\nExperiment: {args.experiment_name}")
        print(f"Total directories: {len(all_dirs)}")
        print(f"Valid training demos: {len(valid_dirs)}")
        print("\nValid demos:")
        for i, d in enumerate(valid_dirs):
            print(f"  {i}: {d}")
        return
    
    print("=" * 60)
    print("Custom Object Evaluation")
    print("=" * 60)
    print(f"Asset directory: {args.asset_dir}")
    print(f"High-level checkpoint: {args.high_level_ckpt}")
    print(f"Low-level checkpoint: {args.low_level_exp_dir}/{args.low_level_ckpt_name}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Mode: {'Random init' if args.use_random_init else f'Experiment states ({args.experiment_name})'}")
    print(f"Demo filter: {'All demos' if args.all_demos else 'Valid training demos only'}")
    print("=" * 60)
    
    # Validate paths
    if not os.path.exists(args.asset_dir):
        print(f"Error: Asset directory not found: {args.asset_dir}")
        sys.exit(1)
    
    if not args.use_random_init and args.experiment_name is None:
        print("Error: Must specify --experiment-name or use --use-random-init")
        sys.exit(1)
    
    # Setup save directory: ./outputs/eval_customs/<object_name>/<model>/
    if args.save_dir is None:
        # Extract object name from asset directory
        object_name = os.path.basename(args.asset_dir.rstrip('/'))
        
        # Extract model name from high-level checkpoint path
        # e.g., "data/high_level_200_obj_ckpt.pth" -> "pretrained"
        # e.g., "weighted_displacement_model/exps/2025-12-03_fewshot.../model_final.pth" -> "2025-12-03_fewshot..."
        ckpt_path = args.high_level_ckpt
        if 'high_level_200_obj_ckpt' in ckpt_path:
            model_name = 'pretrained'
        else:
            # Try to extract experiment name from path
            # Look for pattern like "exps/<exp_name>/model_*.pth"
            path_parts = ckpt_path.replace('\\', '/').split('/')
            if 'exps' in path_parts:
                exp_idx = path_parts.index('exps')
                if exp_idx + 1 < len(path_parts):
                    model_name = path_parts[exp_idx + 1]
                else:
                    model_name = 'finetuned'
            else:
                # Use checkpoint filename without extension
                model_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        
        args.save_dir = os.path.join('outputs', 'eval_customs', object_name, model_name)
    
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Results will be saved to: {args.save_dir}")
    
    # Load policies
    cfg, low_level_policy, high_level_policy = load_policies(
        args.low_level_exp_dir,
        args.low_level_ckpt_name,
        args.high_level_ckpt,
        bool(args.add_one_hot_encoding)
    )
    
    # Run evaluation trials
    all_metrics = []
    
    for trial_idx in range(args.num_trials):
        print(f"\n{'=' * 60}")
        print(f"Trial {trial_idx + 1}/{args.num_trials}")
        print("=" * 60)
        
        try:
            # Construct environment
            if args.use_random_init:
                env, object_name, init_angle = construct_env_random(
                    cfg, args.asset_dir, 
                    config_id=args.config_id,
                    num_points=args.num_points,
                    observation_mode=args.observation_mode
                )
            else:
                env, object_name, init_angle = construct_env_from_experiment(
                    cfg, args.asset_dir,
                    experiment_name=args.experiment_name,
                    trial_idx=trial_idx,
                    num_points=args.num_points,
                    observation_mode=args.observation_mode,
                    only_valid_demos=not args.all_demos
                )
            
            print(f"Object: {object_name}")
            print(f"Initial angle: {init_angle:.4f}" if init_angle else "Initial angle: N/A")
            
            # Run evaluation
            # For experiment states, we already loaded the state via load_env(),
            # so skip reset to avoid needing the temp config file
            
            # Setup debug save directory for this trial
            trial_debug_dir = None
            if args.debug_save_obs:
                trial_debug_dir = os.path.join(args.save_dir, f'debug_trial_{trial_idx:03d}')
                os.makedirs(trial_debug_dir, exist_ok=True)
            
            all_rgbs, metrics = run_evaluation(
                env, object_name, low_level_policy, high_level_policy,
                init_angle,
                horizon=args.horizon,
                update_goal_freq=args.update_goal_freq,
                output_obj_pcd_only=bool(args.output_obj_pcd_only),
                add_one_hot_encoding=bool(args.add_one_hot_encoding),
                save_gif=not args.no_gif,
                skip_reset=not args.use_random_init,  # Skip reset when using saved states
                debug_save_obs=args.debug_save_obs,
                debug_save_dir=trial_debug_dir
            )
            
            metrics['trial'] = trial_idx
            metrics['success'] = True
            all_metrics.append(metrics)
            
            print(f"✓ Trial complete:")
            print(f"  Initial angle: {metrics['initial_angle']:.4f}")
            print(f"  Final angle: {metrics['final_angle']:.4f}")
            print(f"  Improvement: {metrics['improved_angle']:+.4f}")
            
            # Save GIF
            if not args.no_gif and len(all_rgbs) > 0:
                gif_path = os.path.join(
                    args.save_dir,
                    f"trial_{trial_idx:03d}_improve_{metrics['improved_angle']:.4f}.gif"
                )
                save_numpy_as_gif(np.array(all_rgbs), gif_path)
                print(f"  Saved GIF: {gif_path}")
            
        except Exception as e:
            print(f"✗ Trial failed: {e}")
            import traceback
            traceback.print_exc()
            all_metrics.append({
                'trial': trial_idx,
                'success': False,
                'error': str(e)
            })
    
    # Save aggregated metrics
    metrics_path = os.path.join(args.save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'config': {
                'asset_dir': args.asset_dir,
                'experiment_name': args.experiment_name,
                'use_random_init': args.use_random_init,
                'high_level_ckpt': args.high_level_ckpt,
                'low_level_exp_dir': args.low_level_exp_dir,
                'horizon': args.horizon,
                'num_trials': args.num_trials,
            },
            'trials': all_metrics
        }, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print("=" * 60)
    
    # Compute summary statistics
    successful_trials = [m for m in all_metrics if m.get('success', False)]
    if successful_trials:
        improvements = [m['improved_angle'] for m in successful_trials]
        print(f"Successful trials: {len(successful_trials)}/{len(all_metrics)}")
        print(f"Mean improvement: {np.mean(improvements):+.4f}")
        print(f"Std improvement: {np.std(improvements):.4f}")
        print(f"Max improvement: {np.max(improvements):+.4f}")
        print(f"Min improvement: {np.min(improvements):+.4f}")
    else:
        print("No successful trials")
    
    print(f"\nResults saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
