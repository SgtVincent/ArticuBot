#!/usr/bin/env python3
"""
Test script for custom URDF model WITHOUT accessing privileged information.
This script demonstrates how to:
1. Load a custom URDF model using build_up_env_random
2. Run the pretrained policy for manipulation
3. Evaluate results without calling _get_info() which requires mobility_v2.json

Based on test_build_up_env_random.py for successful environment initialization.
"""

import os
import sys
import pathlib
import numpy as np
import argparse
import torch
import hydra
from copy import deepcopy
from omegaconf import OmegaConf
import pybullet as p

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
os.environ['PROJECT_DIR'] = str(PROJECT_ROOT)

from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from train_ddp import TrainDP3Workspace
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env_random, save_numpy_as_gif


class RobogenPointCloudWrapperNoInfo(RobogenPointCloudWrapper):
    """
    Custom wrapper that does NOT call _get_info() during reset or step.
    This allows testing custom objects without mobility_v2.json files.
    """
    
    def reset(self, **kwargs):
        """Reset without calling _get_info()."""
        if "act3d_goal" in self.observation_mode:
            self.grasped_handle = False
        self._env.reset(**kwargs)
        # SKIP: self._env._get_info()  # This requires mobility_v2.json
        self.time_step = 0
        if "goal" in self.observation_mode:
            self.grasped_handle = False
        return self._get_observation(only_object=self.only_object)
    
    def step(self, action, **kwargs):
        """Step without calling _get_info()."""
        import pybullet as p
        from scipy.spatial.transform import Rotation as R
        from manipulation.utils import rotation_transfer_6D_to_matrix, rotation_transfer_matrix_to_6D
        
        # Extract render parameter from kwargs (default True to match parent class)
        render = kwargs.pop('render', True)
        
        pos, orient = self._env.robot.get_pos_orient(self._env.robot.right_end_effector)
        
        ### new_pos = current_pos + action[:3]
        pos = pos + np.array(action[:3])
        
        ### new_orient = current_orient @ delta_orient, which is represented using 6D representation as action[3:9]
        current_rotate_matrix = np.array(p.getMatrixFromQuaternion(orient)).reshape(3, 3)            
        delta_orient = action[3:9]
        delta_rotate_matrix = rotation_transfer_6D_to_matrix(delta_orient)
        after_rotate_matrix = current_rotate_matrix @ delta_rotate_matrix
        orient = R.from_matrix(after_rotate_matrix).as_quat()
        euler = p.getEulerFromQuaternion(orient)

        ### new finger joint = current_finger_joint + action[9]
        cur_joint_angle = p.getJointState(self._env.robot.body, self._env.robot.right_gripper_indices[0], physicsClientId=self._env.id)
        target_joint_angle = action[9] + cur_joint_angle[0]
        
        action = pos.tolist() + list(euler) + [target_joint_angle]
        self._env.take_direct_action(action)
        
        reward, success = self._env.compute_reward()

        # SKIP: info = self._env._get_info()  # This requires mobility_v2.json
        # Return a minimal info dict instead
        info = {'reward': reward, 'success': success}
        
        done = self._env.time_step >= self.horizon
        obs = self._get_observation(render=render, only_object=self.only_object)
        self.time_step += 1
        return obs, reward, done, info


def load_policies(low_level_exp_dir, low_level_ckpt_name, high_level_ckpt_path, add_one_hot_encoding=False):
    """Load pretrained high-level and low-level policies."""
    
    print("=" * 60)
    print("Loading Low-Level Policy...")
    print("=" * 60)
    
    # Load low-level policy
    with hydra.initialize(config_path='3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/config'):
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",
            overrides=OmegaConf.load(f"{low_level_exp_dir}/.hydra/overrides.yaml"),
        )
    
    cfg = recomposed_config
    workspace = TrainDP3Workspace(cfg)
    checkpoint_dir = f"{low_level_exp_dir}/checkpoints/{low_level_ckpt_name}"
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    workspace.load_checkpoint(path=checkpoint_dir)
    
    low_level_policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        low_level_policy = deepcopy(workspace.ema_model)
    
    low_level_policy.eval()
    low_level_policy.reset()
    low_level_policy = low_level_policy.to('cuda')
    
    print("✓ Low-level policy loaded successfully")
    
    # Load high-level policy
    print("\n" + "=" * 60)
    print("Loading High-Level Policy...")
    print("=" * 60)
    
    num_class = 13
    input_channel = 5 if add_one_hot_encoding else 3
    
    from weighted_displacement_model.model_invariant import PointNet2_super
    high_level_policy = PointNet2_super(num_classes=num_class, input_channel=input_channel).to("cuda")
    
    print(f"Loading checkpoint from: {high_level_ckpt_path}")
    high_level_policy.load_state_dict(torch.load(high_level_ckpt_path))
    high_level_policy.eval()
    
    print("✓ High-level policy loaded successfully")
    
    return cfg, low_level_policy, high_level_policy


def construct_custom_env(cfg, config_file, num_points=1280, randomize=True):
    """Construct environment for the custom URDF model with random initialization.
    
    This version does NOT call _get_info() to avoid accessing privileged information
    like mobility_v2.json which may not exist for custom objects.
    
    Args:
        cfg: Hydrated experiment config.
        config_file (str): Path to the task-specific config file.
        num_points (int): Number of points in point cloud.
        randomize (bool): If True, randomly initialize object pose, robot joints, etc.
    
    Returns:
        A MultiStepWrapper-wrapped environment ready for evaluation.
    """
    
    print("\n" + "=" * 60)
    print("Constructing Environment...")
    print("=" * 60)
    print(f"Config file: {config_file}")
    print(f"Randomize initialization: {randomize}")
    
    # Build the base environment with random initialization
    if randomize:
        env, task_config = build_up_env_random(
            task_config=config_file,
            env_name='articulated',
            render=False,
            randomize_object_pose=True,
            randomize_robot_joints=True,
            randomize_initial_joint_angle=True,
            horizon=600,
            max_attempts=100,
            far_distance=0.7,
            near_distance=0.3
        )
        print(f"✓ Base environment created with random initialization")
    else:
        raise NotImplementedError("Non-random initialization requires pre-stored states. Use randomize=True.")
    
    # Get object name from config
    object_name = task_config['object_name'].lower()
    
    # Wrap with custom point cloud wrapper that doesn't call _get_info()
    # Use simpler observation mode that doesn't require goal states or _get_info()
    pointcloud_env = RobogenPointCloudWrapperNoInfo(
        env, 
        object_name,
        num_points=num_points,
        observation_mode="act3d",  # Simple mode without goals
        real_world_camera=False,
        noise_real_world_pcd=False,
    )
    
    print(f"✓ Point cloud wrapper added (no _get_info() calls)")
    
    # Wrap with multi-step wrapper
    env = MultiStepWrapper(
        pointcloud_env, 
        n_obs_steps=cfg.n_obs_steps, 
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=600, 
        reward_agg_method='sum'
    )
    
    print(f"✓ Multi-step wrapper added")
    print(f"✓ Environment construction complete")
    
    return env, object_name


def high_level_policy_infer(obs_dict, high_level_policy, output_obj_pcd_only=True, add_one_hot_encoding=False):
    """Run high-level policy to predict goal point cloud."""
    
    with torch.no_grad():
        pointcloud = obs_dict['point_cloud'][:, -1, :, :]
        gripper_pcd = obs_dict['gripper_pcd'][:, -1, :]
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)
        
        if add_one_hot_encoding:
            # Add one-hot encoding if needed
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
            # Only use object point cloud (exclude gripper points)
            # Filter both outputs and the corresponding input points
            num_obj_points = pointcloud.shape[1]
            outputs = outputs[:, :num_obj_points, :]
            weights = weights[:, :num_obj_points]
            inputs_for_residual = inputs[:, :num_obj_points, :3]
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


def get_joint_metrics(env, object_name):
    """Get joint metrics WITHOUT calling _get_info().
    
    This directly queries PyBullet for joint states without relying on
    privileged information from mobility_v2.json.
    """
    object_id = env.env._env.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=env.env._env.id)
    
    # Find all revolute and prismatic joints
    joint_states = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(object_id, i, physicsClientId=env.env._env.id)
        joint_type = joint_info[2]
        joint_name = joint_info[1].decode('utf-8')
        
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            joint_state = p.getJointState(object_id, i, physicsClientId=env.env._env.id)
            joint_angle = joint_state[0]
            joint_limits = joint_info[8:10]
            
            joint_states.append({
                'joint_id': i,
                'joint_name': joint_name,
                'joint_type': 'revolute' if joint_type == p.JOINT_REVOLUTE else 'prismatic',
                'angle': joint_angle,
                'limits': joint_limits
            })
    
    return joint_states


def run_test_rollout(env, object_name, low_level_policy, high_level_policy, horizon=35, 
                     update_goal_freq=1, output_obj_pcd_only=True, add_one_hot_encoding=False,
                     save_gif=True):
    """Run a test rollout WITHOUT calling _get_info().
    
    This version tracks joint states directly from PyBullet without
    accessing privileged information.
    
    Args:
        save_gif (bool): If True, collect RGB frames for GIF generation.
    """
    
    print("\n" + "=" * 60)
    print("Running Test Rollout...")
    print("=" * 60)
    print(f"Horizon: {horizon} steps")
    print(f"Update goal frequency: {update_goal_freq}")
    
    # Reset environment - this should work without _get_info()
    obs = env.reset()
    
    # Get initial joint states
    initial_joint_states = get_joint_metrics(env, object_name)
    print(f"\nInitial joint states:")
    for js in initial_joint_states:
        print(f"  Joint {js['joint_id']} ({js['joint_name']}): {js['angle']:.4f} ({js['joint_type']})")
    
    # Render initial state (only if saving GIF)
    all_rgbs = []
    if save_gif:
        rgb = env.env.render()
        all_rgbs.append(rgb)
    last_goal = None
    
    for t in range(1, horizon):
        if t % 5 == 0:
            print(f"  Step {t}/{horizon}")
        
        # Prepare input
        parallel_input_dict = obs
        parallel_input_dict = dict_apply(parallel_input_dict, lambda x: torch.from_numpy(x).to('cuda'))
        for key in obs:
            parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)
        
        # Infer high-level goal
        if t % update_goal_freq == 0:
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
        
        # Add required goal observations for the low-level policy
        # goal_eef_points has shape (B, 1, 4, 3)
        # We need to expand it to match the time dimension of observations (B, T, 4, 3)
        T = parallel_input_dict['gripper_pcd'].shape[1]
        goal_expanded = goal_eef_points.expand(-1, T, -1, -1)  # Expand to (B, T, 4, 3)
        
        parallel_input_dict['goal_eef_points'] = goal_expanded
        parallel_input_dict['goal_gripper_pcd'] = goal_expanded  # Use same goal for gripper
        
        # Infer low-level action
        with torch.no_grad():
            action_dict = low_level_policy.predict_action(parallel_input_dict)
        action = action_dict['action'][0].detach().to('cpu').numpy()
        
        # Execute action (render=True is already passed by MultiStepWrapper)
        obs, reward, done, info = env.step(action)
        if save_gif:
            rgb = env.env.render()
            all_rgbs.append(rgb)
        
        if done:
            print(f"  Episode ended at step {t}")
            break
    
    # Get final joint states
    final_joint_states = get_joint_metrics(env, object_name)
    print(f"\nFinal joint states:")
    for js in final_joint_states:
        print(f"  Joint {js['joint_id']} ({js['joint_name']}): {js['angle']:.4f} ({js['joint_type']})")
    
    # Calculate improvements
    print(f"\nJoint improvements:")
    for init_js, final_js in zip(initial_joint_states, final_joint_states):
        improvement = final_js['angle'] - init_js['angle']
        print(f"  Joint {init_js['joint_id']} ({init_js['joint_name']}): {improvement:+.4f}")
    
    env.env._env.close()
    
    print(f"\n✓ Rollout complete!")
    
    # Return summary info
    summary_info = {
        'initial_joints': initial_joint_states,
        'final_joints': final_joint_states,
        'improvements': [
            {
                'joint_id': init_js['joint_id'],
                'joint_name': init_js['joint_name'],
                'improvement': final_js['angle'] - init_js['angle']
            }
            for init_js, final_js in zip(initial_joint_states, final_joint_states)
        ]
    }
    
    return all_rgbs, summary_info


def main():
    parser = argparse.ArgumentParser(description="Test pretrained policy on custom URDF model (no privileged info)")
    parser.add_argument('--low_level_exp_dir', type=str, 
                        default='data/low-level-ckpt',
                        help='Path to low-level policy experiment directory')
    parser.add_argument('--low_level_ckpt_name', type=str, 
                        default='low-level.ckpt',
                        help='Low-level checkpoint name')
    parser.add_argument('--high_level_ckpt_name', type=str, 
                        default='data/high_level_200_obj_ckpt.pth',
                        help='Path to high-level checkpoint')
    parser.add_argument('--config_id', type=int, default=500,
                        help='Config ID to use (500-512)')
    parser.add_argument('--horizon', type=int, default=35,
                        help='Number of steps to run')
    parser.add_argument('--update_goal_freq', type=int, default=1,
                        help='Frequency to update goal')
    parser.add_argument('--output_obj_pcd_only', type=int, default=1,
                        help='Output object point cloud only')
    parser.add_argument('--add_one_hot_encoding', type=int, default=0,
                        help='Add one-hot encoding')
    parser.add_argument('--save_dir', type=str, default='data/test_custom_model',
                        help='Directory to save results')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of random trials to run')
    parser.add_argument('--save_gif', type=int, default=1,
                        help='Save evaluation episodes as GIF (1=yes, 0=no)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing Custom URDF Model WITHOUT Privileged Information")
    print("=" * 60)
    print(f"Model: color_0025_0_v2")
    print(f"Config ID: {args.config_id}")
    print(f"Number of trials: {args.num_trials}")
    
    # Paths
    object_path = os.path.join(PROJECT_ROOT, 'data/custom_objects/color_0025_0_v2')
    config_file = os.path.join(object_path, f'configs/config_larger_randomization_{args.config_id}.yaml')
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    # Load policies
    cfg, low_level_policy, high_level_policy = load_policies(
        args.low_level_exp_dir,
        args.low_level_ckpt_name,
        args.high_level_ckpt_name,
        args.add_one_hot_encoding
    )
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run multiple trials
    all_results = []
    for trial in range(args.num_trials):
        print(f"\n{'=' * 60}")
        print(f"Trial {trial + 1}/{args.num_trials}")
        print(f"{'=' * 60}")
        
        # Construct environment (new random initialization each trial)
        env, object_name = construct_custom_env(cfg, config_file, randomize=True)
        
        # Run test rollout
        all_rgbs, summary_info = run_test_rollout(
            env, 
            object_name,
            low_level_policy, 
            high_level_policy,
            horizon=args.horizon,
            update_goal_freq=args.update_goal_freq,
            output_obj_pcd_only=args.output_obj_pcd_only,
            add_one_hot_encoding=args.add_one_hot_encoding,
            save_gif=bool(args.save_gif)
        )
        
        # Save results for this trial
        trial_suffix = f"_trial{trial}" if args.num_trials > 1 else ""
        
        # Save GIF if enabled
        if args.save_gif and len(all_rgbs) > 0:
            gif_path = os.path.join(args.save_dir, f'color_0025_0_v2_config{args.config_id}{trial_suffix}.gif')
            save_numpy_as_gif(np.array(all_rgbs), gif_path)
            print(f"\n✓ Saved rollout GIF to: {gif_path}")
        elif not args.save_gif:
            print(f"\n⊘ GIF saving disabled (--save_gif=0)")
        
        # Save info
        import json
        info_path = os.path.join(args.save_dir, f'color_0025_0_v2_config{args.config_id}{trial_suffix}_info.json')
        info_dict = {
            "config_id": args.config_id,
            "trial": trial,
            "initial_joints": summary_info['initial_joints'],
            "final_joints": summary_info['final_joints'],
            "improvements": summary_info['improvements']
        }
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        print(f"✓ Saved info to: {info_path}")
        
        all_results.append(summary_info)
    
    # Print summary statistics if multiple trials
    if args.num_trials > 1:
        print(f"\n{'=' * 60}")
        print("Summary Across All Trials")
        print(f"{'=' * 60}")
        
        # Aggregate improvements by joint
        joint_improvements = {}
        for result in all_results:
            for imp in result['improvements']:
                joint_key = f"{imp['joint_name']}"
                if joint_key not in joint_improvements:
                    joint_improvements[joint_key] = []
                joint_improvements[joint_key].append(imp['improvement'])
        
        for joint_name, improvements in joint_improvements.items():
            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            print(f"  {joint_name}: {mean_imp:+.4f} ± {std_imp:.4f}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
