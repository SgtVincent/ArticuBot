#!/usr/bin/env python3
"""
Test script for custom URDF model WITHOUT accessing privileged information.
This script demonstrates how to:
1. Load a custom URDF model using build_up_env_random or curated eval states
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
import yaml
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
from manipulation.utils import build_up_env_random, build_up_env_eval, save_numpy_as_gif


class RobogenPointCloudWrapperNoInfo(RobogenPointCloudWrapper):
    """
    Custom wrapper that does NOT call _get_info() during reset or step.
    This allows testing custom objects without mobility_v2.json files.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize and store whether we should preserve randomization.
        
        Temporarily sets observation_mode to avoid goal file loading,
        then restores it after initialization.
        """
        # Extract observation_mode before calling parent
        observation_mode = kwargs.get('observation_mode', 'act3d_displacement_gripper_to_object')
        
        # Temporarily replace observation_mode to prevent parent from loading goal files
        original_mode = observation_mode
        if 'goal' in observation_mode:
            # Use a mode without 'goal' to skip file loading in parent __init__
            kwargs['observation_mode'] = 'act3d_displacement_gripper_to_object'
        
        super().__init__(*args, **kwargs)
        
        # Restore the original observation mode
        self.observation_mode = original_mode
        
        self._preserve_initial_state = False
        self._initial_state_saved = None
        
        # Initialize goal placeholders when goal observation mode is requested
        if "goal" in self.observation_mode:
            # Initialize zero-valued goal arrays to avoid calling _get_info()
            self.goal_gripper_pcd = np.zeros((4, 3), dtype=np.float32)
            self.grasping_goal = np.zeros((4, 3), dtype=np.float32)
            self.final_goal = np.zeros((4, 3), dtype=np.float32)
            self.grasped_handle = False
    
    def set_preserve_initial_state(self, preserve=True):
        """Set whether to preserve the current state on reset (for randomization)."""
        self._preserve_initial_state = preserve
        if preserve:
            # Save the current state
            from manipulation.utils import save_env
            self._initial_state_saved = save_env(self._env, save_path=None, simplified=False)
    
    def reset(self, **kwargs):
        """Reset without calling _get_info()."""
        if "act3d_goal" in self.observation_mode:
            self.grasped_handle = False
        
        # If we should preserve initial state (randomization), restore it instead of resetting
        if self._preserve_initial_state and self._initial_state_saved is not None:
            from manipulation.utils import load_env
            load_env(self._env, state=self._initial_state_saved, simplified=False)
        else:
            self._env.reset(**kwargs)
        
        # SKIP: self._env._get_info()  # This requires mobility_v2.json
        self.time_step = 0
        if "goal" in self.observation_mode:
            self.grasped_handle = False
        return self._get_observation(only_object=self.only_object)
    
    def _get_observation(self, render=True, only_object=True):
        """Get observation - parent class handles displacement features."""
        # Parent class automatically adds displacement features when
        # observation_mode contains 'displacement_gripper_to_object'
        return super()._get_observation(render=render, only_object=only_object)
    
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


def construct_custom_env(cfg, config_file, num_points=4500, randomize=True,
                         draw_coordinate_frame=False, use_eval_states=False,
                         eval_experiment_name=None, eval_trial_index=0,
                         observation_mode="act3d_displacement_gripper_to_object"):
    """Construct environment for the custom URDF model with random initialization.
    
    This version does NOT call _get_info() to avoid accessing privileged information
    like mobility_v2.json which may not exist for custom objects.
    
    Args:
        cfg: Hydrated experiment config.
        config_file (str): Path to the task-specific config file.
        num_points (int): Number of points in point cloud.
        randomize (bool): If True, randomly initialize object pose, robot joints, etc.
        draw_coordinate_frame (bool): If True, draw world coordinate frame in rendering.
        use_eval_states (bool): If True, load curated evaluation states instead of random resets.
        eval_experiment_name (str): Optional experiment folder to use when loading eval states.
        eval_trial_index (int): Trial index within the experiment to select for eval states.
        observation_mode (str): Observation mode to use for the point cloud wrapper.
    
    Returns:
        A MultiStepWrapper-wrapped environment ready for evaluation.
    """
    
    print("\n" + "=" * 60)
    print("Constructing Environment...")
    print("=" * 60)
    print(f"Config file: {config_file}")
    print(f"Randomize initialization: {randomize}")
    print(f"Use evaluation states: {use_eval_states}")
    print(f"Observation mode: {observation_mode}")
    
    if use_eval_states:
        print("✓ Using curated evaluation states")

        with open(config_file, 'r') as f:
            config_entries = yaml.safe_load(f)

        solution_path_candidates = [entry['solution_path'] for entry in config_entries if 'solution_path' in entry]
        if len(solution_path_candidates) == 0:
            raise ValueError("Config file does not contain a solution_path entry required for evaluation states.")
        solution_path = solution_path_candidates[0]

        object_name_candidates = [entry['name'] for entry in config_entries if 'name' in entry]
        if len(object_name_candidates) == 0:
            raise ValueError("Config file does not specify object name.")
        object_name = object_name_candidates[0].lower()

        project_dir = os.environ['PROJECT_DIR']
        substeps_path = os.path.join(project_dir, solution_path, 'substeps.txt')
        if not os.path.exists(substeps_path):
            raise FileNotFoundError(f"substeps.txt not found at {substeps_path}")

        with open(substeps_path, 'r') as f:
            substeps = [line.strip() for line in f if line.strip()]
        if len(substeps) == 0:
            raise ValueError("substeps.txt is empty, cannot determine task name.")
        first_step = substeps[0]
        task_name = first_step.replace(' ', '_')

        experiment_root = os.path.join(project_dir, solution_path, 'experiment')
        if not os.path.isdir(experiment_root):
            raise FileNotFoundError(f"Experiment directory not found at {experiment_root}")

        experiment_names = sorted([
            name for name in os.listdir(experiment_root)
            if os.path.isdir(os.path.join(experiment_root, name))
        ])
        if len(experiment_names) == 0:
            raise ValueError(f"No experiments found in {experiment_root}")

        if eval_experiment_name is not None:
            if eval_experiment_name not in experiment_names:
                raise ValueError(
                    f"Requested experiment '{eval_experiment_name}' not found. Available: {experiment_names}"
                )
            selected_experiment = eval_experiment_name
        else:
            selected_experiment = experiment_names[0]

        experiment_dir = os.path.join(experiment_root, selected_experiment)
        print(f"✓ Selected experiment: {selected_experiment}")
        trial_dirs = sorted([
            name for name in os.listdir(experiment_dir)
            if os.path.isdir(os.path.join(experiment_dir, name))
        ])
        if len(trial_dirs) == 0:
            raise ValueError(f"No trial directories found in {experiment_dir}")

        if eval_trial_index < 0 or eval_trial_index >= len(trial_dirs):
            raise IndexError(
                f"Trial index {eval_trial_index} is out of range for {len(trial_dirs)} available trials"
            )

        selected_trial = trial_dirs[eval_trial_index]
        trial_dir = os.path.join(experiment_dir, selected_trial)
        print(f"✓ Selected trial [{eval_trial_index}]: {selected_trial}")

        task_config_path = os.path.join(trial_dir, 'task_config.yaml')
        if not os.path.exists(task_config_path):
            raise FileNotFoundError(f"task_config.yaml not found in {trial_dir}")

        primitive_folder = os.path.join(trial_dir, f"{task_name}_primitive")
        state_dir = os.path.join(primitive_folder, 'states')
        init_state_file = os.path.join(state_dir, 'state_0.pkl')
        if not os.path.exists(init_state_file):
            raise FileNotFoundError(f"Initial state file not found at {init_state_file}")

        state_rel = os.path.relpath(init_state_file, project_dir)
        print(f"✓ Restoring from state: {state_rel}")

        env, _ = build_up_env_eval(
            task_config=task_config_path,
            solution_path=solution_path,
            task_name=task_name,
            restore_state_file=init_state_file,
            render=False,
            horizon=600,
        )
        print(f"✓ Loaded evaluation environment from {selected_trial}")
        wrapper_cls = RobogenPointCloudWrapper
    else:
        if randomize:
            env, task_config = build_up_env_random(
                task_config=config_file,
                env_name='articulated',
                render=False,
                # randomize_object_pose=True,
                randomize_object_pose=False,
                # randomize_robot_joints=True,
                randomize_robot_joints=False,
                randomize_initial_joint_angle=True,
                horizon=600,
                max_attempts=100,
                far_distance=0.7,
                near_distance=0.3
            )
            print(f"✓ Base environment created with random initialization")
        else:
            raise NotImplementedError("Non-random initialization requires pre-stored states. Use randomize=True.")

        object_name = task_config['object_name'].lower()
        wrapper_cls = RobogenPointCloudWrapperNoInfo
    
    # Draw world coordinate frame if requested
    if draw_coordinate_frame:
        import pybullet as p
        # X-axis: Red, Y-axis: Green, Z-axis: Blue
        p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], lineWidth=3, lifeTime=0, physicsClientId=env.id)
        p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], lineWidth=3, lifeTime=0, physicsClientId=env.id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], lineWidth=3, lifeTime=0, physicsClientId=env.id)
        print(f"✓ World coordinate frame drawn (X:Red, Y:Green, Z:Blue)")
    
    # Wrap with custom point cloud wrapper that doesn't call _get_info()
    pointcloud_env = wrapper_cls(
        env, 
        object_name,
        num_points=num_points,
        observation_mode=observation_mode,
        real_world_camera=False,
        noise_real_world_pcd=False,
    )

    randomized_state_preserved = False
    if hasattr(pointcloud_env, 'set_preserve_initial_state'):
        pointcloud_env.set_preserve_initial_state(preserve=True)
        randomized_state_preserved = True
    
    if isinstance(pointcloud_env, RobogenPointCloudWrapperNoInfo):
        print(f"✓ Point cloud wrapper added (no _get_info() calls)")
    else:
        print(f"✓ Point cloud wrapper added (full info available)")
    print(f"✓ Observation mode set to '{observation_mode}'")

    if randomized_state_preserved:
        print(f"✓ Randomized state saved and will be preserved on reset")
    if use_eval_states:
        print(f"✓ Environment will reset to curated evaluation state")
    
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
            # Only use object point cloud (exclude last 4 gripper points)
            # This matches eval_robogen.py filtering logic
            outputs = outputs[:, :-4, :]  # Remove last 4 gripper points
            weights = weights[:, :-4]  # Remove last 4 gripper points
            inputs_for_residual = inputs[:, :-4, :3]  # Remove last 4 gripper points
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
        
        # Add required goal observations for the low-level policy
        # goal_eef_points has shape (B, 1, 4, 3)
        # Low-level policy expects (B, 2, 4, 3) - matching eval_robogen.py
        goal_expanded = goal_eef_points.repeat(1, 2, 1, 1)  # Repeat to (B, 2, 4, 3)
        np_goal = goal_eef_points.detach().to('cpu').numpy()
        
        parallel_input_dict['goal_eef_points'] = goal_expanded
        parallel_input_dict['goal_gripper_pcd'] = goal_expanded  # Use same goal for gripper
        
        # Infer low-level action
        with torch.no_grad():
            action_dict = low_level_policy.predict_action(parallel_input_dict)
        action = action_dict['action'][0].detach().to('cpu').numpy()
        
        # Execute action (render=True is already passed by MultiStepWrapper)
        obs, reward, done, info = env.step(action)
        
        # Update goal visualization AFTER stepping (matching eval_robogen.py order)
        env.env.goal_gripper_pcd = np_goal.squeeze(0)[0]
        
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

def parse_args():
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
    parser.add_argument('--model', type=str, default='color_0025_0_v2',
                        help='Model to test: door_40147 or other custom model (e.g., color_0025_0_v2)')
    parser.add_argument('--config_id', type=int, default=500,
                        help='Config ID to use (for color_0025_0_v2: 500-512; for door_40147: 0-299)')
    parser.add_argument('--horizon', type=int, default=70,
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
    parser.add_argument('--draw_coordinate_frame', type=int, default=0,
                        help='Draw world coordinate frame in GIF (1=yes, 0=no)')
    parser.add_argument('--use_eval_states', type=int, default=0,
                        help='Use curated evaluation states instead of random initialization (1=yes, 0=no)')
    parser.add_argument('--eval_experiment_name', type=str, default=None,
                        help='Optional experiment folder to load when using curated evaluation states')
    parser.add_argument('--eval_trial_index', type=int, default=0,
                        help='Zero-based trial index to load when using curated evaluation states')
    parser.add_argument('--observation_mode', type=str,
                        default='act3d_displacement_gripper_to_object',
                        help='Observation mode passed to RobogenPointCloudWrapper')
    
    args = parser.parse_args()
    return args

def main(args):

    
    print("=" * 60)
    print("Testing Custom URDF Model WITHOUT Privileged Information")
    print("=" * 60)
    
    # Leave one model from Part Mobility Dataset to verify custom loading
    if args.model == 'door_40147':
        model_name = 'door_40147'
        object_path = os.path.join(PROJECT_ROOT, 'data/diverse_objects/open_the_door_40147')
        config_file = os.path.join(object_path, f'configs/config_{args.config_id}.yaml')
    else:
        model_name = args.model
        object_path = os.path.join(PROJECT_ROOT, 'data/custom_objects/color_0025_0_v2')
        config_file = os.path.join(object_path, f'configs/config_larger_randomization_{args.config_id}.yaml')
    
    print(f"Model: {model_name}")
    print(f"Config ID: {args.config_id}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Observation mode: {args.observation_mode}")
    
    # Paths
    config_file = config_file
    
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
        env, object_name = construct_custom_env(
            cfg, 
            config_file, 
            randomize=not bool(args.use_eval_states),
            draw_coordinate_frame=bool(args.draw_coordinate_frame),
            use_eval_states=bool(args.use_eval_states),
            eval_experiment_name=args.eval_experiment_name,
            eval_trial_index=args.eval_trial_index,
            observation_mode=args.observation_mode,
        )
        
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
            gif_path = os.path.join(args.save_dir, f'{model_name}_config{args.config_id}{trial_suffix}.gif')
            save_numpy_as_gif(np.array(all_rgbs), gif_path)
            print(f"\n✓ Saved rollout GIF to: {gif_path}")
        elif not args.save_gif:
            print(f"\n⊘ GIF saving disabled (--save_gif=0)")
        
        # Save info
        import json
        info_path = os.path.join(args.save_dir, f'{model_name}_config{args.config_id}{trial_suffix}_info.json')
        info_dict = {
            "model": model_name,
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
    args = parse_args()
    
    ###################################
    # Run for door_40147 with eval states loaded
    ###################################
    # args.model = 'door_40147'
    # args.config_id = 0
    # args.horizon = 35
    # args.save_gif = 1
    # # args.draw_coordinate_frame = 1
    # args.num_trials = 5
    # args.save_dir = 'data/test_custom_model/act3d_goal_displacement_gripper_to_object_curated_state'
    # args.use_eval_states = 1
    # args.eval_experiment_name = "0705-diverse-objects-vary-obj-loc-ori-init-angle-robot-init-joint-near-handle-300-demo-0.4-0.15-translation-first"
    # args.observation_mode = 'act3d_goal_displacement_gripper_to_object'
    
    ###################################
    # Run for door_40147 with random initialization
    ###################################
    args.model = 'door_40147'
    args.config_id = 0
    args.horizon = 35
    args.save_gif = 1
    # args.draw_coordinate_frame = 1
    args.num_trials = 5
    args.save_dir = 'data/test_custom_model/act3d_goal_displacement_gripper_to_object_no_random'
    args.use_eval_states = 0
    args.observation_mode = 'act3d_goal_displacement_gripper_to_object'

    #######################
    # Run for color_0025_0_v2 with random initialization
    #######################
    # args.model = 'color_0025_0_v2'
    # args.config_id = 500
    # args.horizon = 35
    # args.save_gif = 1
    # # args.draw_coordinate_frame = 1
    # args.num_trials = 5
    # args.save_dir = 'data/test_custom_model/act3d_goal_displacement_gripper_to_object'
    # args.observation_mode = 'act3d_goal_displacement_gripper_to_object'

    main(args)
    
