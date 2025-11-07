#!/usr/bin/env python3
"""
High-level test script for custom URDF model.
This script uses ONLY the high-level policy (PointNet2_super) to predict 
4-point end-effector goals for the custom object color_0025_0_v2.

Unlike test_custom_model_no_info.py which uses both high and low-level policies,
this script focuses on evaluating the high-level goal prediction in isolation.
"""

import os
import sys
import pathlib
import numpy as np
import argparse
import torch
import pybullet as p
import json

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
os.environ['PROJECT_DIR'] = str(PROJECT_ROOT)

from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import save_numpy_as_gif

# Import reusable components from test_custom_model_no_info
from test_custom_model_no_info import RobogenPointCloudWrapperNoInfo


def load_high_level_policy(high_level_ckpt_path, add_one_hot_encoding=False):
    """Load only the high-level policy for goal prediction."""
    
    print("=" * 60)
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
    print(f"  - Input channels: {input_channel}")
    print(f"  - Output classes: {num_class} (4 points × 3 coords + weight)")
    
    return high_level_policy


def construct_custom_env_simple(config_file, num_points=4500, randomize=True,
                                draw_coordinate_frame=False, observation_mode="act3d_displacement_gripper_to_object"):
    """Simplified environment constructor for high-level policy testing.
    
    Returns unwrapped point cloud environment (no MultiStepWrapper).
    """
    import yaml
    from manipulation.robogen_wrapper import RobogenPointCloudWrapper
    from manipulation.utils import build_up_env_random
    
    print("\n" + "=" * 60)
    print("Constructing Environment...")
    print("=" * 60)
    print(f"Config file: {config_file}")
    print(f"Randomize initialization: {randomize}")
    print(f"Observation mode: {observation_mode}")
    
    with open(config_file, 'r') as f:
        task_config = yaml.safe_load(f)
    
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
        object_name = task_config['object_name'].lower()
        wrapper_cls = RobogenPointCloudWrapperNoInfo
    else:
        raise NotImplementedError("Non-random initialization not supported in simplified constructor.")
    
    # Draw world coordinate frame if requested
    if draw_coordinate_frame:
        p.addUserDebugLine([0, 0, 0], [0.3, 0, 0], [1, 0, 0], lineWidth=3, lifeTime=0, physicsClientId=env.id)
        p.addUserDebugLine([0, 0, 0], [0, 0.3, 0], [0, 1, 0], lineWidth=3, lifeTime=0, physicsClientId=env.id)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0.3], [0, 0, 1], lineWidth=3, lifeTime=0, physicsClientId=env.id)
        print(f"✓ World coordinate frame drawn (X:Red, Y:Green, Z:Blue)")
    
    # Wrap with point cloud wrapper (no MultiStepWrapper for high-level testing)
    pointcloud_env = wrapper_cls(
        env,
        object_name,
        num_points=num_points,
        observation_mode=observation_mode,
        real_world_camera=False,
        noise_real_world_pcd=False,
    )
    
    print(f"✓ Point cloud wrapper added (no _get_info() calls)")
    print(f"✓ Observation mode set to '{observation_mode}'")
    print(f"✓ Environment construction complete")
    
    return pointcloud_env, object_name


def high_level_policy_infer(obs_dict, high_level_policy, output_obj_pcd_only=True, add_one_hot_encoding=False):
    """Run high-level policy to predict goal point cloud (4 end-effector points).
    
    Reuses the same logic as test_custom_model_no_info.py high_level_policy_infer.
    """
    with torch.no_grad():
        pointcloud = obs_dict['point_cloud']
        gripper_pcd = obs_dict['gripper_pcd']
        
        # Handle both (B, T, N, 3) and (B, N, 3) formats
        if pointcloud.dim() == 4:
            pointcloud = pointcloud[:, -1, :, :]
            gripper_pcd = gripper_pcd[:, -1, :]
        
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


def visualize_goal_prediction(env, object_name, high_level_policy, 
                              num_predictions=5, output_obj_pcd_only=True, 
                              add_one_hot_encoding=False, save_gif=True):
    """Visualize high-level goal predictions over multiple steps."""
    
    print("\n" + "=" * 60)
    print("Visualizing High-Level Goal Predictions...")
    print("=" * 60)
    print(f"Number of predictions: {num_predictions}")
    
    # Get initial observation
    obs = env.reset()
    
    all_rgbs = []
    all_goals = []
    
    if save_gif:
        rgb = env.render()
    
    # Get initial joint states
    object_id = env._env.urdf_ids[object_name]
    num_joints = p.getNumJoints(object_id, physicsClientId=env._env.id)
    
    print(f"\nInitial state captured")
    print(f"Object: {object_name}")
    print(f"Number of joints: {num_joints}")
    
    # Run predictions
    for i in range(num_predictions):
        print(f"\n--- Prediction {i+1}/{num_predictions} ---")
        
        # Prepare input
        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to('cuda'))
        obs_dict = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
        
        # Infer high-level goal
        goal_eef_points = high_level_policy_infer(
            obs_dict, 
            high_level_policy, 
            output_obj_pcd_only=output_obj_pcd_only,
            add_one_hot_encoding=add_one_hot_encoding
        )
        
        # Convert to numpy for storage
        goal_np = goal_eef_points.detach().cpu().numpy()
        all_goals.append(goal_np)
        
        print(f"Predicted goal shape: {goal_np.shape}")
        print(f"Goal points (4 end-effector points):")
        for j, point in enumerate(goal_np[0, 0]):  # [B, 1, 4, 3]
            print(f"  Point {j+1}: [{point[0]:+.4f}, {point[1]:+.4f}, {point[2]:+.4f}]")
        
        # Optionally visualize the goal in PyBullet
        if i == 0:  # Only visualize first prediction to avoid clutter
            print("\nDrawing predicted goal points in simulation...")
            for j, point in enumerate(goal_np[0, 0]):
                # Draw a small sphere at each goal point
                p.addUserDebugLine(
                    point, point + [0.01, 0, 0],
                    [1, 0, 0], lineWidth=5, lifeTime=0,
                    physicsClientId=env._env.id
                )
                p.addUserDebugLine(
                    point, point + [0, 0.01, 0],
                    [0, 1, 0], lineWidth=5, lifeTime=0,
                    physicsClientId=env._env.id
                )
                p.addUserDebugLine(
                    point, point + [0, 0, 0.01],
                    [0, 0, 1], lineWidth=5, lifeTime=0,
                    physicsClientId=env._env.id
                )
        
        if save_gif:
            rgb = env.render()
            all_rgbs.append(rgb)
        
        # Small random action to change the scene slightly for next prediction
        if i < num_predictions - 1:
            random_action = np.random.randn(10) * 0.01
            obs, _, _, _ = env.step(random_action)
    
    env._env.close()
    
    print(f"\n✓ Visualization complete!")
    
    return all_rgbs, all_goals


def parse_args():
    parser = argparse.ArgumentParser(description="Test high-level policy on custom URDF model")
    parser.add_argument('--high_level_ckpt_name', type=str, 
                        default='data/high_level_200_obj_ckpt.pth',
                        help='Path to high-level checkpoint')
    parser.add_argument('--model', type=str, default='color_0025_0_v2',
                        help='Model to test (e.g., color_0025_0_v2)')
    parser.add_argument('--config_id', type=int, default=500,
                        help='Config ID to use (for color_0025_0_v2: 500-512)')
    parser.add_argument('--num_predictions', type=int, default=5,
                        help='Number of goal predictions to generate')
    parser.add_argument('--output_obj_pcd_only', type=int, default=1,
                        help='Output object point cloud only (1=yes, 0=no)')
    parser.add_argument('--add_one_hot_encoding', type=int, default=0,
                        help='Add one-hot encoding (1=yes, 0=no)')
    parser.add_argument('--save_dir', type=str, default='data/test_custom_model_high',
                        help='Directory to save results')
    parser.add_argument('--num_trials', type=int, default=1,
                        help='Number of random trials to run')
    parser.add_argument('--save_gif', type=int, default=1,
                        help='Save visualization as GIF (1=yes, 0=no)')
    parser.add_argument('--draw_coordinate_frame', type=int, default=0,
                        help='Draw world coordinate frame in GIF (1=yes, 0=no)')
    parser.add_argument('--observation_mode', type=str,
                        default='act3d_displacement_gripper_to_object',
                        help='Observation mode passed to RobogenPointCloudWrapper')
    
    args = parser.parse_args()
    return args


def main(args):
    print("=" * 60)
    print("Testing High-Level Policy on Custom URDF Model")
    print("=" * 60)
    
    # Setup paths
    model_name = args.model
    object_path = os.path.join(PROJECT_ROOT, 'data/custom_objects/color_0025_0_v2')
    config_file = os.path.join(object_path, f'configs/config_larger_randomization_{args.config_id}.yaml')
    
    print(f"Model: {model_name}")
    print(f"Config ID: {args.config_id}")
    print(f"Number of trials: {args.num_trials}")
    print(f"Observation mode: {args.observation_mode}")
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    # Load high-level policy
    high_level_policy = load_high_level_policy(
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
        
        # Construct environment (simplified, no eval states)
        env, object_name = construct_custom_env_simple(
            config_file, 
            randomize=True,
            draw_coordinate_frame=bool(args.draw_coordinate_frame),
            observation_mode=args.observation_mode,
        )
        
        # Run visualization
        all_rgbs, all_goals = visualize_goal_prediction(
            env, 
            object_name,
            high_level_policy,
            num_predictions=args.num_predictions,
            output_obj_pcd_only=args.output_obj_pcd_only,
            add_one_hot_encoding=args.add_one_hot_encoding,
            save_gif=bool(args.save_gif)
        )
        
        # Save results for this trial
        trial_suffix = f"_trial{trial}" if args.num_trials > 1 else ""
        
        # Save GIF if enabled
        if args.save_gif and len(all_rgbs) > 0:
            gif_path = os.path.join(args.save_dir, f'{model_name}_high_config{args.config_id}{trial_suffix}.gif')
            save_numpy_as_gif(np.array(all_rgbs), gif_path)
            print(f"\n✓ Saved visualization GIF to: {gif_path}")
        elif not args.save_gif:
            print(f"\n⊘ GIF saving disabled (--save_gif=0)")
        
        # Save predicted goals
        goals_path = os.path.join(args.save_dir, f'{model_name}_high_config{args.config_id}{trial_suffix}_goals.npy')
        np.save(goals_path, np.array(all_goals))
        print(f"✓ Saved predicted goals to: {goals_path}")
        
        # Save summary info
        info_path = os.path.join(args.save_dir, f'{model_name}_high_config{args.config_id}{trial_suffix}_info.json')
        info_dict = {
            "model": model_name,
            "config_id": args.config_id,
            "trial": trial,
            "num_predictions": args.num_predictions,
            "observation_mode": args.observation_mode,
            "goals_shape": [list(g.shape) for g in all_goals]
        }
        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        print(f"✓ Saved info to: {info_path}")
        
        all_results.append({"goals": all_goals})
    
    print(f"\n{'=' * 60}")
    print("High-Level Policy Evaluation Complete")
    print(f"{'=' * 60}")
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
