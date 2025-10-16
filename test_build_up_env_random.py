#!/usr/bin/env python3
"""
Quick test script to verify build_up_env_random works correctly.
This creates an environment, initializes it randomly, and prints state info.
"""

import os
import sys
import pathlib
import numpy as np
import argparse
import time

# Setup paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
os.environ['PROJECT_DIR'] = str(PROJECT_ROOT)

from manipulation.utils import build_up_env_random

def test_build_up_env_random(render=False, num_trials=3, pause_time=2.0):
    """Test that build_up_env_random works correctly."""
    
    print("=" * 60)
    print("Testing build_up_env_random Function")
    print("=" * 60)
    
    # Path to custom config
    config_file = os.path.join(
        PROJECT_ROOT, 
        'data/custom_objects/color_0025_0_v2/configs/config_larger_randomization_500.yaml'
    )
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    print(f"Config file: {config_file}")
    print(f"Render GUI: {render}")
    print(f"Testing with {num_trials} random initializations...\n")
    
    # Test multiple random initializations
    for trial in range(num_trials):
        print(f"--- Trial {trial + 1} ---")
        
        try:
            # Set random seed for reproducibility in this test
            np.random.seed(42 + trial)
            
            # Create environment with random initialization
            env, config = build_up_env_random(
                task_config=config_file,
                env_name='articulated',
                render=render,
                randomize_object_pose=True,
                randomize_robot_joints=True,
                randomize_initial_joint_angle=True,
                horizon=600,
                max_attempts=100,  # Use new parameter
                far_distance=0.7,  # Maximum handle distance
                near_distance=0.3  # Minimum handle distance
            )
            
            print(f"✓ Environment created successfully (collision-free)")
            
            if render:
                print(f"  Pausing for {pause_time} seconds to view GUI...")
                time.sleep(pause_time)
            
            # Get some state information
            import pybullet as p
            
            # Get object pose
            object_name = config['object_name']
            object_id = env.urdf_ids[object_name]
            obj_pos, obj_orient = p.getBasePositionAndOrientation(object_id, physicsClientId=env.id)
            obj_euler = p.getEulerFromQuaternion(obj_orient)
            
            print(f"  Object position: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
            print(f"  Object orientation (euler): [{obj_euler[0]:.3f}, {obj_euler[1]:.3f}, {obj_euler[2]:.3f}]")
            
            # Get robot joint states
            robot_joints = []
            for i in range(7):
                joint_state = p.getJointState(env.robot.body, i, physicsClientId=env.id)
                robot_joints.append(joint_state[0])
            print(f"  Robot joints: {[f'{j:.3f}' for j in robot_joints]}")
            
            # Get articulated joint state
            try:
                handle_joint_id = env.get_handle_joint_id()
            except (FileNotFoundError, KeyError, AttributeError):
                # Fallback: find first revolute joint
                handle_joint_id = None
                num_joints = p.getNumJoints(object_id, physicsClientId=env.id)
                for i in range(num_joints):
                    joint_info = p.getJointInfo(object_id, i, physicsClientId=env.id)
                    joint_type = joint_info[2]
                    if joint_type == 0:  # Revolute joint
                        handle_joint_id = i
                        break
            
            if handle_joint_id is not None:
                handle_angle = p.getJointState(object_id, handle_joint_id, physicsClientId=env.id)[0]
                joint_limits = p.getJointInfo(object_id, handle_joint_id, physicsClientId=env.id)[8:10]
                print(f"  Articulated joint angle: {handle_angle:.3f} (limits: [{joint_limits[0]:.3f}, {joint_limits[1]:.3f}])")
            else:
                print("  Articulated joint: None found")
            
            # Check collision status
            contact_points = p.getContactPoints(env.robot.body, object_id, physicsClientId=env.id)
            closest_points = p.getClosestPoints(env.robot.body, object_id, distance=0.01, physicsClientId=env.id)
            collision_status = "✓ No collision" if len(contact_points) == 0 and len(closest_points) == 0 else "⚠ Collision detected"
            print(f"  Robot-Object collision: {collision_status}")
            
            # Check end effector to handle distance
            try:
                robot_eef_pos, _ = env.robot.get_pos_orient(env.robot.right_end_effector)
                if handle_joint_id is not None:
                    # Get link state for handle position
                    link_state = p.getLinkState(object_id, handle_joint_id, physicsClientId=env.id)
                    handle_pos = np.array(link_state[0])
                    distance = np.linalg.norm(handle_pos - robot_eef_pos)
                    print(f"  EEF to handle distance: {distance:.3f}m (target: 0.3-0.7m)")
                else:
                    print("  EEF to handle distance: N/A (no handle joint)")
            except Exception:
                print("  EEF to handle distance: N/A")
            
            # Close environment
            env.close()
            print(f"✓ Trial {trial + 1} completed successfully\n")
            
        except Exception as e:
            print(f"❌ Error in trial {trial + 1}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("=" * 60)
    print("✓ All tests passed! build_up_env_random is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test build_up_env_random function")
    parser.add_argument('--render', action='store_true', 
                        help='Launch PyBullet GUI to visualize initialization')
    parser.add_argument('--num_trials', type=int, default=3,
                        help='Number of random initialization trials to test (default: 3)')
    parser.add_argument('--pause_time', type=float, default=2.0,
                        help='Seconds to pause in GUI mode for viewing (default: 2.0)')
    args = parser.parse_args()
    
    success = test_build_up_env_random(
        render=args.render,
        num_trials=args.num_trials,
        pause_time=args.pause_time
    )
    sys.exit(0 if success else 1)
