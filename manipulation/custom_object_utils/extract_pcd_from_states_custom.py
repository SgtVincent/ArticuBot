"""
Custom extraction of point cloud states from demonstration trajectories.

This module provides extraction functions specifically designed for custom
articulated objects, using RobogenPointCloudWrapperCustom instead of the
standard wrapper to handle non-standard link naming conventions.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from collections import defaultdict
import tqdm
import yaml
from termcolor import cprint

from manipulation.utils import (
    build_up_env_gen,
    load_env,
    rotation_transfer_6D_to_matrix,
    rotation_transfer_matrix_to_6D,
    save_numpy_as_gif,
)
from manipulation.custom_object_utils.robogen_wrapper_custom import (
    RobogenPointCloudWrapperCustom,
)


def sort_states_file_by_file_number(state_path):
    """Sort state files by their numeric index."""
    ret_files = []
    for file in os.listdir(state_path):
        if file.startswith("state_") and file.endswith(".pkl"):
            ret_files.append(file)
    ret_files = sorted(ret_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    return ret_files


def get_all_expert_angles(all_experiments):
    """Get opening angles and ratios for all experiments."""
    ratios = []
    opened_angles = []
    for experiment_path in all_experiments:
        opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
        if os.path.exists(opened_angle_file):
            with open(opened_angle_file, "r") as f:
                angles = f.readlines()
                opened_angle = float(angles[0].lstrip().rstrip())
                min_angle = float(angles[1].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                opened_angle -= min_angle
                ratio = opened_angle / (max_angle - min_angle)
                ratios.append(ratio)
                opened_angles.append(opened_angle)
    return np.array(ratios), np.array(opened_angles)


def extract_handle_name_from_config(task_config_path):
    """Extract handle name from task config."""
    config = yaml.safe_load(open(task_config_path, "r"))
    for config_dict in config:
        if isinstance(config_dict, dict) and 'handle_name' in config_dict:
            return config_dict['handle_name'].lower()
    return 'handle'


def extract_asset_dir_from_config(task_config_path):
    """Extract asset directory hint from task config."""
    config = yaml.safe_load(open(task_config_path, "r"))
    for config_dict in config:
        if isinstance(config_dict, dict):
            if 'reward_asset_path' in config_dict:
                return config_dict['reward_asset_path']
            if 'solution_path' in config_dict:
                return config_dict['solution_path']
    return None



def _process_single_experiment(
    experiment,
    experiment_folder,
    task_config_path,
    env_name,
    handle_name,
    asset_dir_hint,
    angle_threshold,
    args,
    save_path,
    obs_keys
):
    """Process a single experiment trajectory."""
    traj_result = {}
    
    experiment_path = os.path.join(experiment_folder, experiment)
    state_path = os.path.join(experiment_path, "states")
    
    if not os.path.exists(state_path):
        # State path does not exist
        return None
        
    state_files = sort_states_file_by_file_number(state_path)
    if len(state_files) == 0:
        return None
        
    expert_states = [os.path.join(state_path, f) for f in state_files]
    
    # Load stage lengths
    stage_length_path = os.path.join(experiment_path, "stage_lengths.json")
    if not os.path.exists(stage_length_path):
        return None
        
    with open(stage_length_path, "r") as f:
        stage_lengths = json.load(f)
    
    reach_till_contact_idx = stage_lengths.get('reach_handle', 0) + stage_lengths.get('reach_to_contact', 0)
    open_time_idx = reach_till_contact_idx + stage_lengths.get('close_gripper', 0)
    
    # Check opened angle if available
    opened_angle_file = os.path.join(experiment_path, "opened_angle.txt")
    if os.path.exists(opened_angle_file):
        with open(opened_angle_file, "r") as f:
            angles = f.readlines()
            opened_angle = float(angles[0].lstrip().rstrip())
            min_angle = float(angles[1].lstrip().rstrip())
            max_angle = float(angles[-1].lstrip().rstrip())
            opened_angle -= min_angle
            ratio = opened_angle / (max_angle - min_angle)
        if opened_angle < angle_threshold or ratio < args.min_opened_ratio:
            return None
    
    # Check if already processed
    if os.path.exists(os.path.join(save_path, experiment)):
        print(f"Already saved the data for {experiment}, continue")
        return None
    
    bad_experiment = False
    camera_detected = True
    beg = time.time()
    
    # Get object name from config
    config = yaml.safe_load(open(task_config_path, "r"))
    object_name = None
    for config_dict in config:
        if isinstance(config_dict, dict) and 'name' in config_dict:
            object_name = config_dict['name'].lower()
            break
    
    if object_name is None:
        return None
    
    # Build environment and wrapper
    simulator = None
    try:
        simulator_base, _ = build_up_env_gen(
            task_config=task_config_path,
            env_name=env_name,
            restore_state_file=None,
            render=False,
            randomize=False,
            obj_id=0,
        )
        
        simulator = RobogenPointCloudWrapperCustom(
            simulator_base,
            object_name,
            handle_name=handle_name,
            asset_dir_hint=asset_dir_hint,
            num_points=args.pointcloud_num,
            observation_mode=args.observation_mode,
            noise_real_world_pcd=args.noise_real_world_pcd,
            real_world_camera=args.real_world_camera,
        )
    except Exception as e:
        print(f"Failed to build environment/wrapper for {experiment}: {e}")
        if simulator:
            try:
                simulator._env.close()
            except:
                pass
        return None

    traj_list = defaultdict(list)
    
    reach_till_contact_idx = reach_till_contact_idx // args.combine_action_steps
    open_time_idx = open_time_idx // args.combine_action_steps
    
    expert_states = expert_states[::args.combine_action_steps]
    
    try:
        load_env(simulator._env, load_path=expert_states[0])
    except Exception as e:
        print(f"Failed to load initial state for {experiment}: {e}")
        simulator._env.close()
        return None
    
    # Random camera reset if needed
    if args.randomize_camera:
        try:
            simulator.reset_random_cameras()
        except Exception:
            pass
            
    # Check handle visibility
    min_handle_visibility = 0 
    try:
        handle_visibility = simulator.check_handle_observed_in_pc()
        if min_handle_visibility > 0 and handle_visibility < min_handle_visibility and not args.randomize_camera:
            try:
                simulator.reset_random_cameras()
                handle_visibility = simulator.check_handle_observed_in_pc()
            except Exception:
                pass

        if min_handle_visibility > 0 and handle_visibility < min_handle_visibility:
            camera_detected = False
    except Exception:
        pass
    
    if camera_detected:
        for t_idx, state in enumerate(tqdm.tqdm(expert_states, desc=f"Processing {experiment}")):
            try:
                load_env(simulator._env, load_path=state)
                only_object = True
                observation = simulator._get_observation(only_object=only_object)
                rgb = simulator._env.render()
                
                traj_list['rgb'].append(rgb)
                for key in obs_keys:
                    traj_list[key].append(observation[key].tolist())
            except Exception as e:
                bad_experiment = True
                break
        
        if not bad_experiment and len(traj_list['gripper_pcd']) > open_time_idx:
            goal_gripper_pcd_at_grasping = traj_list['gripper_pcd'][open_time_idx]
            goal_gripper_pcd_at_end = traj_list['gripper_pcd'][-1]
            for t in range(min(open_time_idx, len(traj_list['goal_gripper_pcd']))):
                traj_list['goal_gripper_pcd'][t] = goal_gripper_pcd_at_grasping
            for t in range(open_time_idx, len(traj_list['gripper_pcd'])):
                if t < len(traj_list['goal_gripper_pcd']):
                    traj_list['goal_gripper_pcd'][t] = goal_gripper_pcd_at_end
    
    view_matrices = simulator.view_matrices
    proj_matrices = simulator.project_matrices
    simulator._env.close()
    
    end = time.time()
    cprint(f"Finished extracting {experiment} with length {len(expert_states)} in {end-beg:.1f}s", "green")
    
    if not bad_experiment and camera_detected and len(traj_list['point_cloud']) > 0:
        traj_result['success'] = True
        traj_result['stage_lengths'] = stage_lengths
        traj_result['experiment_path'] = experiment_path
        traj_result['traj_list'] = traj_list
        traj_result['view_matrices'] = view_matrices
        traj_result['proj_matrices'] = proj_matrices
    else:
        traj_result['success'] = False
        traj_result['experiment_path'] = experiment_path
        traj_result['reason'] = f"bad_experiment={bad_experiment}, camera_detected={camera_detected}"
        
    return traj_result


def extract_pc_states_for_all_trajectories_custom(pool_args):
    """Extract point cloud states using the custom wrapper."""
    task_config_path, solution_path, env_name, exp_name, experiments, save_path, angle_threshold, args = pool_args
    
    obs_keys = ['point_cloud', 'agent_pos', 'gripper_pcd', 'goal_gripper_pcd', 'displacement_gripper_to_object']
    all_traj_obs_dict_of_list = defaultdict(list)
    all_traj_stage_lengths = []
    all_traj_store_label_paths = []
    all_view_matrices = []
    all_proj_matrices = []
    
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    
    if not os.path.exists(experiment_folder):
        print("Experiment folder does not exist: ", experiment_folder)
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
    
    # Extract handle name and asset dir from config
    handle_name = extract_handle_name_from_config(task_config_path)
    asset_dir_hint = extract_asset_dir_from_config(task_config_path)
    
    print(f"Using handle_name: {handle_name}, asset_dir_hint: {asset_dir_hint}")
    
    for experiment in experiments:
        result = _process_single_experiment(
            experiment,
            experiment_folder,
            task_config_path,
            env_name,
            handle_name,
            asset_dir_hint,
            angle_threshold,
            args,
            save_path,
            obs_keys
        )
        
        if result is None:
            continue
            
        if result['success']:
            all_traj_stage_lengths.append(result['stage_lengths'])
            all_traj_store_label_paths.append(result['experiment_path'])
            for key in obs_keys + ['rgb']:
                all_traj_obs_dict_of_list[key].append(result['traj_list'][key])
            all_view_matrices.append(result['view_matrices'])
            all_proj_matrices.append(result['proj_matrices'])
        else:
            label_path = os.path.join(result['experiment_path'], "label.json")
            print(f"Skipping {experiment} - {result['reason']}")
            try:
                with open(label_path, "w") as f:
                    json.dump({"good_traj": False, "failure reason": "extraction failed"}, f)
            except:
                pass
    
    return all_traj_obs_dict_of_list, all_traj_stage_lengths, all_traj_store_label_paths, all_view_matrices, all_proj_matrices


def extract_demos_from_a_directory_custom(
    directory_path, 
    exp_name=None, 
    env_name=None, 
    extract_name=None, 
    save_path=None,
    args=None,
):
    """Extract demonstrations from a directory using the custom wrapper.
    
    This is the main entry point for custom object demo extraction.
    """
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)


def _resolve_task_config_path(directory_path, task_path):
    """Find the task configuration file."""
    solution_path = os.path.join(directory_path, task_path)
    # Find task config
    files_and_folders = os.listdir(solution_path)
    task_config_path = None
    for file_or_folder in files_and_folders:
        if file_or_folder.endswith(".yaml") and not file_or_folder.startswith("config_custom"):
            candidate = os.path.join(directory_path, task_path, file_or_folder)
            if os.path.isfile(candidate):
                task_config_path = candidate
                break
    
    if task_config_path is None:
        # Try base_config.yaml specifically
        base_config = os.path.join(solution_path, "base_config.yaml")
        if os.path.exists(base_config):
            task_config_path = base_config
        else:
            raise FileNotFoundError(f"No config YAML found in {solution_path}")
    return task_config_path


def _find_successful_experiments(solution_path, exp_name):
    """Find and filter successful experiments."""
    # Find experiment folder
    if exp_name is None:
        experiment_folder = os.path.join(solution_path, "experiment")
    else:
        experiment_folder = os.path.join(solution_path, "experiment", exp_name)
    
    if not os.path.exists(experiment_folder):
        print(f"Experiment folder does not exist: {experiment_folder}")
        if exp_name is not None:
            experiment_folder = solution_path
        else:
            experiment_folder = os.path.join(solution_path, exp_name)
    
    # Get all experiments
    all_experiments = [
        x for x in os.listdir(experiment_folder) 
        if os.path.isdir(os.path.join(experiment_folder, x)) and not x.startswith(".")
    ]
    all_experiments = sorted(all_experiments)
    print(f"Found {len(all_experiments)} experiment folders")
    
    # Filter to successful experiments
    success_experiments = []
    for exp in all_experiments:
        state_path = os.path.join(experiment_folder, exp, "states")
        if os.path.exists(state_path):
            state_files = [f for f in os.listdir(state_path) if f.startswith("state_") and f.endswith(".pkl")]
            all_gif = os.path.join(experiment_folder, exp, "all.gif")
            if len(state_files) > 1 and os.path.exists(all_gif):
                success_experiments.append(exp)
    
    return success_experiments, experiment_folder


def _save_trajectory_demo(
    traj_idx,
    save_path,
    demo_rgb_save_path,
    all_traj_pc,
    all_traj_pos_ori,
    all_traj_rgbs,
    all_traj_gripper_pcds,
    all_traj_goal_gripper_pcd,
    all_traj_displacement_gripper_to_object,
    all_traj_stage_lengths,
    all_traj_store_label_paths,
    all_view_matrices,
    all_proj_matrices,
    args
):
    """Save a single processed trajectory."""
    traj_pc = all_traj_pc[traj_idx]
    traj_pos_ori = all_traj_pos_ori[traj_idx]
    traj_gripper_pcd = all_traj_gripper_pcds[traj_idx]
    traj_stage_length = all_traj_stage_lengths[traj_idx]
    traj_store_label_path = all_traj_store_label_paths[traj_idx]
    traj_goal_gripper_pcd = all_traj_goal_gripper_pcd[traj_idx]
    traj_displacement_gripper_to_object = all_traj_displacement_gripper_to_object[traj_idx]
    
    good_traj = True
    failure_reason = "null"
    
    traj_actions = []
    
    after_contact_idx = (
        traj_stage_length.get('reach_handle', 0) + 
        traj_stage_length.get('reach_to_contact', 0)
    )
    after_contact_idx = after_contact_idx // args.combine_action_steps
    
    filtered_pcs = []
    filtered_pos_oris = []
    filtered_gripper_pcds = []
    filtered_rgbs = []
    filtered_goal_gripper_pcds = []
    filtered_displacement_gripper_to_objects = []
    
    if len(traj_pos_ori) == 0:
        print(f"Empty trajectory at index {traj_idx}")
        return None
    
    base_pos = traj_pos_ori[0][:3]
    
    for i in range(len(traj_pos_ori) - 1):
        cur_pos = traj_pos_ori[i][:3]
        target_pos = traj_pos_ori[i+1][:3]
        
        single_step_delta_pos = np.array(target_pos) - np.array(cur_pos)
        
        if np.linalg.norm(single_step_delta_pos) > 0.02 * args.combine_action_steps:
            good_traj = False
            failure_reason = "delta movement too large"
            print(f"Not good traj due to delta movement too large at step {i}")
            break
        
        delta_pos = np.array(target_pos) - np.array(base_pos)
        cur_ori_6d = traj_pos_ori[i][3:9]
        target_ori_6d = traj_pos_ori[i+1][3:9]
        
        # Compute orientation difference
        cur_ori_matrix = rotation_transfer_6D_to_matrix(cur_ori_6d)
        target_ori_matrix = rotation_transfer_6D_to_matrix(target_ori_6d)
        delta_ori_matrix = np.linalg.inv(cur_ori_matrix) @ target_ori_matrix
        delta_ori_6d = rotation_transfer_matrix_to_6D(delta_ori_matrix)
        
        # Compute finger angle difference
        cur_finger_angle = traj_pos_ori[i][9]
        target_finger_angle = traj_pos_ori[i+1][9]
        delta_finger_angle = target_finger_angle - cur_finger_angle
        
        # Filter close-to-zero actions before contact
        if i < after_contact_idx and args.filter_close_zero_action:
            if (np.linalg.norm(single_step_delta_pos) < 1e-4 and 
                abs(delta_finger_angle) < args.min_finger_angle_diff):
                continue
        
        action = np.concatenate([delta_pos, delta_ori_6d, [delta_finger_angle]])
        traj_actions.append(action.tolist())
        
        filtered_pcs.append(traj_pc[i])
        filtered_pos_oris.append(traj_pos_ori[i])
        filtered_gripper_pcds.append(traj_gripper_pcd[i])
        filtered_rgbs.append(all_traj_rgbs[traj_idx][i])
        filtered_goal_gripper_pcds.append(
            traj_goal_gripper_pcd[i] if i < len(traj_goal_gripper_pcd) else traj_goal_gripper_pcd[-1]
        )
        filtered_displacement_gripper_to_objects.append(
            traj_displacement_gripper_to_object[i] 
            if i < len(traj_displacement_gripper_to_object) 
            else traj_displacement_gripper_to_object[-1]
        )
    
    if not good_traj:
        label_path = os.path.join(traj_store_label_path, "label.json")
        try:
            with open(label_path, "w") as f:
                json.dump({"good_traj": False, "failure reason": failure_reason}, f)
        except:
            pass
        return None
    
    if len(filtered_pcs) == 0:
        print(f"No valid steps in trajectory {traj_idx}")
        return None
    
    # Save trajectory
    traj_save_path = os.path.join(save_path, os.path.basename(traj_store_label_path))
    os.makedirs(traj_save_path, exist_ok=True)
    
    for step_idx in range(len(filtered_pcs)):
        step_data = {
            'point_cloud': np.array(filtered_pcs[step_idx])[None, :],
            'state': np.array(filtered_pos_oris[step_idx])[None, :],
            'action': np.array(traj_actions[step_idx])[None, :] if step_idx < len(traj_actions) else np.zeros((1, 10)),
            'gripper_pcd': np.array(filtered_gripper_pcds[step_idx])[None, :],
            'goal_gripper_pcd': np.array(filtered_goal_gripper_pcds[step_idx])[None, :],
            'displacement_gripper_to_object': np.array(filtered_displacement_gripper_to_objects[step_idx])[None, :],
        }
        
        step_file = os.path.join(traj_save_path, f"{step_idx}.pkl")
        with open(step_file, "wb") as f:
            pickle.dump(step_data, f)
    
    # Save camera params
    camera_params = {
        'view_matrices': all_view_matrices[traj_idx] if traj_idx < len(all_view_matrices) else [],
        'proj_matrices': all_proj_matrices[traj_idx] if traj_idx < len(all_proj_matrices) else [],
    }
    camera_file = os.path.join(traj_save_path, "camera_params.npz")
    np.savez(camera_file, **camera_params)
    
    # Save demo RGB as gif
    if len(filtered_rgbs) > 0:
        demo_gif_path = os.path.join(demo_rgb_save_path, f"{os.path.basename(traj_store_label_path)}.gif")
        try:
            save_numpy_as_gif(np.array(filtered_rgbs), demo_gif_path)
        except Exception as e:
            print(f"Warning: Could not save demo gif: {e}")
    
    label_path = os.path.join(traj_store_label_path, "label.json")
    try:
        with open(label_path, "w") as f:
            json.dump({"good_traj": True, "failure reason": failure_reason}, f)
    except:
        pass
        
    return traj_save_path


def extract_demos_from_a_directory_custom(
    directory_path, 
    exp_name=None, 
    env_name=None, 
    extract_name=None, 
    save_path=None,
    args=None,
):
    """Extract demonstrations from a directory using the custom wrapper."""
    demo_rgb_save_path = os.path.join(save_path, "demo_rgbs")
    if not os.path.exists(demo_rgb_save_path):
        os.makedirs(demo_rgb_save_path)

    task_path = extract_name
    solution_path = os.path.join(directory_path, task_path)
    
    # Find task config
    task_config_path = _resolve_task_config_path(directory_path, task_path)
    
    print(f"Using task config: {task_config_path}")
    
    # Find experiment folder and experiments
    all_experiments, experiment_folder = _find_successful_experiments(solution_path, exp_name)
    
    print(f"Found {len(all_experiments)} successful experiments with states and all.gif")
    
    if len(all_experiments) == 0:
        print("No successful experiments found!")
        return
    
    num_experiment = len(all_experiments)
    
    # Get angle threshold
    _, all_expert_opened_angles = get_all_expert_angles(
        [os.path.join(experiment_folder, exp) for exp in all_experiments]
    )
    if len(all_expert_opened_angles) > 0:
        angle_threshold = np.quantile(all_expert_opened_angles, 0.1)
    else:
        angle_threshold = 0.0
    
    num_experiment = min(num_experiment, args.num_experiment)
    batch_size = 1
    num_batch = (num_experiment - 1) // batch_size + 1
    
    all_demo_paths = []
    
    for batch_idx in range(num_batch):
        beg_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_experiment)
        
        (
            all_traj_obs_dict_of_list, 
            all_traj_stage_lengths, 
            all_traj_store_label_paths, 
            all_view_matrices, 
            all_proj_matrices
        ) = extract_pc_states_for_all_trajectories_custom([
            task_config_path, 
            solution_path, 
            env_name, 
            exp_name,
            all_experiments[beg_idx:end_idx],
            save_path,
            angle_threshold, 
            args
        ])
        
        if len(all_traj_obs_dict_of_list['point_cloud']) == 0:
            print(f"No valid trajectories in batch {batch_idx}")
            continue
        
        all_traj_pc = all_traj_obs_dict_of_list['point_cloud']
        all_traj_pos_ori = all_traj_obs_dict_of_list['agent_pos']
        all_traj_rgbs = all_traj_obs_dict_of_list['rgb']
        all_traj_gripper_pcds = all_traj_obs_dict_of_list['gripper_pcd']
        all_traj_goal_gripper_pcd = all_traj_obs_dict_of_list['goal_gripper_pcd']
        all_traj_displacement_gripper_to_object = all_traj_obs_dict_of_list['displacement_gripper_to_object']
        
        print(f"Processing {len(all_traj_pc)} trajectories from batch {batch_idx}")
        
        for traj_idx in tqdm.tqdm(range(len(all_traj_pc)), desc="Saving trajectories"):
            traj_path = _save_trajectory_demo(
                traj_idx,
                save_path,
                demo_rgb_save_path,
                all_traj_pc,
                all_traj_pos_ori,
                all_traj_rgbs,
                all_traj_gripper_pcds,
                all_traj_goal_gripper_pcd,
                all_traj_displacement_gripper_to_object,
                all_traj_stage_lengths,
                all_traj_store_label_paths,
                all_view_matrices,
                all_proj_matrices,
                args
            )
            
            if traj_path:
                all_demo_paths.append(traj_path)
    
    # Save all demo paths
    all_demo_path_file = os.path.join(save_path, "all_demo_path.txt")
    with open(all_demo_path_file, "w") as f:
        for path in all_demo_paths:
            f.write(path + "\n")
    
    # Save meta info
    meta_info = {
        "env_name": env_name,
        "extract_name": extract_name,
        "save_path": save_path,
        "pointcloud_num": args.pointcloud_num,
        "observation_mode": args.observation_mode,
        "combine_action_steps": args.combine_action_steps,
        "min_opened_ratio": args.min_opened_ratio,
        "num_demos": len(all_demo_paths),
    }
    meta_info_file = os.path.join(save_path, "meta_info.json")
    with open(meta_info_file, "w") as f:
        json.dump(meta_info, f, indent=4)
    
    cprint(f"\nExtraction complete! Saved {len(all_demo_paths)} demos to {save_path}", "green")
    return all_demo_paths

