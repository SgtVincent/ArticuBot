import os
import sys
import argparse
import json
from copy import deepcopy
import yaml
import pathlib

# ensure repo root is on PYTHONPATH
PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../ArticuBot")
)
os.environ["PROJECT_DIR"] = PROJECT_DIR
sys.path.append(PROJECT_DIR)

import hydra
from omegaconf import OmegaConf
import torch
import pybullet as p
import numpy as np
from termcolor import cprint
import tqdm
import time
import pickle as pkl

from train_ddp import TrainDP3Workspace
from manipulation.robogen_wrapper import RobogenPointCloudWrapper
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.common.pytorch_util import dict_apply
from manipulation.utils import build_up_env_eval, save_numpy_as_gif


def construct_env(
    cfg,
    config_file,
    solution_path,
    task_name,
    init_state_file,
    real_world_camera=False,
    noise_real_world_pcd=False,
    randomize_camera=False,
):
    """Construct and return a wrapped Robogen environment for evaluation.

    This builds the base environment using `build_up_env`, wraps it with
    `RobogenPointCloudWrapper` to provide point-cloud observations, and then
    places it inside `MultiStepWrapper` so the environment returns sequences
    of observations and supports multi-step actions.

    Args:
        cfg: Hydrated experiment config containing environment/task settings.
        config_file (str): Path to the task-specific config file.
        solution_path (str): Solution path (relative to PROJECT_DIR) used by the task.
        task_name (str): The task/primitive name to instantiate.
        init_state_file (str): Path to the initial state file used to seed the env.
        real_world_camera (bool): If True, emulate real-world camera behavior.
        noise_real_world_pcd (bool): If True, add noise to point-clouds.
        randomize_camera (bool): If True, randomize camera poses on reset.

    Returns:
        A `MultiStepWrapper`-wrapped environment that provides point-cloud-based
        observations and supports stepping with low-level actions.
    """
    result = build_up_env_eval(
        config_file,
        solution_path,
        task_name,
        init_state_file,
        render=False,
        horizon=600,
    )
    env = result[0]

    object_name = "StorageFurniture".lower()
    env.reset()
    pointcloud_env = RobogenPointCloudWrapper(
        env,
        object_name,
        num_points=cfg.task.env_runner.num_point_in_pc,
        observation_mode=cfg.task.env_runner.observation_mode,
        real_world_camera=real_world_camera,
        noise_real_world_pcd=noise_real_world_pcd,
    )

    if randomize_camera:
        pointcloud_env.reset_random_cameras()

    env = MultiStepWrapper(
        pointcloud_env,
        n_obs_steps=cfg.n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=600,
        reward_agg_method="sum",
    )
    return env
    """Gather config, init-state paths and expert angles for trials.

    Scans `all_experiments` inside `experiment_path` and returns three lists:
      - config_files: paths to `task_config.yaml` for each valid trial
      - init_state_files: paths to initial state pickle files (state_0.pkl)
      - expert_opened_angles: expert final joint angles parsed from
        `opened_angle.txt` when available

    Trials missing required files or marked as invalid are skipped.

    Args:
        experiment_folder (str): Path where `substeps.txt` resides.
        experiment_path (str): Path containing per-trial folders.
        all_experiments (list[str]): Names of trial subfolders to consider.

    Returns:
        Tuple of (config_files, init_state_files, expert_opened_angles).
    """
    all_substeps_path = os.path.join(experiment_folder, "substeps.txt")
    with open(all_substeps_path, "r") as f:
        substeps = f.readlines()
        first_step = substeps[0].lstrip().rstrip()

    expert_opened_angles = []
    init_state_files = []
    config_files = []
    for experiment in all_experiments:
        if "meta" in experiment:
            continue

        # For this data, states are directly in the trial folder
        first_step_folder = os.path.join(experiment_path, experiment)
        if os.path.exists(os.path.join(first_step_folder, "label.json")):
            with open(os.path.join(first_step_folder, "label.json"), "r") as f:
                label = json.load(f)
            if not label["good_traj"]:
                continue

        first_step_states_path = os.path.join(first_step_folder, "states")
        expert_states = os.listdir(first_step_states_path)
        if len(expert_states) == 0:
            continue

        expert_opened_angle_file = os.path.join(first_step_folder, "opened_angle.txt")
        if os.path.exists(expert_opened_angle_file):
            with open(expert_opened_angle_file, "r") as f:
                angles = f.readlines()
                expert_opened_angle = float(angles[0].lstrip().rstrip())
                max_angle = float(angles[-1].lstrip().rstrip())
                ratio = expert_opened_angle / max_angle
        else:
            expert_opened_angle = 0.0  # or some default

        expert_opened_angles.append(expert_opened_angle)
        init_state_file = os.path.join(first_step_states_path, "state_0.pkl")
        init_state_files.append(init_state_file)
        config_file = os.path.join(first_step_folder, "task_config.yaml")
        config_files.append(config_file)

    return config_files, init_state_files, expert_opened_angles


def load_low_level_policy(exp_dir: str, checkpoint_name: str):
    """Load low-level policy from an experiment directory and checkpoint name.

    Returns a torch module on CUDA in eval mode.
    """
    with hydra.initialize(config_path="diffusion_policy_3d/config"):
        overrides = (
            OmegaConf.load(os.path.join(exp_dir, ".hydra", "overrides.yaml"))
            if os.path.exists(os.path.join(exp_dir, ".hydra", "overrides.yaml"))
            else []
        )
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",
            overrides=list(overrides) if overrides else [],
        )
    cfg = recomposed_config
    workspace = TrainDP3Workspace(OmegaConf.create(dict(cfg)))
    checkpoint_dir = os.path.join(exp_dir, "checkpoints", checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir)
    low_level_policy = deepcopy(workspace.model)
    if low_level_policy is None:
        print("Warning: workspace.model is None")
        return None

    use_ema = False  # Skip EMA for evaluation

    if use_ema:
        ema_model = deepcopy(workspace.ema_model)
        if ema_model is not None:
            low_level_policy = ema_model

    # At this point, low_level_policy is guaranteed to be not None
    low_level_policy.eval()
    try:
        low_level_policy.reset()
    except Exception:
        # some models may not have reset
        pass
    return low_level_policy.to("cuda")


def high_level_policy_infer(
    parallel_input_dict, high_level_policy, output_obj_pcd_only=True, add_one_hot=False
):
    """Run the high-level point-cloud policy and produce a goal point cloud.

    Prepares batched inputs from `parallel_input_dict`, optionally adds a
    one-hot modality encoding, runs `high_level_policy` in no-grad mode, and
    aggregates the per-point predictions using predicted weights into a single
    goal point-cloud tensor.

    Args:
        parallel_input_dict (dict): Batched inputs (point_cloud, gripper_pcd, etc.),
            usually as torch tensors on CUDA.
        high_level_policy (torch.nn.Module): High-level model loaded on CUDA.
        output_obj_pcd_only (bool): If True, exclude gripper points from the
            network outputs and only return the object prediction.

    Returns:
        torch.Tensor: Goal point cloud with shape (B, 1, M, 3).
    """
    with torch.no_grad():
        pointcloud = parallel_input_dict["point_cloud"][:, -1, :, :]
        gripper_pcd = parallel_input_dict["gripper_pcd"][:, -1, :]
        inputs = torch.cat([pointcloud, gripper_pcd], dim=1)

        if add_one_hot:
            # for pointcloud, we add (1, 0)
            # for gripper_pcd, we add (0, 1)
            pointcloud_one_hot = (
                torch.zeros(pointcloud.shape[0], pointcloud.shape[1], 2)
                .float()
                .to(pointcloud.device)
            )
            pointcloud_one_hot[:, :, 0] = 1
            pointcloud_ = torch.cat([pointcloud, pointcloud_one_hot], dim=2)
            gripper_pcd_one_hot = (
                torch.zeros(gripper_pcd.shape[0], gripper_pcd.shape[1], 2)
                .float()
                .to(pointcloud.device)
            )
            gripper_pcd_one_hot[:, :, 1] = 1
            gripper_pcd_ = torch.cat([gripper_pcd, gripper_pcd_one_hot], dim=2)
            inputs = torch.cat([pointcloud_, gripper_pcd_], dim=1)  # B, N+4, 5

        inputs = inputs.to("cuda")
        inputs_ = inputs.permute(0, 2, 1)
        outputs = high_level_policy(inputs_)
        weights = outputs[:, :, -1]  # B, N
        outputs = outputs[:, :, :-1]  # B, N, 12
        if output_obj_pcd_only:
            weights = weights[:, :-4]
            outputs = outputs[:, :-4, :]
            inputs = inputs[:, :-4, :]

        B, N, _ = outputs.shape
        outputs = outputs.view(B, N, 4, 3)

        outputs = outputs + inputs[:, :, :3].unsqueeze(2)
        weights = torch.nn.functional.softmax(weights, dim=1)
        outputs = outputs * weights.unsqueeze(-1).unsqueeze(-1)
        outputs = outputs.sum(dim=1)
        outputs = outputs.unsqueeze(1)


def prepare_env(experiment_folder, experiment_path, all_experiments):
    """Prepare environment configuration files and initial states for evaluation.

    Scans the experiment directory for trial configurations, extracts initial
    state files and expert joint angles, and returns lists of config files,
    initial state files, and expert opened angles.

    Args:
        experiment_folder (str): Base path to the experiment folder.
        experiment_path (str): Path to the specific experiment directory.
        all_experiments (list): List of experiment subdirectories.

    Returns:
        tuple: (config_files, init_state_files, expert_opened_angles)
            - config_files: List of paths to YAML config files.
            - init_state_files: List of paths to initial state files.
            - expert_opened_angles: List of expert joint angles.
    """
    config_files = []
    init_state_files = []
    expert_opened_angles = []

    for exp in all_experiments:
        exp_dir = os.path.join(experiment_path, exp)
        if not os.path.isdir(exp_dir):
            continue

        # Look for config files
        config_candidates = [
            f for f in os.listdir(exp_dir) if f.endswith(".yaml") or f.endswith(".yml")
        ]
        if config_candidates:
            config_file = os.path.join(exp_dir, config_candidates[0])
            config_files.append(config_file)

            # Extract initial state file path from config
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Assume the config has the init state file info
            init_state_file = os.path.join(
                exp_dir, "states", "state_0.pkl"
            )  # Adjust based on actual structure
            init_state_files.append(init_state_file)

            # Extract expert opened angle (placeholder - adjust based on actual data)
            expert_opened_angles.append(1.0)  # Placeholder value

    return config_files, init_state_files, expert_opened_angles


def run_eval_non_parallel(
    cfg,
    low_level_policy,
    high_level_policy,
    save_path,
    exp_beg_idx=0,
    exp_end_idx=1000,
    horizon=150,
    exp_beg_ratio=None,
    exp_end_ratio=None,
    dataset_index=None,
    output_obj_pcd_only=False,
    update_goal_freq=1,
    real_world_camera=False,
    noise_real_world_pcd=False,
    randomize_camera=False,
    add_one_hot=False,
):
    """Evaluate the high-level and low-level policies on configured trials.

    Iterates over datasets described in `cfg.task.env_runner`, constructs
    environments for each trial, runs a closed-loop where the high-level
    policy proposes goals (periodically) and the low-level policy produces
    actions to reach them, steps the environment, and stores statistics and
    GIF visualizations under `save_path`.

    Args:
        cfg: Hydrated experiment configuration object.
        low_level_policy: Low-level controller with `predict_action` method.
        high_level_policy: High-level model that predicts goal point-clouds.
        save_path (str): Directory where results (JSON + GIFs) are saved.
        exp_beg_idx (int): Starting index into available trial configs.
        exp_end_idx (int): One-past-last index into available trial configs.
        horizon (int): Number of environment steps to run per trial.
        exp_beg_ratio/exp_end_ratio (float|None): If provided, interpret begin
            and end indices as ratios of the available configs.
        dataset_index (int|None): If set, override which dataset index to process.
        output_obj_pcd_only (bool): Forwarded to `high_level_policy_infer`.
        update_goal_freq (int): Frequency (in env steps) to recompute the goal.
        real_world_camera, noise_real_world_pcd, randomize_camera (bool): Env flags.

    Side effects:
        Writes per-dataset JSON files named `opened_joint_angles_<dataset_idx>.json`
        and saves GIFs showing the episode rollout under `save_path`.
    """
    ### loop through each test object
    for dataset_idx, (experiment_folder, experiment_name) in enumerate(
        zip(cfg.task.env_runner.experiment_folder, cfg.task.env_runner.experiment_name)
    ):

        if dataset_index is not None:
            dataset_idx = dataset_index

        init_state_files = []
        config_files = []
        experiment_folder = "{}/{}".format(os.environ["PROJECT_DIR"], experiment_folder)
        experiment_name = experiment_name
        experiment_path = os.path.join(experiment_folder, "experiment", experiment_name)
        all_experiments = os.listdir(experiment_path)
        all_experiments = sorted(all_experiments)
        config_files, init_state_files, expert_opened_angles = prepare_env(
            experiment_folder, experiment_path, all_experiments
        )

        opened_joint_angles = {}

        if exp_end_ratio is not None:
            exp_end_idx = int(exp_end_ratio * len(config_files))
        if exp_beg_ratio is not None:
            exp_beg_idx = int(exp_beg_ratio * len(config_files))

        config_files = config_files[exp_beg_idx:exp_end_idx]
        init_state_files = init_state_files[exp_beg_idx:exp_end_idx]
        expert_opened_angles = expert_opened_angles[exp_beg_idx:exp_end_idx]

        ### loop through each test configuration of the object
        for exp_idx, (config_file, init_state_file) in enumerate(
            zip(config_files, init_state_files)
        ):

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            solution_path = [
                x["solution_path"] for x in config if "solution_path" in x
            ][0]
            all_substeps_path = os.path.join(
                os.environ["PROJECT_DIR"], solution_path, "substeps.txt"
            )
            with open(all_substeps_path, "r") as f:
                substeps = f.readlines()
                first_step = substeps[0].lstrip().rstrip()
                task_name = first_step.replace(" ", "_")

            # Construct the real env and run the full evaluation loop (collect frames)
            env = construct_env(
                cfg,
                config_file,
                solution_path,
                task_name,
                init_state_file,
                real_world_camera,
                noise_real_world_pcd,
                randomize_camera,
            )

            obs = env.reset()
            rgb = env.env.render()
            info = env.env._env._get_info()
            all_rgbs = [rgb]
            last_goal = None
            for t in range(1, horizon):
                parallel_input_dict = obs
                parallel_input_dict = dict_apply(
                    parallel_input_dict, lambda x: torch.from_numpy(x).to("cuda")
                )
                for key in obs:
                    parallel_input_dict[key] = parallel_input_dict[key].unsqueeze(0)

                # infer the high-level policy to get the predicted goal
                if t == 1 or t % update_goal_freq == 0:
                    predicted_goal = high_level_policy_infer(
                        parallel_input_dict,
                        high_level_policy,
                        output_obj_pcd_only=output_obj_pcd_only,
                    )
                    last_goal = predicted_goal
                else:
                    predicted_goal = last_goal

                # run the low-level policy to get the robot eef delta transformations
                np_predicted_goal = predicted_goal.detach().to("cpu").numpy()
                predicted_goal = predicted_goal.repeat(1, 2, 1, 1)
                parallel_input_dict["goal_gripper_pcd"] = predicted_goal
                with torch.no_grad():
                    batched_action = low_level_policy.predict_action(
                        parallel_input_dict
                    )
                np_batched_action = dict_apply(
                    batched_action, lambda x: x.detach().to("cpu").numpy()
                )
                np_batched_action = np_batched_action["action"]

                # step the environment with the low-level action
                obs, reward, done, info = env.step(np_batched_action.squeeze(0))
                env.env.goal_gripper_pcd = np_predicted_goal.squeeze(0)[0].reshape(4, 3)
                rgb = env.env.render()
                all_rgbs.append(rgb)

            env.env._env.close()

            # save statistics
            opened_joint_angles[config_file] = {
                "final_door_joint_angle": float(info["opened_joint_angle"][-1]),
                "expert_door_joint_angle": expert_opened_angles[exp_idx],
                "initial_joint_angle": float(info["initial_joint_angle"][-1]),
                "ik_failure": float(info["ik_failure"][-1]),
                "grasped_handle": float(info["grasped_handle"][-1]),
                "exp_idx": exp_idx,
            }

            with open(
                "{}/opened_joint_angles_{}.json".format(save_path, dataset_idx), "w"
            ) as f:
                json.dump(opened_joint_angles, f, indent=4)

            gif_save_exp_name = experiment_folder.split("/")[-2]
            gif_save_folder = "{}/{}".format(save_path, gif_save_exp_name)
            if not os.path.exists(gif_save_folder):
                os.makedirs(gif_save_folder, exist_ok=True)
            gif_save_path = "{}/{}_{}.gif".format(
                gif_save_folder, exp_idx, float(info["improved_joint_angle"][-1])
            )

            # write gif
            try:
                save_numpy_as_gif(np.array(all_rgbs), gif_save_path)
                print(f"Saved gif: {gif_save_path}")
            except Exception as e:
                print(f"Failed to save gif {gif_save_path}: {e}")

    print(f"Evaluation finished. Results in {save_path}")


def load_high_level_policy(ckpt_path: str, add_one_hot: bool = False):
    from weighted_displacement_model.model_invariant import PointNet2_super

    num_class = 13
    input_channel = 5 if add_one_hot else 3
    model = PointNet2_super(num_classes=num_class, input_channel=input_channel).to(
        "cuda"
    )
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--low_level_exp_dir",
        type=str,
        required=True,
        help="Path to low-level experiment directory (exp_dir)",
    )
    parser.add_argument(
        "--low_level_ckpt_name",
        type=str,
        required=True,
        help="Name of low-level checkpoint folder under checkpoints/",
    )
    parser.add_argument(
        "--high_level_ckpt",
        type=str,
        required=True,
        help="Path to high-level checkpoint (pth file)",
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        default="eval_41510",
        help="Name of output eval directory under data/",
    )
    parser.add_argument("--horizon", type=int, default=35)
    parser.add_argument("--update_goal_freq", type=int, default=1)
    parser.add_argument("--output_obj_pcd_only", type=int, default=1)
    parser.add_argument("--add_one_hot_encoding", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Loading low-level policy from {args.low_level_exp_dir} / {args.low_level_ckpt_name}"
    )
    low_level_policy = load_low_level_policy(
        args.low_level_exp_dir, args.low_level_ckpt_name
    )

    print(f"Loading high-level policy from {args.high_level_ckpt}")
    high_level_policy = load_high_level_policy(
        args.high_level_ckpt, add_one_hot=bool(args.add_one_hot_encoding)
    )

    # configure cfg for single model 41510
    # reuse eval_robogen's config loading pattern
    # we need a cfg object; load a default dp3.yaml via hydra then mutate
    with hydra.initialize(config_path="diffusion_policy_3d/config"):
        recomposed_config = hydra.compose(config_name="dp3.yaml")
    cfg = recomposed_config
    OmegaConf.set_struct(cfg, False)

    # point to the single dataset in this repo
    cfg.task.env_runner.experiment_name = ["test_gen_demo"]
    cfg.task.env_runner.experiment_folder = ["data/diverse_objects_all/41510"]
    cfg.task.env_runner.demo_experiment_path = [None]
    cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
    cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"

    save_path = os.path.join(PROJECT_DIR, "data", args.eval_name)
    os.makedirs(save_path, exist_ok=True)

    # dump checkpoint info
    checkpoint_info = {
        "low_level_policy": os.path.join(
            args.low_level_exp_dir, "checkpoints", args.low_level_ckpt_name
        ),
        "low_level_policy_checkpoint": args.low_level_ckpt_name,
        "high_level_policy_checkpoint": args.high_level_ckpt,
    }
    with open(os.path.join(save_path, "checkpoint_info.json"), "w") as f:
        json.dump(checkpoint_info, f, indent=4)

    print("Starting evaluation for model 41510...")
    run_eval_non_parallel(
        cfg,
        low_level_policy,
        high_level_policy,
        save_path,
        exp_beg_idx=0,
        exp_end_idx=1,
        horizon=args.horizon,
        output_obj_pcd_only=args.output_obj_pcd_only,
        update_goal_freq=args.update_goal_freq,
        real_world_camera=False,
        noise_real_world_pcd=False,
        randomize_camera=False,
        add_one_hot=bool(args.add_one_hot_encoding),
    )

    print(f"Evaluation finished. Results in {save_path}")


if __name__ == "__main__":
    main()
