import os
import hydra
import torch
from omegaconf import OmegaConf
import json
import argparse
from copy import deepcopy

import eval_robogen as er

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_level_exp_dir', type=str, required=True)
    parser.add_argument('--low_level_ckpt_name', type=str, required=True)
    parser.add_argument("--high_level_ckpt_name", type=str, required=True)
    parser.add_argument("--eval_exp_name", type=str, default="color_0025_eval")
    parser.add_argument('--output_obj_pcd_only', type=int, default=1)
    parser.add_argument("--update_goal_freq", type=int, default=1)
    parser.add_argument("--noise_real_world_pcd", type=int, default=0)
    parser.add_argument("--randomize_camera", type=int, default=0)
    parser.add_argument("--real_world_camera", type=int, default=0)
    parser.add_argument('--add_one_hot_encoding', type=int, default=0)
    parser.add_argument('--dry_run', action='store_true', help='Only validate configs and dataset layout, do not load models or run evaluation')
    args = parser.parse_args()

    # load low-level policy using the same procedure as eval_robogen
    exp_dir = args.low_level_exp_dir
    checkpoint_name = args.low_level_ckpt_name
    hydra_dir = os.path.join(exp_dir, ".hydra")
    overrides_path = os.path.join(hydra_dir, "overrides.yaml")
    composed_cfg_path = os.path.join(hydra_dir, "config.yaml")

    # Prefer the fully composed config saved by hydra (safer). If not available,
    # fall back to trying to re-compose from overrides. If neither exists, compose
    # the default dp3.yaml.
    if os.path.exists(composed_cfg_path):
        try:
            recomposed_config = OmegaConf.load(composed_cfg_path)
            print(f"Loaded composed config from {composed_cfg_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load composed config at {composed_cfg_path}: {e}")
    else:
        # fallback to composing with hydra (may fail if overrides contain complex values)
        try:
            with hydra.initialize(config_path='diffusion_policy_3d/config'):
                if os.path.exists(overrides_path):
                    # load the overrides file and convert to a plain python list of strings
                    loaded = OmegaConf.load(overrides_path)
                    if isinstance(loaded, (list, tuple)):
                        overrides_list = [str(x) for x in loaded]
                    else:
                        # sometimes the file may be a dict; stringify to single-entry list
                        overrides_list = [str(loaded)]
                    recomposed_config = hydra.compose(
                        config_name="dp3.yaml",
                        overrides=overrides_list,
                    )
                else:
                    print(f"Warning: overrides file not found at {overrides_path}.")
                    print("Composing default 'dp3.yaml' (no experiment overrides).")
                    recomposed_config = hydra.compose(config_name="dp3.yaml")
        except Exception as e:
            raise RuntimeError(
                f"Failed to compose Hydra config for low-level policy from '{exp_dir}'. "
                f"Look for '{overrides_path}' or a valid hydra composed config at '{composed_cfg_path}'. "
                f"Original error: {e}"
            )

    cfg = recomposed_config
    # If the user requested a dry-run, skip model loading and evaluation.
    if args.dry_run:
        print("DRY RUN: composed low-level config loaded (no models will be instantiated)")
        try:
            print("--- low-level config preview ---")
            print(OmegaConf.to_yaml(cfg))
        except Exception:
            pass

        # Patch the cfg to use our custom dataset root and the experiment group name we created
        # Temporarily disable struct mode so we can add keys if they don't exist
        OmegaConf.set_struct(cfg, False)
        cfg.task.env_runner.experiment_name = ['color_0025_v2_demo']
        cfg.task.env_runner.experiment_folder = [
            'data/color_0025_0_v2'
        ]
        cfg.task.env_runner.demo_experiment_path = [None]
        OmegaConf.set_struct(cfg, True)

        print("Patched cfg.task.env_runner with:")
        print("  experiment_folder=", cfg.task.env_runner.experiment_folder)
        print("  experiment_name=", cfg.task.env_runner.experiment_name)

        # validate dataset paths relative to PROJECT_DIR
        project_dir = os.environ.get('PROJECT_DIR', os.getcwd())
        exp_folder = cfg.task.env_runner.experiment_folder[0]
        exp_name = cfg.task.env_runner.experiment_name[0]
        experiment_path = os.path.join(project_dir, exp_folder, 'experiment', exp_name)
        print(f"Checking experiment_path: {experiment_path}")
        if os.path.exists(experiment_path):
            print("Found experiment path. Listing contents:")
            print(os.listdir(experiment_path))
        else:
            print("Experiment path not found. You may need to set PROJECT_DIR or create the dataset layout.")

        # create the output folder for the (hypothetical) evaluation run
        save_path = os.path.join('data', args.eval_exp_name)
        os.makedirs(save_path, exist_ok=True)
        checkpoint_dir = "{}/checkpoints/{}".format(args.low_level_exp_dir, args.low_level_ckpt_name)
        checkpoint_info = {
            "low_level_policy": checkpoint_dir,
            "low_level_policy_checkpoint": args.low_level_ckpt_name,
            "high_level_policy_checkpoint": args.high_level_ckpt_name,
        }
        checkpoint_info.update(args.__dict__)
        with open(os.path.join(save_path, "checkpoint_info.json"), "w") as f:
            json.dump(checkpoint_info, f, indent=4)

        print("Dry-run complete. Created checkpoint_info.json at", save_path)
        return
    workspace = er.TrainDP3Workspace(cfg)
    checkpoint_dir = "{}/checkpoints/{}".format(exp_dir, checkpoint_name)
    workspace.load_checkpoint(path=checkpoint_dir, )
    low_level_policy = deepcopy(workspace.model)
    if workspace.cfg.training.use_ema:
        low_level_policy = deepcopy(workspace.ema_model)
    low_level_policy.eval()
    # reset() may not exist on all model wrappers; ignore if absent
    try:
        low_level_policy.reset()
    except Exception:
        pass

    # Move to GPU if available; if CUDA isn't available this will raise and should be handled by user
    low_level_policy = low_level_policy.to('cuda')

    # load high-level policy
    load_model_path = args.high_level_ckpt_name
    num_class = 13
    input_channel = 5 if args.add_one_hot_encoding else 3
    from weighted_displacement_model.model_invariant import PointNet2_super
    high_level_policy = PointNet2_super(num_classes=num_class, input_channel=input_channel).to("cuda")
    high_level_policy.load_state_dict(torch.load(load_model_path))
    high_level_policy.eval()

    # Patch the cfg to use our custom dataset root and the experiment group name we created
    OmegaConf.set_struct(cfg, False)
    cfg.task.env_runner.experiment_name = ['color_0025_v2_demo']
    cfg.task.env_runner.experiment_folder = [
        'data/color_0025_0_v2'
    ]
    cfg.task.env_runner.demo_experiment_path = [None]
    OmegaConf.set_struct(cfg, True)

    # dump evaluation configuration
    save_path = "data/{}".format(args.eval_exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_info = {
        "low_level_policy": checkpoint_dir,
        "low_level_policy_checkpoint": checkpoint_name,
        "high_level_policy_checkpoint": args.high_level_ckpt_name,
    }
    checkpoint_info.update(args.__dict__)
    with open("{}/checkpoint_info.json".format(save_path), "w") as f:
        json.dump(checkpoint_info, f, indent=4)

    cfg.task.env_runner.observation_mode = "act3d_goal_displacement_gripper_to_object"
    cfg.task.dataset.observation_mode = "act3d_goal_displacement_gripper_to_object"

    # run the evaluation (shorter horizon by default for debugging)
    er.run_eval_non_parallel(
            cfg, low_level_policy, high_level_policy,
            save_path,
            horizon=35,
            exp_beg_idx=0,
            exp_end_idx=25,
            output_obj_pcd_only=args.output_obj_pcd_only,
            update_goal_freq=args.update_goal_freq,
            real_world_camera=args.real_world_camera,
            noise_real_world_pcd=args.noise_real_world_pcd,
            randomize_camera=args.randomize_camera,
    )

if __name__ == '__main__':
    main()
