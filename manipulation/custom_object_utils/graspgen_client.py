"""Client for calling GraspGen inside a Docker container.

GraspGen predicts grasps in world frame for the current object state (mesh, joint angles).
This module provides utilities to:
1. Call GraspGen inside a running Docker container
2. Parse the predicted grasps
3. Handle temporary file management for URDF state

Usage:
    from manipulation.custom_object_utils.graspgen_client import (
        predict_grasps_for_state,
        GraspGenConfig,
    )
    
    config = GraspGenConfig(container_name="friendly_galileo")
    grasps, confidences = predict_grasps_for_state(
        urdf_path="/path/to/object.urdf",
        config=config,
    )
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import yaml


def _default_host_graspgen_root() -> str:
    """Default host root expected to be mounted to ``/workspace/data``.

    See scripts/run_graspgen_container.sh.
    """
    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir:
        candidate = Path(project_dir).expanduser().resolve() / "data"
        return str(candidate)

    # Fallback: keep a reasonable default relative to this file.
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / "data").resolve())


@dataclass
class GraspGenConfig:
    """Configuration for GraspGen Docker calls."""
    
    # Docker settings
    container_name: str = "friendly_galileo"
    
    # GraspGen script settings
    gripper_config: str = "GraspGenModels/checkpoints/graspgen_franka_panda.yml"
    num_grasps: int = 400
    num_sample_points: int = 2000
    grasp_threshold: float = -1.0
    use_aabb_sampling: bool = True
    skip_aabb_filter: bool = True
    no_remove_outliers: bool = True
    dump_grasp_locations: int = 0
    aabb_padding: float = 0.2
    
    # Path mapping: host path -> container path
    # The GraspGen container mounts certain directories
    host_graspgen_root: str = field(default_factory=_default_host_graspgen_root)
    container_graspgen_root: str = "/workspace/data"
    
    # Timeout for Docker command
    timeout_seconds: float = 120.0
    # Container working directory where GraspGen scripts live (GraspGen code is mounted at /code)
    container_workdir: str = "/code"


def _host_to_container_path(host_path: str, config: GraspGenConfig) -> str:
    """Convert a host path to a container path."""
    host_path = str(Path(host_path).resolve())
    if host_path.startswith(config.host_graspgen_root):
        relative = host_path[len(config.host_graspgen_root):]
        if relative.startswith("/"):
            relative = relative[1:]
        return f"{config.container_graspgen_root}/{relative}"
    raise ValueError(f"Path {host_path} is not under GraspGen root {config.host_graspgen_root}")


def _container_to_host_path(container_path: str, config: GraspGenConfig) -> str:
    """Convert a container path to a host path."""
    if container_path.startswith(config.container_graspgen_root):
        relative = container_path[len(config.container_graspgen_root):]
        if relative.startswith("/"):
            relative = relative[1:]
        return f"{config.host_graspgen_root}/{relative}"
    raise ValueError(f"Path {container_path} is not under container root {config.container_graspgen_root}")


def load_predicted_grasps_yaml(yaml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load grasps from an Isaac-format YAML file.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (grasps, confidences)
        - grasps: (N, 4, 4) homogeneous matrices
        - confidences: (N,) float scores
    """
    from scipy.spatial.transform import Rotation as R
    
    path = Path(yaml_path)
    if not path.exists():
        return np.empty((0, 4, 4)), np.empty((0,))
    
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    grasps = []
    confidences = []
    
    for key, value in data.get("grasps", {}).items():
        if value is None:
            continue
        conf = value.get("confidence", 0.0)
        if conf is None:
            continue
        confidences.append(conf)
        
        pos = value.get("position", [0, 0, 0])
        ori = value.get("orientation", {"w": 1, "xyz": [0, 0, 0]})
        
        if pos is None or ori is None:
            continue
        
        # Convert quaternion to rotation matrix
        quat = [ori["xyz"][0], ori["xyz"][1], ori["xyz"][2], ori["w"]]
        rot = R.from_quat(quat).as_matrix()
        
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = pos
        grasps.append(mat)
        
    if not grasps:
        return np.empty((0, 4, 4)), np.empty((0,))
        
    return np.array(grasps), np.array(confidences)


def run_graspgen_in_docker(
    urdf_folder_container: str,
    config: GraspGenConfig,
) -> Tuple[bool, str]:
    """Run GraspGen inside the Docker container.
    
    Args:
        urdf_folder_container: Path to URDF folder inside container.
        config: GraspGen configuration.
        
    Returns:
        (success, output/error message)
    """
    # Build the command
    cmd_parts = [
        "python", "scripts/generate_grasps_from_urdf.py",
        "--urdf-root", urdf_folder_container,
        "--gripper-config", config.gripper_config,
        "--num-grasps", str(config.num_grasps),
        "--num-sample-points", str(config.num_sample_points),
    ]
    
    if config.grasp_threshold >= 0:
        cmd_parts.extend(["--grasp-threshold", str(config.grasp_threshold)])
    if config.use_aabb_sampling:
        cmd_parts.append("--use-aabb-sampling")
    if config.skip_aabb_filter:
        cmd_parts.append("--skip-aabb-filter")
    if config.no_remove_outliers:
        cmd_parts.append("--no-remove-outliers")
    if config.dump_grasp_locations > 0:
        cmd_parts.extend(["--dump-grasp-locations", str(config.dump_grasp_locations)])
    if config.aabb_padding > 0:
        cmd_parts.extend(["--aabb-padding", str(config.aabb_padding)])
    
    # Full Docker command
    inner_cmd = " ".join(cmd_parts)
    docker_cmd = [
        "docker", "exec", "-w", config.container_workdir,
        config.container_name,
        "bash", "-c", inner_cmd,
    ]
    
    print(f"[GraspGen] Running: docker exec {config.container_name} bash -c '{inner_cmd}'")
    
    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
        
        if result.returncode != 0:
            error_msg = f"GraspGen failed with return code {result.returncode}\n"
            error_msg += f"STDOUT: {result.stdout}\n"
            error_msg += f"STDERR: {result.stderr}"
            print(f"[GraspGen] Error: {error_msg}")
            return False, error_msg
        
        print(f"[GraspGen] Success: {result.stdout[-500:] if len(result.stdout) > 500 else result.stdout}")
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        error_msg = f"GraspGen timed out after {config.timeout_seconds}s"
        print(f"[GraspGen] {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"GraspGen exception: {e}"
        print(f"[GraspGen] {error_msg}")
        return False, error_msg


def predict_grasps_for_urdf_folder(
    urdf_folder_host: str,
    config: Optional[GraspGenConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict grasps for a URDF in a folder.
    
    The folder must contain:
    - A .urdf file
    - annotation.json (optional but recommended)
    - base_config.yaml (optional, for scale info)
    
    Args:
        urdf_folder_host: Path to URDF folder on host.
        config: GraspGen configuration.
        
    Returns:
        (grasps, confidences) where grasps is (N, 4, 4) and confidences is (N,).
    """
    if config is None:
        config = GraspGenConfig()
    
    urdf_folder_host = str(Path(urdf_folder_host).resolve())
    
    # Convert to container path
    try:
        urdf_folder_container = _host_to_container_path(urdf_folder_host, config)
    except ValueError as e:
        print(f"[GraspGen] Path conversion error: {e}")
        return np.empty((0, 4, 4)), np.empty((0,))
    
    # Run GraspGen
    success, output = run_graspgen_in_docker(urdf_folder_container, config)
    
    if not success:
        return np.empty((0, 4, 4)), np.empty((0,))
    
    # Load results
    predicted_grasps_path = Path(urdf_folder_host) / "predicted_grasps.yml"
    if not predicted_grasps_path.exists():
        print(f"[GraspGen] No predicted_grasps.yml found at {predicted_grasps_path}")
        return np.empty((0, 4, 4)), np.empty((0,))
    
    return load_predicted_grasps_yaml(str(predicted_grasps_path))


def prepare_urdf_with_joint_state(
    original_urdf_path: str,
    joint_states: dict,
    output_folder: Union[str, Path],
    scale: float = 1.0,
) -> str:
    """Prepare a URDF with specific joint states for grasp prediction.
    
    For articulated objects, we need to "bake" the joint state into the mesh
    since GraspGen operates on static meshes.
    
    NOTE: This is a simplified approach that copies the URDF folder.
    For a full solution, you'd need to:
    1. Load the URDF with the specified joint states
    2. Export the meshes in the posed configuration
    3. Create a new URDF with the posed meshes
    
    For now, this just copies the folder and updates base_config.yaml
    to record the intended state.
    
    Args:
        original_urdf_path: Path to original URDF file.
        joint_states: Dict mapping joint names to angles.
        output_folder: Where to create the temporary URDF folder.
        scale: Scale factor for the object.
        
    Returns:
        Path to the prepared URDF folder.
    """
    original_folder = Path(original_urdf_path).parent
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from original folder
    for item in original_folder.iterdir():
        if item.is_file():
            shutil.copy2(item, output_folder_path / item.name)
        elif item.is_dir():
            # Copy subdirectories (meshes, textures, etc.)
            dest = output_folder_path / item.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
    
    # Update base_config.yaml with joint state info
    base_config_path = output_folder_path / "base_config.yaml"
    if base_config_path.exists():
        with open(base_config_path) as f:
            config = yaml.safe_load(f)
        
        # Add joint state info
        for block in config:
            if isinstance(block, dict) and "size" in block:
                block["size"] = scale
                block["joint_states_for_grasp"] = joint_states
                break
        
        with open(base_config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)
    
    return str(output_folder_path)


def predict_grasps_on_demand(
    asset_dir: Union[str, Path],
    config: Optional[GraspGenConfig] = None,
    force_regenerate: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """High-level function to predict grasps for an asset directory.
    
    This is the main entry point for the demo generation pipeline.
    
    Args:
        asset_dir: Path to asset directory (must be under GraspGen folder structure).
        config: GraspGen configuration.
        force_regenerate: If True, regenerate even if predicted_grasps.yml exists.
        
    Returns:
        (grasps, confidences).
    """
    if config is None:
        config = GraspGenConfig()
    
    asset_dir_path = Path(asset_dir).resolve()
    predicted_grasps_path = asset_dir_path / "predicted_grasps.yml"
    
    # Check if we need to regenerate
    if predicted_grasps_path.exists() and not force_regenerate:
        print(f"[GraspGen] Loading existing grasps from {predicted_grasps_path}")
        return load_predicted_grasps_yaml(str(predicted_grasps_path))
    
    # Generate new grasps
    print(f"[GraspGen] Generating new grasps for {asset_dir_path}")
    return predict_grasps_for_urdf_folder(str(asset_dir_path), config)

