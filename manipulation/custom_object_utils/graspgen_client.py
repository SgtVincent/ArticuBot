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
import uuid
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import yaml


def _default_host_graspgen_root() -> str:
    """Pick a host directory that is *actually visible* inside the GraspGen container.

    In this repo's default setup, the running GraspGen container mounts:
    - ``<...>/GraspGen/GraspGenModels`` -> ``/models``
    - ``<...>/GraspGen`` -> ``/code``

    So the safest default is the host ``GraspGenModels`` folder.
    """
    override = os.environ.get("GRASPGEN_HOST_ROOT")
    if override:
        override_path = Path(override).expanduser().resolve()
        if override_path.exists():
            return str(override_path)

    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir:
        project_dir_path = Path(project_dir).expanduser().resolve()

        # Common layouts:
        # 1) <...>/ArticuBot (PROJECT_DIR) and <...>/GraspGen/GraspGenModels (sibling)
        # 2) <...>/ArticuBot (PROJECT_DIR) and <...>/ArticuBot/GraspGen/GraspGenModels (nested)
        for candidate in [
            (project_dir_path.parent / "GraspGen" / "GraspGenModels").resolve(),
            (project_dir_path / "GraspGen" / "GraspGenModels").resolve(),
        ]:
            if candidate.exists():
                return str(candidate)

        sibling_models = (project_dir_path.parent / "GraspGen" / "GraspGenModels").resolve()
        if sibling_models.exists():
            return str(sibling_models)
        # Fallback: some setups mount data directly (not the default in this repo).
        candidate = (project_dir_path / "data").resolve()
        if candidate.exists():
            return str(candidate)

    # Fallback: keep a reasonable default relative to this file.
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in [
        (repo_root.parent / "GraspGen" / "GraspGenModels").resolve(),
        (repo_root / "GraspGen" / "GraspGenModels").resolve(),
    ]:
        if candidate.exists():
            return str(candidate)
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
    # In the default container used by this repo, GraspGenModels is mounted to /models.
    host_graspgen_root: str = field(default_factory=_default_host_graspgen_root)
    container_graspgen_root: str = "/models"
    
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

    return load_predicted_grasps_yaml_data(data)


def load_predicted_grasps_yaml_data(data: Optional[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Load grasps from an already-parsed Isaac-format YAML dict."""
    from scipy.spatial.transform import Rotation as R

    if not data:
        return np.empty((0, 4, 4)), np.empty((0,))
    
    grasps: list[np.ndarray] = []
    confidences: list[float] = []
    
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


def _docker_exec(config: GraspGenConfig, shell_cmd: str, *, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """Run a shell command inside the GraspGen container."""
    docker_cmd = [
        "docker",
        "exec",
        "-w",
        config.container_workdir,
        config.container_name,
        "bash",
        "-lc",
        shell_cmd,
    ]
    return subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
        timeout=timeout if timeout is not None else config.timeout_seconds,
    )


def _container_path_exists(config: GraspGenConfig, container_path: str) -> bool:
    """Check if a path exists inside the container."""
    result = _docker_exec(config, f"test -d {shlex_quote(container_path)} && echo OK || echo NO", timeout=10.0)
    return result.returncode == 0 and "OK" in (result.stdout or "")


def shlex_quote(s: str) -> str:
    """Small, local shlex.quote to avoid importing shlex at module import time."""
    import shlex

    return shlex.quote(s)


def _docker_cp_dir_contents_to_container(config: GraspGenConfig, src_dir_host: str, dst_dir_container: str) -> bool:
    """Copy directory contents from host -> container without relying on bind mounts."""
    src_dir = str(Path(src_dir_host).resolve())
    if not Path(src_dir).is_dir():
        print(f"[GraspGen] docker cp source is not a directory: {src_dir}")
        return False
    # Ensure destination exists.
    mkdir_res = _docker_exec(config, f"mkdir -p {shlex_quote(dst_dir_container)}", timeout=30.0)
    if mkdir_res.returncode != 0:
        print(f"[GraspGen] Failed to mkdir in container: {mkdir_res.stderr}")
        return False

    docker_cp = [
        "docker",
        "cp",
        f"{src_dir}/.",
        f"{config.container_name}:{dst_dir_container}",
    ]
    try:
        cp_res = subprocess.run(docker_cp, capture_output=True, text=True, timeout=300.0)
        if cp_res.returncode != 0:
            print(f"[GraspGen] docker cp failed: {cp_res.stderr or cp_res.stdout}")
            return False
        return True
    except Exception as e:
        print(f"[GraspGen] docker cp exception: {e}")
        return False


def _docker_cp_file_to_container(config: GraspGenConfig, src_file_host: str, dst_file_container: str) -> bool:
    """Copy a single file from host -> container."""
    src_file = str(Path(src_file_host).resolve())
    if not Path(src_file).is_file():
        print(f"[GraspGen] docker cp source is not a file: {src_file}")
        return False

    dst_parent = str(Path(dst_file_container).parent)
    mkdir_res = _docker_exec(config, f"mkdir -p {shlex_quote(dst_parent)}", timeout=30.0)
    if mkdir_res.returncode != 0:
        print(f"[GraspGen] Failed to mkdir in container: {mkdir_res.stderr}")
        return False

    docker_cp = [
        "docker",
        "cp",
        src_file,
        f"{config.container_name}:{dst_file_container}",
    ]
    try:
        cp_res = subprocess.run(docker_cp, capture_output=True, text=True, timeout=120.0)
        if cp_res.returncode != 0:
            print(f"[GraspGen] docker cp file failed: {cp_res.stderr or cp_res.stdout}")
            return False
        return True
    except Exception as e:
        print(f"[GraspGen] docker cp file exception: {e}")
        return False


def _container_dir_has_urdf(config: GraspGenConfig, container_dir: str) -> bool:
    """Best-effort check whether a container directory contains at least one .urdf file."""
    res = _docker_exec(
        config,
        f"test -d {shlex_quote(container_dir)} && find {shlex_quote(container_dir)} -maxdepth 2 -name '*.urdf' -print -quit",
        timeout=10.0,
    )
    return res.returncode == 0 and bool((res.stdout or "").strip())


def _read_predicted_grasps_yaml_from_container_dir(
    config: GraspGenConfig, root_dir_container: str
) -> Optional[dict]:
    """Read predicted_grasps.yml from a container directory.

    GraspGen sometimes writes the output YAML next to the discovered URDF file,
    which may be nested under ``root_dir_container``.
    """
    # Fast path: root dir.
    cat_res = _docker_exec(
        config,
        f"cat {shlex_quote(root_dir_container + '/predicted_grasps.yml')}",
        timeout=30.0,
    )
    if cat_res.returncode == 0 and (cat_res.stdout or "").strip():
        return yaml.safe_load(cat_res.stdout)

    # Fallback: search for the first match.
    find_res = _docker_exec(
        config,
        (
            f"find {shlex_quote(root_dir_container)} -maxdepth 6 -name predicted_grasps.yml -print "
            "2>/dev/null | head -n 1"
        ),
        timeout=30.0,
    )
    candidate = (find_res.stdout or "").strip()
    if not candidate:
        return None

    cat_res = _docker_exec(config, f"cat {shlex_quote(candidate)}", timeout=30.0)
    if cat_res.returncode != 0 or not (cat_res.stdout or "").strip():
        return None
    return yaml.safe_load(cat_res.stdout)


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

    # Preferred path: host<->container mapping (bind mounts), but only if the
    # container can actually see the mapped path.
    urdf_folder_container: Optional[str] = None
    try:
        urdf_folder_container = _host_to_container_path(urdf_folder_host, config)
    except ValueError:
        urdf_folder_container = None

    use_docker_cp_fallback = True
    if urdf_folder_container is not None:
        try:
            use_docker_cp_fallback = not _container_path_exists(config, urdf_folder_container)
        except Exception:
            use_docker_cp_fallback = True

    if use_docker_cp_fallback:
        # Fallback: Use a persistent cache directory in the container.
        # We only need to update base_config.yaml per attempt (joint state / scale),
        # so avoid copying large mesh folders on every call.
        # NOTE: urdf_folder_host is typically a per-attempt temp folder (e.g. _tmp_xxx).
        # Key off the URDF identity instead so caching actually hits across attempts.
        urdf_candidates = sorted(Path(urdf_folder_host).glob("*.urdf"))
        if urdf_candidates:
            urdf_path = urdf_candidates[0]
            try:
                urdf_hash = hashlib.sha1(urdf_path.read_bytes()).hexdigest()[:10]
            except Exception:
                urdf_hash = "nohash"
            key_material = f"{urdf_path.stem}:{urdf_hash}"
        else:
            key_material = urdf_folder_host

        cache_key = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:10]
        cache_container_dir = f"/tmp/articubot_graspgen_cache_{cache_key}"

        base_config_host = str(Path(urdf_folder_host) / "base_config.yaml")
        base_config_container = f"{cache_container_dir}/base_config.yaml"

        cache_ready = _container_dir_has_urdf(config, cache_container_dir)
        if not cache_ready:
            if urdf_folder_container is None:
                print(
                    f"[GraspGen] Using docker-cp cache (populate). "
                    f"host_graspgen_root={config.host_graspgen_root} urdf_folder_host={urdf_folder_host}"
                )
            else:
                print(
                    f"[GraspGen] Using docker-cp cache (populate). "
                    f"urdf_folder_container={urdf_folder_container}"
                )

            # First time: copy the full folder contents.
            if not _docker_cp_dir_contents_to_container(config, urdf_folder_host, cache_container_dir):
                return np.empty((0, 4, 4)), np.empty((0,))
        else:
            # Fast path: only refresh base_config.yaml (joint state / scale).
            if Path(base_config_host).is_file():
                if not _docker_cp_file_to_container(config, base_config_host, base_config_container):
                    # If the small file copy fails, fall back to a full refresh.
                    if not _docker_cp_dir_contents_to_container(config, urdf_folder_host, cache_container_dir):
                        return np.empty((0, 4, 4)), np.empty((0,))
            else:
                # If base_config.yaml is missing, refresh everything.
                if not _docker_cp_dir_contents_to_container(config, urdf_folder_host, cache_container_dir):
                    return np.empty((0, 4, 4)), np.empty((0,))

        try:
            success, _ = run_graspgen_in_docker(cache_container_dir, config)
            if not success:
                return np.empty((0, 4, 4)), np.empty((0,))

            data = _read_predicted_grasps_yaml_from_container_dir(config, cache_container_dir)
            if not data:
                print(
                    f"[GraspGen] No predicted_grasps.yml found inside container under {cache_container_dir}"
                )
                return np.empty((0, 4, 4)), np.empty((0,))

            return load_predicted_grasps_yaml_data(data)
        finally:
            # Keep cache_container_dir for reuse.
            pass

    # Run GraspGen via mapped path.
    assert urdf_folder_container is not None
    success, _ = run_graspgen_in_docker(urdf_folder_container, config)
    if not success:
        return np.empty((0, 4, 4)), np.empty((0,))

    predicted_grasps_path = Path(urdf_folder_host) / "predicted_grasps.yml"
    if predicted_grasps_path.exists():
        return load_predicted_grasps_yaml(str(predicted_grasps_path))

    # If the bind mount exists but the host file isn't visible, try reading the
    # file directly from the container.
    data = _read_predicted_grasps_yaml_from_container_dir(config, urdf_folder_container)
    if not data:
        print(f"[GraspGen] No predicted_grasps.yml found at {predicted_grasps_path}")
        return np.empty((0, 4, 4)), np.empty((0,))

    return load_predicted_grasps_yaml_data(data)


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

    original_folder_resolved = original_folder.resolve()
    output_folder_resolved = output_folder_path.resolve()
    
    # Copy all files from original folder.
    # IMPORTANT: output_folder can live under original_folder (common for v2 assets
    # where we create temp folders under <asset>/urdf/). Avoid copying the temp
    # folder into itself, and skip previous temp folders to prevent recursive
    # nesting that can confuse downstream GraspGen URDF discovery.
    for item in original_folder.iterdir():
        try:
            item_resolved = item.resolve()
        except Exception:
            item_resolved = (original_folder_resolved / item.name)

        if item_resolved == output_folder_resolved:
            continue

        if item.is_dir() and item.name.startswith("_tmp_"):
            continue

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

