#!/usr/bin/env python3
"""
Visualize articulated object manipulation trajectories using viser.

This script loads multiple trajectories from different object poses, 
transforms all end-effector trajectories into the object's local frame,
and visualizes them together with the object mesh at the origin.

Key insight: Each trajectory has the object at a different world pose.
To visualize all trajectories together, we transform EEF positions from
world frame to object-local frame, so all trajectories appear relative
to the object placed at the origin.
"""

import argparse
import os
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R
import yaml

from manipulation.utils.env_utils import parse_center

# Color palette for different trajectories
TRAJECTORY_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring green
    (255, 0, 128),    # Pink
    (128, 255, 0),    # Lime
    (0, 128, 255),    # Sky blue
    (255, 128, 128),  # Light red
    (128, 255, 128),  # Light green
    (128, 128, 255),  # Light blue
    (255, 255, 128),  # Light yellow
]


def load_state(state_path: str) -> dict:
    """Load a single state pickle file."""
    with open(state_path, 'rb') as f:
        return pickle.load(f)


def get_successful_trajectories(experiment_path: str) -> List[str]:
    """Get paths to all successful trajectory directories."""
    successful = []
    for d in sorted(os.listdir(experiment_path)):
        dp = os.path.join(experiment_path, d)
        if not os.path.isdir(dp):
            continue
        if not os.path.exists(os.path.join(dp, 'all.gif')):
            continue
        states_dir = os.path.join(dp, 'states')
        if not os.path.exists(states_dir):
            continue
        state_files = [f for f in os.listdir(states_dir) if f.endswith('.pkl')]
        if len(state_files) > 10:
            successful.append(dp)
    return successful


def get_object_pose(state: dict, object_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract object position and orientation from state.
    Returns (position, quaternion_xyzw).
    """
    pos = np.array(state['object_base_position'][object_name])
    orn = np.array(state['object_base_orientation'][object_name])
    return pos, orn


def world_to_object_frame(
    point_world: np.ndarray,
    obj_pos: np.ndarray,
    obj_orn_xyzw: np.ndarray
) -> np.ndarray:
    """
    Transform a point from world frame to object's local frame.
    
    The object's local frame has the object at the origin.
    Given:
    - point_world: point in world coordinates
    - obj_pos: object position in world frame
    - obj_orn_xyzw: object orientation as quaternion (xyzw format)
    
    To transform to object frame:
    1. Translate: point_rel = point_world - obj_pos
    2. Rotate by inverse of object rotation: point_obj = R_obj^T @ point_rel
    """
    # Get rotation matrix from quaternion
    R_obj = R.from_quat(obj_orn_xyzw).as_matrix()
    
    # Transform: first translate, then rotate by inverse
    point_rel = point_world - obj_pos
    point_obj = R_obj.T @ point_rel
    
    return point_obj


def load_trajectory(traj_dir: str, object_name: str, subsample: int = 5) -> dict:
    """
    Load a full trajectory from a directory.
    
    Returns a dict with:
    - states: list of state dicts
    - object_pose: (pos, orn) tuple - constant throughout trajectory
    - object_scale: float - scale factor for the object
    - robot_joints: list of joint angle arrays
    """
    states_dir = os.path.join(traj_dir, 'states')
    state_files = sorted(
        [f for f in os.listdir(states_dir) if f.endswith('.pkl')],
        key=lambda x: int(x.replace('state_', '').replace('.pkl', ''))
    )
    
    # Subsample for visualization
    state_files = state_files[::subsample]
    
    states = []
    robot_joints = []
    object_pose = None
    object_scale = 1.0
    
    for i, sf in enumerate(state_files):
        state = load_state(os.path.join(states_dir, sf))
        states.append(state)
        
        # Get object pose and scale from first state (they're constant)
        if i == 0:
            object_pose = get_object_pose(state, object_name)
            # Get object scale from state
            if 'object_sizes' in state and object_name in state['object_sizes']:
                object_scale = state['object_sizes'][object_name]
        
        robot_joint_angles = state['object_joint_angle_dicts']['robot']
        robot_joints.append(np.array(robot_joint_angles))
    
    return {
        'states': states,
        'object_pose': object_pose,
        'object_scale': object_scale,
        'robot_joints': robot_joints,
        'dir': traj_dir
    }


def compute_all_eef_positions(traj_data: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute EEF positions and orientations for all timesteps using PyBullet FK.
    Returns (positions, orientations) where orientations are xyzw quaternions.
    """
    import pybullet as p
    import pybullet_data
    
    # Connect to PyBullet once
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Load the Panda robot - try multiple possible paths
    script_dir = os.path.dirname(os.path.realpath(__file__))
    possible_paths = [
        os.path.join(script_dir, "manipulation/assets/panda_bullet/panda.urdf"),
        os.path.join(script_dir, "bullet3/examples/pybullet/gym/pybullet_data/franka_panda/panda.urdf"),
        os.path.join(script_dir, "manipulation/assets/franka_mobile/panda.urdf"),
    ]
    
    robot_urdf = None
    for path in possible_paths:
        if os.path.exists(path):
            robot_urdf = path
            break
    
    if robot_urdf is None:
        raise FileNotFoundError(f"Could not find panda.urdf in any of: {possible_paths}")
    
    robot_id = p.loadURDF(robot_urdf, [0, 0, 0], useFixedBase=True)
    
    eef_positions = []
    eef_orientations = []
    
    for robot_joints in traj_data['robot_joints']:
        # Set joint angles (first 7 are arm joints)
        for i in range(min(7, len(robot_joints))):
            p.resetJointState(robot_id, i, robot_joints[i])
        
        # Also set finger joints if available
        if len(robot_joints) > 9:
            p.resetJointState(robot_id, 9, robot_joints[9])   # finger 1
            p.resetJointState(robot_id, 10, robot_joints[10]) # finger 2
        
        # Get end effector link state
        # For Panda: link 8 is panda_hand (end effector)
        eef_link = 8
        link_state = p.getLinkState(robot_id, eef_link)
        eef_positions.append(np.array(link_state[0]))
        eef_orientations.append(np.array(link_state[1]))
    
    p.disconnect()
    
    return eef_positions, eef_orientations


def load_object_mesh(urdf_path: Path, scale: float = 1.0) -> trimesh.Trimesh:
    """
    Load object mesh from URDF directory.
    Combines all mesh files into a single mesh.
    """
    mesh_dir = urdf_path.parent / "mesh"
    combined_mesh = None
    
    if mesh_dir.exists():
        for mesh_file in mesh_dir.glob("*.obj"):
            try:
                mesh = trimesh.load(str(mesh_file), force='mesh')
                if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
                    # Apply scale
                    mesh.vertices *= scale
                    if combined_mesh is None:
                        combined_mesh = mesh
                    else:
                        combined_mesh = trimesh.util.concatenate([combined_mesh, mesh])
            except Exception as e:
                print(f"  Warning: Could not load {mesh_file.name}: {e}")
    
    return combined_mesh


def load_root_link_rotation(urdf_path: Path) -> np.ndarray:
    """Extract the fixed rotation between the URDF root link and its first child."""
    if not urdf_path.exists():
        return np.eye(3)
    try:
        tree = ET.parse(urdf_path)
    except Exception as exc:
        print(f"Warning: failed to parse {urdf_path}: {exc}")
        return np.eye(3)

    robot = tree.getroot()
    link_names = {link.get("name", "") for link in robot.findall("link")}
    child_links = set()
    for joint in robot.findall("joint"):
        child_elem = joint.find("child")
        if child_elem is None:
            continue
        link_name = child_elem.attrib.get("link")
        if link_name:
            child_links.add(link_name)
    root_candidates = [name for name in link_names if name and name not in child_links]
    if not root_candidates:
        return np.eye(3)
    root_link = root_candidates[0]

    for joint in robot.findall("joint"):
        parent_elem = joint.find("parent")
        if parent_elem is None or parent_elem.get("link") != root_link:
            continue
        origin = joint.find("origin")
        rpy_str = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        try:
            rpy = [float(val) for val in rpy_str.split()]
        except ValueError:
            rpy = [0.0, 0.0, 0.0]
        return R.from_euler('xyz', rpy).as_matrix()

    return np.eye(3)


def load_base_rotation(config_path: Path, object_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the canonical object rotation (quaternion + matrix) from base_config."""
    if not config_path.exists():
        print(f"Warning: config file not found at {config_path}; using identity rotation")
        return np.eye(3), None
    try:
        config_data = yaml.safe_load(config_path.open("r", encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: failed to parse {config_path}: {exc}; using identity rotation")
        return np.eye(3), None
    target_name = object_name.lower()
    base_quat = None
    if isinstance(config_data, list):
        for block in config_data:
            name = str(block.get("name", "")).lower()
            if name != target_name:
                continue
            orientation_val = block.get("orientation")
            if orientation_val is not None:
                if isinstance(orientation_val, str):
                    base_quat = parse_center(orientation_val)
                else:
                    base_quat = np.array(orientation_val, dtype=float)
            else:
                euler_val = block.get("euler")
                if euler_val is not None:
                    if isinstance(euler_val, str):
                        euler_np = parse_center(euler_val)
                    else:
                        euler_np = np.array(euler_val, dtype=float)
                    base_quat = R.from_euler('xyz', euler_np).as_quat()
            break
    if base_quat is None:
        return np.eye(3), None
    base_quat = np.asarray(base_quat, dtype=float)
    if base_quat.shape[0] != 4:
        print(f"Warning: orientation for {object_name} has unexpected shape {base_quat.shape}; using identity")
        return np.eye(3), None
    norm = np.linalg.norm(base_quat)
    if norm > 0:
        base_quat = base_quat / norm
    base_rot = R.from_quat(base_quat).as_matrix()
    return base_rot, base_quat


def main():
    parser = argparse.ArgumentParser(description="Visualize manipulation trajectories in object frame")
    parser.add_argument(
        "--experiment-path",
        type=str,
        default="data/custom_objects/sim_microwave_good/experiment/traj_100",
        help="Path to experiment directory containing trajectory folders"
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="data/custom_objects/sim_microwave_good/color_0025_0.urdf",
        help="Path to object URDF"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="data/custom_objects/sim_microwave_good/base_config.yaml",
        help="Path to base_config.yaml for the object"
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="sim microwave good",
        help="Name of the object in the state dict"
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=16,
        help="Maximum number of trajectories to visualize"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=5,
        help="Subsample factor for trajectory points"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server"
    )
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent
    experiment_path = project_root / args.experiment_path
    urdf_path = project_root / args.urdf_path
    config_path = project_root / args.config_path
    
    print(f"Loading trajectories from: {experiment_path}")
    print(f"Object URDF: {urdf_path}")
    print(f"Config path: {config_path}")

    base_rot_mat, base_quat = load_base_rotation(config_path, args.object_name)
    if base_quat is not None:
        base_euler = R.from_quat(base_quat).as_euler('xyz', degrees=True)
        print(f"Loaded base rotation quaternion: {base_quat}")
        print(f"Base rotation euler (deg): {base_euler}")
    else:
        print("No base rotation found; defaulting to identity")

    root_rot_mat = load_root_link_rotation(urdf_path)
    root_euler = R.from_matrix(root_rot_mat).as_euler('xyz', degrees=True)
    print(f"Root joint rotation (deg): {root_euler}")

    # Find successful trajectories
    traj_dirs = get_successful_trajectories(str(experiment_path))
    print(f"Found {len(traj_dirs)} successful trajectories")
    
    if len(traj_dirs) == 0:
        print("No successful trajectories found!")
        return
    
    # Limit number of trajectories
    traj_dirs = traj_dirs[:args.max_trajectories]
    print(f"Visualizing {len(traj_dirs)} trajectories")
    
    # Load all trajectories
    print("Loading trajectory data...")
    trajectories = []
    for i, traj_dir in enumerate(traj_dirs):
        print(f"  Loading trajectory {i+1}/{len(traj_dirs)}: {os.path.basename(traj_dir)}")
        traj_data = load_trajectory(traj_dir, args.object_name, subsample=args.subsample)
        trajectories.append(traj_data)
    
    # Print object poses for verification
    print("\nObject poses across trajectories:")
    for i, traj in enumerate(trajectories):
        pos, orn = traj['object_pose']
        scale = traj['object_scale']
        print(f"  Traj {i}: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
              f"orn=[{orn[0]:.3f}, {orn[1]:.3f}, {orn[2]:.3f}, {orn[3]:.3f}], scale={scale:.4f}")
    
    # Get the common object scale (should be same across all trajectories)
    object_scale = trajectories[0]['object_scale']
    print(f"\nObject scale: {object_scale}")
    
    # Compute EEF positions for all trajectories in world frame
    print("\nComputing end-effector positions...")
    all_eef_world = []
    for i, traj in enumerate(trajectories):
        print(f"  Computing FK for trajectory {i+1}/{len(trajectories)}")
        eef_pos, eef_orn = compute_all_eef_positions(traj)
        all_eef_world.append(eef_pos)
    
    # Transform EEF positions from world frame to object-local frame (mesh coordinates)
    # The transformation pipeline:
    # 1. Translate by -obj_pos (move to object origin)
    # 2. Rotate by R_obj^T (align with object base frame)
    # 3. Apply base_config rotation to align with canonical mesh orientation
    # 4. Divide by scale (convert to mesh coordinates)
    print("\nTransforming trajectories to object mesh frame...")
    all_eef_object_frame = []
    for i, (traj, eef_positions) in enumerate(zip(trajectories, all_eef_world)):
        obj_pos, obj_orn = traj['object_pose']
        scale = traj['object_scale']
        
        transformed = []
        for eef_pos in eef_positions:
            # Transform from world frame to object-local frame
            eef_in_obj = world_to_object_frame(eef_pos, obj_pos, obj_orn)
            # Align with canonical base rotation and convert to mesh coordinates
            eef_aligned = base_rot_mat @ eef_in_obj
            eef_mesh = eef_aligned / scale
            transformed.append(eef_mesh)
        
        all_eef_object_frame.append(transformed)
        
        # Print first and last points for verification
        if i < 3:
            print(f"  Traj {i}: start_mesh={transformed[0]}, end_mesh={transformed[-1]}")
    
    # Start viser server
    print(f"\nStarting viser server on port {args.port}...")
    server = viser.ViserServer(port=args.port)
    
    # Configure lighting
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    
    # Add ground plane grid at z=0
    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1, position=(0, 0, 0))
    
    # Add world coordinate frame at origin
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.005)
    
    # Load and add object mesh at origin
    print("Loading object mesh...")
    
    # Try to load mesh using trimesh
    mesh = load_object_mesh(urdf_path)
    if mesh is not None:
        # Print mesh bounds for verification
        mesh_vertices = np.array(mesh.vertices, dtype=np.float32)
        mesh_vertices = (root_rot_mat @ mesh_vertices.T).T
        mesh_vertices = (base_rot_mat @ mesh_vertices.T).T.astype(np.float32)
        bounds = np.array([
            mesh_vertices.min(axis=0),
            mesh_vertices.max(axis=0)
        ])
        print(f"  Mesh bounds: min={bounds[0]}, max={bounds[1]}")
        print(f"  Mesh center: {mesh_vertices.mean(axis=0)}")
        
        # Add mesh at origin (object-local frame, mesh coordinates)
        server.scene.add_mesh_simple(
            "/object/mesh",
            vertices=mesh_vertices,
            faces=np.array(mesh.faces, dtype=np.uint32),
            color=(180, 180, 180),
            opacity=0.9,
        )
        print(f"  Loaded object mesh with {len(mesh.vertices)} vertices")
        
        # Add approximate handle location marker (for microwave, handle is typically on front)
        # Based on analysis: handle center ~[-0.01, 0.48, -0.01] in mesh coords
        handle_approx = np.array([-0.01, 0.48, -0.01])
        handle_approx = base_rot_mat @ (root_rot_mat @ handle_approx)
        server.scene.add_icosphere(
            "/object/handle_approx",
            radius=0.03,
            color=(255, 255, 0),  # Yellow
            position=tuple(handle_approx.tolist()),
        )
        print(f"  Added approximate handle marker at {handle_approx}")
    else:
        print("  Warning: Could not load object mesh")
        # Add a placeholder box
        server.scene.add_box(
            "/object/placeholder",
            color=(180, 180, 180),
            dimensions=(0.3, 0.3, 0.3),
            position=(0, 0, 0.15),
            opacity=0.5,
        )
    
    # Visualize trajectories in object frame
    print("\nAdding trajectory visualizations...")
    for i, eef_in_obj in enumerate(all_eef_object_frame):
        color_idx = i % len(TRAJECTORY_COLORS)
        color = TRAJECTORY_COLORS[color_idx]
        
        points = np.array(eef_in_obj)
        
        if len(points) < 2:
            continue
        
        # Add trajectory as connected line segments using splines
        # Group points into segments for smoother visualization
        segment_size = 5
        for j in range(0, len(points) - 1, segment_size):
            end_idx = min(j + segment_size + 1, len(points))
            segment_points = points[j:end_idx]
            if len(segment_points) >= 2:
                server.scene.add_spline_catmull_rom(
                    f"/trajectories/traj_{i}/segment_{j}",
                    positions=segment_points.astype(np.float32),
                    color=color,
                    line_width=3.0,
                )
        
        # Add start marker (green)
        server.scene.add_icosphere(
            f"/trajectories/traj_{i}/start",
            radius=0.015,
            color=(0, 255, 0),
            position=tuple(points[0].tolist()),
        )
        
        # Add end marker (red)
        server.scene.add_icosphere(
            f"/trajectories/traj_{i}/end",
            radius=0.015,
            color=(255, 0, 0),
            position=tuple(points[-1].tolist()),
        )
        
        # Add intermediate point markers (every few points)
        for j in range(1, len(points) - 1, 5):
            server.scene.add_icosphere(
                f"/trajectories/traj_{i}/point_{j}",
                radius=0.006,
                color=color,
                position=tuple(points[j].tolist()),
            )
    
    # Add GUI controls
    with server.gui.add_folder("Trajectory Info"):
        server.gui.add_markdown(f"**Trajectories:** {len(trajectories)}")
        server.gui.add_markdown(f"**Subsample:** {args.subsample}")
        server.gui.add_markdown("Object is at origin (0,0,0)")
        server.gui.add_markdown("All trajectories transformed to object frame")
    
    # Add visibility toggles for individual trajectories
    with server.gui.add_folder("Trajectory Visibility"):
        trajectory_checkboxes = []
        for i in range(len(all_eef_object_frame)):
            cb = server.gui.add_checkbox(f"Trajectory {i}", initial_value=True)
            trajectory_checkboxes.append(cb)
            
            # Create closure for callback
            def make_callback(traj_idx):
                def callback(_):
                    # Toggle visibility of all elements in this trajectory
                    visible = trajectory_checkboxes[traj_idx].value
                    # Note: viser doesn't have direct visibility toggle, 
                    # but we can track this for future reference
                return callback
            
            cb.on_update(make_callback(i))
    
    print(f"\n{'='*60}")
    print(f"Viser server running at http://localhost:{args.port}")
    print(f"{'='*60}")
    print("\nVisualization shows:")
    print("  - Object mesh at origin (0,0,0)")
    print("  - All EEF trajectories in object's local coordinate frame")
    print("  - Green spheres = trajectory start")
    print("  - Red spheres = trajectory end")
    print("  - If manipulation is correct, trajectories should show")
    print("    the gripper approaching and pulling the handle")
    print("\nPress Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
