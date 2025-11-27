"""Visualization utilities for custom articulated objects.

Provides functions to visualize grasps, coordinate frames, and other debug
information in PyBullet environments using actual visual shapes (visible in camera renders).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None


def _create_cylinder(
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
    color: Sequence[float],
    physics_client: int,
) -> int:
    """Create a cylinder visual shape between two points.
    
    Returns the body ID of the created visual object.
    """
    start = np.array(start)
    end = np.array(end)
    
    # Compute cylinder parameters
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return -1
    
    center = (start + end) / 2
    
    # Compute orientation: align Z-axis with direction
    z_axis = direction / length
    # Find a perpendicular vector for x-axis
    if abs(z_axis[2]) < 0.9:
        x_axis = np.cross([0, 0, 1], z_axis)
    else:
        x_axis = np.cross([0, 1, 0], z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Build rotation matrix and convert to quaternion
    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    from scipy.spatial.transform import Rotation as R
    quat = R.from_matrix(rot_matrix).as_quat()  # x, y, z, w
    
    # Create visual shape
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=length,
        rgbaColor=[color[0], color[1], color[2], 1.0],
        physicsClientId=physics_client,
    )
    
    # Create body with no collision
    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=center.tolist(),
        baseOrientation=quat.tolist(),
        physicsClientId=physics_client,
    )
    
    return body_id


def _create_sphere(
    position: Sequence[float],
    radius: float,
    color: Sequence[float],
    physics_client: int,
) -> int:
    """Create a sphere visual shape at a position."""
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[color[0], color[1], color[2], 1.0],
        physicsClientId=physics_client,
    )
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=list(position),
        physicsClientId=physics_client,
    )
    
    return body_id


def _create_box(
    position: Sequence[float],
    orientation: Sequence[float],
    half_extents: Sequence[float],
    color: Sequence[float],
    physics_client: int,
) -> int:
    """Create a box visual shape."""
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=[color[0], color[1], color[2], 1.0],
        physicsClientId=physics_client,
    )
    
    body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape,
        basePosition=list(position),
        baseOrientation=list(orientation),
        physicsClientId=physics_client,
    )
    
    return body_id


def draw_pose_axes(
    position: Sequence[float],
    orientation: Sequence[float],
    physics_client: int,
    length: float = 0.1,
    radius: float = 0.003,
    **kwargs,  # Accept but ignore legacy params like line_width, lifetime
) -> List[int]:
    """Draw XYZ coordinate axes at a given 6D pose using cylinder shapes.
    
    Uses actual visual shapes visible in camera renders.
    
    Args:
        position: (x, y, z) position in world frame.
        orientation: Quaternion (x, y, z, w) in world frame.
        physics_client: PyBullet physics client ID.
        length: Length of each axis.
        radius: Radius of the axis cylinders.
        
    Returns:
        List of body IDs for the created visual objects.
    """
    if p is None:
        return []
    
    position = list(position)
    body_ids = []
    
    # Define local axis directions and colors (RGB for XYZ)
    axes = [
        ([length, 0, 0], [1, 0, 0]),  # X: red
        ([0, length, 0], [0, 1, 0]),  # Y: green
        ([0, 0, length], [0, 0, 1]),  # Z: blue
    ]
    
    for local_dir, color in axes:
        # Transform to world frame
        world_dir = p.rotateVector(orientation, local_dir)
        end = [position[i] + world_dir[i] for i in range(3)]
        
        body_id = _create_cylinder(position, end, radius, color, physics_client)
        if body_id >= 0:
            body_ids.append(body_id)
    
    # Add a small sphere at origin
    sphere_id = _create_sphere(position, radius * 2, [1, 1, 1], physics_client)
    if sphere_id >= 0:
        body_ids.append(sphere_id)
    
    return body_ids


def draw_gripper(
    position: Sequence[float],
    orientation: Sequence[float],
    physics_client: int,
    gripper_depth: float = 0.1,
    gripper_width: float = 0.08,
    color: Sequence[float] = (1, 0.5, 0),
    radius: float = 0.004,
    **kwargs,  # Accept but ignore legacy params
) -> List[int]:
    """Draw a simplified parallel-jaw gripper using cylinder shapes.
    
    The gripper is visualized as a fork shape with:
    - Approach line along -Z axis
    - Two parallel fingers along Y axis
    
    Args:
        position: Grasp position (x, y, z) in world frame.
        orientation: Grasp orientation quaternion (x, y, z, w) in world frame.
        physics_client: PyBullet physics client ID.
        gripper_depth: Length of the gripper/finger lines.
        gripper_width: Width between finger tips.
        color: RGB color for the gripper.
        radius: Radius of the cylinder lines.
        
    Returns:
        List of body IDs for the created visual objects.
    """
    if p is None:
        return []
    
    position = list(position)
    body_ids = []
    
    # Get world-frame directions using p.rotateVector
    z_world = p.rotateVector(orientation, [0, 0, gripper_depth])
    y_world = p.rotateVector(orientation, [0, gripper_width / 2, 0])
    
    # Wrist position (back along -Z from grasp point)
    wrist = [position[i] - z_world[i] for i in range(3)]
    
    # Finger tip positions
    left_tip = [position[i] - y_world[i] for i in range(3)]
    right_tip = [position[i] + y_world[i] for i in range(3)]
    
    # Wrist corners
    wrist_left = [wrist[i] - y_world[i] for i in range(3)]
    wrist_right = [wrist[i] + y_world[i] for i in range(3)]
    
    # Create cylinders for gripper structure
    lines = [
        (wrist, position),           # Approach line
        (wrist_left, wrist_right),   # Wrist bar
        (wrist_left, left_tip),      # Left finger
        (wrist_right, right_tip),    # Right finger
    ]
    
    for start, end in lines:
        body_id = _create_cylinder(start, end, radius, color, physics_client)
        if body_id >= 0:
            body_ids.append(body_id)
    
    # Add spheres at grasp point and finger tips
    for pt in [position, left_tip, right_tip]:
        sphere_id = _create_sphere(pt, radius * 1.5, color, physics_client)
        if sphere_id >= 0:
            body_ids.append(sphere_id)
    
    return body_ids


def visualize_grasp(
    position: Sequence[float],
    orientation: Sequence[float],
    physics_client: int,
    color: Sequence[float] = (0, 1, 0),
    label: Optional[str] = None,
    show_gripper: bool = True,
    show_axes: bool = True,
    axis_length: float = 0.08,
    gripper_depth: float = 0.1,
    gripper_width: float = 0.08,
    radius: float = 0.004,
    **kwargs,  # Accept but ignore legacy params
) -> List[int]:
    """Visualize a single grasp pose with axes and gripper using visual shapes.
    
    Args:
        position: Grasp position in world frame.
        orientation: Grasp orientation quaternion (x, y, z, w).
        physics_client: PyBullet physics client ID.
        color: RGB color for gripper visualization.
        label: Optional text label (NOTE: text labels not visible in renders).
        show_gripper: Whether to draw gripper.
        show_axes: Whether to draw coordinate axes.
        axis_length: Length of coordinate axes.
        gripper_depth: Gripper visualization depth.
        gripper_width: Gripper finger width.
        radius: Radius of visualization cylinders.
        
    Returns:
        List of body IDs for created visual objects.
    """
    if p is None:
        return []
    
    body_ids = []
    
    # Draw coordinate axes
    if show_axes:
        axis_ids = draw_pose_axes(
            position, orientation, physics_client,
            length=axis_length, radius=radius * 0.75
        )
        body_ids.extend(axis_ids)
    
    # Draw gripper
    if show_gripper:
        gripper_ids = draw_gripper(
            position, orientation, physics_client,
            gripper_depth=gripper_depth, gripper_width=gripper_width,
            color=color, radius=radius
        )
        body_ids.extend(gripper_ids)
    
    # Note: text labels are only visible in GUI mode, not in camera renders
    # We skip adding debug text as it won't be visible
    
    return body_ids


def visualize_grasp_candidates(
    grasps: np.ndarray,
    confidences: np.ndarray,
    object_pos: Sequence[float],
    object_orn: Sequence[float],
    physics_client: int,
    top_n: int = 5,
    show_gripper: bool = True,
    show_axes: bool = True,
    show_labels: bool = True,
    radius: float = 0.004,
    **kwargs,  # Accept but ignore legacy params
) -> dict:
    """Visualize multiple grasp candidates ranked by confidence.
    
    Args:
        grasps: (N, 4, 4) array of grasp poses in object frame.
        confidences: (N,) array of confidence scores.
        object_pos: Object position in world frame.
        object_orn: Object orientation quaternion in world frame.
        physics_client: PyBullet physics client ID.
        top_n: Number of top grasps to visualize.
        show_gripper: Whether to draw gripper.
        show_axes: Whether to draw coordinate axes.
        show_labels: Whether to show confidence labels (GUI only).
        radius: Radius of visualization cylinders.
        
    Returns:
        Dict with world-frame grasp info and body IDs.
    """
    if p is None or len(grasps) == 0:
        return {"grasps": [], "body_ids": []}
    
    from scipy.spatial.transform import Rotation as R
    
    sorted_idx = np.argsort(confidences)[::-1]
    top_idx = sorted_idx[:min(top_n, len(grasps))]
    
    results = {"grasps": [], "body_ids": []}
    
    for rank, idx in enumerate(top_idx):
        grasp_mat = grasps[idx]
        conf = confidences[idx]
        
        # Extract pose from 4x4 matrix
        pos_local = grasp_mat[:3, 3].tolist()
        quat_local = R.from_matrix(grasp_mat[:3, :3]).as_quat().tolist()
        
        # Transform to world frame
        pos_world, quat_world = p.multiplyTransforms(
            object_pos, object_orn, pos_local, quat_local
        )
        
        # Color: green (best) -> red (worst)
        t = rank / max(1, top_n - 1)
        color = (t, 1 - t, 0)
        
        ids = visualize_grasp(
            pos_world, quat_world, physics_client,
            color=color,
            show_gripper=show_gripper, show_axes=show_axes,
            axis_length=0.05, radius=radius
        )
        results["body_ids"].extend(ids)
        
        results["grasps"].append({
            "rank": rank + 1,
            "confidence": float(conf),
            "position": pos_world,
            "orientation": quat_world,
        })
    
    return results


def visualize_motion_target(
    target_pos: Sequence[float],
    target_orn: Sequence[float],
    physics_client: int,
    pre_grasp_offset: float = 0.1,
    show_pre_grasp: bool = True,
    radius: float = 0.005,
    **kwargs,  # Accept but ignore legacy params
) -> dict:
    """Visualize motion planning target with pre-grasp pose using visual shapes.
    
    Shows:
    1. Final grasp pose (green)
    2. Pre-grasp pose (yellow) offset along approach direction
    3. Approach trajectory (blue line)
    
    Args:
        target_pos: Target grasp position in world frame.
        target_orn: Target orientation quaternion in world frame.
        physics_client: PyBullet physics client ID.
        pre_grasp_offset: Distance to offset pre-grasp.
        show_pre_grasp: Whether to show pre-grasp pose.
        radius: Radius of visualization cylinders.
        
    Returns:
        Dict with poses and body IDs.
    """
    if p is None:
        return {}
    
    body_ids = []
    target_pos = list(target_pos)
    
    # Draw grasp target (green)
    grasp_ids = visualize_grasp(
        target_pos, target_orn, physics_client,
        color=(0, 1, 0),
        show_gripper=True, show_axes=True,
        radius=radius
    )
    body_ids.extend(grasp_ids)
    
    result = {
        "grasp_pos": target_pos,
        "grasp_orn": list(target_orn),
        "body_ids": body_ids,
    }
    
    if show_pre_grasp:
        # Pre-grasp: offset back along -Z (approach direction)
        approach = p.rotateVector(target_orn, [0, 0, -pre_grasp_offset])
        pre_pos = [target_pos[i] + approach[i] for i in range(3)]
        
        # Draw pre-grasp (yellow)
        pre_ids = visualize_grasp(
            pre_pos, target_orn, physics_client,
            color=(1, 1, 0),
            show_gripper=True, show_axes=False,
            gripper_depth=0.08, radius=radius
        )
        body_ids.extend(pre_ids)
        
        # Draw approach line (blue cylinder)
        line_id = _create_cylinder(
            pre_pos, target_pos, radius * 0.8, [0, 0.5, 1], physics_client
        )
        if line_id >= 0:
            body_ids.append(line_id)
        
        result["pre_grasp_pos"] = pre_pos
    
    return result


def clear_visualization(body_ids: List[int], physics_client: int) -> None:
    """Remove visualization objects by their body IDs."""
    if p is None:
        return
    for body_id in body_ids:
        try:
            p.removeBody(body_id, physicsClientId=physics_client)
        except:
            pass


# Legacy alias for backwards compatibility
def clear_debug_items(debug_ids: List[int], physics_client: int) -> None:
    """Remove debug visualizations by ID (legacy - handles both debug items and bodies)."""
    if p is None:
        return
    for item_id in debug_ids:
        try:
            # Try removing as body first (new visual shapes)
            p.removeBody(item_id, physicsClientId=physics_client)
        except:
            try:
                # Fall back to debug item removal (legacy)
                p.removeUserDebugItem(item_id, physicsClientId=physics_client)
            except:
                pass

