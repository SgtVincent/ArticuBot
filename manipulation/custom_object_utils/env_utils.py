"""Custom-environment helpers for ArticuBot's bespoke articulated assets.

These utilities mirror the logic in ``manipulation.utils.env_utils`` but relax the
assumptions about PartNet-Mobility directory layouts. They are used by the
``articulated_custom_object`` environment to load mobility metadata, handle
point clouds, and other annotations produced by the custom-object pipeline.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import numpy as np
import pybullet as p

from manipulation.utils.env_utils import get_joint_id_from_name, parse_center
from manipulation.utils.defaults import default_config, data_dir
from manipulation.utils.object_utils import down_load_single_object, preprocess_urdf
from objaverse_utils.utils import partnet_mobility_dict
import copy
import os.path as osp
from manipulation.utils.geometry_utils import sample_point_inside_triangle
from manipulation.utils.handle_utiils import (
    find_nearest_point_on_line,
    load_obj,
    rotate_point_around_axis,
)

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path.cwd()))


def _derive_asset_root(urdf_path: str, asset_dir_hint: Optional[str]) -> Path:
    """Return the directory that stores mobility/json metadata for an object."""
    if asset_dir_hint:
        hint_path = Path(asset_dir_hint)
        if hint_path.exists():
            return hint_path
    urdf_parent = Path(urdf_path).parent
    dataset_idx = str(urdf_parent).find("data/dataset")
    if dataset_idx != -1:
        rel = str(urdf_parent)[dataset_idx:]
        return (PROJECT_DIR / rel).resolve()
    return urdf_parent.resolve()


def _load_mobility_info(asset_root: Path) -> List[dict]:
    mobility_path = asset_root / "mobility_v2.json"
    if not mobility_path.exists():
        raise FileNotFoundError(f"mobility_v2.json not found under {asset_root}")
    with mobility_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_handle_points(
    handle_id: int,
    target_object: str,
    asset_root: Path,
    scaling: float,
    handle_pts_override: Optional[str],
) -> np.ndarray:
    """Load handle point cloud either from override .npy or OBJ mesh."""
    if handle_pts_override:
        override_path = Path(handle_pts_override)
        if override_path.exists():
            pts = np.load(override_path)
            return pts * scaling  # Apply scaling to NPY override

    # Prefer pre-computed point clouds when available.
    # These are commonly produced by the custom-object pipeline and are reliable
    # even when OBJ exports contain only vertices (no faces).
    npy_candidates = [
        asset_root / "parts_render" / f"{handle_id}{target_object}_points.npy",
        asset_root / "parts_render" / f"{handle_id}{target_object}.npy",
        asset_root / "parts_render" / f"{handle_id}handle_points.npy",
    ]
    for npy_path in npy_candidates:
        if npy_path.exists():
            pts = np.load(npy_path)
            return pts * scaling

    # Fall back to OBJ mesh.
    handle_obj_path = asset_root / "parts_render" / f"{handle_id}{target_object}.obj"
    if handle_obj_path.exists():
        handle_pts, handle_faces = load_obj(str(handle_obj_path))
        handle_pts = handle_pts * scaling
        added_points = []
        for face in handle_faces:
            v1, v2, v3 = face
            v1 = handle_pts[v1 - 1]
            v2 = handle_pts[v2 - 1]
            v3 = handle_pts[v3 - 1]
            a = np.linalg.norm(v1 - v2)
            b = np.linalg.norm(v2 - v3)
            c = np.linalg.norm(v3 - v1)
            s = (a + b + c) / 2
            temp = max(0, s * (s - a) * (s - b) * (s - c))
            surface = np.sqrt(temp)
            num_points = int(surface * 1e6)
            num_points = np.clip(num_points, 0, 5)
            added_points.extend([sample_point_inside_triangle(v1, v2, v3) for _ in range(num_points)])
        if added_points:
            handle_pts = np.concatenate((handle_pts, np.array(added_points)), axis=0)
        return handle_pts

    raise FileNotFoundError(f"Could not find handle points for {handle_id}{target_object} in {asset_root}/parts_render")


def get_handle_pos_custom(
    simulator,
    obj_name: str,
    *,
    asset_dir_hint: Optional[str] = None,
    handle_pts_override: Optional[str] = None,
    return_median: bool = True,
    handle_pts_obj_frame: Optional[Sequence[np.ndarray]] = None,
    mobility_info: Optional[List[dict]] = None,
    return_info: bool = False,
    target_object: str = "handle",
):
    """Variant of ``get_handle_pos`` that works for custom asset layouts."""
    obj_name = obj_name.lower()
    scaling = simulator.simulator_sizes[obj_name]
    obj_id = simulator.urdf_ids[obj_name]

    asset_root = _derive_asset_root(simulator.urdf_paths[obj_name], asset_dir_hint)
    if mobility_info is None:
        mobility_info = _load_mobility_info(asset_root)

    ret_handle_pt_list: List[np.ndarray] = []
    ret_joint_idx_list: List[int] = []
    axis_world_list = []
    axis_end_world_list = []
    all_handle_pts_object_frame: List[np.ndarray] = []
    handle_idx = 0

    for joint_info in mobility_info:
        all_parts = [part["name"] for part in joint_info["parts"]]
        if target_object in all_parts:
            all_ids = [part["id"] for part in joint_info["parts"]]
            index = all_parts.index(target_object)
            handle_id = all_ids[index]
            joint_name = f"joint_{joint_info['id']}"
            parent_joint_name = f"joint_{joint_info['parent']}"
            
            joint_data = joint_info['jointData']
            axis_body = np.array(joint_data["axis"]["origin"]) * scaling
            axis_dir_body = np.array(joint_data["axis"]["direction"])
            joint_limit = joint_data["limit"]
            if joint_limit['a'] > joint_limit['b']:
                axis_dir_body = -axis_dir_body

            joint_idx = get_joint_id_from_name(simulator, obj_name, joint_name)
            if joint_idx is None:
                # Try without underscore
                joint_name = "joint{}".format(joint_info["id"])
                joint_idx = get_joint_id_from_name(simulator, obj_name, joint_name)

            parent_joint_idx = get_joint_id_from_name(simulator, obj_name, parent_joint_name)
            if parent_joint_idx is None:
                # Try without underscore
                parent_joint_name = "joint{}".format(joint_info["parent"])
                parent_joint_idx = get_joint_id_from_name(simulator, obj_name, parent_joint_name)
            
            if parent_joint_idx is None and joint_info["parent"] == -1:
                 # If parent is -1 (base), use -1 as index? 
                 # But getLinkState expects a valid link index. 
                 # Usually base link has index -1 in some contexts, but getLinkState might behave differently.
                 # However, for now let's assume if it's not found it might be the base if parent is -1.
                 # But wait, if parent is -1, we are at the root joint.
                 pass

            if parent_joint_idx is None:
                 # Fallback or error reporting
                 print(f"WARNING: Could not find parent joint {parent_joint_name} for object {obj_name}")
                 # We might crash in getLinkState, so let's try to avoid it or let it crash with a better message
            
            parent_link_state = p.getLinkState(obj_id, parent_joint_idx, physicsClientId=simulator.id)
            link_urdf_world_pos, link_urdf_world_orn = parent_link_state[0], parent_link_state[1]
            
            T_body_to_world = np.eye(4)
            T_body_to_world[:3, :3] = np.array(p.getMatrixFromQuaternion(link_urdf_world_orn)).reshape(3, 3)
            T_body_to_world[:3, 3] = link_urdf_world_pos
            
            axis_world = T_body_to_world[:3, :3] @ axis_body + T_body_to_world[:3, 3]   
            axis_pt2_body = np.array(axis_body) + axis_dir_body
            axis_end_world = T_body_to_world[:3, :3] @ axis_pt2_body + T_body_to_world[:3, 3]
            axis_dir_world = axis_end_world - axis_world

            if handle_pts_obj_frame is None:
                handle_pts = _load_handle_points(handle_id, target_object, asset_root, scaling, handle_pts_override)
                all_handle_pts_object_frame.append(handle_pts)
            else:
                handle_pts = handle_pts_obj_frame[handle_idx]

            handle_points_world = T_body_to_world[:3, :3] @ handle_pts.T + T_body_to_world[:3, 3].reshape(3, 1)
            if return_median:
                handle_point_median = np.median(handle_points_world, axis=1)
            else:
                handle_point_median = handle_points_world.T

            project_on_rotation_axis = find_nearest_point_on_line(axis_world, axis_end_world, handle_point_median)
            
            joint_info_pb = p.getJointInfo(obj_id, joint_idx, physicsClientId=simulator.id)
            joint_type = joint_info_pb[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                rotation_angle = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0]
                rotated_handle_pt_local = rotate_point_around_axis(handle_point_median - project_on_rotation_axis, axis_dir_world, rotation_angle)
                rotated_handle_pt = project_on_rotation_axis + rotated_handle_pt_local
            elif joint_type == p.JOINT_PRISMATIC:
                translation = p.getJointState(obj_id, joint_idx, physicsClientId=simulator.id)[0]
                rotated_handle_pt = handle_point_median + axis_dir_world * translation
            else:
                rotated_handle_pt = handle_point_median

            if return_median:
                ret_handle_pt_list.append(rotated_handle_pt.flatten())
            else:
                ret_handle_pt_list.append(rotated_handle_pt)
            ret_joint_idx_list.append(joint_idx)
            axis_world_list.append(axis_world)
            axis_end_world_list.append(axis_end_world)
            handle_idx += 1

    if return_info:
        return ret_handle_pt_list, ret_joint_idx_list, all_handle_pts_object_frame, mobility_info
    
    return ret_handle_pt_list, ret_joint_idx_list, axis_world_list, axis_end_world_list


def parse_config_custom_object(config, obj_id=None, use_vhacd=True, default_initial_joint_angle=0, custom_asset_dir=None):
    """
    Parse configuration dictionary to extract object URDF paths and properties.
    Supports both standard PartNet-Mobility objects and custom objects with flexible paths.
    """
    urdf_paths = []
    urdf_sizes = []
    urdf_locations = []
    urdf_orientations = []
    urdf_names = []
    urdf_types = []
    urdf_on_tables = []
    urdf_movables = []
    urdf_crop_sizes = []
    use_table = False
    articulated_joint_angles = {}
    spatial_relationships = []
    distractor_config_path = None

    robot_initial_joint_angles = [0.0, 0.0, 0.0, -0.4, 0.0, 0.4, 0.0]
    initial_finger_angle = default_initial_joint_angle

    for obj in config:
        if "use_table" in obj.keys():
            use_table = obj['use_table']

        if "set_joint_angle_object_name" in obj.keys():
            new_obj = copy.deepcopy(obj)
            new_obj.pop('set_joint_angle_object_name')
            articulated_joint_angles[obj['set_joint_angle_object_name']] = new_obj

        if "spatial_relationships" in obj.keys():
            spatial_relationships = obj['spatial_relationships']

        if 'task_name' in obj.keys() or 'task_description' in obj.keys():
            continue

        if "distractor_config_path" in obj.keys():
            distractor_config_path = obj['distractor_config_path']

        if 'initial_joint_angles' in obj.keys():
            initial_joint_angles = obj['initial_joint_angles']
            initial_joint_angles = parse_center(initial_joint_angles)
            robot_initial_joint_angles = initial_joint_angles

        if 'initial_finger_angle' in obj.keys():
            initial_finger_angle = obj['initial_finger_angle']
            initial_finger_angle = float(initial_finger_angle)

        if "type" not in obj.keys():
            if "urdf_path" in obj.keys():
                obj['type'] = 'urdf'
            else:
                continue
        
        if obj['type'] == 'mesh':
            if 'uid' not in obj.keys():
                continue
            if obj_id is None:
                uid = obj['uid'][np.random.randint(len(obj['uid']))]
            else:
                uid = obj['uid'][obj_id]
                
            urdf_file_path = osp.join("objaverse_utils/data/obj", "{}".format(uid), "material.urdf")
            if not os.path.exists(urdf_file_path):
                down_load_single_object(name=obj['lang'], uids=[uid])
            
            if not use_vhacd:
                new_urdf_file_path = urdf_file_path.replace("material.urdf", "material_non_vhacd.urdf")
                if not os.path.exists(new_urdf_file_path):
                    new_urdf_lines = []
                    with open(urdf_file_path, 'r') as f:
                        urdf_lines = f.readlines()
                    for line in urdf_lines:
                        if 'vhacd' in line:
                            new_line = line.replace("_vhacd", "")
                            new_urdf_lines.append(new_line)
                        else:
                            new_urdf_lines.append(line)
                    with open(new_urdf_file_path, 'w') as f:
                        f.writelines(new_urdf_lines)
                urdf_file_path = new_urdf_file_path
                
            urdf_paths.append(urdf_file_path)
            urdf_types.append('mesh')
            urdf_movables.append(True)
           
        elif obj['type'] == 'urdf':
            if 'reward_asset_path' in obj.keys():
                obj_path = obj['reward_asset_path']
            else:
                try:
                    category = obj['lang']
                    possible_obj_path = partnet_mobility_dict[category]
                except:
                    category = obj['name']
                    if category == 'Computer display':
                        category = 'Display'
                    possible_obj_path = partnet_mobility_dict[category]
                
                obj_path = np.random.choice(possible_obj_path)
                if category == 'Toaster':
                    obj_path = str(103486)
                if category == 'Microwave':
                    obj_path = str(7310)
                if category == "Oven":
                    obj_path = str(101808)
                if category == 'Refrigerator':
                    obj_path = str(10638)
            
            # Determine if it's a custom object or standard PartNet-Mobility
            is_custom = 'custom_objects' in obj_path or osp.isabs(obj_path) or 'urdf_path' in obj or (custom_asset_dir and not obj_path.startswith('/'))
            
            if is_custom:
                # Resolve base path
                base_path = None
                
                # 1. Try absolute path
                if osp.isabs(obj_path) and osp.exists(obj_path):
                    base_path = obj_path
                
                # 2. Try relative to custom_asset_dir
                if base_path is None and custom_asset_dir:
                    candidate = osp.abspath(osp.join(custom_asset_dir, obj_path))
                    if osp.exists(candidate):
                        base_path = candidate

                # 3. Try relative to data_dir (standard location)
                if base_path is None:
                    candidate = osp.abspath(osp.join(f"{data_dir}", obj_path))
                    if osp.exists(candidate):
                        base_path = candidate

                # 4. Try custom_objects logic
                if base_path is None:
                    # Extract the object name from the path if it looks like a path
                    # e.g. "../data/color_0025_0" -> "color_0025_0"
                    obj_name_from_path = osp.basename(obj_path)
                    
                    # Try data/custom_objects/<name>
                    candidate = osp.join(f"{data_dir}/custom_objects", obj_name_from_path)
                    if osp.exists(candidate):
                        base_path = candidate
                    
                    # Try data/custom_objects/<name>_v2
                    if base_path is None:
                        candidate_v2 = osp.join(f"{data_dir}/custom_objects", f"{obj_name_from_path}_v2")
                        if osp.exists(candidate_v2):
                            base_path = candidate_v2

                if base_path is None:
                     # If we still haven't found it, default to the first candidate for error reporting
                     if custom_asset_dir:
                         base_path = osp.abspath(osp.join(custom_asset_dir, obj_path))
                     else:
                         base_path = osp.abspath(osp.join(f"{data_dir}", obj_path))
                
                base_path = osp.abspath(base_path)

                # Determine URDF filename
                urdf_file_path = None
                possible_urdf_names = ['mobility.urdf', 'mobility_vhacd.urdf']
                
                explicit_urdf = obj.get('urdf_path')
                if explicit_urdf:
                    possible_urdf_names.insert(0, explicit_urdf)
                    if not osp.isabs(explicit_urdf):
                         # If explicit_urdf is relative, try joining with base_path
                         test_path = osp.join(base_path, explicit_urdf)
                         if osp.exists(test_path):
                             urdf_file_path = test_path
                
                if urdf_file_path is None:
                    # Try finding in base_path
                    for urdf_name in possible_urdf_names:
                        test_path = osp.join(base_path, urdf_name)
                        if osp.exists(test_path):
                            urdf_file_path = test_path
                            break
                
                if urdf_file_path is None:
                     # Fallback: search for any .urdf in base_path
                     if osp.exists(base_path):
                        candidates = [f for f in os.listdir(base_path) if f.endswith('.urdf')]
                        if candidates:
                            urdf_file_path = osp.join(base_path, candidates[0])

                if urdf_file_path is None:
                     # If still not found, and we have an explicit URDF, assume it might be relative to CWD or absolute
                     if explicit_urdf and osp.exists(explicit_urdf):
                         urdf_file_path = explicit_urdf
                
                if urdf_file_path is None:
                     raise FileNotFoundError(f"Could not find URDF file for object {obj.get('name', 'unknown')} in {base_path} or with path {explicit_urdf}")

                urdf_paths.append(urdf_file_path)

            else:
                # Standard PartNet-Mobility path
                urdf_file_path = osp.join(f"{data_dir}/dataset", obj_path, "mobility.urdf")
                if use_vhacd:
                    new_urdf_file_path = urdf_file_path.replace("mobility.urdf", "mobility_vhacd.urdf")
                    if not osp.exists(new_urdf_file_path):
                        new_urdf_file_path = preprocess_urdf(urdf_file_path)
                    urdf_paths.append(new_urdf_file_path)
                else:
                    urdf_paths.append(urdf_file_path)

            urdf_types.append('urdf')
            urdf_movables.append(obj.get('movable', False))

        urdf_sizes.append(obj['size'])
        urdf_locations.append(parse_center(obj['center']))
        ori = obj.get('orientation', [0, 0, 0, 1])
        if type(ori) == str:
            ori = parse_center(ori)
        urdf_orientations.append(ori)
        
        if 'name' not in obj:
            obj['name'] = obj.get('lang', 'custom_object')

        urdf_names.append(obj['name'])
        urdf_on_tables.append(obj.get('on_table', False))
        urdf_crop_sizes.append(obj.get('is_crop_size', True))

    return urdf_paths, urdf_sizes, urdf_locations, urdf_orientations, urdf_names, urdf_types, urdf_on_tables, use_table, urdf_crop_sizes, \
        articulated_joint_angles, spatial_relationships, distractor_config_path, urdf_movables, robot_initial_joint_angles, initial_finger_angle
