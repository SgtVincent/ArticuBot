"""Utilities for preparing custom articulated objects from URDF + affordance annotations.

These helpers convert a raw URDF folder plus an affordance annotation (3D points that
identify the graspable region) into the assets expected by the ArticuBot pipeline:

- ``mobility_v2.json`` capturing joint metadata
- ``parts_render/<id><handle_name>.obj`` containing a handle mesh
- basic geometric metadata (center, euler, size) used for config generation

The functions are intentionally standalone so that scripts such as
``generate_base_config_from_urdf_custom.py`` and ``gen_demo_custom.py`` can reuse the
same logic without duplicating code.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from manipulation.utils.handle_utiils import load_obj

try:  # Optional dependency for tighter size estimation
    import open3d as o3d  # type: ignore
except Exception:  # pragma: no cover - optional
    o3d = None


def _ensure_path(path_like: os.PathLike | str) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def parse_float_list(text: Optional[str], length: int = 3, default: Optional[Sequence[float]] = None) -> List[float]:
    """Parse a whitespace-separated list of floats."""
    if text is None:
        if default is not None:
            return list(default)
        return [0.0] * length
    parts = text.strip().split()
    out: List[float] = []
    for idx, part in enumerate(parts):
        if idx >= length:
            break
        try:
            out.append(float(part))
        except ValueError:
            out.append(0.0)
    while len(out) < length:
        out.append(0.0)
    return out


def find_first_urdf(folder: os.PathLike | str) -> Path:
    """Return the first ``*.urdf`` file inside ``folder``."""
    folder_path = _ensure_path(folder)
    for entry in sorted(folder_path.iterdir()):
        if entry.suffix.lower() == ".urdf" and entry.is_file():
            return entry
    raise FileNotFoundError(f"No URDF file found under {folder_path}")


def collect_mesh_info(base_dir: Path, urdf_path: Path, link_name: str):
    """Collect mesh file paths and transforms referenced by a URDF link."""
    try:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()
    except Exception:
        return []
    
    mesh_info = []
    for link in root.findall('link'):
        if link.get('name') == link_name:
            for visual in link.findall('visual'):
                geometry = visual.find('geometry')
                if geometry is not None:
                    mesh = geometry.find('mesh')
                    if mesh is not None:
                        filename = mesh.get('filename')
                        if not filename:
                            continue
                        if filename.startswith('package://'):
                            filename = filename[len('package://') :]
                            parts = filename.split('/', 1)
                            filename = parts[1] if len(parts) > 1 else parts[0]
                        
                        p = (base_dir / filename).resolve(strict=False)
                        
                        origin = visual.find('origin')
                        xyz = [0.0, 0.0, 0.0]
                        rpy = [0.0, 0.0, 0.0]
                        if origin is not None:
                            xyz_str = origin.get('xyz')
                            if xyz_str:
                                xyz = [float(x) for x in xyz_str.split()]
                            rpy_str = origin.get('rpy')
                            if rpy_str:
                                rpy = [float(x) for x in rpy_str.split()]
                        
                        mesh_info.append((p, xyz, rpy))
    return mesh_info


def collect_mesh_paths(urdf_root: Path, tree: ET.ElementTree) -> List[Path]:
    paths: List[Path] = []
    for mesh in tree.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf_root / mesh_path).resolve()
        paths.append(mesh_path)
    return paths


def ensure_support_files(asset_dir: Path, args, handle_name: str) -> None:
    annotation_path = args.annotation_path or (asset_dir / "annotation.json")
    if annotation_path.exists():
        ensure_handle_mesh_from_annotation(
            asset_dir,
            handle_name,
            annotation_path,
            part_id=args.handle_part_id,
            padding=args.handle_padding,
            overwrite=args.overwrite_support,
        )
    mobility_path = asset_dir / "mobility_v2.json"
    if (args.prepare_mobility or not mobility_path.exists()) and args.urdf_path:
        joint_info = extract_joint_metadata(args.urdf_path, joint_name=args.joint_name)
        ensure_mobility_file(asset_dir, joint_info, overwrite=True)

    # If annotation exists, compute a handle point cloud from mesh vertices inside bbox
    if args.annotation_path is not None and args.urdf_path is not None:
        try:
            ann_pts = load_annotation_points(args.annotation_path)
        except Exception:
            ann_pts = None
        if ann_pts is not None:
            min_corner = np.min(ann_pts, axis=0) - args.handle_dilate
            max_corner = np.max(ann_pts, axis=0) + args.handle_dilate

            print(f"Extracting handle points for {handle_name} from {args.urdf_path}")
            mesh_infos = collect_mesh_info(args.urdf_path.parent, args.urdf_path, handle_name)
            print(f"Found {len(mesh_infos)} meshes for link {handle_name}")
            
            handle_pts_list = []
            for mesh_path, xyz, rpy in mesh_infos:
                if not mesh_path.exists():
                    continue
                try:
                    v, _ = load_obj(mesh_path)
                except Exception:
                    continue
                
                # Apply transform
                rot = R.from_euler('xyz', rpy).as_matrix()
                v = (rot @ v.T).T + np.array(xyz)

                mask = np.all((v >= min_corner) & (v <= max_corner), axis=1)
                if np.any(mask):
                    handle_pts_list.append(v[mask])
            if handle_pts_list:
                handle_pts = np.vstack(handle_pts_list)
                parts_render = asset_dir / "parts_render"
                parts_render.mkdir(exist_ok=True)
                pts_file = parts_render / f"{args.handle_part_id}{handle_name}_points.npy"
                print(f"Saving handle points to {pts_file}")
                np.save(pts_file, handle_pts)
                obj_out = parts_render / f"{args.handle_part_id}{handle_name}.obj"
                print(f"Saving handle obj to {obj_out}")
                with obj_out.open("w", encoding="ascii") as f:
                    for p in handle_pts:
                        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            else:
                print("No handle points found inside annotation bbox!")


def estimate_size_from_meshes(mesh_paths: Sequence[Path]) -> Optional[float]:
    if not mesh_paths or o3d is None:
        return None
    min_corner = np.array([math.inf, math.inf, math.inf])
    max_corner = np.array([-math.inf, -math.inf, -math.inf])
    any_valid = False
    for mesh_path in mesh_paths:
        if not mesh_path.exists():
            continue
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))  # type: ignore[attr-defined]
        except Exception:
            continue
        if mesh.is_empty():  # type: ignore[attr-defined]
            continue
        bbox = mesh.get_axis_aligned_bounding_box()  # type: ignore[attr-defined]
        min_corner = np.minimum(min_corner, bbox.get_min_bound())
        max_corner = np.maximum(max_corner, bbox.get_max_bound())
        any_valid = True
    if not any_valid:
        return None
    extent = max_corner - min_corner
    extent = np.maximum(extent, 1e-6)
    size = float(np.cbrt(np.prod(extent)))
    return size if math.isfinite(size) else None


def compute_center_and_euler(tree: ET.ElementTree) -> Tuple[np.ndarray, np.ndarray]:
    positions: List[List[float]] = []
    eulers: List[List[float]] = []
    for tag_name in ("visual", "collision", "joint"):
        for elem in tree.iter(tag_name):
            origin = elem.find("origin")
            if origin is None:
                continue
            xyz = parse_float_list(origin.get("xyz"))
            rpy = parse_float_list(origin.get("rpy"))
            positions.append(xyz)
            eulers.append(rpy)
    if positions:
        pos_arr = np.array(positions, dtype=float)
        center = pos_arr.mean(axis=0)
    else:
        center = np.zeros(3)
    if eulers:
        euler_arr = np.array(eulers, dtype=float)
        euler = euler_arr.mean(axis=0)
    else:
        euler = np.zeros(3)
    return center.astype(float), euler.astype(float)


def compute_object_size(urdf_path: Path, override: Optional[float] = None) -> float:
    if override is not None:
        return float(override)
    tree = ET.parse(urdf_path)
    mesh_paths = collect_mesh_paths(urdf_path.parent, tree)
    size = estimate_size_from_meshes(mesh_paths)
    if size is not None:
        return size
    return 1.0


def load_annotation_points(annotation_path: os.PathLike | str) -> np.ndarray:
    annotation_file = _ensure_path(annotation_path)
    with annotation_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    points = data.get("affordable_points") or data.get("affordance_points")
    if not points:
        raise ValueError(f"No affordable_points found in {annotation_file}")
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Annotation points must be Nx3, got shape {pts.shape}")
    return pts


def compute_aabb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    min_corner = points.min(axis=0)
    max_corner = points.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    dims = max_corner - min_corner
    return min_corner, max_corner, center, dims


def create_box_vertices(min_corner: Sequence[float], max_corner: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    min_corner = np.asarray(min_corner, dtype=float)
    max_corner = np.asarray(max_corner, dtype=float)
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner
    vertices = np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ], dtype=float)
    # 12 triangles (two per face) using 1-indexed OBJ convention
    faces = np.array([
        [1, 2, 3], [1, 3, 4],  # bottom
        [5, 6, 7], [5, 7, 8],  # top
        [1, 2, 6], [1, 6, 5],  # front
        [2, 3, 7], [2, 7, 6],  # right
        [3, 4, 8], [3, 8, 7],  # back
        [4, 1, 5], [4, 5, 8],  # left
    ], dtype=int)
    return vertices, faces


def write_obj(vertices: np.ndarray, faces: np.ndarray, out_path: os.PathLike | str) -> None:
    path = _ensure_path(out_path)
    with path.open("w", encoding="ascii") as fp:
        for v in vertices:
            fp.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f_idx in faces:
            fp.write(f"f {int(f_idx[0])} {int(f_idx[1])} {int(f_idx[2])}\n")


def ensure_handle_mesh_from_annotation(
    asset_dir: os.PathLike | str,
    handle_name: str,
    annotation_path: os.PathLike | str,
    part_id: int = 1,
    padding: float = 0.0,
    overwrite: bool = False,
) -> Path:
    asset_dir_path = _ensure_path(asset_dir)
    parts_render = asset_dir_path / "parts_render"
    parts_render.mkdir(exist_ok=True, parents=True)
    handle_name_clean = handle_name or "handle"
    mesh_path = parts_render / f"{part_id}{handle_name_clean}.obj"
    if mesh_path.exists() and not overwrite:
        return mesh_path
    points = load_annotation_points(annotation_path)
    min_corner, max_corner, _, _ = compute_aabb(points)
    if padding > 0:
        min_corner = min_corner - padding
        max_corner = max_corner + padding
    vertices, faces = create_box_vertices(min_corner, max_corner)
    write_obj(vertices, faces, mesh_path)
    return mesh_path


def extract_joint_metadata(urdf_path: os.PathLike | str, joint_name: Optional[str] = None) -> dict:
    path = _ensure_path(urdf_path)
    tree = ET.parse(path)
    root = tree.getroot()
    for idx, joint in enumerate(root.iter("joint")):
        j_type = joint.get("type", "fixed").lower()
        if j_type not in {"revolute", "continuous", "prismatic"}:
            continue
        if joint_name and joint.get("name") != joint_name:
            continue
        parent_tag = joint.find("parent")
        child_tag = joint.find("child")
        if parent_tag is None or child_tag is None:
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        origin_xyz = parse_float_list(origin.get("xyz") if origin is not None else None)
        axis_dir = parse_float_list(axis.get("xyz") if axis is not None else None)
        if np.linalg.norm(axis_dir) > 0:
            axis_dir = (np.asarray(axis_dir) / np.linalg.norm(axis_dir)).tolist()
        lower = float(limit.get("lower", "-0.5")) if limit is not None else -0.5
        upper = float(limit.get("upper", "0.5")) if limit is not None else 0.5
        return {
            "joint_name": joint.get("name", f"joint{idx}"),
            "joint_type": j_type,
            "parent_link": parent_tag.get("link", "base"),
            "child_link": child_tag.get("link", f"link{idx}"),
            "axis_origin": origin_xyz,
            "axis_direction": axis_dir,
            "limit_lower": lower,
            "limit_upper": upper,
            "joint_kind": "hinge" if j_type in {"revolute", "continuous"} else "slider",
        }
    raise RuntimeError("No revolute/prismatic joint found in URDF")


def ensure_mobility_file(asset_dir: os.PathLike | str, joint_info: dict, overwrite: bool = False) -> Path:
    asset_dir_path = _ensure_path(asset_dir)
    mobility_path = asset_dir_path / "mobility_v2.json"
    if mobility_path.exists() and not overwrite:
        return mobility_path
    parent_part = {
        "id": 0,
        "parent": -1,
        "joint": "free",
        "name": "base",
        "parts": [
            {"id": 0, "name": joint_info.get("parent_link", "link0"), "children": []},
        ],
    }
    child_part = {
        "id": 1,
        "parent": 0,
        "joint": joint_info.get("joint_kind", "hinge"),
        "name": joint_info.get("joint_name", "door"),
        "parts": [
            {"id": 1, "name": joint_info.get("child_link", "link1"), "children": []},
        ],
        "jointData": {
            "axis": {
                "origin": joint_info.get("axis_origin", [0.0, 0.0, 0.0]),
                "direction": joint_info.get("axis_direction", [0.0, 0.0, 1.0]),
            },
            "limit": {
                "a": joint_info.get("limit_lower", -0.5),
                "b": joint_info.get("limit_upper", 0.5),
                "noLimit": False,
                "rotates": True,
                "noRotationLimit": False,
            },
        },
    }
    with mobility_path.open("w", encoding="utf-8") as fp:
        json.dump([parent_part, child_part], fp, indent=4)
    return mobility_path


def sanitize_object_name(name: str) -> str:
    words = name.replace("_", " ").split()
    return " ".join(w.capitalize() for w in words) if words else name.title()


def determine_reward_asset_path(dest_dir: Path) -> str:
    dest_path = Path(os.path.abspath(dest_dir))
    dest_real_path = Path(dest_dir).resolve()
    project_dir = os.environ.get("PROJECT_DIR")
    if project_dir:
        project_path = Path(os.path.abspath(project_dir))
        try:
            rel = dest_path.relative_to(project_path)
            rel_parts = rel.parts[1:] if rel.parts and rel.parts[0] == "data" else rel.parts
            if rel_parts:
                return Path(*rel_parts).as_posix()
        except ValueError:
            pass
        data_dir = Path(os.path.abspath(project_path / "data"))
    else:
        data_dir = Path(os.path.abspath("data"))

    try:
        rel = dest_path.relative_to(data_dir)
        return rel.as_posix()
    except ValueError:
        return dest_real_path.as_posix()


def compute_config_metadata(urdf_path: os.PathLike | str, size_override: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    path = _ensure_path(urdf_path)
    tree = ET.parse(path)
    center, euler = compute_center_and_euler(tree)
    size = compute_object_size(path, override=size_override)
    return center, euler, size


def quaternion_from_euler(euler: Sequence[float]) -> List[float]:
    roll, pitch, yaw = euler
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


def serialize_vector(vec: Sequence[float]) -> str:
    return str(tuple(float(x) for x in vec))


def ensure_annotation_copy(src_annotation: os.PathLike | str, dest_dir: os.PathLike | str) -> Path:
    src = _ensure_path(src_annotation)
    dest = _ensure_path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    dest_file = dest / src.name
    if src.resolve() != dest_file.resolve():
        dest_file.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest_file


def load_predicted_grasps(grasp_path: os.PathLike | str) -> Tuple[np.ndarray, np.ndarray]:
    """Load grasps from an Isaac-format YAML file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (grasps, confidences)
        - grasps: (N, 4, 4) homogeneous matrices
        - confidences: (N,) float scores
    """
    path = _ensure_path(grasp_path)
    if not path.exists():
        return np.empty((0, 4, 4)), np.empty((0,))
    
    import yaml
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    grasps = []
    confidences = []
    
    for key, value in data.get("grasps", {}).items():
        confidences.append(value.get("confidence", 0.0))
        
        pos = value.get("position", [0, 0, 0])
        ori = value.get("orientation", {"w": 1, "xyz": [0, 0, 0]})
        
        # Convert quaternion to rotation matrix
        # Note: scipy uses scalar-last (x, y, z, w) by default for from_quat,
        # but the YAML has w separate.
        quat = [ori["xyz"][0], ori["xyz"][1], ori["xyz"][2], ori["w"]]
        rot = R.from_quat(quat).as_matrix()
        
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = pos
        grasps.append(mat)
        
    if not grasps:
        return np.empty((0, 4, 4)), np.empty((0,))
        
    return np.array(grasps), np.array(confidences)


__all__ = [
    "find_first_urdf",
    "compute_config_metadata",
    "compute_object_size",
    "determine_reward_asset_path",
    "ensure_handle_mesh_from_annotation",
    "ensure_mobility_file",
    "extract_joint_metadata",
    "load_annotation_points",
    "compute_aabb",
    "quaternion_from_euler",
    "serialize_vector",
    "ensure_annotation_copy",
    "sanitize_object_name",
    "load_predicted_grasps",
]
