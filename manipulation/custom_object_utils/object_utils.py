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
            print(f"Extracting handle points for {handle_name} from {args.urdf_path}")
            mesh_infos = collect_mesh_info(args.urdf_path.parent, args.urdf_path, handle_name)
            print(f"Found {len(mesh_infos)} meshes for link {handle_name}")
            
            all_vertices = []
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
                all_vertices.append(v)

            if not all_vertices:
                return

            v_all = np.vstack(all_vertices)

            # IMPORTANT (custom env): get_handle_pos_custom interprets handle point clouds in the
            # *parent link* frame (see env_utils.get_handle_pos_custom: it transforms points by
            # the parent joint's link pose). For PartNet-style URDFs, that's often the base link;
            # for custom assets like sim_microwave_good, it's typically `link0`.
            #
            # We infer the intended parent-frame link from mobility_v2.json when available.
            frame_link_name = handle_name
            try:
                mobility_json = asset_dir / "mobility_v2.json"
                if mobility_json.exists():
                    with mobility_json.open("r", encoding="utf-8") as fp:
                        mobility_info = json.load(fp)
                    handle_entry = next(
                        (
                            j for j in mobility_info
                            if any(p.get("name") == handle_name for p in j.get("parts", []) if isinstance(p, dict))
                        ),
                        None,
                    )
                    if handle_entry is not None:
                        parent_id = handle_entry.get("parent")
                        if isinstance(parent_id, int) and parent_id >= 0:
                            parent_entry = next((j for j in mobility_info if j.get("id") == parent_id), None)
                            if parent_entry is not None:
                                parent_parts = parent_entry.get("parts") or []
                                if parent_parts and isinstance(parent_parts[0], dict):
                                    frame_link_name = str(parent_parts[0].get("name") or frame_link_name)
            except Exception:
                frame_link_name = handle_name

            def _mean_nn_dist(a: np.ndarray, b: np.ndarray) -> float:
                from scipy.spatial import cKDTree
                tree = cKDTree(b)
                d, _ = tree.query(a, k=1)
                return float(np.mean(d))

            def _try_transform_points_and_vertices_to_frame(
                urdf_path: Path,
                mesh_link: str,
                frame_link: str,
                ann_points: np.ndarray,
                mesh_vertices_link: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray]:
                """Return (ann_pts_in_frame, mesh_vertices_in_frame)."""
                import pybullet as p
                import pybullet_data

                cid = None
                try:
                    cid = p.connect(p.DIRECT)
                    p.setAdditionalSearchPath(pybullet_data.getDataPath())
                    body = p.loadURDF(str(urdf_path), [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

                    base_link_name = p.getBodyInfo(body)[0].decode("utf-8")

                    def _link_pose_world(link_name: str):
                        if link_name == base_link_name:
                            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
                        link_idx = None
                        for ji in range(p.getNumJoints(body)):
                            child_link_name = p.getJointInfo(body, ji)[12].decode("utf-8")
                            if child_link_name == link_name:
                                link_idx = ji
                                break
                        if link_idx is None:
                            return None
                        ls = p.getLinkState(body, link_idx, computeForwardKinematics=True)
                        return ls[4], ls[5]

                    frame_pose = _link_pose_world(frame_link)
                    mesh_pose = _link_pose_world(mesh_link)
                    if frame_pose is None or mesh_pose is None:
                        return ann_points, mesh_vertices_link

                    frame_pos, frame_orn = frame_pose
                    mesh_pos, mesh_orn = mesh_pose
                    inv_frame_pos, inv_frame_orn = p.invertTransform(frame_pos, frame_orn)

                    # Transform annotation points from base/world into frame_link.
                    ann_out = []
                    for pt in np.asarray(ann_points, dtype=float):
                        lp, _ = p.multiplyTransforms(inv_frame_pos, inv_frame_orn, pt.tolist(), [0, 0, 0, 1])
                        ann_out.append(lp)
                    ann_out = np.asarray(ann_out, dtype=float)

                    # Transform mesh vertices from mesh_link frame into frame_link frame.
                    rel_pos, rel_orn = p.multiplyTransforms(inv_frame_pos, inv_frame_orn, mesh_pos, mesh_orn)
                    v_out = []
                    for v in np.asarray(mesh_vertices_link, dtype=float):
                        vp, _ = p.multiplyTransforms(rel_pos, rel_orn, v.tolist(), [0, 0, 0, 1])
                        v_out.append(vp)
                    v_out = np.asarray(v_out, dtype=float)

                    return ann_out, v_out
                except Exception:
                    return ann_points, mesh_vertices_link
                finally:
                    if cid is not None:
                        try:
                            p.disconnect(cid)
                        except Exception:
                            pass

            ann_pts_used = np.asarray(ann_pts, dtype=float)
            # Try interpreting annotation points in the inferred frame link.
            ann_candidate, v_candidate = _try_transform_points_and_vertices_to_frame(
                args.urdf_path,
                mesh_link=handle_name,
                frame_link=frame_link_name,
                ann_points=ann_pts_used,
                mesh_vertices_link=v_all,
            )
            mean_before = _mean_nn_dist(ann_pts_used, v_all)
            mean_after = _mean_nn_dist(ann_candidate, v_candidate)
            if mean_after < mean_before:
                print(
                    f"Using handle points in frame '{frame_link_name}' (inferred from mobility_v2.json); "
                    f"mean NN dist {mean_before:.4f} -> {mean_after:.4f}"
                )
                ann_pts_used = ann_candidate
                v_all = v_candidate

            def _save_handle_points(handle_pts: np.ndarray, *, suffix: str = "") -> None:
                parts_render = asset_dir / "parts_render"
                parts_render.mkdir(exist_ok=True)
                pts_file = parts_render / f"{args.handle_part_id}{handle_name}_points.npy"
                obj_out = parts_render / f"{args.handle_part_id}{handle_name}.obj"
                label = f" {suffix}" if suffix else ""
                print(f"Saving handle points{label} to {pts_file} (n={handle_pts.shape[0]})")
                np.save(pts_file, handle_pts)
                print(f"Saving handle obj{label} to {obj_out}")
                with obj_out.open("w", encoding="ascii") as f:
                    for p in handle_pts:
                        f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

            def _collect_nearby_vertex_indices(
                tree,
                query_points: np.ndarray,
                *,
                r0: float,
                max_r: float,
                target_count: int,
                growth: float,
                max_iters: int,
            ) -> tuple[set[int], float]:
                r = float(r0)
                collected: set[int] = set()
                for _ in range(max_iters):
                    matches = tree.query_ball_point(query_points, r)
                    for lst in matches:
                        collected.update(lst)
                    if len(collected) >= target_count:
                        break
                    r *= float(growth)
                    if r > max_r:
                        break
                return collected, r

            min_corner = np.min(ann_pts_used, axis=0) - args.handle_dilate
            max_corner = np.max(ann_pts_used, axis=0) + args.handle_dilate

            mask = np.all((v_all >= min_corner) & (v_all <= max_corner), axis=1)
            if np.any(mask):
                handle_pts = v_all[mask]

                # If the bbox is too tight (common with sparse annotations),
                # densify via a radius search around the annotation points.
                min_handle_pts = 200
                if handle_pts.shape[0] < min_handle_pts:
                    print(
                        f"Handle bbox returned {handle_pts.shape[0]} points; "
                        f"densifying with radius search (target >= {min_handle_pts})."
                    )
                    from scipy.spatial import cKDTree
                    tree = cKDTree(v_all)
                    r = max(float(args.handle_dilate), 0.01)
                    max_r = 0.10
                    collected_idx, r = _collect_nearby_vertex_indices(
                        tree,
                        ann_pts_used,
                        r0=r,
                        max_r=max_r,
                        target_count=min_handle_pts,
                        growth=1.6,
                        max_iters=8,
                    )
                    if collected_idx:
                        handle_pts = v_all[np.fromiter(collected_idx, dtype=int)]
                        print(f"Densified handle points (radius): n={handle_pts.shape[0]}, r={r:.4f}")

                    # If radius search is still sparse (coarse meshes / very small handles),
                    # fall back to k-NN around annotation points to guarantee a minimum density.
                    if handle_pts.shape[0] < min_handle_pts:
                        k = min(200, int(v_all.shape[0]))
                        if k >= 1:
                            _, nn_idx = tree.query(ann_pts_used, k=k)
                            nn_idx = np.asarray(nn_idx).reshape(-1)
                            nn_idx = np.unique(nn_idx)
                            handle_pts = v_all[nn_idx]
                            print(f"Densified handle points (kNN): n={handle_pts.shape[0]}, k={k}")
                _save_handle_points(handle_pts)
            else:
                # Fallback: if annotation points are slightly misaligned with the
                # mesh frame (common in custom assets), use a radius search around
                # annotation points to find nearby mesh vertices.
                print("No handle points found inside annotation bbox! Falling back to radius search.")
                try:
                    from scipy.spatial import cKDTree

                    tree = cKDTree(v_all)
                    r0 = max(float(args.handle_dilate), 0.01)
                    max_r = 0.10
                    collected_idx, r = _collect_nearby_vertex_indices(
                        tree,
                        ann_pts_used,
                        r0=r0,
                        max_r=max_r,
                        target_count=50,
                        growth=2.0,
                        max_iters=6,
                    )

                    if collected_idx and r <= max_r:
                        handle_pts = v_all[np.fromiter(collected_idx, dtype=int)]
                        _save_handle_points(handle_pts, suffix=f"(fallback, r={r:.4f})")
                    else:
                        if collected_idx and r > max_r:
                            print(
                                f"Fallback radius search would require r={r:.4f} (> {max_r:.4f}); "
                                "using transformed annotation points as handle point cloud instead."
                            )
                        else:
                            print("Fallback radius search found no nearby vertices; using transformed annotation points instead.")
                        handle_pts = np.asarray(ann_pts_used, dtype=float)
                        _save_handle_points(handle_pts, suffix="(annotation)")
                except Exception as e:
                    print(f"Fallback handle point extraction failed: {e}")


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
