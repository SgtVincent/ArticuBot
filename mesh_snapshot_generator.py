# %%
import argparse
import json
import threading
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import imageio.v3 as iio
import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

try:
    from manipulation.custom_object_utils.object_utils import load_scale_from_file
except ImportError:
    def load_scale_from_file(asset_dir: Path) -> Optional[float]:
        candidates = [
            asset_dir / "urdf" / "scale.txt",
            asset_dir / "urdf" / "scale.text",
            asset_dir / "scale.txt",
            asset_dir / "scale.text",
        ]
        for scale_path in candidates:
            if scale_path.exists():
                try:
                    with scale_path.open("r") as f:
                        return float(f.read().strip())
                except ValueError:
                    continue
        return None

FRANKA_URDF_PATH = Path("manipulation/assets/panda_bullet/panda.urdf")


def _resolve_urdf_packages(src_path: Path) -> Path:
    """Resolve package:// URDF paths to absolute asset paths."""
    if not src_path.exists():
        raise FileNotFoundError(f"URDF not found: {src_path}")

    text = src_path.read_text()
    replacements = {
        "package://meshes/": (src_path.parent / "meshes").as_posix() + "/",
        "package://franka_description/": (
            Path.cwd() / "pybullet_ompl/models/franka_description"
        ).as_posix() + "/",
    }

    updated = text
    for token, prefix in replacements.items():
        if token in updated:
            updated = updated.replace(token, prefix)

    tmp_path = Path(tempfile.gettempdir()) / f"{src_path.stem}_viser_resolved.urdf"
    tmp_path.write_text(updated)
    return tmp_path


def _create_scaled_urdf(src_path: Path, scale_factor: float) -> Path:
    """Create a temporary URDF with explicit mesh scaling."""
    if not src_path.exists():
        return src_path
    
    try:
        tree = ET.parse(src_path)
        root = tree.getroot()
        
        for mesh in root.iter("mesh"):
            # Fix relative paths to be absolute
            filename = mesh.get("filename")
            if filename:
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    abs_path = (src_path.parent / mesh_path).resolve()
                    mesh.set("filename", str(abs_path))

            current_scale_str = mesh.get("scale", "1 1 1")
            try:
                parts = current_scale_str.replace(",", " ").split()
                if len(parts) >= 3:
                     sx, sy, sz = float(parts[0]), float(parts[1]), float(parts[2])
                elif len(parts) == 1:
                     sx = sy = sz = float(parts[0])
                else:
                     sx, sy, sz = 1.0, 1.0, 1.0
            except ValueError:
                sx, sy, sz = 1.0, 1.0, 1.0
            
            new_scale = f"{sx * scale_factor} {sy * scale_factor} {sz * scale_factor}"
            mesh.set("scale", new_scale)
            
        tmp_path = Path(tempfile.gettempdir()) / f"{src_path.stem}_scaled_{scale_factor:.4f}.urdf"
        tree.write(tmp_path)
        return tmp_path
    except Exception as e:
        print(f"Error scaling URDF: {e}")
        return src_path


ROOT = Path.cwd()
DEFAULT_SIM_DIR = ROOT / "data/custom_objects_v2/sim"
DEFAULT_OUTPUT_DIR = ROOT / "data/custom_objects_v2/sim_snapshots"


def _parse_text_affordance_file(path: Path) -> np.ndarray:
    """Parse a plain-text affordance file (whitespace/comma-separated or Python-list literal).

    Returns an (N, 3) float array. Lines starting with '#' are ignored.
    """
    text = path.read_text().strip()
    if not text:
        return np.zeros((0, 3), dtype=float)
    try:
        arr = np.loadtxt(path, comments="#", ndmin=2)
    except Exception:
        import ast

        try:
            parsed = ast.literal_eval(text)
            arr = np.asarray(parsed, dtype=float)
        except Exception as exc:
            raise ValueError(f"Could not parse text affordance file {path}: {exc}") from exc
    if arr.ndim == 1 and arr.size == 3:
        arr = arr.reshape(1, 3)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.shape[1] < 3:
        raise ValueError("Affordance point file must contain at least 3 columns (x y z)")
    return arr[:, :3].astype(float)


def load_affordance_points(annotation_path: Path) -> np.ndarray:
    """Load affordance points from JSON or plain-text file and return an (N,3) array.

    JSON accepted structures:
      - list of [x,y,z]
      - dict with keys: 'affordable_points', 'affordance_points', 'points', 'affordances'
    """
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    suffix = annotation_path.suffix.lower()
    if suffix == ".json":
        with annotation_path.open("r") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            pts = np.asarray(data, dtype=float)
        elif isinstance(data, dict):
            for key in ("affordable_points", "affordance_points", "points", "affordances"):
                if key in data:
                    pts = np.asarray(data[key], dtype=float)
                    break
            else:
                pts = None
                for v in data.values():
                    if isinstance(v, list):
                        cand = np.asarray(v, dtype=float)
                        if cand.ndim >= 2 and cand.shape[1] >= 3:
                            pts = cand
                            break
                if pts is None:
                    raise ValueError("JSON does not contain affordance points under a recognized key")
        else:
            raise ValueError("Unsupported JSON structure for affordance annotations")
    else:
        pts = _parse_text_affordance_file(annotation_path)

    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1 and pts.size == 3:
        pts = pts.reshape(1, 3)
    if pts.size == 0:
        return pts.reshape(0, 3)
    if pts.shape[1] < 3:
        raise ValueError("Affordance points must have >=3 columns (x,y,z)")
    return pts[:, :3]


def transform_affordance_points(pts: np.ndarray, rotations_deg: Optional[Sequence[float]] = None) -> np.ndarray:
    """Apply Euler rotations (degrees) to affordance points.

    rotations_deg: sequence-like of three angles [rx, ry, rz] in degrees
    representing rotations about the x, y, z axes. The rotations are applied
    in X then Y then Z order (i.e., R = Rz @ Ry @ Rx). If `rotations_deg` is
    None or all zeros, the input points are returned unchanged.
    """
    if rotations_deg is None:
        return pts
    rot_arr = np.asarray(rotations_deg, dtype=float)
    if rot_arr.size == 0:
        return pts
    if rot_arr.shape[0] != 3:
        raise ValueError("rotations_deg must be a sequence of three values (rx, ry, rz)")
    if np.allclose(rot_arr, 0.0, atol=1e-8):
        return pts

    rx, ry, rz = np.deg2rad(rot_arr)
    cosx, sinx = np.cos(rx), np.sin(rx)
    cosy, siny = np.cos(ry), np.sin(ry)
    cosz, sinz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cosx, -sinx], [0.0, sinx, cosx]])
    Ry = np.array([[cosy, 0.0, siny], [0.0, 1.0, 0.0], [-siny, 0.0, cosy]])
    Rz = np.array([[cosz, -sinz, 0.0], [sinz, cosz, 0.0], [0.0, 0.0, 1.0]])

    R = Rz @ Ry @ Rx
    return pts @ R.T


def add_affordance_visuals(server: viser.ViserServer, points: np.ndarray) -> None:
    for point_idx, point in enumerate(points):
        server.scene.add_icosphere(
            f"/affordance/point_{point_idx}",
            radius=0.008,
            color=(255, 80, 80),
            subdivisions=2,
            position=tuple(point.tolist()),
        )


def configure_ground_grid(server: viser.ViserServer, viser_urdf: ViserUrdf) -> None:
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene  # noqa: SLF001
    ground_z = trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        cell_size=0.1,
        position=(0.0, 0.0, float(ground_z)),
    )


def compute_scene_bounds(viser_urdf: ViserUrdf) -> Optional[Tuple[np.ndarray, float]]:
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene  # noqa: SLF001
    if trimesh_scene is None or trimesh_scene.bounds is None:
        return None
    bounds = trimesh_scene.bounds
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    radius = float(np.linalg.norm(extents) * 0.5)
    return center, radius


def compute_camera_pose(
    center: np.ndarray,
    radius: float,
    fov_rad: float,
    distance_multiplier: float,
    view_direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if radius <= 1e-6:
        radius = 0.5
    view_dir = np.asarray(view_direction, dtype=float)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-9)
    fov_rad = max(1e-3, min(np.pi - 1e-3, float(fov_rad)))
    distance = radius / np.sin(fov_rad / 2.0)
    distance *= distance_multiplier
    position = center + view_dir * distance
    return position, center


def find_urdf_file(obj_dir: Path) -> Optional[Path]:
    urdf_dir = obj_dir / "urdf"
    if not urdf_dir.exists():
        return None
    preferred = urdf_dir / f"{obj_dir.name}.urdf"
    if preferred.exists():
        return preferred
    candidates = sorted(urdf_dir.glob("*.urdf"))
    if not candidates:
        return None
    return candidates[0]


def find_affordance_file(obj_dir: Path) -> Optional[Path]:
    candidates = [
        obj_dir / "urdf" / "affordance.txt",
        obj_dir / "urdf" / "annotation.json",
        obj_dir / "annotation.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def iter_object_dirs(sim_dir: Path, name_filter: Optional[str]) -> Iterable[Tuple[Path, Path]]:
    if not sim_dir.exists():
        return []
    for obj_dir in sorted(sim_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        if obj_dir.name == "assets":
            continue
        if name_filter and name_filter not in obj_dir.name:
            continue
        urdf_path = find_urdf_file(obj_dir)
        if urdf_path is None:
            continue
        yield obj_dir, urdf_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate mesh snapshots for sim assets using Viser.")
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=DEFAULT_SIM_DIR,
        help="Root directory for sim assets (contains per-object folders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write snapshot images.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Snapshot width in pixels.")
    parser.add_argument("--height", type=int, default=768, help="Snapshot height in pixels.")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Camera vertical FOV in degrees.")
    parser.add_argument(
        "--distance-multiplier",
        type=float,
        default=1.25,
        help="Scale camera distance to fit the object with extra margin.",
    )
    parser.add_argument(
        "--view-direction",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 0.7),
        metavar=("X", "Y", "Z"),
        help="View direction vector for the camera.",
    )
    parser.add_argument(
        "--include-affordances",
        action="store_true",
        help="Overlay affordance points when available.",
    )
    parser.add_argument(
        "--load-collision-meshes",
        action="store_true",
        help="Render collision meshes in addition to visuals.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip snapshots that already exist in the output directory.",
    )
    parser.add_argument(
        "--name-filter",
        type=str,
        default="",
        help="Only process object directories whose name contains this substring.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Limit number of processed models (0 means no limit).",
    )
    parser.add_argument(
        "--apply-affordance-transform",
        action="store_true",
        help="Apply (0, 0, -90) degree rotation to affordance points",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Time to wait after setting camera before capturing (seconds).",
    )
    parser.add_argument(
        "--add-franka",
        action="store_true",
        help="Load the Franka arm URDF into the scene",
    )
    parser.add_argument(
        "--apply-mesh-scale",
        action="store_true",
        help="Read scale.txt/scale.text from the asset directory and apply it to the mesh and affordance points.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    return parser.parse_args()


def generate_snapshots(server: viser.ViserServer, client: viser.ClientHandle, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = list(iter_object_dirs(args.sim_dir, args.name_filter or None))
    if args.max_models and args.max_models > 0:
        entries = entries[: args.max_models]

    print(f"Found {len(entries)} model(s) to render.")

    for idx, (obj_dir, urdf_path) in enumerate(entries, start=1):
        out_path = args.output_dir / f"{obj_dir.name}.png"
        if args.skip_existing and out_path.exists():
            print(f"[{idx}/{len(entries)}] Skipping {obj_dir.name} (snapshot exists).")
            continue

        print(f"[{idx}/{len(entries)}] Rendering {obj_dir.name}...")
        server.scene.reset()
        server.scene.configure_default_lights(enabled=True, cast_shadow=True)

        # 1. Determine Scale
        obj_scale = 1.0
        if args.apply_mesh_scale:
            s = load_scale_from_file(obj_dir)
            if s is not None:
                obj_scale = float(s)
            else:
                # Try fallback locations if needed, but obj_dir is usually sufficient for v1/v2 via load_scale_from_file logic
                pass
        
        # 2. Setup Object Root Frame
        object_root_path = "/object"
        server.scene.add_frame(
            object_root_path,
        )
        # Apply scaled URDF
        final_urdf_path = urdf_path
        if args.apply_mesh_scale and obj_scale != 1.0:
             final_urdf_path = _create_scaled_urdf(urdf_path, obj_scale)

        viser_urdf = ViserUrdf(
            server,
            urdf_or_path=final_urdf_path,
            load_meshes=True,
            load_collision_meshes=args.load_collision_meshes,
            collision_mesh_color_override=(1.0, 0.0, 0.0, 0.4),
            root_node_name=f"{object_root_path}/urdf"
        )
        configure_ground_grid(server, viser_urdf)

        # 3. Load Franka
        if args.add_franka and FRANKA_URDF_PATH.exists():
            try:
                viser_franka_path = _resolve_urdf_packages(FRANKA_URDF_PATH)
                franka_urdf = ViserUrdf(
                    server,
                    urdf_or_path=viser_franka_path,
                    load_meshes=True,
                    root_node_name="/franka",
                )
                
                 # Set Franka to home configuration to avoid twisted look
                home_pose = {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785,  # -45 deg
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356,  # -135 deg
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.571,   # 90 deg
                    "panda_joint7": 0.785,   # 45 deg
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                }
                
                # Apply home pose
                franka_limits = franka_urdf.get_actuated_joint_limits()
                franka_cfg = []
                for name in franka_limits.keys():
                    franka_cfg.append(home_pose.get(name, 0.0))
                franka_urdf.update_cfg(np.array(franka_cfg))
                
            except Exception as e:
                print(f"Failed to load Franka URDF: {e}")

        # Set joints to neutral if present.
        joint_limits = viser_urdf.get_actuated_joint_limits()
        if joint_limits:
            cfg = []
            for _, (lower, upper) in joint_limits.items():
                lower_val = lower if lower is not None else -np.pi
                upper_val = upper if upper is not None else np.pi
                if lower_val > upper_val:
                    lower_val, upper_val = upper_val, lower_val
                cfg.append((lower_val + upper_val) / 2.0)
            viser_urdf.update_cfg(np.array(cfg, dtype=float))

        if args.include_affordances:
            affordance_path = find_affordance_file(obj_dir)
            if affordance_path is not None:
                try:
                    pts = load_affordance_points(affordance_path)
                    if pts.size:
                        if args.apply_mesh_scale and obj_scale != 1.0:
                            pts = pts * obj_scale
                        if args.apply_affordance_transform:
                            # Apply (0, 0, -90) degree rotation
                            pts = transform_affordance_points(pts, (0.0, 0.0, -90.0))
                        add_affordance_visuals(server, pts)
                except Exception as exc:
                    print(f"  Warning: failed to load affordances from {affordance_path}: {exc}")

        bounds = compute_scene_bounds(viser_urdf)
        if bounds is None:
            print(f"  Warning: no bounds found for {obj_dir.name}, skipping.")
            continue
        center, radius = bounds
        position, look_at = compute_camera_pose(
            center,
            radius,
            np.deg2rad(args.fov_deg),
            args.distance_multiplier,
            np.asarray(args.view_direction, dtype=float),
        )

        client.camera.position = position
        client.camera.look_at = look_at
        client.camera.fov = float(np.deg2rad(args.fov_deg))

        server.flush()
        time.sleep(max(0.0, args.settle_seconds))

        image = client.get_render(args.height, args.width, transport_format="png")
        iio.imwrite(out_path, image)
        print(f"  Saved {out_path}")

    print("Snapshot generation complete.")


def main() -> None:
    args = parse_args()

    server = viser.ViserServer(host=args.host, port=args.port)
    viewer_url = getattr(server, "viewer_url", f"http://localhost:{args.port}")
    print(f"Viser server running at {viewer_url}")
    print("Open the URL in a browser to start snapshot generation.")

    started = threading.Event()

    def _on_client_connect(client: viser.ClientHandle) -> None:
        if started.is_set():
            return
        started.set()
        try:
            generate_snapshots(server, client, args)
        finally:
            server.stop()

    server.on_client_connect(_on_client_connect)

    try:
        while not started.is_set():
            time.sleep(0.2)
        # Keep alive until server.stop() is called.
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted. Stopping server.")
        server.stop()


if __name__ == "__main__":
    main()
