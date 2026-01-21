# %%
import argparse
import json
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union, Optional

import numpy as np
try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency
    trimesh = None
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
    # Assume we are at repo root for __file__ relative paths if needed, 
    # but here we mostly rely on src_path locality or explicit repo paths.
    # For manipulation/assets/panda_bullet/panda.urdf, meshes are in valid location rel to URDF.
    replacements = {
        "package://meshes/": (src_path.parent / "meshes").as_posix() + "/",
        # Add other package replacements if encountered
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
        return src_path # Or raise error
    
    try:
        tree = ET.parse(src_path)
        root = tree.getroot()
        
        # Scale all mesh visual/collision elements
        # Note: This only scales the mesh geometry. Joint origins are not scaled.
        # This assumes the scale.txt is intended for visual mesh correction.
        for mesh in root.iter("mesh"):
            # Fix relative paths to be absolute
            filename = mesh.get("filename")
            if filename:
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    # Resolve relative to the original URDF directory
                    abs_path = (src_path.parent / mesh_path).resolve()
                    mesh.set("filename", str(abs_path))

            current_scale_str = mesh.get("scale", "1 1 1")
            try:
                # Handle cases with commas or extra spaces if any (though URDF standard is space separated)
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
            
        # Write to temp file
        # Use a consistent name based on hash or just overwriting to avoid disk bloat? 
        # tempfile name is safer.
        tmp_path = Path(tempfile.gettempdir()) / f"{src_path.stem}_scaled_{scale_factor:.4f}.urdf"
        tree.write(tmp_path)
        return tmp_path
    except Exception as e:
        print(f"Error scaling URDF: {e}")
        return src_path



ROOT = Path.cwd()
# DEFAULT_URDF = ROOT / "data/custom_objects_v2/sim/microwave_7167/urdf/microwave_7167.urdf"
# DEFAULT_ANNOTATION = ROOT / "data/custom_objects_v2/sim/microwave_7167/urdf/affordance.txt"
object_name = "microwave_7167"
DEFAULT_URDF = ROOT / f"data/custom_objects_v2/sim/{object_name}/urdf/{object_name}.urdf"
DEFAULT_ANNOTATION = ROOT / f"data/custom_objects_v2/sim/{object_name}/urdf/affordance.txt"


print(f"Viser version: {viser.__version__}")
print("Update `DEFAULT_URDF` and `DEFAULT_ANNOTATION` if you want to visualize another asset.")

def tinted_mesh(mesh: Any, rgb: Tuple[float, float, float]):
    if trimesh is None:
        raise RuntimeError("tinted_mesh requires 'trimesh' package to be installed")
    colored = mesh.copy()
    rgba = np.clip(np.array([rgb[0], rgb[1], rgb[2], 255]), 0, 255).astype(np.uint8)
    colored.visual.vertex_colors = np.tile(rgba, (len(colored.vertices), 1))  # type: ignore[attr-defined]
    return colored


def compute_axis_aligned_box(points: Union[Sequence[Sequence[float]], np.ndarray]):
    pts = np.asarray(points)
    if pts.size == 0:
        return None
    min_corner = pts.min(axis=0)
    max_corner = pts.max(axis=0)
    center = (min_corner + max_corner) / 2.0
    dimensions = max_corner - min_corner
    return center, dimensions


def compute_oriented_box_mesh(points: Union[Sequence[Sequence[float]], np.ndarray]):
    if trimesh is None:
        print("Warning: 'trimesh' not available; skipping oriented bounding box computation")
        return None
    pts = np.asarray(points)
    if pts.shape[0] < 3:
        return None
    point_cloud = trimesh.points.PointCloud(pts)
    return point_cloud.bounding_box_oriented.to_mesh()


def create_joint_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> Tuple[List[Any], List[float]]:
    slider_handles: List[Any] = []
    initial_config: List[float] = []
    joint_limits = viser_urdf.get_actuated_joint_limits()
    if not joint_limits:
        server.gui.add_markdown("No actuated joints detected in the URDF.")
        return slider_handles, initial_config

    def _update_cfg(_: Any) -> None:
        if not slider_handles:
            return
        cfg = np.array([slider.value for slider in slider_handles], dtype=float)
        viser_urdf.update_cfg(cfg)

    for joint_name, (lower, upper) in joint_limits.items():
        lower_val = lower if lower is not None else -np.pi
        upper_val = upper if upper is not None else np.pi
        if lower_val > upper_val:
            lower_val, upper_val = upper_val, lower_val
        if lower_val < -0.1 and upper_val > 0.1:
            initial_pos = 0.0
        else:
            initial_pos = (lower_val + upper_val) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower_val,
            max=upper_val,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider_handles.append(slider)
        slider.on_update(_update_cfg)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def add_visibility_controls(
    server: viser.ViserServer,
    viser_urdf: ViserUrdf,
    load_meshes: bool,
    load_collision_meshes: bool,
) -> None:
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            initial_value=viser_urdf.show_visual,
        )
        show_collision_cb = server.gui.add_checkbox(
            "Show collision meshes",
            initial_value=viser_urdf.show_collision,
        )

    @show_meshes_cb.on_update
    def _(_: Any) -> None:
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_cb.on_update
    def _(_: Any) -> None:
        viser_urdf.show_collision = show_collision_cb.value

    show_meshes_cb.visible = load_meshes
    show_collision_cb.visible = load_collision_meshes


def add_reset_button(
    server: viser.ViserServer,
    slider_handles: List[Any],
    initial_config: List[float],
) -> None:
    if not slider_handles:
        return
    reset_button = server.gui.add_button("Reset joints")

    @reset_button.on_click
    def _(_: Any) -> None:
        for slider, init_q in zip(slider_handles, initial_config):
            slider.value = init_q


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


def clear_affordance_handles(handles: List[Any]) -> None:
    for handle in list(handles):
        try:
            handle.remove()
        except Exception:
            pass
    handles.clear()


def add_annotation_visuals(
    server: viser.ViserServer,
    annotation_path: Path,
    rotation_deg: Optional[Sequence[float]] = None,
    existing_handles: Optional[List[Any]] = None,
    scale: float = 1.0,
) -> List[Any]:
    if existing_handles is None:
        existing_handles = []
    if existing_handles:
        clear_affordance_handles(existing_handles)

    if not annotation_path.exists():
        print(f"Warning: annotation file not found at {annotation_path}")
        return []
    try:
        pts = load_affordance_points(annotation_path)
    except Exception as exc:
        print(f"Warning: failed to load affordance annotations from {annotation_path}: {exc}")
        return []
    if pts.size == 0:
        return []

    # 1. Apply Scale
    if scale != 1.0:
        pts = pts * scale

    # 2. Apply Rotation
    pts = transform_affordance_points(pts, rotation_deg)
    
    if rotation_deg is not None and not np.allclose(rotation_deg, 0.0):
        print(f"Applied rotation {rotation_deg} (deg) to affordance points")
    if scale != 1.0:
        print(f"Applied scale {scale:.4f} to affordance points")
    
    print(f"Loaded {len(pts)} affordance point(s) from {annotation_path.name}")

    handles: List[Any] = []
    for point_idx, point in enumerate(pts):
        handles.append(
            server.scene.add_icosphere(
                f"/affordance/point_{point_idx}",
                radius=0.008,
                color=(255, 80, 80),
                subdivisions=2,
                position=tuple(point.tolist()),
            )
        )
    aabb = compute_axis_aligned_box(pts)
    if aabb is not None:
        center, dims = aabb
        handles.append(
            server.scene.add_box(
                "/affordance/aabb",
                color=(80, 200, 120),
                dimensions=tuple(dims.tolist()),
                wireframe=True,
                position=tuple(center.tolist()),
                opacity=0.1,
            )
        )
    obb_mesh = compute_oriented_box_mesh(pts)
    if obb_mesh is not None:
        tinted_obb = tinted_mesh(obb_mesh, (64, 128, 255))
        handles.append(
            server.scene.add_mesh_trimesh(
                "/affordance/obb",
                mesh=tinted_obb,
                cast_shadow=False,
                receive_shadow=False,
            )
        )
    return handles


def configure_ground_grid(server: viser.ViserServer, viser_urdf: ViserUrdf) -> None:
    # Safely get trimesh scene
    trimesh_scene = None
    if hasattr(viser_urdf, "_urdf"):
        trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    
    ground_z = 0.0
    if trimesh_scene is not None:
         # Check if bounds are available
         if hasattr(trimesh_scene, "bounds") and trimesh_scene.bounds is not None:
             ground_z = trimesh_scene.bounds[0, 2]
    
    server.scene.add_grid(
        "/grid",
        width=2.0,
        height=2.0,
        cell_size=0.1,
        position=(0.0, 0.0, float(ground_z)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Viser URDF viewer with affordances")
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=DEFAULT_URDF,
        help="Path to the URDF to visualize",
    )
    parser.add_argument(
        "--annotation-path",
        type=Path,
        default=DEFAULT_ANNOTATION,
        help="Path to the affordance annotation (JSON or plain-text .txt with one 'x y z' point per line)",
    )
    parser.add_argument(
        "--load-collision-meshes",
        action="store_true",
        help="Render collision meshes in addition to visuals",
    )
    parser.add_argument(
        "--apply-affordance-transform",
        action="store_true",
        help="Apply the transformation xyzrot2mat(np.zeros(3), [[1,0,0],[0,0,1],[0,1,0]]) to affordance points",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = viser.ViserServer()
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)

    # 1. Determine Scale
    obj_scale = 1.0
    if args.apply_mesh_scale:
        # Try finding scale.txt in urdf_path parent or grandparent (in case of urdf/ subdir)
        s = load_scale_from_file(args.urdf_path.parent)
        if s is None:
            s = load_scale_from_file(args.urdf_path.parent.parent)
        
        if s is not None:
            obj_scale = float(s)
            print(f"Loaded scale {obj_scale} from {args.urdf_path.parent}")
        else:
            print(f"Warning: --apply-mesh-scale requested but no scale.txt found for {args.urdf_path}")

    # 2. Setup Object Root Frame
    # We use a frame to apply scale to the URDF visuals
    object_root_path = "/object"
    server.scene.add_frame(
        object_root_path,
    )
    # Note: Setting frame_handle.scale here proved ineffective for ViserUrdf meshes in some versions.
    # We now use _create_scaled_urdf below.

    # Prepare URDF path (scaled if needed)
    final_urdf_path = args.urdf_path
    if args.apply_mesh_scale and obj_scale != 1.0:
        print(f"Applying explicit scale {obj_scale} to URDF meshes...")
        final_urdf_path = _create_scaled_urdf(args.urdf_path, obj_scale)

    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=final_urdf_path,
        load_meshes=True,
        load_collision_meshes=args.load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.4),
        root_node_name=f"{object_root_path}/urdf"
    )

    configure_ground_grid(server, viser_urdf)

    # 3. Load Franka if requested
    if args.add_franka:
        if FRANKA_URDF_PATH.exists():
            try:
                viser_franka_path = _resolve_urdf_packages(FRANKA_URDF_PATH)
                print(f"Created Viser-compatible Franka URDF at {viser_franka_path}")
                
                franka_urdf = ViserUrdf(
                    server,
                    urdf_or_path=viser_franka_path,
                    load_meshes=True,
                    root_node_name="/franka",
                )
                print(f"Loaded Franka URDF from {viser_franka_path}")

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
                joint_limits = franka_urdf.get_actuated_joint_limits()
                initial_cfg = []
                for name in joint_limits.keys():
                    initial_cfg.append(home_pose.get(name, 0.0))
                franka_urdf.update_cfg(np.array(initial_cfg))

            except Exception as e:
                print(f"Failed to load Franka URDF: {e}")
        else:
            print(f"Warning: Franka URDF not found at {FRANKA_URDF_PATH}")

    with server.gui.add_folder("Joint position control"):
        slider_handles, initial_config = create_joint_control_sliders(server, viser_urdf)

    add_visibility_controls(server, viser_urdf, load_meshes=True, load_collision_meshes=args.load_collision_meshes)
    add_reset_button(server, slider_handles, initial_config)

    initial_rotation = (0.0, 0.0, 0.0)
    affordance_handles: List[Any] = []
    
    affordance_handles = add_annotation_visuals(
        server,
        args.annotation_path,
        rotation_deg=initial_rotation if args.apply_affordance_transform else (0.0, 0.0, 0.0),
        existing_handles=affordance_handles,
        scale=obj_scale
    )

    with server.gui.add_folder("Affordances"):
        affordance_visible_cb = server.gui.add_checkbox(
            "Show affordances",
            initial_value=True,
        )
        affordance_transform_cb = server.gui.add_checkbox(
            "Apply affordance transform",
            initial_value=args.apply_affordance_transform,
        )
        affordance_rotation = server.gui.add_vector3(
            "Affordance rotation (deg)",
            initial_value=initial_rotation,
            step=1.0,
        )
        reload_affordance_btn = server.gui.add_button("Reload affordances")

    # Disable rotation control when the checkbox is unchecked
    affordance_rotation.disabled = not affordance_transform_cb.value

    def _get_rotation() -> tuple:
        return tuple(affordance_rotation.value) if affordance_transform_cb.value else (0.0, 0.0, 0.0)

    def _set_affordance_visibility(_: Any) -> None:
        for handle in affordance_handles:
            try:
                handle.visible = affordance_visible_cb.value
            except Exception:
                pass

    def _reload_affordances(_: Optional[Any] = None) -> None:
        nonlocal affordance_handles
        affordance_handles = add_annotation_visuals(
            server,
            args.annotation_path,
            rotation_deg=_get_rotation(),
            existing_handles=affordance_handles,
            scale=obj_scale
        )
        _set_affordance_visibility(None)

    def _on_transform_checkbox_update(_: Any) -> None:
        # Enable/disable the rotation control and reload visuals
        affordance_rotation.disabled = not affordance_transform_cb.value
        _reload_affordances(None)

    affordance_visible_cb.on_update(_set_affordance_visibility)
    affordance_transform_cb.on_update(_on_transform_checkbox_update)
    affordance_rotation.on_update(_reload_affordances)
    reload_affordance_btn.on_click(_reload_affordances)
    _set_affordance_visibility(None)
    server.scene.add_frame("/world", axes_length=0.05)

    if slider_handles:
        viser_urdf.update_cfg(np.array(initial_config, dtype=float))

    viewer_url = getattr(server, "viewer_url", "http://localhost:8080")
    print(f"Viser server is running at {viewer_url}")
    print("Press Ctrl+C to exit")

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        print("Shutting down viewer")


if __name__ == "__main__":
    main()



