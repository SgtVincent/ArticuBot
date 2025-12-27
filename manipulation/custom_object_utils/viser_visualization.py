"""Interactive viser-based visualization for integrated inverse map debugging.

This module provides real-time 3D visualization using viser for:
1. Trajectory waypoints with reachability indicators
2. Per-waypoint inverse reachability heatmaps
3. Integrated inverse map distribution
4. Interactive controls to step through waypoints and theta slices
5. Robot URDF and object mesh visualization

Usage:
    visualizer = IntegratedInverseMapVisualizer(port=8080)
    visualizer.visualize_attempt(
        trajectory_obj, sampler, object_pos, object_quat,
        x_grid, y_grid, theta_grid, integrated_scores,
        base_euler, z_height
    )
    # Blocks until user presses 'n' for next or 'q' to quit
"""
from __future__ import annotations

import numpy as np
import tempfile
from typing import List, Optional, Sequence, Tuple, Any
from pathlib import Path
from scipy.spatial.transform import Rotation as R_scipy
import threading

try:
    import viser
    from viser.extras import ViserUrdf
    VISER_AVAILABLE = True
except ImportError:
    VISER_AVAILABLE = False
    ViserUrdf = None
    print("Warning: viser not installed. Install with: pip install viser")


# Default paths
_DEFAULT_ROBOT_URDF = Path(__file__).resolve().parent.parent / "assets" / "panda_bullet" / "panda.urdf"


def _make_transform_matrix(
    position: np.ndarray,
    quat_xyzw: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Create a 4x4 transform with uniform scale.

    Notes:
        - Uses quaternion in xyzw ordering.
        - Scale matches PyBullet loadURDF(globalScaling=...).
    """
    mat = np.eye(4, dtype=np.float64)
    rot = R_scipy.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_matrix()
    mat[:3, :3] = rot * float(scale)
    mat[:3, 3] = np.asarray(position, dtype=np.float64)
    return mat


def _resolve_urdf_packages(src_path: Path, base_path: Optional[Path] = None) -> Path:
    """Resolve package:// URDF paths to absolute asset paths.
    
    Args:
        src_path: Path to the URDF file.
        base_path: Base path for resolving relative mesh paths.
        
    Returns:
        Path to resolved URDF (may be a temp file).
    """
    if not src_path.exists():
        raise FileNotFoundError(f"URDF not found: {src_path}")

    text = src_path.read_text()
    
    # Common replacements for package:// paths
    if base_path is None:
        base_path = src_path.parent
    
    replacements = {
        "package://meshes/": (base_path / "meshes").as_posix() + "/",
        "package://franka_description/": (
            Path(__file__).resolve().parent.parent / "assets" / "panda_bullet"
        ).as_posix() + "/",
    }

    updated = text
    needs_temp = False
    for token, prefix in replacements.items():
        if token in updated:
            updated = updated.replace(token, prefix)
            needs_temp = True

    # Also handle relative paths like "meshes/..." without package://
    if 'filename="meshes/' in updated:
        updated = updated.replace('filename="meshes/', f'filename="{(base_path / "meshes").as_posix()}/')
        needs_temp = True
    
    if needs_temp:
        tmp_path = Path(tempfile.gettempdir()) / f"{src_path.stem}_viser_resolved.urdf"
        tmp_path.write_text(updated)
        return tmp_path
    
    return src_path


class IntegratedInverseMapVisualizer:
    """Interactive viser-based visualizer for integrated inverse map debugging."""
    
    def __init__(
        self, 
        port: int = 8080, 
        share: bool = False,
        robot_urdf_path: Optional[Path] = None,
    ):
        """Initialize viser server.
        
        Args:
            port: Port for viser server.
            share: Whether to create a shareable link.
            robot_urdf_path: Path to robot URDF (defaults to Franka Panda).
        """
        if not VISER_AVAILABLE:
            raise ImportError("viser is required. Install with: pip install viser")

        # Import locally so type checkers don't treat `viser` as unbound.
        import viser as _viser

        self.server = _viser.ViserServer(port=port, verbose=False)
        self.port = port
        self.share = share
        
        # URDF paths
        self.robot_urdf_path = Path(robot_urdf_path) if robot_urdf_path else _DEFAULT_ROBOT_URDF
        self.object_urdf_path: Optional[Path] = None
        
        # Loaded URDF handles
        # NOTE: ViserUrdf is conditionally imported; keep these as Any.
        self.robot_urdf: Optional[Any] = None
        self.object_urdf: Optional[Any] = None
        self.robot_joint_sliders: List = []
        self.robot_initial_cfg: List[float] = []

        # Internal flags
        self._robot_slider_callbacks_registered = False
        
        # State
        self.continue_event = threading.Event()
        self.quit_event = threading.Event()
        self.current_waypoint_idx = 0
        self.current_theta_idx = 0
        
        # UI controls
        self._setup_ui_controls()
        
        print(f"Viser server started at http://localhost:{port}")
        if share:
            get_share_url = getattr(self.server, "get_share_url", None)
            if callable(get_share_url):
                print(f"Shareable link: {get_share_url()}")
    
    def _setup_ui_controls(self):
        """Set up UI controls in viser."""
        # Folder for controls
        with self.server.gui.add_folder("Controls"):
            self.next_button = self.server.gui.add_button("Next Attempt (n)")
            self.quit_button = self.server.gui.add_button("Quit (q)")

            self.waypoint_slider = self.server.gui.add_slider(
                "Waypoint Index", min=0, max=1, step=1, initial_value=0
            )
            self.theta_slider = self.server.gui.add_slider(
                "Theta Index", min=0, max=1, step=1, initial_value=0
            )

            self.show_trajectory = self.server.gui.add_checkbox("Show Trajectory", initial_value=True)
            self.show_heatmap = self.server.gui.add_checkbox("Show Inverse Map", initial_value=True)
            self.show_integrated = self.server.gui.add_checkbox("Show Integrated Map", initial_value=True)
            self.show_per_waypoint = self.server.gui.add_checkbox("Show Per-Waypoint Map", initial_value=False)
        
        # Button callbacks
        @self.next_button.on_click
        def _on_next(_):
            self.continue_event.set()
        
        @self.quit_button.on_click
        def _on_quit(_):
            self.quit_event.set()
            self.continue_event.set()
        
        # Slider callbacks
        @self.waypoint_slider.on_update
        def _on_waypoint_change(_):
            self.current_waypoint_idx = self.waypoint_slider.value
            self._update_per_waypoint_visualization()
        
        @self.theta_slider.on_update
        def _on_theta_change(_):
            self.current_theta_idx = self.theta_slider.value
            self._update_heatmap_visualization()
    
    def _load_robot_urdf(
        self,
        base_position: Optional[np.ndarray] = None,
        base_quat_xyzw: Optional[np.ndarray] = None,
    ) -> None:
        """Load robot URDF into viser with joint control sliders."""
        # Note: we may clear the scene between attempts. In that case we need to
        # recreate the ViserUrdf, but should avoid duplicating GUI sliders.
        if self.robot_urdf is None:
            if not self.robot_urdf_path.exists():
                print(f"[VISER] Warning: Robot URDF not found: {self.robot_urdf_path}")
                return
        
            try:
                from viser.extras import ViserUrdf as _ViserUrdf
                resolved = _resolve_urdf_packages(self.robot_urdf_path)
                self.robot_urdf = _ViserUrdf(
                    self.server,
                    urdf_or_path=resolved,
                    root_node_name="/robot_urdf",
                    load_meshes=True,
                    load_collision_meshes=False,
                )
                print(f"[VISER] Loaded robot URDF: {self.robot_urdf_path.name}")
            except Exception as exc:
                print(f"[VISER] Warning: Failed to load robot URDF: {exc}")
                self.robot_urdf = None
                return
        
        if self.robot_urdf is None:
            return

        # Align the robot base pose to match PyBullet.
        if base_position is not None and base_quat_xyzw is not None:
            try:
                quat = np.asarray(base_quat_xyzw, dtype=float)
                wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)
                pos = np.asarray(base_position, dtype=float)
                for attr in ("_visual_root_frame", "_collision_root_frame"):
                    root_frame = getattr(self.robot_urdf, attr, None)
                    if root_frame is not None:
                        root_frame.position = pos
                        root_frame.wxyz = wxyz
            except Exception:
                pass

        # Create sliders only once.
        if not self.robot_joint_sliders:
            joint_limits = self.robot_urdf.get_actuated_joint_limits()
            self.robot_joint_sliders = []
            self.robot_initial_cfg = []

            with self.server.gui.add_folder("Robot Joints"):
                for joint_name, (lower, upper) in joint_limits.items():
                    lower_val = lower if lower is not None else -np.pi
                    upper_val = upper if upper is not None else np.pi
                    init = 0.0 if lower_val < -0.1 and upper_val > 0.1 else (lower_val + upper_val) / 2.0
                    slider = self.server.gui.add_slider(
                        label=joint_name,
                        min=lower_val,
                        max=upper_val,
                        step=1e-3,
                        initial_value=init,
                    )
                    self.robot_joint_sliders.append(slider)
                    self.robot_initial_cfg.append(init)
        
        # Update callback
        if self.robot_joint_sliders and not self._robot_slider_callbacks_registered:
            def _update_robot_cfg(_=None) -> None:
                if not self.robot_joint_sliders or self.robot_urdf is None:
                    return
                cfg = np.array([s.value for s in self.robot_joint_sliders], dtype=float)
                self.robot_urdf.update_cfg(cfg)

            for s in self.robot_joint_sliders:
                s.on_update(_update_robot_cfg)
            self._robot_slider_callbacks_registered = True
        
        # Reset button
        reset_btn = self.server.gui.add_button("Reset Robot Joints")
        
        @reset_btn.on_click
        def _(_: object) -> None:
            for slider, init in zip(self.robot_joint_sliders, self.robot_initial_cfg):
                slider.value = init
        
        # Apply current config (or the stored initial one).
        if self.robot_joint_sliders:
            cfg = np.array([s.value for s in self.robot_joint_sliders], dtype=float)
            self.robot_urdf.update_cfg(cfg)
        elif self.robot_initial_cfg:
            self.robot_urdf.update_cfg(np.array(self.robot_initial_cfg, dtype=float))
    
    def _load_object_urdf(
        self,
        urdf_path: Path,
        position: np.ndarray,
        quat: np.ndarray,
        scale: float = 1.0,
    ) -> None:
        """Load object URDF into viser at specified pose.
        
        Args:
            urdf_path: Path to object URDF.
            position: Object position in world frame.
            quat: Object quaternion (xyzw) in world frame.
            scale: Uniform scaling factor (matches PyBullet globalScaling).
        """
        if not urdf_path.exists():
            print(f"[VISER] Warning: Object URDF not found: {urdf_path}")
            return
        
        try:
            from viser.extras import ViserUrdf as _ViserUrdf
            resolved = _resolve_urdf_packages(urdf_path, base_path=urdf_path.parent)

            # Use ViserUrdf's built-in `scale=` (supported by current viser).
            # Note: some viser builds do not expose scene.set_object_transform().
            self.object_urdf = _ViserUrdf(
                self.server,
                urdf_or_path=resolved,
                scale=float(scale),
                root_node_name="/object_urdf",
                load_meshes=True,
                load_collision_meshes=False,
            )
            print(f"[VISER] Loaded object URDF: {urdf_path.name}")

            # Apply pose to the URDF root frame(s). ViserUrdf creates frames under
            # /object_urdf/visual and optionally /object_urdf/collision.
            wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)
            for attr in ("_visual_root_frame", "_collision_root_frame"):
                root_frame = getattr(self.object_urdf, attr, None)
                if root_frame is not None:
                    try:
                        root_frame.position = np.asarray(position, dtype=float)
                        root_frame.wxyz = wxyz
                    except Exception:
                        pass
        except Exception as exc:
            print(f"[VISER] Warning: Failed to load object URDF: {exc}")
    
    def set_object_urdf(self, urdf_path: Path) -> None:
        """Set the object URDF path for visualization.
        
        Args:
            urdf_path: Path to object URDF file.
        """
        self.object_urdf_path = Path(urdf_path) if urdf_path else None
    
    def set_robot_joint_config(self, joint_angles: np.ndarray) -> None:
        """Update robot visualization to show specific joint configuration.
        
        Args:
            joint_angles: Array of joint angles for the robot arm (7 DOF).
                         Will be padded with zeros for gripper joints if needed.
        """
        if self.robot_urdf is None:
            return
        
        # Get the expected number of actuated joints from the URDF
        n_actuated = len(self.robot_joint_sliders)
        n_provided = len(joint_angles)
        
        if n_provided == n_actuated:
            # Exact match, use as-is
            cfg = joint_angles
        elif n_provided < n_actuated:
            # Pad with zeros for additional joints (e.g., gripper fingers)
            cfg = np.zeros(n_actuated)
            cfg[:n_provided] = joint_angles
        else:
            # More provided than expected, truncate
            cfg = joint_angles[:n_actuated]
        
        # Update sliders to match
        for i, (slider, angle) in enumerate(zip(self.robot_joint_sliders, cfg)):
            slider.value = float(angle)
        
        # Update visualization
        self.robot_urdf.update_cfg(cfg)
    
    def visualize_attempt(
        self,
        trajectory_obj: Sequence[Tuple[np.ndarray, np.ndarray]],
        sampler,  # IntegratedInverseMapSampler or None
        object_pos: np.ndarray,
        object_quat: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        theta_grid: np.ndarray,
        integrated_scores: np.ndarray,
        base_euler: np.ndarray,
        z_height: float,
        attempt_number: int = 0,
        object_urdf_path: Optional[Path] = None,
        robot_joint_angles: Optional[np.ndarray] = None,
        object_scale: float = 1.0,
        robot_base_pos: Optional[np.ndarray] = None,
        robot_base_quat: Optional[np.ndarray] = None,
        grasp_pos_obj: Optional[np.ndarray] = None,
        grasp_quat_obj: Optional[np.ndarray] = None,
        grasp_candidates_obj: Optional[Sequence[Tuple[np.ndarray, np.ndarray, float]]] = None,
    ) -> str:
        """Visualize a demo generation attempt and wait for user input.
        
        Args:
            trajectory_obj: Object-frame trajectory.
            sampler: IntegratedInverseMapSampler instance.
            object_pos: Sampled object position.
            object_quat: Sampled object quaternion.
            x_grid, y_grid, theta_grid: Grids for distribution.
            integrated_scores: 3D integrated scores array.
            base_euler: Base euler angles.
            z_height: Object z-height.
            attempt_number: Current attempt number.
            object_urdf_path: Optional path to object URDF for mesh visualization.
            robot_joint_angles: Optional robot joint configuration to display.
            
        Returns:
            'next' to continue, 'quit' to stop.
        """
        # Clear previous visualization.
        # NOTE: scene.reset() clears *all* scene nodes (including URDF meshes).
        self.server.scene.reset()
        self.object_urdf = None
        self.robot_urdf = None
        
        # Store data for interactive updates
        self.trajectory_obj = trajectory_obj
        self.sampler = sampler
        self.object_pos = object_pos
        self.object_quat = object_quat
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.theta_grid = theta_grid
        self.integrated_scores = integrated_scores
        self.base_euler = base_euler
        self.z_height = z_height

        # Optional pose alignment / grasp visualization inputs.
        self.robot_base_pos = robot_base_pos
        self.robot_base_quat = robot_base_quat
        self.grasp_pos_obj = grasp_pos_obj
        self.grasp_quat_obj = grasp_quat_obj
        self.grasp_candidates_obj = grasp_candidates_obj
        
        # Update slider ranges
        self.waypoint_slider.max = max(len(trajectory_obj) - 1, 0)
        self.theta_slider.max = max(len(theta_grid) - 1, 0)
        
        # Transform trajectory to world frame
        R_obj = R_scipy.from_quat(object_quat).as_matrix()
        self.trajectory_world = []
        for R_wp, p_wp in trajectory_obj:
            R_world = R_obj @ R_wp
            p_world = R_obj @ p_wp + object_pos
            self.trajectory_world.append((R_world, p_world))
        
        # Compute waypoint scores
        if sampler is not None and hasattr(sampler, 'rmap'):
            self.waypoint_scores = self._compute_waypoint_scores(sampler.rmap, self.trajectory_world)
        else:
            self.waypoint_scores = [0.0] * len(trajectory_obj)
        
        # Add title
        n_reachable = sum(1 for s in self.waypoint_scores if s > 0)
        self.server.gui.add_markdown(
            f"## Attempt {attempt_number}\n"
            f"**Waypoints reachable:** {n_reachable}/{len(self.waypoint_scores)}\n"
            f"**Object position:** ({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f})\n"
            f"**Max integrated score:** {np.max(integrated_scores):.3f}\n"
        )
        
        # Load robot URDF (re-created after scene reset)
        self._load_robot_urdf(base_position=robot_base_pos, base_quat_xyzw=robot_base_quat)
        
        # Update robot joint configuration if provided
        if robot_joint_angles is not None:
            self.set_robot_joint_config(robot_joint_angles)
        
        # Load object URDF if provided
        if object_urdf_path is not None:
            self._load_object_urdf(Path(object_urdf_path), object_pos, object_quat, scale=object_scale)
        
        # Visualize components
        self._visualize_coordinate_system()
        self._visualize_trajectory()
        self._visualize_object(object_pos, object_quat)
        self._visualize_grasps()
        self._visualize_integrated_map()
        
        # Pre-compute per-waypoint maps if enabled
        if self.show_per_waypoint.value and sampler is not None:
            print("  [VISER] Pre-computing per-waypoint maps...")
            self.per_waypoint_scores = self._compute_per_waypoint_scores()
            self._update_per_waypoint_visualization()
        
        # Reset event and wait for user
        self.continue_event.clear()
        print(f"\n[VISER] Visualization ready. Press 'Next Attempt' button or 'n' key to continue, 'q' to quit.")
        print(f"[VISER] View at: http://localhost:{self.port}")
        
        # Wait for user input
        self.continue_event.wait()
        
        if self.quit_event.is_set():
            return 'quit'
        return 'next'
    
    def _compute_waypoint_scores(self, rmap, trajectory_world):
        """Compute reachability score for each waypoint."""
        scores = []
        for R_wp, p_wp in trajectory_world:
            tf = np.eye(4)
            tf[:3, :3] = R_wp
            tf[:3, 3] = p_wp
            
            try:
                indices = rmap.get_indices_for_ee_pose(tf)
                is_reachable = float(rmap.is_reachable(indices))
                scores.append(is_reachable)
            except (IndexError, ValueError):
                scores.append(0.0)
        
        return scores
    
    def _compute_per_waypoint_scores(self):
        """Compute per-waypoint reachability for each grid cell."""
        n_waypoints = len(self.trajectory_obj)
        nx, ny, ntheta = len(self.x_grid), len(self.y_grid), len(self.theta_grid)
        
        per_waypoint_scores = np.zeros((n_waypoints, nx, ny, ntheta), dtype=np.float32)
        
        # Pre-compute trajectory as 4x4 transforms
        traj_obj_tf = []
        for R_o, p_o in self.trajectory_obj:
            tf = np.eye(4)
            tf[:3, :3] = np.asarray(R_o)
            tf[:3, 3] = np.asarray(p_o)
            traj_obj_tf.append(tf)
        
        # Pre-compute rotation matrices
        R_wo_all = np.zeros((ntheta, 3, 3), dtype=np.float64)
        for ith, theta in enumerate(self.theta_grid):
            euler = self.base_euler.copy()
            euler[2] += theta
            R_wo_all[ith] = R_scipy.from_euler('xyz', euler).as_matrix()
        
        # Compute scores
        for wp_idx in range(n_waypoints):
            if wp_idx % 10 == 0:
                print(f"    Computing waypoint {wp_idx}/{n_waypoints}...")
            
            tf_obj = traj_obj_tf[wp_idx]
            
            for ith in range(ntheta):
                R_wo = R_wo_all[ith]
                
                for ix, x in enumerate(self.x_grid):
                    for iy, y in enumerate(self.y_grid):
                        T_wo = np.eye(4)
                        T_wo[:3, :3] = R_wo
                        T_wo[:3, 3] = [x, y, self.z_height]
                        
                        tf_world = T_wo @ tf_obj
                        
                        try:
                            indices = self.sampler.rmap.get_indices_for_ee_pose(tf_world)
                            is_reachable = float(self.sampler.rmap.is_reachable(indices))
                        except (IndexError, ValueError):
                            is_reachable = 0.0
                        
                        per_waypoint_scores[wp_idx, ix, iy, ith] = is_reachable
        
        return per_waypoint_scores
    
    def _visualize_coordinate_system(self):
        """Add world coordinate frame."""
        # Ground grid: project uses z-up.
        self.server.scene.add_grid(
            "/ground",
            width=2.0,
            height=2.0,
            cell_size=0.1,
            position=(0.0, 0.0, 0.0),
        )
        
        # World axes
        axis_length = 0.3
        self.server.scene.add_frame(
            "/world",
            axes_length=axis_length,
            axes_radius=0.01,
        )
    
    def _visualize_trajectory(self):
        """Visualize trajectory waypoints."""
        if not self.show_trajectory.value:
            return
        
        positions = np.array([p for R, p in self.trajectory_world])
        
        # Trajectory line
        if len(positions) > 1:
            self.server.scene.add_spline_catmull_rom(
                "/trajectory/path",
                positions=positions,
                color=(100, 100, 100),
                line_width=2.0,
            )
        
        # Waypoints as spheres
        for i, ((R, p), score) in enumerate(zip(self.trajectory_world, self.waypoint_scores)):
            # Color based on reachability
            if score > 0:
                color = (0, 255, 0)  # Green = reachable
            else:
                color = (255, 0, 0)  # Red = unreachable
            
            self.server.scene.add_icosphere(
                f"/trajectory/waypoint_{i}",
                radius=0.015,
                color=color,
                position=p,
            )
            
            # Add frame for every 5th waypoint
            if i % 5 == 0:
                self.server.scene.add_frame(
                    f"/trajectory/frame_{i}",
                    wxyz=R_scipy.from_matrix(R).as_quat()[[3, 0, 1, 2]],  # viser uses wxyz
                    position=p,
                    axes_length=0.03,
                    axes_radius=0.002,
                )
    
    def _visualize_object(self, pos, quat):
        """Visualize object position."""
        self.server.scene.add_icosphere(
            "/object/center",
            radius=0.05,
            color=(128, 0, 128),
            position=pos,
        )
        
        # Object frame
        R_obj = R_scipy.from_quat(quat)
        self.server.scene.add_frame(
            "/object/frame",
            wxyz=R_obj.as_quat()[[3, 0, 1, 2]],
            position=pos,
            axes_length=0.1,
            axes_radius=0.005,
        )

    def _visualize_grasps(self) -> None:
        """Visualize selected grasp pose and optional grasp candidates."""
        if getattr(self, "grasp_pos_obj", None) is None or getattr(self, "grasp_quat_obj", None) is None:
            return

        obj_pos = np.asarray(self.object_pos, dtype=float)
        obj_quat = np.asarray(self.object_quat, dtype=float)
        grasp_pos_obj = np.asarray(self.grasp_pos_obj, dtype=float)
        grasp_quat_obj = np.asarray(self.grasp_quat_obj, dtype=float)

        R_obj = R_scipy.from_quat(obj_quat)
        pos_world = R_obj.apply(grasp_pos_obj) + obj_pos
        quat_world = (R_obj * R_scipy.from_quat(grasp_quat_obj)).as_quat()  # xyzw

        self.server.scene.add_icosphere(
            "/grasp/selected/pos",
            radius=0.012,
            color=(255, 0, 255),
            position=pos_world,
        )
        self.server.scene.add_frame(
            "/grasp/selected/frame",
            wxyz=quat_world[[3, 0, 1, 2]],
            position=pos_world,
            axes_length=0.08,
            axes_radius=0.004,
        )

        # Optional candidates (e.g., from GraspGen). Render as smaller points.
        candidates = getattr(self, "grasp_candidates_obj", None)
        if not candidates:
            return

        # Normalize confidences for coloring.
        confs = [float(c[2]) for c in candidates]
        cmin = float(min(confs))
        cmax = float(max(confs))
        denom = (cmax - cmin) if (cmax - cmin) > 1e-9 else 1.0

        for i, (p_obj_i, q_obj_i, conf_i) in enumerate(candidates):
            p_obj_i = np.asarray(p_obj_i, dtype=float)
            q_obj_i = np.asarray(q_obj_i, dtype=float)
            p_w_i = R_obj.apply(p_obj_i) + obj_pos
            q_w_i = (R_obj * R_scipy.from_quat(q_obj_i)).as_quat()

            t = (float(conf_i) - cmin) / denom
            # Color: red (low) -> green (high)
            color = (int(255 * (1 - t)), int(255 * t), 0)
            self.server.scene.add_icosphere(
                f"/grasp/candidates/{i}",
                radius=0.008,
                color=color,
                position=p_w_i,
            )
            if i < 5:
                self.server.scene.add_frame(
                    f"/grasp/candidates/{i}/frame",
                    wxyz=q_w_i[[3, 0, 1, 2]],
                    position=p_w_i,
                    axes_length=0.04,
                    axes_radius=0.002,
                )
    
    def _visualize_integrated_map(self):
        """Visualize integrated inverse map as heatmap."""
        if not self.show_integrated.value:
            return
        
        theta_idx = int(self.current_theta_idx)
        score_slice = self.integrated_scores[:, :, theta_idx]
        
        # Create point cloud for heatmap
        points = []
        colors = []
        
        for ix, x in enumerate(self.x_grid):
            for iy, y in enumerate(self.y_grid):
                score = score_slice[ix, iy]
                if score > 0:
                    # x/y are object base positions; z is fixed at z_height.
                    points.append([x, y, float(self.z_height) + 0.001])
                    
                    # Color: red (0) -> yellow (0.5) -> green (1)
                    if score < 0.5:
                        r, g, b = 255, int(510 * score), 0
                    else:
                        r, g, b = int(510 * (1 - score)), 255, 0
                    colors.append((r, g, b))
        
        if points:
            self.server.scene.add_point_cloud(
                "/heatmap/integrated",
                points=np.array(points),
                colors=np.array(colors),
                point_size=0.02,
            )
    
    def _update_heatmap_visualization(self):
        """Update heatmap when theta slider changes."""
        # Remove old heatmap
        remove_fn = getattr(self.server.scene, "remove", None)
        if callable(remove_fn):
            try:
                remove_fn("/heatmap/integrated")
            except Exception:
                pass
        # Re-visualize with new theta
        self._visualize_integrated_map()
    
    def _update_per_waypoint_visualization(self):
        """Update per-waypoint heatmap when waypoint slider changes."""
        if not self.show_per_waypoint.value or not hasattr(self, 'per_waypoint_scores'):
            return
        
        # Remove old per-waypoint heatmap
        remove_fn = getattr(self.server.scene, "remove", None)
        if callable(remove_fn):
            try:
                remove_fn("/heatmap/per_waypoint")
            except Exception:
                pass
        
        wp_idx = int(self.current_waypoint_idx)
        theta_idx = int(self.current_theta_idx)
        score_slice = self.per_waypoint_scores[wp_idx, :, :, theta_idx]
        
        # Create point cloud
        points = []
        colors = []
        
        for ix, x in enumerate(self.x_grid):
            for iy, y in enumerate(self.y_grid):
                score = score_slice[ix, iy]
                if score > 0:
                    points.append([x, y, float(self.z_height) + 0.002])
                    
                    # Color for per-waypoint (blue tint)
                    if score < 0.5:
                        r, g, b = 0, int(255 * score), 255
                    else:
                        r, g, b = 0, 255, int(255 * (1 - score))
                    colors.append((r, g, b))
        
        if points:
            self.server.scene.add_point_cloud(
                "/heatmap/per_waypoint",
                points=np.array(points),
                colors=np.array(colors),
                point_size=0.02,
            )
    
    def wait_for_input(self):
        """Block until user presses next or quit."""
        self.continue_event.wait()
        return 'quit' if self.quit_event.is_set() else 'next'
    
    def close(self):
        """Close the viser server."""
        print("[VISER] Closing server...")
        # Server will be garbage collected


def create_viser_visualizer(port: int = 8080) -> Optional[IntegratedInverseMapVisualizer]:
    """Create a viser visualizer if available.
    
    Args:
        port: Port for viser server.
        
    Returns:
        IntegratedInverseMapVisualizer instance or None if viser unavailable.
    """
    if not VISER_AVAILABLE:
        print("Warning: viser not available. Skipping interactive visualization.")
        return None
    
    return IntegratedInverseMapVisualizer(port=port)
