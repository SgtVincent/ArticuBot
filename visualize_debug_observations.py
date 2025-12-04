#!/usr/bin/env python3
"""
Visualize debug observations from evaluation using viser.

This script loads the debug_observations.pkl file saved during evaluation
(with --debug-save-obs flag) and visualizes the point clouds, gripper positions,
goal predictions, and actions over time.

Usage:
    python visualize_debug_observations.py \
        --obs-path outputs/eval_customs/sim_microwave_good/pretrained/debug_trial_000/debug_observations.pkl

    # With specific timestep
    python visualize_debug_observations.py \
        --obs-path outputs/eval_customs/sim_microwave_good/pretrained/debug_trial_000/debug_observations.pkl \
        --timestep 10
"""

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import viser

# Color palette for visualization
COLORS = {
    'point_cloud': (100, 150, 200),      # Light blue for object points
    'gripper': (0, 255, 0),               # Green for gripper
    'goal': (255, 0, 0),                  # Red for goal
    'action_start': (255, 255, 0),        # Yellow for action start
    'action_end': (255, 128, 0),          # Orange for action end
}


def load_debug_observations(obs_path: str) -> List[Dict[str, Any]]:
    """Load debug observations from pickle file."""
    with open(obs_path, 'rb') as f:
        return pickle.load(f)


def add_point_cloud(server: viser.ViserServer, name: str, points: np.ndarray, 
                    color: tuple = (100, 150, 200), point_size: float = 0.003):
    """Add a point cloud to the scene."""
    if points is None or len(points) == 0:
        return
    
    # Ensure points are 2D (N, 3)
    if points.ndim == 3:
        # Take the last observation step if shape is (n_obs_steps, N, 3)
        points = points[-1]
    
    # Create colors array
    colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    
    server.scene.add_point_cloud(
        name,
        points=points.astype(np.float32),
        colors=colors,
        point_size=point_size,
    )


def add_gripper_visualization(server: viser.ViserServer, name: str, 
                               gripper_pcd: np.ndarray, color: tuple = (0, 255, 0)):
    """Add gripper point cloud visualization with spheres and connections."""
    if gripper_pcd is None:
        return
    
    # Take last observation step if needed
    if gripper_pcd.ndim == 3:
        gripper_pcd = gripper_pcd[-1]
    
    # Add spheres for each gripper point
    for i, point in enumerate(gripper_pcd):
        server.scene.add_icosphere(
            f"{name}/point_{i}",
            radius=0.008,
            color=color,
            position=tuple(point.tolist()),
        )
    
    # Connect gripper points with lines (assuming 4 points: hand, 2 fingers, eef)
    if len(gripper_pcd) >= 4:
        # Connect hand to fingers and eef
        connections = [(0, 1), (0, 2), (0, 3)]
        for j, (i1, i2) in enumerate(connections):
            server.scene.add_spline_catmull_rom(
                f"{name}/line_{j}",
                positions=np.array([gripper_pcd[i1], gripper_pcd[i2]], dtype=np.float32),
                color=color,
                line_width=2.0,
            )


def add_goal_visualization(server: viser.ViserServer, name: str,
                           goal_points: np.ndarray, color: tuple = (255, 0, 0)):
    """Add goal end-effector points visualization."""
    if goal_points is None:
        return
    
    # Handle different shapes: (1, 1, 4, 3) or (1, 4, 3) or (4, 3)
    while goal_points.ndim > 2:
        goal_points = goal_points[0]
    
    # Add spheres for each goal point
    for i, point in enumerate(goal_points):
        server.scene.add_icosphere(
            f"{name}/point_{i}",
            radius=0.01,
            color=color,
            position=tuple(point.tolist()),
        )
    
    # Connect goal points
    if len(goal_points) >= 4:
        connections = [(0, 1), (0, 2), (0, 3)]
        for j, (i1, i2) in enumerate(connections):
            server.scene.add_spline_catmull_rom(
                f"{name}/line_{j}",
                positions=np.array([goal_points[i1], goal_points[i2]], dtype=np.float32),
                color=color,
                line_width=2.0,
            )


def visualize_timestep(server: viser.ViserServer, obs: Dict[str, Any], 
                       timestep: int, show_goal: bool = True):
    """Visualize a single timestep's observation."""
    # Clear previous visualization (by adding with same names, they get replaced)
    
    # Add point cloud
    if obs.get('point_cloud') is not None:
        add_point_cloud(
            server, 
            f"/timestep/point_cloud",
            obs['point_cloud'],
            color=COLORS['point_cloud'],
            point_size=0.003
        )
    
    # Add gripper visualization
    if obs.get('gripper_pcd') is not None:
        add_gripper_visualization(
            server,
            f"/timestep/gripper",
            obs['gripper_pcd'],
            color=COLORS['gripper']
        )
    
    # Add goal visualization (if available)
    if show_goal and obs.get('goal_eef_points') is not None:
        add_goal_visualization(
            server,
            f"/timestep/goal",
            obs['goal_eef_points'],
            color=COLORS['goal']
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize debug observations from evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize debug observations with interactive timestep slider
    python visualize_debug_observations.py \\
        --obs-path outputs/eval_customs/sim_microwave_good/pretrained/debug_trial_000/debug_observations.pkl

    # Start at specific timestep
    python visualize_debug_observations.py \\
        --obs-path outputs/eval_customs/sim_microwave_good/pretrained/debug_trial_000/debug_observations.pkl \\
        --timestep 30
        """
    )
    
    parser.add_argument(
        '--obs-path', type=str, required=True,
        help='Path to debug_observations.pkl file'
    )
    parser.add_argument(
        '--timestep', type=int, default=0,
        help='Initial timestep to visualize'
    )
    parser.add_argument(
        '--port', type=int, default=8080,
        help='Port for viser server'
    )
    parser.add_argument(
        '--point-size', type=float, default=0.003,
        help='Size of points in point cloud'
    )
    
    args = parser.parse_args()
    
    # Load observations
    print(f"Loading observations from: {args.obs_path}")
    observations = load_debug_observations(args.obs_path)
    print(f"Loaded {len(observations)} timesteps")
    
    # Print observation info
    if len(observations) > 0:
        obs0 = observations[0]
        print(f"\nObservation keys: {list(obs0.keys())}")
        if obs0.get('point_cloud') is not None:
            print(f"Point cloud shape: {obs0['point_cloud'].shape}")
        if obs0.get('gripper_pcd') is not None:
            print(f"Gripper pcd shape: {obs0['gripper_pcd'].shape}")
    
    # Start viser server
    print(f"\nStarting viser server on port {args.port}...")
    server = viser.ViserServer(port=args.port)
    
    # Configure scene
    server.scene.configure_default_lights(enabled=True, cast_shadow=True)
    
    # Add coordinate frame at origin
    server.scene.add_frame("/world", axes_length=0.1, axes_radius=0.003)
    
    # Add ground grid
    server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.1, position=(0, 0, 0))
    
    # Current state
    current_timestep = [args.timestep]  # Use list to allow modification in callbacks
    show_goal = [True]
    show_trajectory = [False]
    
    # Initial visualization
    if len(observations) > 0:
        visualize_timestep(server, observations[current_timestep[0]], current_timestep[0], show_goal[0])
    
    # Add GUI controls
    with server.gui.add_folder("Observation Info"):
        server.gui.add_markdown(f"**Total timesteps:** {len(observations)}")
        timestep_text = server.gui.add_markdown(f"**Current timestep:** {current_timestep[0]}")
    
    with server.gui.add_folder("Controls"):
        # Timestep slider
        timestep_slider = server.gui.add_slider(
            "Timestep",
            min=0,
            max=len(observations) - 1,
            step=1,
            initial_value=current_timestep[0],
        )
        
        # Show goal checkbox
        goal_checkbox = server.gui.add_checkbox("Show Goal", initial_value=True)
        
        # Show full trajectory checkbox
        trajectory_checkbox = server.gui.add_checkbox("Show Full Trajectory", initial_value=False)
        
        # Playback controls
        play_button = server.gui.add_button("Play")
        playback_speed = server.gui.add_slider("Playback Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
    
    with server.gui.add_folder("Legend"):
        server.gui.add_markdown("🔵 **Blue**: Object point cloud")
        server.gui.add_markdown("🟢 **Green**: Gripper position")
        server.gui.add_markdown("🔴 **Red**: Goal position")
    
    # Callback for timestep slider
    @timestep_slider.on_update
    def on_timestep_change(_):
        current_timestep[0] = int(timestep_slider.value)
        timestep_text.content = f"**Current timestep:** {current_timestep[0]}"
        if current_timestep[0] < len(observations):
            visualize_timestep(
                server, 
                observations[current_timestep[0]], 
                current_timestep[0],
                show_goal[0]
            )
    
    # Callback for goal checkbox
    @goal_checkbox.on_update
    def on_goal_toggle(_):
        show_goal[0] = goal_checkbox.value
        if current_timestep[0] < len(observations):
            visualize_timestep(
                server, 
                observations[current_timestep[0]], 
                current_timestep[0],
                show_goal[0]
            )
    
    # Callback for trajectory checkbox
    @trajectory_checkbox.on_update
    def on_trajectory_toggle(_):
        show_trajectory[0] = trajectory_checkbox.value
        if show_trajectory[0]:
            # Add full trajectory visualization
            add_full_trajectory(server, observations)
        else:
            # Remove trajectory visualization
            try:
                server.scene.remove("/trajectory")
            except:
                pass
    
    # Play button state
    is_playing = [False]
    
    @play_button.on_click
    def on_play_click(_):
        is_playing[0] = not is_playing[0]
        play_button.name = "Pause" if is_playing[0] else "Play"
    
    def add_full_trajectory(server: viser.ViserServer, observations: List[Dict]):
        """Add visualization of the full gripper trajectory."""
        gripper_positions = []
        goal_positions = []
        
        for obs in observations:
            if obs.get('gripper_pcd') is not None:
                gripper = obs['gripper_pcd']
                if gripper.ndim == 3:
                    gripper = gripper[-1]
                # Use the center of gripper (average of all points)
                center = gripper.mean(axis=0)
                gripper_positions.append(center)
            
            if obs.get('goal_eef_points') is not None:
                goal = obs['goal_eef_points']
                while goal.ndim > 2:
                    goal = goal[0]
                center = goal.mean(axis=0)
                goal_positions.append(center)
        
        # Add gripper trajectory line
        if len(gripper_positions) >= 2:
            positions = np.array(gripper_positions, dtype=np.float32)
            server.scene.add_spline_catmull_rom(
                "/trajectory/gripper",
                positions=positions,
                color=COLORS['gripper'],
                line_width=2.0,
            )
            
            # Add start and end markers
            server.scene.add_icosphere(
                "/trajectory/gripper_start",
                radius=0.012,
                color=(0, 200, 0),
                position=tuple(positions[0].tolist()),
            )
            server.scene.add_icosphere(
                "/trajectory/gripper_end",
                radius=0.012,
                color=(0, 100, 0),
                position=tuple(positions[-1].tolist()),
            )
        
        # Add goal trajectory line
        if len(goal_positions) >= 2:
            positions = np.array(goal_positions, dtype=np.float32)
            server.scene.add_spline_catmull_rom(
                "/trajectory/goal",
                positions=positions,
                color=COLORS['goal'],
                line_width=2.0,
            )
    
    print(f"\n{'='*60}")
    print(f"Viser server running at http://localhost:{args.port}")
    print(f"{'='*60}")
    print("\nVisualization shows:")
    print("  - Blue points: Object point cloud")
    print("  - Green spheres: Gripper position")
    print("  - Red spheres: Goal end-effector position")
    print("\nControls:")
    print("  - Use the timestep slider to navigate through time")
    print("  - Toggle 'Show Goal' to show/hide goal positions")
    print("  - Toggle 'Show Full Trajectory' to see gripper path")
    print("  - Click 'Play' for automatic playback")
    print("\nPress Ctrl+C to exit")
    
    try:
        while True:
            # Handle playback
            if is_playing[0]:
                next_timestep = (current_timestep[0] + 1) % len(observations)
                current_timestep[0] = next_timestep
                timestep_slider.value = next_timestep
                timestep_text.content = f"**Current timestep:** {current_timestep[0]}"
                visualize_timestep(
                    server, 
                    observations[current_timestep[0]], 
                    current_timestep[0],
                    show_goal[0]
                )
                time.sleep(0.1 / playback_speed.value)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
