"""RM4D-based inverse reachability filtering for articulated object manipulation.

This module provides utilities for filtering initial object-robot base relative
transformations using the RM4D (Reachability Map 4D) library's inverse reachability
map. This enables efficient filtering of object poses that would place the 
manipulation trajectory outside the robot's workspace.

Key classes:
- TrajectoryReachabilityFilter: Filter object poses based on trajectory reachability
- IntegratedInverseMapSampler: Sample object poses from integrated inverse map
"""
from .trajectory_filter import (
    TrajectoryReachabilityFilter,
    load_or_create_rm4d_map,
    score_trajectory_reachability,
    filter_object_pose_by_trajectory,
    load_object_trajectory,
    generate_straight_line_trajectory,
    IntegratedInverseMapSampler,
    create_integrated_inverse_sampler,
)

__all__ = [
    "TrajectoryReachabilityFilter",
    "load_or_create_rm4d_map",
    "score_trajectory_reachability",
    "filter_object_pose_by_trajectory",
    "load_object_trajectory",
    "generate_straight_line_trajectory",
    "IntegratedInverseMapSampler",
    "create_integrated_inverse_sampler",
]
