"""Custom object utilities for ArticuBot demo generation.

This package provides utilities for generating demonstrations with custom
articulated objects (URDFs with affordance annotations).

Main modules:
- demo_utils: Core demo generation utilities (heuristic method)
- demo_utils_integrated: Integrated inverse map sampling (direct sampling)
- object_utils: URDF/annotation processing utilities
- contact_trajectory: Compute gripper trajectories in object frame
- integrated_sampling: Integrated inverse map sampling
- graspgen_client: Docker-based grasp prediction
- visualization_utils: Debug visualization helpers
"""
from manipulation.custom_object_utils.demo_utils import (
    custom_gen_init_state,
    custom_execute,
    create_variant_config,
    parse_config_metadata,
    resolve_relative_path,
)

from manipulation.custom_object_utils.demo_utils_integrated import (
    custom_gen_init_state_integrated,
)

from manipulation.custom_object_utils.object_utils import (
    find_first_urdf,
    ensure_support_files,
    load_predicted_grasps,
    ensure_handle_mesh_from_annotation,
    ensure_mobility_file,
    extract_joint_metadata,
)

from manipulation.custom_object_utils.contact_trajectory import (
    JointKinematics,
    compute_contact_trajectory,
    compute_full_manipulation_trajectory,
    parse_joint_kinematics_from_urdf,
    parse_joint_kinematics_from_mobility,
)


from manipulation.custom_object_utils.integrated_sampling import (
    IntegratedSamplingConfig,
    integrated_sample_initial_state,
    create_sampler_from_map,
)

__all__ = [
    # demo_utils
    "custom_gen_init_state",
    "custom_execute",
    "create_variant_config",
    "parse_config_metadata",
    "resolve_relative_path",
    # demo_utils_integrated
    "custom_gen_init_state_integrated",
    # object_utils
    "find_first_urdf",
    "ensure_support_files",
    "load_predicted_grasps",
    "ensure_handle_mesh_from_annotation",
    "ensure_mobility_file",
    "extract_joint_metadata",
    # contact_trajectory
    "JointKinematics",
    "compute_contact_trajectory",
    "compute_full_manipulation_trajectory",
    "parse_joint_kinematics_from_urdf",
    "parse_joint_kinematics_from_mobility",
    # integrated_sampling
    "IntegratedSamplingConfig",
    "integrated_sample_initial_state",
    "create_sampler_from_map",
]
