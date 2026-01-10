#!/usr/bin/env python3
"""Generate a base_config.yaml for a custom URDF + affordance annotation.

Compared to ``generate_base_config_from_urdf.py`` this script additionally:

* copies the raw URDF folder into ``data/custom_objects/<name>`` (configurable)
* creates ``mobility_v2.json`` from the URDF joint definition
* converts an affordance annotation (3D bbox) into ``parts_render/<id><handle>.obj``
* stores the handle metadata inside the generated config so downstream scripts can
  register the correct handle name.

Supports both v1 and v2 asset layouts:
- v1: URDF and mesh/ at root level
- v2: URDF and mesh/ inside urdf/ subfolder, with optional scale.txt
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import yaml

from manipulation.custom_object_utils.object_utils import (
    compute_config_metadata,
    detect_asset_version,
    determine_reward_asset_path,
    ensure_annotation_copy,
    ensure_handle_mesh_from_annotation,
    ensure_mobility_file,
    extract_joint_metadata,
    find_first_urdf,
    get_urdf_parent_dir,
    quaternion_from_euler,
    sanitize_object_name,
    serialize_vector,
)


def copy_asset_tree(src_dir: Path, dst_dir: Path, overwrite: bool) -> None:
    """Copy the entire asset directory while keeping user edits intact."""
    if dst_dir.exists() and overwrite:
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        dst_item = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)


def build_config_dict(
    center,
    euler,
    size: float,
    reward_asset_path: str,
    object_name: str,
    handle_name: str,
    solution_path: Path,
    annotation_path: Path,
    urdf_path: Path,
    use_table: bool,
) -> list[dict]:
    # For custom objects, center represents the target world placement position
    # The z coordinate should be 0 (ground level) - the adjust_object_positions 
    # function will compute the proper height to place the object on the ground.
    # Keep the computed x,y from URDF center as a reasonable default, but set z=0.
    placement_center = (center[0], center[1], 0.0)
    
    return [
        {"use_table": bool(use_table)},
        {
            "center": serialize_vector(placement_center),
            "euler": serialize_vector(euler),
            "orientation": serialize_vector(quaternion_from_euler(euler)),
            "lang": "a custom object",
            "name": object_name,
            "on_table": False,
            "reward_asset_path": reward_asset_path,
            "size": float(size),
            "type": "urdf",
            "handle_name": handle_name,
            "annotation_path": str(annotation_path),
            "urdf_path": str(urdf_path),
        },
        {"solution_path": str(solution_path)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate base_config for custom URDF")
    parser.add_argument("--asset-folder", default="data/URDF_with_affordance", help="Folder containing URDF, mesh/, annotation.json (supports both v1 and v2 layouts)")
    parser.add_argument("--output-root", default="data/custom_objects_v2", help="Root folder for processed assets")
    parser.add_argument("--dataset-name", default=None, help="Name for the processed asset folder")
    parser.add_argument("--annotation", default=None, help="Path to annotation.json (defaults to <asset-folder>/annotation.json)")
    parser.add_argument("--joint-name", default=None, help="Name of revolute/prismatic joint to treat as handle")
    parser.add_argument("--handle-name", default=None, help="Override handle name (defaults to child link name)")
    parser.add_argument("--handle-part-id", type=int, default=1, help="Part id used when writing parts_render/<id><handle>.obj")
    parser.add_argument("--handle-padding", type=float, default=0.0, help="Extra padding (meters) added to affordance bbox")
    parser.add_argument("--size", type=float, default=None, help="Optional manual size override")
    parser.add_argument("--object-name", default=None, help="Name stored in config (defaults to prettified dataset name)")
    parser.add_argument("--overwrite-assets", action="store_true", help="Delete target folder before copying assets")
    parser.add_argument("--overwrite-support", action="store_true", help="Regenerate mobility/handle meshes even if they exist")
    parser.add_argument("--use-table", action="store_true", help="Set use_table: true in the config")
    parser.add_argument("--in-place", action="store_true", help="Generate config in the source folder instead of copying to output-root")
    args = parser.parse_args()

    src_dir = Path(args.asset_folder).expanduser().resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Asset folder not found: {src_dir}")
    
    # Detect asset version and find URDF
    asset_version = detect_asset_version(src_dir)
    print(f"Detected asset version: v{asset_version}")
    
    urdf_path = find_first_urdf(src_dir)
    print(f"Found URDF: {urdf_path}")

    dataset_name = args.dataset_name or src_dir.name
    
    # Determine destination directory
    if args.in_place:
        dst_dir = src_dir
    else:
        dst_root = Path(args.output_root).expanduser()
        dst_dir = (dst_root / dataset_name)
        copy_asset_tree(src_dir, dst_dir, overwrite=args.overwrite_assets)
    
    # Find annotation (try root level first, then standard locations)
    annotation_src = None
    if args.annotation:
        annotation_src = Path(args.annotation).expanduser().resolve()
    else:
        # Try root level first (both v1 and v2)
        annotation_src = src_dir / "annotation.json"
        if not annotation_src.exists():
            # v2 might have it at root level
            annotation_src = src_dir / "annotation.json"
    
    if not annotation_src.exists():
        raise FileNotFoundError(f"Annotation JSON not found at {annotation_src}")
    annotation_dst = ensure_annotation_copy(annotation_src, dst_dir)

    # Compute target URDF path in destination directory
    # Preserve the relative path structure (e.g., urdf/object.urdf for v2)
    urdf_rel = urdf_path.relative_to(src_dir)
    target_urdf = dst_dir / urdf_rel
    
    if not target_urdf.exists():
        raise FileNotFoundError(f"Expected URDF not found at {target_urdf}")
    
    joint_info = extract_joint_metadata(target_urdf, joint_name=args.joint_name)
    handle_name = (args.handle_name or joint_info["child_link"]).lower()

    ensure_mobility_file(dst_dir, joint_info, overwrite=args.overwrite_support)
    ensure_handle_mesh_from_annotation(
        dst_dir,
        handle_name,
        annotation_dst,
        part_id=args.handle_part_id,
        padding=args.handle_padding,
        overwrite=args.overwrite_support,
    )

    center, euler, size = compute_config_metadata(target_urdf, size_override=args.size, asset_dir=dst_dir)
    reward_asset_path = determine_reward_asset_path(dst_dir)
    object_name = args.object_name or sanitize_object_name(dataset_name)

    config = build_config_dict(
        center,
        euler,
        size,
        reward_asset_path,
        object_name,
        handle_name,
        dst_dir,
        annotation_dst,
        target_urdf,
        use_table=args.use_table,
    )

    base_config_path = dst_dir / "base_config.yaml"
    with base_config_path.open("w", encoding="utf-8") as fp:
        yaml.dump(config, fp, sort_keys=False)

    print("Saved base config to", base_config_path)
    print("  Asset folder:", dst_dir)
    print("  Asset version:", f"v{asset_version}")
    print("  URDF file:", target_urdf)
    print("  Handle name:", handle_name)
    print("  Reward asset path:", reward_asset_path)
    print("  Size:", f"{size:.4f}")
    print("  Center:", center)
    print("  Euler:", euler)


if __name__ == "__main__":
    main()
