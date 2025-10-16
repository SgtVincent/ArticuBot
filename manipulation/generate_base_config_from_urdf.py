#!/usr/bin/env python3
"""
Generate a base_config.yaml for a dataset folder that contains a URDF.

Usage:
    python manipulation/generate_base_config_from_urdf.py /path/to/data/<asset_folder>

The script finds the first URDF file in the folder (by extension .urdf), parses
visual origin positions and rpy if available and writes a minimal
base_config.yaml mirroring the repository's expected format.

It writes:
- use_table: false
- a dict with center, euler, lang, name, on_table, reward_asset_path, size, type, solution_path

Notes:
- If no mesh-loading libraries are available this script uses conservative
  defaults (size=1.0, euler=[0,0,0]). If open3d is installed it will try to
  compute a tighter size estimate from the visual meshes.
"""

import sys
import os
import argparse
import xml.etree.ElementTree as ET
import yaml
from math import isfinite


def find_urdf(folder):
    for entry in os.listdir(folder):
        if entry.lower().endswith('.urdf'):
            return os.path.join(folder, entry)
    return None


def parse_origin(origin_str):
    # origin_str example: "0.73630002 0.04670638 0.0092915"
    if origin_str is None:
        return [0.0, 0.0, 0.0]
    parts = origin_str.strip().split()
    try:
        vals = [float(p) for p in parts]
    except Exception:
        return [0.0, 0.0, 0.0]
    if len(vals) == 3:
        return vals
    return [0.0, 0.0, 0.0]


def parse_rpy(rpy_str):
    if rpy_str is None:
        return [0.0, 0.0, 0.0]
    parts = rpy_str.strip().split()
    try:
        vals = [float(p) for p in parts]
    except Exception:
        return [0.0, 0.0, 0.0]
    if len(vals) == 3:
        return vals
    return [0.0, 0.0, 0.0]


def estimate_size_from_meshes(mesh_paths):
    # optional: try to compute bbox extents if open3d is available
    try:
        import open3d as o3d
    except Exception:
        return None

    min_corner = [float('inf')] * 3
    max_corner = [float('-inf')] * 3
    for p in mesh_paths:
        if not os.path.exists(p):
            continue
        try:
            mesh = o3d.io.read_triangle_mesh(p)
            if mesh.is_empty():
                continue
            bbox = mesh.get_axis_aligned_bounding_box()
            minc = bbox.get_min_bound()
            maxc = bbox.get_max_bound()
            for i in range(3):
                min_corner[i] = min(min_corner[i], float(minc[i]))
                max_corner[i] = max(max_corner[i], float(maxc[i]))
        except Exception:
            continue

    if not all(isfinite(x) for x in min_corner + max_corner):
        return None

    extent = [max_corner[i] - min_corner[i] for i in range(3)]
    # use geometric mean of extents as a proxy for size
    prod = 1.0
    for e in extent:
        prod *= max(e, 1e-6)
    size = prod ** (1.0/3.0)
    # avoid degenerate sizes
    if size <= 0 or not isfinite(size):
        return None
    return float(size)


def collect_mesh_paths(folder, urdf_root, urdf_tree):
    # collect mesh filenames referenced in <mesh filename="..."/>
    mesh_files = []
    for mesh in urdf_tree.iter('mesh'):
        fn = mesh.get('filename')
        if fn is None:
            continue
        # if path is absolute keep it; if relative, join with folder
        if os.path.isabs(fn):
            mesh_files.append(fn)
        else:
            # try relative to the urdf folder
            mesh_files.append(os.path.normpath(os.path.join(urdf_root, fn)))
    return mesh_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset_folder', help='Path to dataset asset folder (contains URDF and mesh/).')
    parser.add_argument('--out', default='base_config.yaml', help='Output YAML filename inside the asset folder')
    parser.add_argument('--size', type=float, default=None, help='Manual size override')
    args = parser.parse_args()

    folder = args.asset_folder
    if not os.path.isdir(folder):
        print('Provided path is not a directory:', folder)
        sys.exit(2)

    urdf_path = find_urdf(folder)
    if urdf_path is None:
        print('No .urdf file found in', folder)
        sys.exit(2)

    print('Using URDF:', urdf_path)
    urdf_root = os.path.dirname(urdf_path)

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    positions = []
    rpy_list = []
    mesh_paths = []

    # search visual/origin and joint/origin
    for visual in root.iter('visual'):
        origin = visual.find('origin')
        if origin is not None:
            pos = parse_origin(origin.get('xyz'))
            positions.append(pos)
            rpy_list.append(parse_rpy(origin.get('rpy')))

    for collision in root.iter('collision'):
        origin = collision.find('origin')
        if origin is not None:
            pos = parse_origin(origin.get('xyz'))
            positions.append(pos)

    for joint in root.iter('joint'):
        origin = joint.find('origin')
        if origin is not None:
            pos = parse_origin(origin.get('xyz'))
            positions.append(pos)
            rpy_list.append(parse_rpy(origin.get('rpy')))

    mesh_paths = collect_mesh_paths(folder, urdf_root, root)

    # default values
    center = [0.0, 0.0, 0.0]
    euler = [0.0, 0.0, 0.0]

    if positions:
        arr = list(zip(*positions))
        # average each coordinate
        center = [float(sum(xs) / len(xs)) for xs in arr]

    if rpy_list:
        arr = list(zip(*rpy_list))
        euler = [float(sum(xs)/len(xs)) for xs in arr]

    # attempt to estimate size if not provided
    size = args.size
    if size is None:
        size = estimate_size_from_meshes(mesh_paths)
    if size is None:
        size = 1.0

    # assemble YAML structure
    asset_basename = os.path.basename(os.path.normpath(folder))
    urdf_basename = os.path.splitext(os.path.basename(urdf_path))[0]
    # make a friendly name from URDF base name
    name = urdf_basename.replace('_', ' ').title()

    solution_path = args.asset_folder # use folder as solution path

    config = [
        {'use_table': False},
        {
            'center': str([round(float(c), 6) for c in center]),
            'euler': str([round(float(e), 6) for e in euler]),
            'lang': 'a custom object',
            'name': name,
            'on_table': False,
            'reward_asset_path': asset_basename,
            'size': float(size),
            'type': 'urdf',
        },
        {'solution_path': solution_path}
    ]

    out_path = os.path.join(folder, args.out)
    if os.path.exists(out_path):
        # back up
        import shutil
        bak = out_path + '.bak'
        print('Backing up existing', out_path, 'to', bak)
        shutil.copyfile(out_path, bak)

    with open(out_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print('Wrote base_config to', out_path)
    print('Generated values:')
    print('  name:', name)
    print('  center:', center)
    print('  euler:', euler)
    print('  size:', size)
    print('  reward_asset_path:', asset_basename)


if __name__ == '__main__':
    main()
