import os
import yaml
import numpy as np
from PIL import Image, ImageSequence
from moviepy.editor import ImageSequenceClip
import os.path as osp
import pybullet as p
import json
import multiprocessing
import multiprocessing.pool
from multiprocessing import Process
import objaverse
import trimesh
from objaverse_utils.utils import text_to_uid_dict, partnet_mobility_dict, sapaien_cannot_vhacd_part_dict

from .defaults import *

def normalize_obj(obj_file_path):
    """
    Normalize an OBJ file by centering and scaling vertices.
    
    This function reads an OBJ file, centers the vertices around the origin,
    and scales them to fit within a unit sphere (normalized to [-1, 1] range).
    The normalized mesh is saved with '_normalized' suffix.
    
    Args:
        obj_file_path (str): Path to the input OBJ file.
    
    Returns:
        None
    
    Side Effects:
        Creates a new file: {obj_file_path}_normalized.obj
    
    Example:
        >>> normalize_obj('models/chair.obj')
        # Creates models/chair_normalized.obj
    """
    vertices = []
    with open(osp.join(obj_file_path), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:]])
    
    vertices = np.array(vertices).reshape(-1, 3)
    vertices = vertices - np.mean(vertices, axis=0) # center to zero
    vertices = vertices / np.max(np.linalg.norm(vertices, axis=1)) # normalize to -1, 1

    with open(osp.join(obj_file_path.replace(".obj", "_normalized.obj")), 'w') as f:
        vertex_idx = 0
        for line in lines:
            if line.startswith("v "):
                line = "v " + " ".join([str(x) for x in vertices[vertex_idx]]) + "\n"
                vertex_idx += 1
            f.write(line)

def down_load_single_object(name, uids=None, candidate_num=5, vhacd=True, debug=False, task_name=None, task_description=None):
    """
    Download a single object from Objaverse dataset and convert to URDF.
    
    Searches for objects matching the name/description, downloads candidate models,
    and converts them to URDF format with optional V-HACD collision meshes.
    
    Args:
        name (str): Object name or search query.
        uids (list, optional): Pre-specified object UIDs to download. Defaults to None.
        candidate_num (int, optional): Number of candidates to find. Defaults to 5.
        vhacd (bool, optional): Apply V-HACD decomposition. Defaults to True.
        debug (bool, optional): Enable debug output. Defaults to False.
        task_name (str, optional): Task name for context. Defaults to None.
        task_description (str, optional): Task description for search. Defaults to None.
    
    Returns:
        list: Paths to downloaded and processed URDF directories.
    
    Side Effects:
        - Downloads mesh files from Objaverse
        - Creates URDF files and collision meshes
        - May create data/diverse_objects/ directory structure
    
    Example:
        >>> paths = down_load_single_object('cup', candidate_num=3, vhacd=True)
        >>> print(f"Downloaded {len(paths)} cup models")
    """
    if uids is None:
        if name in text_to_uid_dict:
            uids = text_to_uid_dict[name]
        else:
            from objaverse_utils.find_uid_utils import find_uid
            uids = find_uid(name, candidate_num=candidate_num, debug=debug, task_name=task_name, task_description=task_description)
            if uids is None:
                return False

    processes = multiprocessing.cpu_count()
   
    for uid in uids:
        save_path = osp.join(os.environ["PROJECT_DIR"], "objaverse_utils/data/obj", "{}".format(uid))
        if not osp.exists(save_path):
            os.makedirs(save_path)
        if osp.exists(save_path + "/material.urdf"):
            continue

        objects = objaverse.load_objects(
            uids=[uid],
            download_processes=processes
        )
        
        test_obj = (objects[uid])
        scene = trimesh.load(test_obj)

        try:
            trimesh.exchange.export.export_mesh(
                scene, osp.join(save_path, "material.obj")
            )
        except:
            if debug:
                return False
            # print("cannot export obj for uid: ", uid)
            uids.remove(uid)
            if name in text_to_uid_dict and uid in text_to_uid_dict[name]:
                text_to_uid_dict[name].remove(uid)
            continue

        # we need to further parse the obj to normalize the size to be within -1, 1
        if not osp.exists(osp.join(save_path, "material_normalized.obj")):
            normalize_obj(osp.join(save_path, "material.obj"))

        # we also need to parse the obj to vhacd
        if vhacd:
            if not osp.exists(osp.join(save_path, "material_normalized_vhacd.obj")):
                run_vhacd(save_path)

        # for pybullet, we have to additionally parse it to urdf
        obj_to_urdf(save_path, scale=1, vhacd=vhacd) 

    return True

def download_and_parse_objavarse_obj_from_yaml_config(config_path, candidate_num=10, vhacd=True):

    config = None
    while config is None:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

    task_name = None
    task_description = None
    for obj in config:
        if 'task_name' in obj.keys():
            task_name = obj['task_name']
            task_description = obj['task_description']
            break

    for obj in config:
        if 'type' in obj.keys() and obj['type'] == 'mesh' and 'uid' not in obj.keys():
            print("{} trying to download object: {} {}".format("=" * 20, obj['lang'], "=" * 20))
            success = down_load_single_object(obj["lang"], candidate_num=candidate_num, vhacd=vhacd, 
                                              task_name=task_name, task_description=task_description)
            if not success:
                print("failed to find suitable object to download {} quit building this task".format(obj["lang"]))
                return False
            obj['uid'] = text_to_uid_dict[obj["lang"]]
            obj['all_uid'] = text_to_uid_dict[obj["lang"] + "_all"]

            with open(config_path, 'w') as f:
                yaml.dump(config, f, indent=4)

    return True

def load_gif(gif_path):
    """
    Load a GIF file and extract all frames as numpy arrays.
    
    Args:
        gif_path (str): Path to the GIF file to load.
    
    Returns:
        list[np.ndarray]: List of numpy arrays, each representing a frame
            in RGB format with shape (height, width, 3).
    
    Example:
        >>> frames = load_gif('demo.gif')
        >>> print(f"Loaded {len(frames)} frames")
        >>> print(f"Frame shape: {frames[0].shape}")
    """
    img = Image.open(gif_path)
    # Extract each frame from the GIF and convert to RGB
    frames = [frame.convert('RGB') for frame in ImageSequence.Iterator(img)]
    # Convert each frame to a numpy array
    frames_arrays = [np.array(frame) for frame in frames]
    return frames_arrays

class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""
            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc

def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

def obj_to_urdf(obj_file_path, scale=1, vhacd=True, normalized=True, obj_name='material'):
    """
    Convert OBJ mesh file to URDF format with collision mesh.
    
    Creates a URDF robot description from an OBJ mesh file, optionally applying
    V-HACD decomposition for improved collision detection.
    
    Args:
        obj_file_path (str): Directory path containing the OBJ file.
        scale (float, optional): Scale factor for the mesh. Defaults to 1.
        vhacd (bool, optional): Whether to use V-HACD decomposition. Defaults to True.
        normalized (bool, optional): Whether the OBJ is normalized. Defaults to True.
        obj_name (str, optional): Name for the material. Defaults to 'material'.
    
    Returns:
        str: Path to the generated URDF file.
    
    Side Effects:
        - Creates a URDF file in obj_file_path directory
        - May create collision mesh files if vhacd=True
    
    Example:
        >>> urdf_path = obj_to_urdf('data/my_object/', scale=0.5, vhacd=True)
        >>> print(urdf_path)  # 'data/my_object/model.urdf'
    """
    header = """<?xml version="1.0" ?>
<robot name="cube.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
      <rolling_friction value="0.0"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0.0 0.02 0.0"/>
       <mass value=".1"/>
       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
"""

    all_files = os.listdir(obj_file_path)
    png_file = None
    for x in all_files:
        if x.endswith(".png"):
            png_file = x
            break

    if png_file is not None:
        material = """
         <material name="texture">
        <texture filename="{}"/>
      </material>""".format(osp.join(obj_file_path, png_file))        
    else:
        material = """
        <material name="yellow">
            <color rgba="1 1 0.4 1"/>
        </material>
        """

    obj_file = "{}.obj".format(obj_name) if not normalized else "{}_normalized.obj".format(obj_name)
    visual = """
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
      {}
    </visual>
    """.format(osp.join(obj_file_path, obj_file), scale, scale, scale, material)

    if normalized:
        collision_file = '{}_normalized_vhacd.obj'.format(obj_name) if vhacd else "{}_normalized.obj".format(obj_name)
    else:
        collision_file = '{}_vhacd.obj'.format(obj_name) if vhacd else "{}.obj".format(obj_name)
    collision = """
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
             <mesh filename="{}" scale="{} {} {}"/>
      </geometry>
    </collision>
  </link>
  </robot>
  """.format(osp.join(obj_file_path, collision_file), scale, scale, scale)
    


    urdf =  "".join([header, visual, collision])
    with open(osp.join(obj_file_path, "{}.urdf".format(obj_name)), 'w') as f:
        f.write(urdf)

def run_vhacd(input_obj_file_path, normalized=True, obj_name="material"):
    """
    Run V-HACD (Voxel-based Hierarchical Approximate Convex Decomposition) on an OBJ file.
    
    Decomposes a complex mesh into approximate convex hulls for efficient collision detection
    using PyBullet's built-in V-HACD implementation.
    
    Args:
        input_obj_file_path (str): Directory path containing the input OBJ file.
        normalized (bool, optional): Whether the mesh is normalized. Determines input/output
            filenames (_normalized.obj vs .obj). Defaults to True.
        obj_name (str, optional): Base name for the mesh files. Defaults to "material".
    
    Returns:
        None
    
    Side Effects:
        - Creates decomposed OBJ file (*_vhacd.obj) in input_obj_file_path directory
        - Creates log.txt file with V-HACD decomposition details
        - Initializes PyBullet DIRECT connection
    
    Example:
        >>> run_vhacd('data/my_object/', normalized=True, obj_name='mesh')
        >>> # Creates data/my_object/mesh_normalized_vhacd.obj
    """
    p.connect(p.DIRECT)
    if normalized:
        name_in = os.path.join(input_obj_file_path, "{}_normalized.obj".format(obj_name))
        name_out = os.path.join(input_obj_file_path, "{}_normalized_vhacd.obj".format(obj_name))
        name_log = os.path.join(input_obj_file_path, "log.txt")
    else:
        name_in = os.path.join(input_obj_file_path, "{}.obj".format(obj_name))
        name_out = os.path.join(input_obj_file_path, "{}_vhacd.obj".format(obj_name))
        name_log = os.path.join(input_obj_file_path, "log.txt")
    p.vhacd(name_in, name_out, name_log)


def run_vhacd_with_timeout(args):
    name_in, name_out, name_log, urdf_file_path, obj_file_name = args
    id = p.connect(p.DIRECT)
    proc = Process(target=p.vhacd, args=(name_in, name_out, name_log))

    proc.start()

    # Wait for 10 seconds or until process finishes
    proc.join(200)

    # If thread is still active
    if proc.is_alive():
        print("running too long... let's kill it...")

        # Terminate
        proc.kill()
        proc.join()

        if urdf_file_path not in sapaien_cannot_vhacd_part_dict.keys():
            sapaien_cannot_vhacd_part_dict[urdf_file_path] = []
        sapaien_cannot_vhacd_part_dict[urdf_file_path].append(obj_file_name)

        p.disconnect(id)
        return False

    else:
        print("process finished")
        p.disconnect(id)
        return True

def preprocess_urdf(urdf_file_path, num_processes=6):
    new_lines = []
    with open(urdf_file_path, 'r') as f:
        lines = f.readlines()
        
    num_lines = len(lines)
    l_idx = 0
    to_process_args = []
    while l_idx < num_lines:
        line_1 = lines[l_idx]

        if "<collision>" in line_1:
            new_lines.append(line_1)

            for l_idx_2 in range(l_idx + 1, num_lines):
                line_2 = lines[l_idx_2]

                if ".obj" in line_2:
                    start_idx = line_2.find('filename="') + len('filename="')
                    end_idx = line_2.find('.obj') + len('.obj')
                    obj_file_name = line_2[start_idx:end_idx]
                    obj_file_path = osp.join(osp.dirname(urdf_file_path), obj_file_name)
                    # import pdb; pdb.set_trace()
                    name_in = obj_file_path
                    name_out = obj_file_path[:-4] + "_vhacd.obj"
                    name_log = obj_file_path[:-4] + "_log.txt"

                    if not osp.exists(name_out) and obj_file_name not in sapaien_cannot_vhacd_part_dict.get(urdf_file_path, []):
                        to_process_args.append([name_in, name_out, name_log, urdf_file_path, obj_file_name])
                        new_lines.append("to_be_processed, {}".format(line_2))
                    else:
                        new_name = line_2.replace(obj_file_name, obj_file_name[:-4] + '_vhacd.obj')
                        new_lines.append(new_name)
                
                elif "</collision>" in line_2:
                    new_lines.append(line_2)
                    l_idx = l_idx_2 
                    break

                else:
                    new_lines.append(line_2)
            
        else:
            new_lines.append(line_1)

        l_idx += 1

    # do vhacd in parallel, each has a timeout of 200 seconds
    with NonDaemonPool(processes=num_processes) as pool: 
        results = pool.map(run_vhacd_with_timeout, to_process_args)

    processed_idx = 0
    for l_idx in range(len(new_lines)):
        if "to_be_processed" in new_lines[l_idx]:
            if results[processed_idx]:
                new_name = new_lines[l_idx].replace("to_be_processed, ", "")
                new_name = new_name.replace(".obj", "_vhacd.obj")
                new_lines[l_idx] = new_name
            else:
                new_name = new_lines[l_idx].replace("to_be_processed, ", "")
                new_lines[l_idx] = new_name
            processed_idx += 1

    new_path = urdf_file_path.replace(".urdf", "_vhacd.urdf")    
    with open(new_path, 'w') as f:
        f.writelines("".join(new_lines))

    with open(f"{data_dir}/sapien_cannot_vhacd_part.json", 'w') as f:
        json.dump(sapaien_cannot_vhacd_part_dict, f, indent=4)

    return new_path
     
      