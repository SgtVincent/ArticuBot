import numpy as np
import pybullet as p
import random
import scipy
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R

def xyzw2wxyz(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[3], quat[0], quat[1], quat[2]])

def wxyz2xyzw(quat : np.ndarray) -> np.ndarray:
    assert len(quat) == 4, f'quaternion size must be 4, got {len(quat)}'
    return np.asarray([quat[1], quat[2], quat[3], quat[0]])

def pose_6d_to_7d(pose) -> np.ndarray:
    if len(pose) == 7:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_rotvec(pose[3:]).as_quat()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)

def pose_7d_to_6d(pose) -> np.ndarray:
    """
    Convert 7D pose (position + quaternion) to 6D pose (position + rotation vector).
    
    Args:
        pose (array-like): 7D pose [x, y, z, qx, qy, qz, qw] or 6D pose [x, y, z, rx, ry, rz].
            If already 6D, returns unchanged.
    
    Returns:
        np.ndarray: 6D pose [x, y, z, rx, ry, rz] where (rx, ry, rz) is rotation vector.
    
    Example:
        >>> pose_7d = [1, 2, 3, 0, 0, 0, 1]  # Identity rotation quaternion
        >>> pose_6d = pose_7d_to_6d(pose_7d)
        >>> print(pose_6d)  # [1, 2, 3, 0, 0, 0] - zero rotation vector
    """
    if len(pose) == 6:
        return np.array(pose)
    pos = np.asarray(pose[:3])
    rot = R.from_quat(pose[3:]).as_rotvec()
    pose_ret = list(pos) + list(rot)

    return np.array(pose_ret)

def get_matrix_from_pose(pose) -> np.ndarray:
    """
    Convert pose representation to 4x4 homogeneous transformation matrix.
    
    Supports multiple pose formats:
    - 6D: [x, y, z, rx, ry, rz] (rotation vector)
    - 7D: [x, y, z, qx, qy, qz, qw] (quaternion)
    - 9D: [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z] (first two rotation matrix columns)
    
    Args:
        pose (array-like): Pose in 6D, 7D, or 9D format.
    
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
    
    Example:
        >>> pose_6d = [1, 0, 0, 0, 0, 1.57]  # 90° rotation around z-axis
        >>> T = get_matrix_from_pose(pose_6d)
        >>> print(T.shape)  # (4, 4)
    """
    assert len(pose) == 6 or len(pose) == 7 or len(pose) == 9, f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3])
    rot_m = np.identity(3)

    if len(pose) == 6:
        rot_m = R.from_rotvec(pose[3:]).as_matrix()
    elif len(pose) == 7:
        rot_m = R.from_quat(pose[3:]).as_matrix()
    elif len(pose) == 9:
        rot_xy = pose[3:].reshape(2, 3)
        rot_m = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T
            
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m

    return ret_m

def rot_6d_to_3d(rot) -> np.ndarray:
    """
    Convert 6D rotation representation to 3D rotation vector (axis-angle).
    
    Converts the continuous 6D rotation (two 3D vectors) to a compact
    3D rotation vector representation (axis-angle form).
    
    Args:
        rot (array-like): 6D rotation [r1x, r1y, r1z, r2x, r2y, r2z], shape (6,).
    
    Returns:
        np.ndarray: 3D rotation vector [rx, ry, rz], shape (3,).
    
    Example:
        >>> rot_6d = [1, 0, 0, 0, 1, 0]  # Identity-like rotation
        >>> rot_3d = rot_6d_to_3d(rot_6d)
        >>> print(rot_3d.shape)  # (3,)
    """
    rot_xy = np.asarray(rot)

    assert rot_xy.shape == (6,), f'dimension of rot should be (6,), but got {rot_xy.shape}'

    rot_xy = rot_xy.reshape(2, 3)
    rot_mat = np.vstack((rot_xy, np.cross(rot_xy[0], rot_xy[1]))).T 

    return R.from_matrix(rot_mat).as_rotvec()

def get_pose_from_matrix(matrix, pose_size : int = 7) -> np.ndarray:
    """
    Extract pose from 4x4 homogeneous transformation matrix.
    
    Converts a transformation matrix to various pose representations.
    
    Args:
        matrix (array-like): 4x4 homogeneous transformation matrix.
        pose_size (int, optional): Desired pose format:
            - 6: [x, y, z, rx, ry, rz] (rotation vector)
            - 7: [x, y, z, qx, qy, qz, qw] (quaternion)
            - 9: [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z] (6D rotation)
            Defaults to 7.
    
    Returns:
        np.ndarray: Pose in requested format, shape (pose_size,).
    
    Example:
        >>> T = np.eye(4)
        >>> T[:3, 3] = [1, 2, 3]
        >>> pose = get_pose_from_matrix(T, pose_size=7)
        >>> print(pose[:3])  # [1, 2, 3] - position
    """
    mat = np.array(matrix)
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'
    
    pos = matrix[:3, 3]
    rot = None

    if pose_size == 6:
        rot = R.from_matrix(matrix[:3, :3]).as_rotvec()
    elif pose_size == 7:
        rot = R.from_matrix(matrix[:3, :3]).as_quat()
    elif pose_size == 9:
        rot = (matrix[:3, :2].T).reshape(-1)
    else:
        raise ValueError(f"Invalid pose_size: {pose_size}. Must be 6, 7, or 9.")
            
    pose = list(pos) + list(rot)

    return np.array(pose)

def get_matrix_from_pos_rot(pos, rot) -> np.ndarray:
    """
    Construct 4x4 transformation matrix from position and rotation.
    
    Combines position and rotation (quaternion or rotation vector) into
    a homogeneous transformation matrix.
    
    Args:
        pos (array-like): Position [x, y, z], shape (3,).
        rot (array-like): Rotation as quaternion [qx, qy, qz, qw] (4,) or
            rotation vector [rx, ry, rz] (3,).
    
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix.
    
    Example:
        >>> pos = [1, 0, 0]
        >>> rot_quat = [0, 0, 0, 1]  # Identity quaternion
        >>> T = get_matrix_from_pos_rot(pos, rot_quat)
        >>> print(T[:3, 3])  # [1, 0, 0] - position
    """
    assert (len(pos) == 3 and len(rot) == 4) or (len(pos) == 3 and len(rot) == 3)
    pos_m = np.asarray(pos)
    rot_m = np.identity(3)  # Initialize rot_m
    if len(rot) == 3:
        rot_m = R.from_rotvec(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(rot))).reshape((3, 3))
    elif len(rot) == 4: # x, y, z, w
        rot_m = R.from_quat(rot).as_matrix()
        # rot_m = np.asarray(p.getMatrixFromQuaternion(rot)).reshape((3, 3))
    ret_m = np.identity(4)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m

def cross(a:np.ndarray,b:np.ndarray)->np.ndarray:
    return np.cross(a,b)

def get_pos_rot_from_matrix(pose : np.ndarray) -> np.ndarray:
    assert pose.shape == (4, 4)
    pos = pose[:3, 3]
    rot = R.from_matrix(pose[:3, :3]).as_quat()
    return pos, rot

def get_projmat_and_intrinsic(width, height, fx, fy, far, near):
    cx = width / 2
    cy = height / 2
    fov = 2 * np.arctan(height / (2 * fy)) * 180.0 / np.pi

    project_matrix = p.computeProjectionMatrixFOV(
                        fov=fov,
                        aspect=width/height,
                        nearVal=near,
                        farVal=far
                      )
    
    intrinsic = np.array([
                    [ fx, 0.0,  cx],
                    [0.0,  fy,  cy],
                    [0.0, 0.0, 1.0],
                  ])
    
    return project_matrix, intrinsic

def get_viewmat_and_extrinsic(cameraEyePosition, cameraTargetPosition, cameraUpVector):

    view_matrix = p.computeViewMatrix(
                    cameraEyePosition=cameraEyePosition,
                    cameraTargetPosition=cameraTargetPosition,
                    cameraUpVector=cameraUpVector
                  )

    # rotation vector extrinsic
    z = np.asarray(cameraTargetPosition) - np.asarray(cameraEyePosition)
    norm = np.linalg.norm(z, ord=2)
    assert norm > 0, 'cameraTargetPosition and cameraEyePosition is at same location'
    z /= norm
   
    y = -np.asarray(cameraUpVector)
    y -= (np.dot(z, y)) * z
    norm = np.linalg.norm(y, ord=2)
    assert norm > 0, 'cameraUpVector is parallel to z axis'
    y /= norm
    
    x = cross(y, z)

    # extrinsic
    extrinsic = np.identity(4)
    extrinsic[:3, 0] = x
    extrinsic[:3, 1] = y
    extrinsic[:3, 2] = z
    extrinsic[:3, 3] = np.asarray(cameraEyePosition)

    return view_matrix, extrinsic

def draw_coordinate(pose, size, color : np.ndarray=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])):
    assert (type(pose) == np.ndarray and pose.shape == (4, 4)) or (len(pose) == 7) or (len(pose) == 6)

    if len(pose) == 7 or len(pose) == 6:
        pose = get_matrix_from_pose(pose)

    origin = pose[:3, 3]
    x = origin + pose[:3, 0] * size
    y = origin + pose[:3, 1] * size
    z = origin + pose[:3, 2] * size
    p.addUserDebugLine(origin, x, color[0], 2, 0)
    p.addUserDebugLine(origin, y, color[1], 2, 0)
    p.addUserDebugLine(origin, z, color[2], 2, 0)

def sample_point_inside_triangle(v1,v2,v3):
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    while r1 + r2 >= 1:
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
    r3 = 1 - r1 - r2

    # Calculate the point using barycentric coordinates
    x = r1 * v1[0] + r2 * v2[0] + r3 * v3[0]
    y = r1 * v1[1] + r2 * v2[1] + r3 * v3[1]
    z = r1 * v1[2] + r2 * v2[2] + r3 * v3[2]
    return [x, y, z]

def get_link_handle(all_handle_pos, handle_joint_id, link_pc, threshold=0.02):
    handle_median_points = np.array([np.median(handle_pos, axis=0) for handle_pos in all_handle_pos]).reshape(-1, 3)
    distance_handle_median_to_link_pc = scipy.spatial.distance.cdist(handle_median_points, link_pc)
    min_distance = np.min(distance_handle_median_to_link_pc, axis=1)
    min_distance_handle_idx = np.argmin(min_distance)
    handle_joint_id = handle_joint_id[min_distance_handle_idx]
    handle_pc = all_handle_pos[min_distance_handle_idx]
    handle_median = handle_median_points[min_distance_handle_idx]
    pc_to_handle_distance = scipy.spatial.distance.cdist(link_pc, handle_pc).min(axis=1)
    handle_pc = link_pc[pc_to_handle_distance < threshold]
    # use the pointcloud of link instead of the handle itself. (partially occluded)
    return handle_pc, handle_joint_id, handle_median, min_distance_handle_idx

def get_pc_num_within_gripper(cur_eef_pos, cur_eef_orient, pc_points):
    
    cur_pos, cur_orient = cur_eef_pos, cur_eef_orient

    X_GW = p.invertTransform(cur_pos, cur_orient)
    translation = np.array(X_GW[0])
    rotation = np.array(p.getMatrixFromQuaternion(X_GW[1])).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation ### this is the transformation from world frame to gripper frame

    pc_homogeneous = np.hstack((pc_points, np.ones((pc_points.shape[0], 1))))  # Convert to homogeneous coordinates Nx4
    pc_transformed_homogeneous = T @ pc_homogeneous.T # 4x4 @ 4xN = 4xN
    p_GC = pc_transformed_homogeneous[:3, :] # 3xN

    ### Crop to a region inside of the finger box.
    crop_min = [-0.02, -0.06, -0.01] 
    crop_max = [0.02, 0.06, 0.01]
    indices = np.all(
        (
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ),
        axis=0,
    )
    
    within_bbox_handle_pc = pc_points[indices]
    if len(within_bbox_handle_pc) == 0:
        return 0
    score = np.sum(indices) 
    return score

def get_handle_orient(handle_pc):
    # get axis aligned bounding box of the handle pc
    min_xyz = np.min(handle_pc, axis=0)
    max_xyz = np.max(handle_pc, axis=0)
    x_range = max_xyz[0] - min_xyz[0]
    y_range = max_xyz[1] - min_xyz[1]
    z_range = max_xyz[2] - min_xyz[2]
    horizontal_range = np.max([x_range, y_range])
    vertical_range = z_range
    if horizontal_range > vertical_range:
        handle_orient = "horizontal"
    else:
        handle_orient = "vertical"
    
    return handle_orient

def pc_to_line_distance(pc, line_start, line_end):
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    line_vec = line_end - line_start
    pc_vecs = pc - line_start
    cross_prod = np.cross(line_vec, pc_vecs)
    distances = np.linalg.norm(cross_prod, axis=1) / np.linalg.norm(line_vec)
    return distances

def filter_pointcloud_by_line(pointcloud, point1, point2):
    """
    Filter point cloud based on line projection in xoy plane, keeping points on the same side as origin [0,0]
    
    Args:
        pointcloud: numpy array, shape [n, 3], each row is [x, y, z]
        point1: list/array, first point [x1, y1, z1] (only x,y coordinates are used)
        point2: list/array, second point [x2, y2, z2] (only x,y coordinates are used)
    
    Returns:
        numpy array: filtered point cloud
    """
    
    # Extract xy coordinates
    point1_xy = np.array(point1[:2])  # [x1, y1]
    point2_xy = np.array(point2[:2])  # [x2, y2]
    origin = np.array([0.0, 0.0])
    
    # Calculate direction vector of the line
    direction = point2_xy - point1_xy
    
    # Calculate normal vector (perpendicular to the line)
    # If direction vector is [a, b], then normal vector is [-b, a] or [b, -a]
    normal = np.array([-direction[1], direction[0]])
    
    # Ensure normal vector is not zero vector
    if np.linalg.norm(normal) == 0:
        raise ValueError("Two points are coincident or on the same vertical line, cannot form a valid dividing line")
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    # Calculate signed distance from origin to the line
    # Line equation: normal[0] * (x - x1) + normal[1] * (y - y1) = 0
    origin_distance = np.dot(normal, origin - point1_xy)
    
    # Calculate signed distance from each point in point cloud to the line
    pointcloud_xy = pointcloud[:, :2]  # Extract only x,y coordinates
    point_distances = np.dot(pointcloud_xy - point1_xy, normal)
    
    # Filter points on the same side as origin (same sign)
    if origin_distance >= 0:
        # Origin is on positive side of line, select points on positive side
        mask = point_distances >= 0
    else:
        # Origin is on negative side of line, select points on negative side
        mask = point_distances <= 0
    
    return pointcloud[mask]


def estimate_line_direction(pc):
    centered = pc - np.mean(pc, axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    principal_axis /= np.linalg.norm(principal_axis)
    return principal_axis


def in_bbox(pos, bbox_min, bbox_max):
    if (pos[0] <= bbox_max[0] and pos[0] >= bbox_min[0] and \
        pos[1] <= bbox_max[1] and pos[1] >= bbox_min[1] and \
        pos[2] <= bbox_max[2] and pos[2] >= bbox_min[2]):
        return True
    return False

def draw_bbox(start, end):

    assert len(start) == 3 and len(end) == 3, 'infeasible size of position, len(position) must be 3'

    points_bb = [
        [start[0], start[1], start[2]],
        [end[0], start[1], start[2]],
        [end[0], end[1], start[2]],
        [start[0], end[1], start[2]],
        [start[0], start[1], end[2]],
        [end[0], start[1], end[2]],
        [end[0], end[1], end[2]],
        [start[0], end[1], end[2]],
    ]

    for i in range(4):
        p.addUserDebugLine(points_bb[i], points_bb[(i + 1) % 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i + 4], points_bb[(i + 1) % 4 + 4], [1, 0, 0])
        p.addUserDebugLine(points_bb[i], points_bb[i + 4], [1, 0, 0])

def radial_shift(x_coord: float, y_coord: float, noise_bounds: List[float]):
    theta = np.arctan2(y_coord, x_coord)
    theta_noise = np.random.uniform(-0.1, 0.1)
    dist = np.linalg.norm([x_coord, y_coord])
    dist_noise = np.random.uniform(noise_bounds[0],noise_bounds[1])
    theta += theta_noise
    dist += dist_noise
    perturbed_x = dist * np.cos(theta)
    perturbed_y = dist * np.sin(theta)
    return perturbed_x, perturbed_y

def get_pc(proj_matrix, view_matrix, depth, width, height, mask_infinite=False):
    """
    Convert depth image to 3D point cloud in world coordinates.
    
    This function transforms a depth map into a 3D point cloud using camera
    projection and view matrices.
    
    Args:
        proj_matrix (array-like): 4x4 projection matrix (flattened or 2D).
        view_matrix (array-like): 4x4 view matrix (flattened or 2D).
        depth (np.ndarray): Depth image with shape (height, width).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        mask_infinite (bool, optional): If True, mask out infinite depth values.
            Defaults to False.
    
    Returns:
        np.ndarray: Point cloud with shape (N, 3) where N = height * width.
            Each row is an (x, y, z) coordinate in world space.
    
    Example:
        >>> depth = camera.get_depth()
        >>> pc = get_pc(proj_mat, view_mat, depth, 640, 480)
        >>> print(pc.shape)  # (307200, 3)
    """
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    if mask_infinite:
        pixels = pixels[z < 0.9999]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points

def get_pixel_location(proj_matrix, view_matrix, point_3d, width, height):
    """
    Project a 3D world point to 2D pixel coordinates.
    
    This function transforms a 3D point in world coordinates to pixel
    coordinates in the image plane using camera matrices.
    
    Args:
        proj_matrix (array-like): 4x4 projection matrix.
        view_matrix (array-like): 4x4 view matrix.
        point_3d (array-like): 3D point coordinates (x, y, z).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
    
    Returns:
        tuple: (x_img, y_img) pixel coordinates in the image.
            Note: y-axis is inverted (origin at top-left).
    
    Example:
        >>> point_3d = [0.5, 0.3, 1.2]
        >>> x, y = get_pixel_location(proj_mat, view_mat, point_3d, 640, 480)
        >>> print(f"Pixel location: ({x:.1f}, {y:.1f})")
    """
    # Ensure matrices are in the correct shape
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    
    # Combine the projection and view matrices
    tran_world_pix = np.matmul(proj_matrix, view_matrix)

    # Add homogeneous coordinate to the 3D point
    point_3d_h = np.append(point_3d, 1.0)
    
    # Transform the 3D point to pixel coordinates
    pixel_h = np.matmul(tran_world_pix, point_3d_h)
    
    # Normalize by the homogeneous coordinate
    pixel_h /= pixel_h[3]
    
    # Convert from normalized device coordinates to pixel coordinates
    x_ndc, y_ndc, z_ndc = pixel_h[:3]
    
    # Transform normalized device coordinates to image coordinates
    x_img = (x_ndc * 0.5 + 0.5) * width
    y_img = (1.0 - (y_ndc * 0.5 + 0.5)) * height  # Note: y-axis is inverted
    
    return int(x_img), int(y_img), z_ndc

def get_pc_in_camera_frame(proj_matrix, view_matrix, depth, width, height, mask_infinite=False):
    """
    Convert depth image to 3D point cloud in camera frame coordinates.
    
    Similar to get_pc() but returns points in the camera reference frame
    instead of world coordinates.
    
    Args:
        proj_matrix (array-like): 4x4 projection matrix.
        view_matrix (array-like): 4x4 view matrix.
        depth (np.ndarray): Depth image with shape (height, width).
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        mask_infinite (bool, optional): Mask infinite depths. Defaults to False.
    
    Returns:
        np.ndarray: Point cloud in camera frame with shape (N, 3).
    
    Example:
        >>> depth = camera.get_depth()
        >>> pc_cam = get_pc_in_camera_frame(proj, view, depth, 640, 480)
        >>> # Points are in camera coordinates (z-forward, y-down, x-right)
    """
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    if mask_infinite:
        pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    # turn pixels to camera cooridnates
    points = np.matmul(np.linalg.inv(proj_matrix), pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]
    return points

    
def setup_camera_ben(client_id, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], camera_width=1920//4, camera_height=1080//4, 
                 z_near=0.01, z_far=100):
    view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=client_id)
    focal_length = 450 # CAMERA_INTRINSICS[0, 0]
    fov = (np.arctan((camera_height / 2) / focal_length) * 2 / np.pi) * 180
    projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, z_near, z_far, physicsClientId=client_id)
    return view_matrix, projection_matrix

def get_pc_ben(depth, view_matrix, projection_matrix, znear, zfar):
    height, width = depth.shape
    CAMERA_INTRINSICS = np.array(
        [
            [450, 0, width / 2],
            [0, 450, height / 2],
            [0, 0, 1],
        ]
    )

    T_CAMGL_2_CAM = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )

    depth = zfar + znear - (2.0 * depth - 1.0) * (zfar - znear)
    depth = (2.0 * znear * zfar) / depth

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - CAMERA_INTRINSICS[0, 2]) * (depth / CAMERA_INTRINSICS[0, 0])
    py = (py - CAMERA_INTRINSICS[1, 2]) * (depth / CAMERA_INTRINSICS[1, 1])
    P_cam = np.float32([px, py, depth]).transpose(1, 2, 0).reshape(-1, 3)

    T_camgl2world = np.asarray(view_matrix).reshape(4, 4).T
    T_world2camgl = np.linalg.inv(T_camgl2world)
    T_world2cam = T_world2camgl @ T_CAMGL_2_CAM

    Ph_cam = np.concatenate([P_cam, np.ones((len(P_cam), 1))], axis=1)
    Ph_world = (T_world2cam @ Ph_cam.T).T
    P_world = Ph_world[:, :3]

    return P_world
