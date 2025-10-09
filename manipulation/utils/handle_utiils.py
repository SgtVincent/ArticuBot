import numpy as np
import pybullet as p

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []; faces = [];
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.array(line.split()[1:4], dtype=np.float32))
        elif line.startswith('f '):
            faces.append(np.array([item.split('/')[0] for item in line.split()[1:4]], dtype=np.int32))

    f = np.vstack(faces)
    v = np.vstack(vertices)

    return v, f

def find_nearest_point_on_line(line_pt1, line_pt2, target_pt):
    """
    Find the nearest point(s) on a line to target point(s).
    
    Projects target points onto lines defined by two points using vector projection.
    Supports batched operations for multiple lines/points.
    
    Args:
        line_pt1 (array-like): Starting point(s) of line(s), shape (N, 3) or (3,).
        line_pt2 (array-like): Ending point(s) of line(s), shape (N, 3) or (3,).
        target_pt (array-like): Target point(s) to project, shape (N, 3) or (3,).
    
    Returns:
        np.ndarray: Nearest point(s) on the line(s), shape (N, 3).
    
    Example:
        >>> line_start = [0, 0, 0]
        >>> line_end = [1, 0, 0]
        >>> target = [0.5, 1.0, 0]
        >>> nearest = find_nearest_point_on_line(line_start, line_end, target)
        >>> print(nearest)  # [[0.5, 0.0, 0.0]] - projection onto x-axis
    """
    line_pt1 = np.array(line_pt1).reshape(-1, 3)
    line_pt2 = np.array(line_pt2).reshape(-1, 3)
    target_pt = np.array(target_pt).reshape(-1, 3)
    
    # Step 1: Compute the vector along the line
    line_vec = line_pt2 - line_pt1
    
    # Step 2: Compute the vector from line_pt1 to target_pt
    pt_vec = target_pt - line_pt1
    
    # Step 3: Project pt_vec onto line_vec to find the projection scalar
    # dot_product(pt_vec, line_vec) / dot_product(line_vec, line_vec) gives the scalar
    # by which to multiply line_vec to get the projection vector.
    projection_scalar = np.sum(pt_vec * line_vec, axis=1) / np.sum(line_vec * line_vec)
    
    
    # Step 4: Find the nearest point on the line by scaling line_vec and adding it to line_pt1
    nearest_pt = line_pt1 + projection_scalar.reshape(-1, 1) * line_vec.repeat(len(projection_scalar), axis=0)
    
    return nearest_pt # (-1, 3)

def rotate_point_around_axis(pt, ax, theta_rad):
    """
    Rotate a point around a given axis by theta radiance.
    
    :param pt: The point to rotate (3D coordinates).
    :param ax: The rotation axis (3D unit vector).
    :param theta: The rotation angle in radians.
    :return: The rotated point's coordinates.
    """
    # Ensure ax is a unit vector
    ax = ax / np.linalg.norm(ax)
    ax = ax.reshape(-1, 3)
    
    # Rodrigues' rotation formula
    v_rot = (pt * np.cos(theta_rad) +
             np.cross(ax, pt) * np.sin(theta_rad) +
             ax * np.sum(ax.repeat(pt.shape[0], axis=0) * pt, axis=1, keepdims=True) * (1 - np.cos(theta_rad)))
    
    return v_rot

def add_sphere(position, radius=0.05, rgba=[0, 1, 1, 1]):
    """
    Create a visual sphere in the PyBullet simulation.
    
    Adds a sphere with collision and visual geometry at the specified position.
    Useful for debugging and visualization.
    
    Args:
        position (array-like): (x, y, z) position for the sphere center.
        radius (float, optional): Sphere radius. Defaults to 0.05.
        rgba (list, optional): RGBA color [r, g, b, a], values in [0, 1].
            Defaults to [0, 1, 1, 1] (cyan, fully opaque).
    
    Returns:
        int: PyBullet body ID of the created sphere.
    
    Side Effects:
        - Creates a sphere object in the active PyBullet simulation
    
    Example:
        >>> sphere_id = add_sphere([0.5, 0.3, 1.0], radius=0.1, rgba=[1, 0, 0, 1])
        >>> # Creates a red sphere of radius 0.1 at (0.5, 0.3, 1.0)
    """
    sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius) 
    sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    mass = 0.1
    body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                             basePosition=position)
    return body

def rotation_transfer_6D_to_matrix(orient):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Uses the continuous 6D rotation representation (Zhou et al.) which encodes
    rotation as two 3D vectors. Applies Gram-Schmidt orthonormalization to
    construct an orthonormal basis.
    
    Args:
        orient (array-like): 6D rotation representation, shape (6,) or (2, 3).
            First 3 values are first basis vector (a1).
            Last 3 values are second basis vector (a2).
    
    Returns:
        np.ndarray: 3x3 rotation matrix with orthonormal columns.
    
    Example:
        >>> orient_6d = [1, 0, 0, 0, 1, 0]  # Identity-like
        >>> rot_mat = rotation_transfer_6D_to_matrix(orient_6d)
        >>> print(rot_mat.shape)  # (3, 3)
    """
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D(rotate_matrix):
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    
    Extracts the first two columns of the rotation matrix as the 6D representation.
    This is the inverse of rotation_transfer_6D_to_matrix.
    
    Args:
        rotate_matrix (array-like): 3x3 rotation matrix, shape (3, 3) or flattened (9,).
    
    Returns:
        np.ndarray: 6D rotation representation, shape (6,). Contains first two
            column vectors of the rotation matrix flattened.
    
    Example:
        >>> rot_mat = np.eye(3)  # Identity rotation
        >>> orient_6d = rotation_transfer_matrix_to_6D(rot_mat)
        >>> print(orient_6d)  # [1, 0, 0, 0, 1, 0]
    """
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def rotation_transfer_6D_to_matrix_batch(orient):
    """
    Convert batched 6D rotation representations to rotation matrices.
    
    Vectorized version of rotation_transfer_6D_to_matrix for processing
    multiple rotations simultaneously.
    
    Args:
        orient (array-like): Batched 6D rotations, shape (B, 6) where B is batch size.
    
    Returns:
        np.ndarray: Batched rotation matrices, shape (B, 3, 3).
    
    Example:
        >>> orients = np.array([[1, 0, 0, 0, 1, 0], [0, 1, 0, -1, 0, 0]])
        >>> rot_mats = rotation_transfer_6D_to_matrix_batch(orients)
        >>> print(rot_mats.shape)  # (2, 3, 3)
    """
    # orient shape = (B, 6)
    # return shape = (3, B * 3)

    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)
    
    assert orient.shape[-1] == 6

    orient = orient.reshape(-1, 2, 3)
    a1 = orient[:,0]
    a2 = orient[:,1]

    b1 = a1 / np.linalg.norm(a1, axis=-1).reshape(-1,1)
    b2 = a2 - (np.sum(a2*b1, axis=-1).reshape(-1,1) * b1)
    b2 = b2 / np.linalg.norm(b2, axis=-1).reshape(-1,1)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.hstack((b1, b2, b3))
    rotate_matrix = rotate_matrix.reshape(-1, 3).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D_batch(rotate_matrix):

    # rotate_matrix.shape = (B, 9) or (B x 3, 3) rotation transpose (i.e., row vectors instead of column vectors)
    # return shape = (B, 6)

    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(-1, 9)
    rotate_matrix = rotate_matrix.reshape(-1, 9)

    return rotate_matrix[:,:6]
