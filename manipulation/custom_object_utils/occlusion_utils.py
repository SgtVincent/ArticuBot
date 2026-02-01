
import numpy as np
import pybullet as p
from termcolor import cprint

def check_trajectory_occlusion(env, object_id, handle_joint_id, threshold=0.5):
    """
    Checks if the opening trajectory is 'outwards' (towards robot) or obstructed.
    Returns True if trajectory is VALID, False if occluded/bad.
    
    Args:
        env: Environment instance.
        object_id: PyBullet body ID.
        handle_joint_id: Moving joint ID.
        threshold: alignment threshold (0.5 means reject if dot < -0.5, i.e., >120deg away).
                   User calls this 'trajectory_occlusion_rate' in args.
    """
    if handle_joint_id is None:
        return True 

    # 1. Get current handle pos (closed)
    ls = p.getLinkState(object_id, handle_joint_id, physicsClientId=env.id)
    p_closed = np.array(ls[0])
    
    # 2. Kinematically open to check the path
    jinfo = p.getJointInfo(object_id, handle_joint_id, physicsClientId=env.id)
    lower, upper = jinfo[8], jinfo[9]
    target_open = lower + 0.8 * (upper - lower) 

    current_q = p.getJointState(object_id, handle_joint_id, physicsClientId=env.id)[0]
    p.resetJointState(object_id, handle_joint_id, target_open, physicsClientId=env.id)
    p.performCollisionDetection(physicsClientId=env.id) 
    
    ls_open = p.getLinkState(object_id, handle_joint_id, physicsClientId=env.id)
    p_open = np.array(ls_open[0])
    
    # Restore
    p.resetJointState(object_id, handle_joint_id, current_q, physicsClientId=env.id)
    
    # 3. Check Direction (Dot Product)
    opening_vec = p_open - p_closed
    dist = np.linalg.norm(opening_vec)
    if dist < 0.05: 
        return True 
    
    opening_dir = opening_vec / dist
    
    # Get ACTUAL robot base position
    # env.robot.body is the ID. Link -1 is base.
    robot_pos, _ = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)
    robot_pos = np.array(robot_pos)
    
    vec_to_robot = (robot_pos - p_closed) 
    vec_to_robot /= np.linalg.norm(vec_to_robot)
    
    # If opening_dir points broadly AWAY from robot, reject.
    # threshold 0.5 -> reject if dot < -0.5
    dot = np.dot(opening_dir, vec_to_robot)
    if dot < -threshold: 
        # cprint(f"[Occlusion] Rejecting: Opens away from robot (dot={dot:.2f} < -{threshold})", "yellow")
        return False
        
    # 4. Raycast Check (Line of Sight)
    # Check if the robot can see the handle at the OPEN position.
    res = p.rayTest(robot_pos, p_open, physicsClientId=env.id)
    hit_obj, hit_link, _, _, _ = res[0]
    
    if hit_obj == object_id and hit_link != handle_joint_id:
        return False
            
    return True

def check_handle_facing(env, object_id, handle_joint_id, handle_pos_world=None):
    """
    Checks if the handle (affordance point) is on the side of the object facing the robot.
    Geometry: Dot(Vec(Robot->ObjCenter), Vec(ObjCenter->Handle)) < 0.
    Also performs a Raycast visibility check to ensure the body doesn't block the view.
    
    Returns: True if VALID (handle is on robot side), False if invalid.
    """
    if handle_joint_id is None and handle_pos_world is None:
        return True
        
    # 1. Get Object Center (AABB of the whole object, excluding handle if possible? 
    # Actually AABB of the root link is strictly better for "Body" center if the door is huge.
    # But getAABB(obj) gives the union. Let's stick to Union for "Object Center" general approximator.
    aabb_min, aabb_max = p.getAABB(object_id, physicsClientId=env.id)
    obj_center = (np.array(aabb_min) + np.array(aabb_max)) / 2.0
    
    # 2. Get Handle Position (Use affordance median if provided, else AABB center of the link)
    # The joint frame is the hinge. The AABB center captures the physical "Door/Handle" location better.
    if handle_pos_world is not None:
        handle_pos = np.array(handle_pos_world, dtype=float)
    else:
        if handle_joint_id is None:
            return True
        h_min, h_max = p.getAABB(object_id, handle_joint_id, physicsClientId=env.id)
        handle_pos = (np.array(h_min) + np.array(h_max)) / 2.0
    
    # 3. Get Robot Base Position
    robot_pos, _ = p.getBasePositionAndOrientation(env.robot.body, physicsClientId=env.id)
    robot_pos = np.array(robot_pos)
    
    # 4. Compute Vectors
    vec_robot_to_obj = obj_center - robot_pos
    # 4. Compute Vectors (XY Projection Only)
    # We ignore Z because top-manipulation is valid even if robot is "below" the object center.
    vec_robot_to_obj_xy = obj_center[:2] - robot_pos[:2]
    vec_obj_to_handle_xy = handle_pos[:2] - obj_center[:2]
    
    # Debug info
    # print(f"  [Debug] ObjCenter: {obj_center}, HandlePos: {handle_pos}, RobotPos: {robot_pos}")
    
    # 5. Check Geom Alignment
    # 5. Check Geom Alignment (XY Plane)
    dot = np.dot(vec_robot_to_obj_xy, vec_obj_to_handle_xy)
    
    # Relaxed threshold: Allow small positive dot products (handle slightly "behind" center)
    # This accounts for object geometry where handle might be on the side/top but slightly offset back
    if dot >= 0.15:
        cprint(f"  [Reject] Handle behind center (xy-dot={dot:.3f})", "yellow")
        return False
        
    # 6. Raycast Visibility Check (Horizontal / XY-Plane Logic)
    # User Request: "Only compute the raycasting on xy-plane ... vector from robot arm base... still passes object body"
    # To simulate XY-plane raycast, we cast a ray from the robot's XY position AT THE HANDLE'S Z HEIGHT
    # to the handle's position. This avoids the ray passing through the object body "diagonally".
    
    ray_start = np.array([robot_pos[0], robot_pos[1], handle_pos[2]])
    ray_end = handle_pos
    
    res = p.rayTest(ray_start, ray_end, physicsClientId=env.id)
    hit_obj, hit_link, _, _, _ = res[0]
    
    # Valid if:
    # 1. Hits the handle (hit_link == handle_joint_id)
    # 2. Hits nothing (-1)
    # 3. Hits the object but NOT the fixed base? 
    # Ideally, if it hits the fixed base (link -1) of the object, that's an occlusion.
    
    if hit_obj == object_id:
        if hit_link == -1: # Hit the static base/body
            cprint(f"  [Reject] Handle occluded by object body (hit link -1)", "yellow")
            return False
        if handle_joint_id is not None and hit_link != handle_joint_id:
            return False
        # If it hits another link (not handle, not base), it might be okay or bad.
        # But if it hits the handle (hit_link == handle_joint_id), strictly good.
        
    return True
