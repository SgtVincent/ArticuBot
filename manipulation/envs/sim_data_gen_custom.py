"""Dataset generation simulation environment built on the shared base."""

import imageio
import numpy as np
import open3d as o3d
import pybullet as p

from manipulation.envs.sim_base import SimpleEnvBase
from manipulation.utils import get_pc


class SimpleEnvDataGenCustom(SimpleEnvBase):
    """Environment variant used for dataset generation with perception helpers."""

    def step(self, action):
        raise NotImplementedError("SimpleEnvDataGenCustom is used for offline data collection and does not support step().")

    def take_round_images(
        self,
        center,
        distance,
        elevation=30,
        azimuth_interval=30,
        camera_width=640,
        camera_height=480,
        return_camera_matrices=False,
    ):
        """Capture RGB/depth images from evenly spaced viewpoints on a circle."""

        camera_target = center
        delta_z = distance * np.sin(np.deg2rad(elevation))
        xy_distance = distance * np.cos(np.deg2rad(elevation))

        prev_view_matrix, prev_projection_matrix = self.view_matrix, self.projection_matrix

        rgbs = []
        depths = []
        view_camera_matrices = []
        project_camera_matrices = []
        for azimuth in range(0, 360, azimuth_interval):
            delta_x = xy_distance * np.cos(np.deg2rad(azimuth))
            delta_y = xy_distance * np.sin(np.deg2rad(azimuth))
            camera_position = [
                camera_target[0] + delta_x,
                camera_target[1] + delta_y,
                camera_target[2] + delta_z,
            ]
            self.setup_camera(  # type: ignore[attr-defined]
                camera_position,
                camera_target,
                camera_width=camera_width,
                camera_height=camera_height,
            )

            rgb, depth = self.render(return_depth=True)  # type: ignore[call-arg]
            rgbs.append(rgb)
            depths.append(depth)
            view_camera_matrices.append(self.view_matrix)
            project_camera_matrices.append(self.projection_matrix)

        self.view_matrix, self.projection_matrix = prev_view_matrix, prev_projection_matrix

        if not return_camera_matrices:
            return rgbs, depths
        return rgbs, depths, view_camera_matrices, project_camera_matrices

    def get_link_pc(self, object_name, urdf_link_name):
        """Extract a point cloud around a specified link by masking other geometry."""

        object_name = object_name.lower()
        object_id = self.urdf_ids[object_name]

        previous_rgba = []
        for other_name, other_id in self.urdf_ids.items():
            if other_name == object_name:
                continue
            num_links = p.getNumJoints(other_id, physicsClientId=self.id)
            for link_idx in range(-1, num_links):
                shape_data = p.getVisualShapeData(other_id, link_idx, physicsClientId=self.id)
                rgba = shape_data[0][14:18]
                previous_rgba.append(rgba)
                p.changeVisualShape(other_id, link_idx, rgbaColor=[0, 0, 0, 0], physicsClientId=self.id)

        prev_view_matrix, prev_projection_matrix = self.view_matrix, self.projection_matrix
        camera_width = 640
        camera_height = 480
        min_aabb, max_aabb = self.get_aabb(object_id)
        camera_target = (max_aabb + min_aabb) / 2
        distance = np.linalg.norm(max_aabb - min_aabb)
        distance *= 1.2 if self.robot_name == "panda" else 0.8
        elevation = 30

        capture = self.take_round_images(
            camera_target,
            distance,
            elevation,
            camera_width=camera_width,
            camera_height=camera_height,
            return_camera_matrices=True,
        )
        if not isinstance(capture, (list, tuple)) or len(capture) != 4:
            raise RuntimeError("Camera capture did not return view/projection matrices.")
        imgs, depths, view_mats, proj_mats = capture

        link_id = self.get_link_id_from_name(object_name, urdf_link_name)
        link_shape = p.getVisualShapeData(object_id, link_id, physicsClientId=self.id)
        prev_link_rgba = link_shape[0][14:18]
        p.changeVisualShape(object_id, link_id, rgbaColor=[0, 0, 0, 0], physicsClientId=self.id)

        invisible_capture = self.take_round_images(
            camera_target,
            distance,
            elevation,
            camera_width=camera_width,
            camera_height=camera_height,
            return_camera_matrices=True,
        )
        if not isinstance(invisible_capture, (list, tuple)) or len(invisible_capture) != 4:
            raise RuntimeError("Invisible capture did not return view/projection matrices.")
        img_invisible, depth_invisible = invisible_capture[0], invisible_capture[1]

        best_idx = 0
        max_diff_pixels = 0
        all_pc = []
        for idx, (depth, depth_mask) in enumerate(zip(depths, depth_invisible)):
            diff = np.abs(depth - depth_mask)
            mask = diff > 0
            diff_pixels = np.sum(mask)
            if diff_pixels > max_diff_pixels:
                max_diff_pixels = diff_pixels
                best_idx = idx

            pc = get_pc(
                proj_mats[idx],
                view_mats[idx],
                depths[idx],
                camera_width,
                camera_height,
            )
            pc = pc.reshape((camera_height, camera_width, 3))
            all_pc.append(pc[mask])

        if all_pc:
            all_pc = np.concatenate(all_pc, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_pc)
            downsampled = pcd.voxel_down_sample(voxel_size=0.005)
            all_pc = np.asarray(downsampled.points)
        else:
            all_pc = np.zeros((0, 3))

        imageio.imwrite("img_invisible.png", img_invisible[best_idx])

        p.changeVisualShape(object_id, link_id, rgbaColor=prev_link_rgba, physicsClientId=self.id)

        cnt = 0
        for other_name, other_id in self.urdf_ids.items():
            if other_name == object_name:
                continue
            num_links = p.getNumJoints(other_id, physicsClientId=self.id)
            for link_idx in range(-1, num_links):
                rgba = previous_rgba[cnt]
                p.changeVisualShape(other_id, link_idx, rgbaColor=rgba, physicsClientId=self.id)
                cnt += 1

        self.view_matrix, self.projection_matrix = prev_view_matrix, prev_projection_matrix

        return (
            all_pc,
            view_mats[best_idx],
            proj_mats[best_idx],
            imgs[best_idx],
        )

    def get_link_id_from_name(self, object_name, link_name):
        object_id = self.urdf_ids[object_name]
        num_joints = p.getNumJoints(object_id, physicsClientId=self.id)
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(object_id, joint_idx, physicsClientId=self.id)
            joint_link_name = joint_info[12].decode("utf-8")
            if joint_link_name == link_name:
                return joint_idx
        return None

    def get_joint_id_from_name(self, object_name, joint_name):
        object_id = self.urdf_ids[object_name]
        num_joints = p.getNumJoints(object_id, physicsClientId=self.id)
        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(object_id, joint_idx, physicsClientId=self.id)
            name = joint_info[1].decode("utf-8")
            if name == joint_name:
                return joint_idx
        return None

    def get_bounding_box(self, object_name):
        object_name = object_name.lower()
        if object_name == "init_table":
            return self.table_bbox_min, self.table_bbox_max
        object_id = self.urdf_ids[object_name]
        return self.get_aabb(object_id)

    def get_bounding_box_link(self, object_name, link_name):
        object_id = self.urdf_ids[object_name.lower()]
        link_id = self.get_link_id_from_name(object_name.lower(), link_name)
        if link_id is None:
            return None, None
        return self.get_aabb_link(object_id, link_id)

    def get_link_pose(self, object_name, custom_link_name):
        link_id = self.get_link_id_from_name(object_name.lower(), custom_link_name)
        if link_id is None:
            return None, None
        object_id = self.urdf_ids[object_name.lower()]
        link_pos, link_orient = p.getLinkState(object_id, link_id, physicsClientId=self.id)[:2]
        return np.array(link_pos), np.array(link_orient)
