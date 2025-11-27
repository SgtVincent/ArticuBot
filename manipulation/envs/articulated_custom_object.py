"""Custom articulated environment for user-defined URDF assets."""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional
import gym
import numpy as np

from manipulation.envs.articulated import articulated
from manipulation.envs.sim_data_gen_custom import SimpleEnvDataGenCustom
from manipulation.custom_object_utils.env_utils import get_handle_pos_custom, parse_config_custom_object
from manipulation.utils.env_utils import parse_center


class articulated_custom_object(SimpleEnvDataGenCustom, articulated):
    """Variant of ``articulated`` that reads custom annotations/paths."""

    def __init__(self, task_name, object_name, link_name, init_angle, *args, **kwargs):
        self.custom_asset_dir: Optional[str] = kwargs.pop("custom_asset_dir", None)
        self.custom_handle_pts_path: Optional[str] = kwargs.pop("handle_pts_path", None)
        self.custom_annotation_path: Optional[str] = kwargs.pop("annotation_path", None)
        
        config_path = kwargs.get("config_path")
        self._config_dir: Optional[Path] = None
        if config_path:
            self._config_dir = Path(config_path).resolve().parent

        super().__init__(task_name, object_name, link_name, init_angle, *args, **kwargs)
        
        self._ingest_custom_metadata()

    def _parse_config(self):
        """Override to use custom config parser."""
        asset_dir = self.custom_asset_dir
        if not asset_dir and self._config_dir:
            asset_dir = str(self._config_dir)
            
        return parse_config_custom_object(
            self.config,
            obj_id=self.obj_id,
            use_vhacd=True,
            custom_asset_dir=asset_dir
        )

    def _resolve_path(self, raw_value: Optional[str]) -> Optional[str]:
        if not raw_value:
            return None
        candidate = Path(raw_value).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        if self._config_dir is not None:
            cfg_candidate = (self._config_dir / candidate).resolve()
            if cfg_candidate.exists():
                return str(cfg_candidate)
        project_dir = Path(os.environ.get("PROJECT_DIR", str(Path.cwd()))).resolve()
        return str((project_dir / candidate).resolve())

    def _ingest_custom_metadata(self) -> None:
        if not getattr(self, "config", None):
            return
        object_name_lower = self.object_name.lower()
        for block in self.config:
            if str(block.get("name", "")).lower() != object_name_lower:
                continue
            handle_name = block.get("handle_name")
            if handle_name:
                self.handle_name = handle_name
            asset_dir_candidate = block.get("reward_asset_path") or block.get("solution_path")
            resolved_asset_dir = self._resolve_path(asset_dir_candidate)
            if resolved_asset_dir:
                self.custom_asset_dir = resolved_asset_dir
            handle_pts_path = block.get("handle_pts_path")
            resolved_handle_pts = self._resolve_path(handle_pts_path)
            if resolved_handle_pts:
                self.custom_handle_pts_path = resolved_handle_pts
            annotation_path = block.get("annotation_path")
            resolved_annotation = self._resolve_path(annotation_path)
            if resolved_annotation:
                self.custom_annotation_path = resolved_annotation


    def get_handle_pos(
        self,
        return_median: bool = True,
        handle_pts_obj_frame=None,
        mobility_info=None,
        return_info: bool = False,
        custom_joint_name: Optional[str] = None,
    ):
        target_joint_name = custom_joint_name or self.handle_name
        return get_handle_pos_custom(
            self,
            self.object_name,
            asset_dir_hint=self.custom_asset_dir,
            handle_pts_override=self.custom_handle_pts_path,
            return_median=return_median,
            handle_pts_obj_frame=handle_pts_obj_frame,
            mobility_info=mobility_info,
            return_info=return_info,
            target_object=target_joint_name,
        )


gym.register(
    id="articulated-custom-object-v0",
    entry_point=articulated_custom_object,
)
