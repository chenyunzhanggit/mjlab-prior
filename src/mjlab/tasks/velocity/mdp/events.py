"""Velocity-task specific MDP events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def force_forward_only_command_for_terrain_class(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  terrain_classes: Sequence[int],
  min_lin_vel_x: float | None = None,
) -> None:
  """Zero out lateral and yaw command for envs on the given terrain classes.

  Runs every step (``mode="step"``). Reads ``terrain.env_terrain_class`` and
  masks ``vel_command_b[:, 1]`` (lin_vel_y) and ``vel_command_b[:, 2]``
  (ang_vel_z) to zero for matching envs. If ``min_lin_vel_x`` is given,
  also forces ``vel_command_b[:, 0]`` to be at least ``min_lin_vel_x``
  (i.e. strictly positive, no backward / no near-zero forward), so the
  robot is always commanded to step forward on those terrains.
  """
  del env_ids  # step-mode events receive None; we mask all envs.
  terrain = env.scene.terrain
  if terrain is None:
    return
  env_class = getattr(terrain, "env_terrain_class", None)
  if env_class is None:
    return

  cmd_term = env.command_manager.get_term(command_name)
  if cmd_term is None:
    return
  vel_b = getattr(cmd_term, "vel_command_b", None)
  if vel_b is None:
    return

  classes = torch.tensor(
    list(terrain_classes), device=env_class.device, dtype=env_class.dtype
  )
  mask = (env_class.unsqueeze(-1) == classes).any(dim=-1)
  if not mask.any():
    return

  vel_b[mask, 1] = 0.0
  vel_b[mask, 2] = 0.0
  if min_lin_vel_x is not None:
    vel_b[mask, 0] = torch.clamp(vel_b[mask, 0], min=min_lin_vel_x)

  vel_w = getattr(cmd_term, "vel_command_w", None)
  if vel_w is not None:
    vel_w[mask, 1] = 0.0
    vel_w[mask, 2] = 0.0
    if min_lin_vel_x is not None:
      vel_w[mask, 0] = vel_b[mask, 0]

  is_heading = getattr(cmd_term, "is_heading_env", None)
  if is_heading is not None:
    is_heading[mask] = False


def reset_yaw_for_terrain_class(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  terrain_classes: Sequence[int],
  yaw_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Re-sample root yaw within ``yaw_range`` for envs on the given terrain classes.

  Runs in ``mode="reset"`` after ``reset_root_state_uniform``. The default
  ``reset_base`` randomizes yaw over [-pi, pi], which conflicts with
  ``cheat_penalty(mode="world_heading")`` on stair terrains (which assumes the
  world +x axis is the forward direction). This event overwrites the yaw of
  matching envs while keeping their position and roll/pitch intact.

  Implementation reads the just-written pose directly from ``qpos`` rather than
  from ``root_link_pos_w`` / ``root_link_quat_w`` (which alias ``xpos`` /
  ``xquat`` -- forward-kinematics outputs that are not refreshed until the next
  ``mj_forward``). Re-uses the existing quaternion's roll/pitch components and
  only overwrites yaw.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  if env_ids.numel() == 0:
    return

  terrain = env.scene.terrain
  if terrain is None:
    return
  env_class = getattr(terrain, "env_terrain_class", None)
  if env_class is None:
    return

  classes = torch.tensor(
    list(terrain_classes), device=env_class.device, dtype=env_class.dtype
  )
  mask = (env_class[env_ids].unsqueeze(-1) == classes).any(dim=-1)
  target_ids = env_ids[mask]
  if target_ids.numel() == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  q_adr = asset.data.indexing.free_joint_q_adr  # length 7: pos(3) + quat(4)
  qpos = asset.data.data.qpos

  pos = qpos[target_ids][:, q_adr[0:3]]
  quat = qpos[target_ids][:, q_adr[3:7]]  # (w, x, y, z)

  # Extract roll/pitch from current quat, replace yaw with a fresh sample.
  w, x, y, z = quat.unbind(dim=-1)
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = torch.atan2(sinr_cosp, cosr_cosp)
  sinp = 2.0 * (w * y - z * x)
  pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

  new_yaw = sample_uniform(
    torch.tensor(yaw_range[0], device=env.device),
    torch.tensor(yaw_range[1], device=env.device),
    (target_ids.numel(),),
    device=env.device,
  )

  new_quat = quat_from_euler_xyz(roll, pitch, new_yaw)
  pose = torch.cat([pos, new_quat], dim=-1)
  asset.write_root_link_pose_to_sim(pose, env_ids=target_ids)
