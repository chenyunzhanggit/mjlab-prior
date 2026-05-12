"""Velocity-task specific MDP events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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
