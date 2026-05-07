"""Multi-motion tracking observation terms.

Adds the per-frame error / heuristic observations consumed by the upstream
``motionprior`` reference's teacher actor that are not already present in
``mjlab.tasks.tracking.mdp.observations``:

* ``motion_anchor_ang_vel`` — anchor angular velocity in world frame.
* ``anchor_pos_error`` — world-frame ``anchor_pos_w − robot_anchor_pos_w``.
* ``relative_body_pos_error`` — per-body world-frame position residual,
  flattened to ``(N, 3 * num_bodies)``.
* ``relative_body_orientation_error`` — per-body quaternion error
  magnitude, ``(N, num_bodies)``.
* ``anchor_height`` — robot anchor's world-frame Z, ``(N, 1)``.

Each term reads its data from the ``MotionCommand`` interface (whose
property surface is shared by the single- and multi-motion command
implementations — see ``single_motion_migration_audit.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_ang_vel(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Reference anchor angular velocity in world frame, ``(N, 3)``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.anchor_ang_vel_w.view(env.num_envs, -1)


def anchor_pos_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """World-frame position residual ``anchor_pos_w − robot_anchor_pos_w``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (command.anchor_pos_w - command.robot_anchor_pos_w).view(env.num_envs, -1)


def relative_body_pos_error(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Per-body world-frame position residual, flattened to ``(N, 3·B)``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = command.body_pos_relative_w - command.robot_body_pos_w
  return error.reshape(env.num_envs, -1)


def relative_body_orientation_error(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Per-body quaternion error magnitude, ``(N, B)``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(
    command.body_quat_relative_w, command.robot_body_quat_w
  )
  return error.reshape(env.num_envs, -1)


def anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Robot anchor world-frame Z, ``(N, 1)``."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.robot_anchor_pos_w[:, 2:3]
