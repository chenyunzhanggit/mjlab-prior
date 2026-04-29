"""Motion-reference velocity observations for the Teleopit teacher_a actor.

The Teleopit ``General-Tracking-G1`` actor consumes three reference signals
derived from the motion command (see prior.md, "Teleopit Teacher 实情"):

  * ``ref_base_lin_vel_b`` — motion ref anchor linear velocity in the
    robot's current anchor frame.
  * ``ref_base_ang_vel_b`` — motion ref anchor angular velocity in the
    robot's current anchor frame.
  * ``ref_projected_gravity_b`` — gravity projected into the motion ref's
    own anchor frame (independent of robot state).

These are equivalent to Teleopit's ``observation.py`` formulas:
``rotate_by_quat_inverse(robot_anchor_quat, anchor_*_vel_w)`` and
``rotate_by_quat_inverse(motion_anchor_quat, gravity_unit_world)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def ref_base_lin_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return quat_apply_inverse(command.robot_anchor_quat_w, command.anchor_lin_vel_w)


def ref_base_ang_vel_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return quat_apply_inverse(command.robot_anchor_quat_w, command.anchor_ang_vel_w)


def ref_projected_gravity_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  asset = env.scene[command.cfg.entity_name]
  return quat_apply_inverse(command.anchor_quat_w, asset.data.gravity_vec_w)
