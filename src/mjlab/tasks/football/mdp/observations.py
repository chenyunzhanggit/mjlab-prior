"""Ball- and goal-related observation terms.

Ports of the ball observations from
``motionprior/.../mdp/observations.py``. All terms read the ball entity
by name via ``env.scene[ball_name]`` and transform into the robot's body
frame using ``quat_apply_inverse(robot_quat, ...)`` when relative
information is exposed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def dribbling_goal_position(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """``(dx, dy, dist)`` of the goal point in world XY frame relative to
  the robot. Shape ``(num_envs, 3)``."""
  command = env.command_manager.get_term(command_name)
  robot: Entity = env.scene["robot"]
  robot_xy = robot.data.root_link_pos_w[:, :2]
  goal_xy = command.goal_pos[:, :2]
  rel = goal_xy - robot_xy
  dist = torch.norm(rel, dim=-1, keepdim=True)
  return torch.cat([rel, dist], dim=-1)


def ball_relative_position(
  env: ManagerBasedRlEnv,
  ball_name: str,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Ball position relative to the robot, in the robot's body frame
  (so XY rotates with yaw). Shape ``(num_envs, 3)``."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene[asset_name]
  rel_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, rel_w)


def ball_velocity(env: ManagerBasedRlEnv, ball_name: str) -> torch.Tensor:
  """Ball linear velocity in the world frame. Shape ``(num_envs, 3)``."""
  ball: Entity = env.scene[ball_name]
  return ball.data.root_link_lin_vel_w


def ball_to_goal_vector(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
) -> torch.Tensor:
  """XY vector from ball to goal, expressed in the robot body frame.
  Shape ``(num_envs, 2)``."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_term(command_name)

  ball_xy = ball.data.root_link_pos_w[:, :2]
  goal_xy = command.goal_pos[:, :2]
  diff = goal_xy - ball_xy  # (N, 2)

  diff_3d = torch.cat(
    [diff, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
  )
  diff_b = quat_apply_inverse(robot.data.root_link_quat_w, diff_3d)
  return diff_b[:, :2]


def ball_absolute_position(
  env: ManagerBasedRlEnv, ball_name: str
) -> torch.Tensor:
  """Ball position in the world frame. Shape ``(num_envs, 3)``.

  Privileged obs only — for multi-env training each env has its own
  origin, so this carries a large absolute offset.
  """
  ball: Entity = env.scene[ball_name]
  return ball.data.root_link_pos_w


def passing_source_position(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """``(dx, dy, dist)`` of the passing source position relative to the
  robot, in the robot body frame. Shape ``(num_envs, 3)``."""
  command = env.command_manager.get_term(command_name)
  robot: Entity = env.scene["robot"]
  robot_xy = robot.data.root_link_pos_w[:, :2]
  src_xy = command.source_pos[:, :2]
  rel = src_xy - robot_xy
  dist = torch.norm(rel, dim=-1, keepdim=True)

  rel_3d = torch.cat([rel, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
  rel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_3d)
  return torch.cat([rel_b[:, :2], dist], dim=-1)


def depth_image(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  miss_value: float | None = None,
  scale: float = 1.0,
) -> torch.Tensor:
  """Flattened depth image from a :class:`RayCastSensor`.

  Returns the raw per-ray distances (metres) as a flat ``(num_envs, N)``
  vector, where ``N = num_frames * num_rays_per_frame``. Miss rays
  (``distance < 0`` from the raycaster) are clamped to ``miss_value``
  (defaults to the sensor's ``max_distance``). The output is multiplied
  by ``scale`` so the caller can normalise into ``[0, 1]`` for the
  policy network if desired (e.g. ``scale=1/max_distance``).

  Use with a :class:`PinholeCameraPatternCfg` raycast sensor to simulate
  a forward-facing depth camera / LiDAR on a humanoid — the perception
  signal we use to replace direct ball-state observations in the
  perception-only passing task.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  if miss_value is None:
    miss_value = sensor.cfg.max_distance
  distances = sensor.data.distances  # [B, N]
  miss_mask = distances < 0
  out = torch.where(
    miss_mask, torch.full_like(distances, miss_value), distances
  )
  if scale != 1.0:
    out = out * scale
  return out
