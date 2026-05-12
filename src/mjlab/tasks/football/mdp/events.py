"""Reset events that lay out robot + ball + (optional goal) along a random
direction θ at episode reset.

These replace the reference's ``G1DribblingEnv._reset_scene_along_line``
method by registering as ``mode="reset"`` EventTerms instead of forking
the env class.

Layout per task (see reference for context):

* Dribbling — robot at env_origin (±0.3m noise) facing θ, ball 0.4-0.7m
  ahead along θ, goal 8-15m ahead along θ.
* Kicking — robot at env_origin (±0.3m noise) facing θ, ball 0.8-1.8m
  ahead along θ with ±(0.3-0.6m) lateral offset, goal 6-10m ahead.
* Passing — robot at env_origin (±0.2m noise) facing θ, ball at source
  position (3-6m ahead with ±0.3m lateral offset) moving toward robot at
  5-9 m/s (the speed range comes from the command cfg).

The robot is teleported using ``write_root_link_pose_to_sim``; the ball
state is written via ``write_joint_state_to_sim`` (3 slide positions + 3
hinge zeros, plus the matching velocity vector).

Robot joints / actuator state are NOT reset here — that's expected to be
covered by the standard ``reset_robot_joints`` event already on the
inherited flat-velocity env. Only the floating base of the robot, and
the ball's joints, are touched.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.tasks.football.soccer_ball import BALL_JOINT_NAMES
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _resolve_env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  return env_ids.to(torch.long)


def _resolve_ball_joint_ids(env: ManagerBasedRlEnv, ball_name: str) -> torch.Tensor:
  """Cache & return the ball's 6 joint indices (slide x/y/z, hinge x/y/z)."""
  ball: Entity = env.scene[ball_name]
  key = f"_ball_joint_ids_{ball_name}"
  ids = getattr(env, key, None)
  if ids is None:
    ids, _names = ball.find_joints(list(BALL_JOINT_NAMES), preserve_order=True)
    ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
    setattr(env, key, ids)
  return ids


def _write_robot_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  xy: torch.Tensor,           # (n, 2)
  yaw: torch.Tensor,          # (n,)
  z: float = 0.78,
  asset_name: str = "robot",
) -> None:
  """Teleport the robot floating base; joints are not touched."""
  n = xy.shape[0]
  zeros_n = torch.zeros(n, device=env.device)
  quat = quat_from_euler_xyz(zeros_n, zeros_n, yaw)
  pos = torch.empty(n, 3, device=env.device)
  pos[:, :2] = xy
  pos[:, 2] = z
  pose = torch.cat([pos, quat], dim=-1)
  robot: Entity = env.scene[asset_name]
  robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
  # Zero base velocity so the new pose doesn't carry stale momentum.
  robot.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=env.device), env_ids=env_ids
  )


def _write_ball_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  pos_w: torch.Tensor,        # (n, 3)
  vel_w: torch.Tensor | None, # (n, 3) or None
  ball_name: str,
) -> None:
  """Write the ball's 6-joint state from a desired world pos / linear vel."""
  ball: Entity = env.scene[ball_name]
  joint_ids = _resolve_ball_joint_ids(env, ball_name)

  n = pos_w.shape[0]
  # qpos: 3 slides = world x/y/z, 3 hinges = 0.
  qpos = torch.zeros(n, 6, device=env.device)
  qpos[:, 0:3] = pos_w
  qvel = torch.zeros(n, 6, device=env.device)
  if vel_w is not None:
    qvel[:, 0:3] = vel_w
  ball.write_joint_state_to_sim(qpos, qvel, joint_ids=joint_ids, env_ids=env_ids)


# ---------------------------------------------------------------------------
# Per-task reset functions (mode="reset" EventTerm func)
# ---------------------------------------------------------------------------


def reset_ball_along_line_dribbling(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ball_name: str = "soccer_ball",
  command_name: str = "dribbling_commands",
  ball_radius: float = 0.11,
  robot_xy_noise: float = 0.3,
  ball_forward_range: tuple[float, float] = (0.4, 0.7),
  goal_forward_range: tuple[float, float] = (8.0, 15.0),
  robot_init_height: float = 0.78,
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  n = len(env_ids)
  if n == 0:
    return

  theta = torch.rand(n, device=env.device) * 2.0 * math.pi
  cos_t = torch.cos(theta)
  sin_t = torch.sin(theta)

  origins = env.scene.env_origins[env_ids]
  xy_noise = (torch.rand(n, 2, device=env.device) - 0.5) * (2.0 * robot_xy_noise)
  robot_xy = origins[:, :2] + xy_noise

  _write_robot_pose(env, env_ids, robot_xy, theta, z=robot_init_height)

  # Ball forward of robot along θ.
  d_ball = (
    torch.rand(n, device=env.device) * (ball_forward_range[1] - ball_forward_range[0])
    + ball_forward_range[0]
  )
  ball_pos = torch.empty(n, 3, device=env.device)
  ball_pos[:, 0] = robot_xy[:, 0] + d_ball * cos_t
  ball_pos[:, 1] = robot_xy[:, 1] + d_ball * sin_t
  ball_pos[:, 2] = ball_radius
  _write_ball_state(env, env_ids, ball_pos, vel_w=None, ball_name=ball_name)

  # Goal further forward along θ; commit into the command's buffer.
  d_goal = (
    torch.rand(n, device=env.device) * (goal_forward_range[1] - goal_forward_range[0])
    + goal_forward_range[0]
  )
  command = env.command_manager.get_term(command_name)
  goal_pos = torch.zeros(n, 3, device=env.device)
  goal_pos[:, 0] = robot_xy[:, 0] + d_goal * cos_t
  goal_pos[:, 1] = robot_xy[:, 1] + d_goal * sin_t
  command.goal_pos[env_ids] = goal_pos


def reset_ball_along_line_kicking(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ball_name: str = "soccer_ball",
  command_name: str = "kicking_commands",
  ball_radius: float = 0.11,
  robot_xy_noise: float = 0.3,
  ball_forward_range: tuple[float, float] = (0.8, 1.8),
  ball_lateral_range: tuple[float, float] = (0.3, 0.6),
  goal_forward_range: tuple[float, float] = (6.0, 10.0),
  robot_init_height: float = 0.78,
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  n = len(env_ids)
  if n == 0:
    return

  theta = torch.rand(n, device=env.device) * 2.0 * math.pi
  cos_t = torch.cos(theta)
  sin_t = torch.sin(theta)

  origins = env.scene.env_origins[env_ids]
  xy_noise = (torch.rand(n, 2, device=env.device) - 0.5) * (2.0 * robot_xy_noise)
  robot_xy = origins[:, :2] + xy_noise

  _write_robot_pose(env, env_ids, robot_xy, theta, z=robot_init_height)

  # Ball: random forward + signed lateral offset along the perpendicular.
  d_fwd = (
    torch.rand(n, device=env.device)
    * (ball_forward_range[1] - ball_forward_range[0])
    + ball_forward_range[0]
  )
  d_lat = (
    torch.rand(n, device=env.device)
    * (ball_lateral_range[1] - ball_lateral_range[0])
    + ball_lateral_range[0]
  )
  sign = torch.where(
    torch.rand(n, device=env.device) > 0.5,
    torch.ones(n, device=env.device),
    -torch.ones(n, device=env.device),
  )
  d_lat = d_lat * sign

  ball_pos = torch.empty(n, 3, device=env.device)
  ball_pos[:, 0] = robot_xy[:, 0] + d_fwd * cos_t - d_lat * sin_t
  ball_pos[:, 1] = robot_xy[:, 1] + d_fwd * sin_t + d_lat * cos_t
  ball_pos[:, 2] = ball_radius
  _write_ball_state(env, env_ids, ball_pos, vel_w=None, ball_name=ball_name)

  # Goal along θ, plus ball_init_pos snapshot for the kicking command.
  d_goal = (
    torch.rand(n, device=env.device) * (goal_forward_range[1] - goal_forward_range[0])
    + goal_forward_range[0]
  )
  command = env.command_manager.get_term(command_name)
  goal_pos = torch.zeros(n, 3, device=env.device)
  goal_pos[:, 0] = robot_xy[:, 0] + d_goal * cos_t
  goal_pos[:, 1] = robot_xy[:, 1] + d_goal * sin_t
  command.goal_pos[env_ids] = goal_pos
  if hasattr(command, "ball_init_pos"):
    command.ball_init_pos[env_ids] = ball_pos


def reset_ball_along_line_passing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ball_name: str = "soccer_ball",
  command_name: str = "passing_commands",
  ball_radius: float = 0.11,
  robot_xy_noise: float = 0.2,
  robot_init_height: float = 0.78,
) -> None:
  env_ids = _resolve_env_ids(env, env_ids)
  n = len(env_ids)
  if n == 0:
    return

  theta = torch.rand(n, device=env.device) * 2.0 * math.pi
  cos_t = torch.cos(theta)
  sin_t = torch.sin(theta)

  origins = env.scene.env_origins[env_ids]
  xy_noise = (torch.rand(n, 2, device=env.device) - 0.5) * (2.0 * robot_xy_noise)
  robot_xy = origins[:, :2] + xy_noise

  _write_robot_pose(env, env_ids, robot_xy, theta, z=robot_init_height)

  command = env.command_manager.get_term(command_name)
  cfg = command.cfg
  lo_d, hi_d = cfg.source_distance_range
  lo_l, hi_l = cfg.source_lateral_range
  lo_v, hi_v = cfg.ball_speed_range
  d_src = torch.rand(n, device=env.device) * (hi_d - lo_d) + lo_d
  lat = torch.rand(n, device=env.device) * (hi_l - lo_l) + lo_l

  src_x = robot_xy[:, 0] + cos_t * d_src - sin_t * lat
  src_y = robot_xy[:, 1] + sin_t * d_src + cos_t * lat
  src_z = torch.full((n,), ball_radius, device=env.device)
  src_pos = torch.stack([src_x, src_y, src_z], dim=-1)

  # Velocity: from source toward robot (XY only).
  to_robot_xy = robot_xy - src_pos[:, :2]
  to_robot_norm = to_robot_xy / to_robot_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  speed = torch.rand(n, device=env.device) * (hi_v - lo_v) + lo_v
  vel_xy = to_robot_norm * speed.unsqueeze(-1)
  vel_w = torch.cat([vel_xy, torch.zeros(n, 1, device=env.device)], dim=-1)

  _write_ball_state(env, env_ids, src_pos, vel_w=vel_w, ball_name=ball_name)
  command.source_pos[env_ids] = src_pos
