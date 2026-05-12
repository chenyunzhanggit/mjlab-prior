"""Ball / kick reward terms.

Port of the football reward functions in
``motionprior/.../mdp/rewards.py``. Each term is a stateless function
that reads ball/robot state from the env scene and returns a
``[num_envs]`` reward tensor.

Per-env stateful caches (e.g. last-step ball distance for progress
rewards) are stashed on the env instance under unique attribute names so
running multiple football tasks in the same process doesn't cross-talk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# Progress / approach
# ---------------------------------------------------------------------------


def ball_to_goal_progress(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
) -> torch.Tensor:
  """Reward per-step decrease in ball-to-goal distance (XY)."""
  ball: Entity = env.scene[ball_name]
  command = env.command_manager.get_term(command_name)

  cur = torch.norm(
    ball.data.root_link_pos_w[:, :2] - command.goal_pos[:, :2], dim=-1
  )
  key = f"_ball_progress_{command_name}"
  prev = getattr(env, key, None)
  if prev is None:
    setattr(env, key, cur.clone())
    return torch.zeros(env.num_envs, device=env.device)

  progress = prev - cur
  setattr(env, key, cur.clone())
  return progress


def approach_ball_reward(
  env: ManagerBasedRlEnv,
  ball_name: str,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Reward the robot for moving closer to the ball (only positive progress)."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene[asset_name]

  cur = torch.norm(
    ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1
  )
  prev = getattr(env, "_robot_ball_distance", None)
  if prev is None:
    env._robot_ball_distance = cur.clone()
    return torch.zeros(env.num_envs, device=env.device)

  progress = prev - cur
  env._robot_ball_distance = cur.clone()
  return torch.clamp(progress, min=0.0)


# ---------------------------------------------------------------------------
# Distance / proximity
# ---------------------------------------------------------------------------


def ball_distance_penalty(
  env: ManagerBasedRlEnv,
  ball_name: str,
  max_distance: float,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Squared penalty when ball is further than ``max_distance`` from the robot."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene[asset_name]

  dist = torch.norm(
    ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1
  )
  excess = torch.clamp(dist - max_distance, min=0.0)
  return torch.square(excess)


def ball_too_far_penalty(
  env: ManagerBasedRlEnv,
  ball_name: str,
  critical_distance: float,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Indicator penalty (1.0) when ball is beyond ``critical_distance``."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene[asset_name]
  dist = torch.norm(
    ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1
  )
  return (dist > critical_distance).float()


def foot_ball_proximity_reward(
  env: ManagerBasedRlEnv,
  ball_name: str,
  foot_body_names: list[str],
  contact_distance: float = 0.15,
) -> torch.Tensor:
  """``exp(-5 * min_foot_to_ball_distance)`` — soft proximity reward."""
  del contact_distance  # kept for cfg parity; hard threshold disabled by default
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene["robot"]

  ball_pos = ball.data.root_link_pos_w  # (N, 3)

  body_ids, _ = robot.find_bodies(list(foot_body_names))
  if not body_ids:
    return torch.zeros(env.num_envs, device=env.device)
  body_ids_t = torch.as_tensor(body_ids, device=env.device, dtype=torch.long)
  # body_link_pos_w: (N, num_bodies, 3) -> (N, F, 3)
  foot_pos = robot.data.body_link_pos_w.index_select(1, body_ids_t)
  diff = foot_pos - ball_pos.unsqueeze(1)  # (N, F, 3)
  dists = torch.norm(diff, dim=-1)  # (N, F)
  min_dist = dists.min(dim=-1).values  # (N,)
  return torch.exp(-5.0 * min_dist)


# ---------------------------------------------------------------------------
# Contact / impact
# ---------------------------------------------------------------------------


def foot_ball_contact_reward(
  env: ManagerBasedRlEnv,
  ball_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward foot↔ball contact, but only when force is mostly horizontal.

  Mirrors reference: ``(force_xy > 5.0) & (force_z < 5.0)``. The sensor
  must be configured with ``reduce="netforce"`` so that ``data.force`` is
  in the global frame and sums all primary contacts on the ball.
  """
  sensor: ContactSensor = env.scene[ball_sensor_cfg.name]
  data = sensor.data
  force = data.force
  if force is None:
    return torch.zeros(env.num_envs, device=env.device)
  # force shape: [B, N, 3] — sum over primary contacts (N) into one net wrench.
  net = force.sum(dim=1)  # [B, 3]
  force_xy = torch.norm(net[..., :2], dim=-1)
  force_z = torch.abs(net[..., 2])
  return ((force_xy > 5.0) & (force_z < 5.0)).float()


def ball_impact_reward(
  env: ManagerBasedRlEnv,
  ball_name: str,
  velocity_change_threshold: float = 0.5,
) -> torch.Tensor:
  """Indicator reward when ball velocity changes by more than threshold per step."""
  ball: Entity = env.scene[ball_name]
  vel = ball.data.root_link_lin_vel_w.clone()
  prev = getattr(env, "_prev_ball_velocity", None)
  if prev is None:
    env._prev_ball_velocity = vel
    return torch.zeros(env.num_envs, device=env.device)

  delta = torch.norm(vel - prev, dim=-1)
  env._prev_ball_velocity = vel
  return (delta > velocity_change_threshold).float()


# ---------------------------------------------------------------------------
# Goal reaching (gaussian distance + bonus when within threshold)
# ---------------------------------------------------------------------------


def ball_reach_goal_reward(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
  threshold: float,
  sigma: float = 2.0,
  bonus_scale: float = 10.0,
) -> torch.Tensor:
  """``exp(-d²/2σ²) * (1 + bonus_scale * 1[d<threshold])``."""
  ball: Entity = env.scene[ball_name]
  command = env.command_manager.get_term(command_name)

  dist = torch.norm(
    ball.data.root_link_pos_w[:, :2] - command.goal_pos[:, :2], dim=-1
  )
  distance_reward = torch.exp(-(dist**2) / (2 * sigma**2))
  bonus = (dist < threshold).float() * bonus_scale
  return distance_reward * (1.0 + bonus)


# ---------------------------------------------------------------------------
# Kicking-specific
# ---------------------------------------------------------------------------


def align_to_kick_reward(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
  asset_name: str = "robot",
  kick_distance: float = 0.45,
) -> torch.Tensor:
  """Reward standing on the ball→goal axis behind the ball.

  Ideal robot xy ≈ ball + normalize(ball - goal) * kick_distance.
  Returns ``exp(-4 * dist_to_ideal)``.
  """
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene[asset_name]
  command = env.command_manager.get_term(command_name)

  ball_xy = ball.data.root_link_pos_w[:, :2]
  goal_xy = command.goal_pos[:, :2]
  robot_xy = robot.data.root_link_pos_w[:, :2]

  goal_to_ball = ball_xy - goal_xy
  norm = torch.norm(goal_to_ball, dim=-1, keepdim=True).clamp(min=1e-6)
  approach_dir = goal_to_ball / norm
  ideal = ball_xy + approach_dir * kick_distance
  dist_to_ideal = torch.norm(robot_xy - ideal, dim=-1)
  return torch.exp(-4.0 * dist_to_ideal)


def kick_ball_toward_goal(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
  speed_threshold: float = 0.8,
) -> torch.Tensor:
  """Velocity-aligned kick reward: ``speed * exp(-4*(1-cosθ)) * 1[speed>thr]``."""
  ball: Entity = env.scene[ball_name]
  command = env.command_manager.get_term(command_name)

  ball_xy = ball.data.root_link_pos_w[:, :2]
  goal_xy = command.goal_pos[:, :2]
  ball_vel_xy = ball.data.root_link_lin_vel_w[:, :2]

  ball_to_goal = goal_xy - ball_xy
  norm = torch.norm(ball_to_goal, dim=-1, keepdim=True).clamp(min=1e-6)
  ball_to_goal_dir = ball_to_goal / norm

  speed = torch.norm(ball_vel_xy, dim=-1)
  cos_theta = (ball_vel_xy * ball_to_goal_dir).sum(-1) / (speed + 1e-6)
  active = (speed > speed_threshold).float()
  direction_weight = torch.exp(-4.0 * (1.0 - cos_theta.clamp(max=1.0)))
  return speed * direction_weight * active


def ball_scored_goal_reward(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
  goal_width: float = 3.66,
  goal_height: float = 2.44,
  ball_radius: float = 0.11,
  bonus: float = 1.0,
) -> torch.Tensor:
  """Latching goal-scored bonus: stays at ``bonus`` for the rest of the episode
  once the ball crosses the goal line within the goal box.
  """
  ball: Entity = env.scene[ball_name]
  command = env.command_manager.get_term(command_name)

  ball_pos = ball.data.root_link_pos_w
  goal_pos = command.goal_pos

  scored = getattr(env, "_kicking_scored", None)
  if scored is None or scored.shape[0] != env.num_envs:
    scored = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._kicking_scored = scored
  # Clear on the first step of each new episode.
  env._kicking_scored &= env.episode_length_buf != 1

  past_x = ball_pos[:, 0] >= (goal_pos[:, 0] - ball_radius)
  in_width = torch.abs(ball_pos[:, 1] - goal_pos[:, 1]) <= (goal_width / 2.0 + ball_radius)
  in_height = ball_pos[:, 2] <= (goal_height + ball_radius)
  env._kicking_scored |= past_x & in_width & in_height
  return env._kicking_scored.float() * bonus
