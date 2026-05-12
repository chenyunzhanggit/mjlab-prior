"""Ball / kick termination terms.

Mirrors ``motionprior/.../mdp/terminations.py`` (ball-related entries).
Stateful counters (no-progress timeout, kick-stop latch, redirect latch)
are stashed on the env instance and cleared on episode reset via the
``episode_length_buf == 1`` rule used throughout the reference code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def ball_too_far_termination(
  env: ManagerBasedRlEnv,
  ball_name: str,
  max_distance: float,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Terminate when ball-to-robot XY distance exceeds ``max_distance``."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene[asset_name]
  dist = torch.norm(
    ball.data.root_link_pos_w[:, :2] - robot.data.root_link_pos_w[:, :2], dim=-1
  )
  return dist > max_distance


def ball_reach_goal_termination(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  """Terminate when ball reaches within ``threshold`` of the goal."""
  ball: Entity = env.scene[ball_name]
  command = env.command_manager.get_term(command_name)
  dist = torch.norm(
    ball.data.root_link_pos_w[:, :2] - command.goal_pos[:, :2], dim=-1
  )
  return dist < threshold


def ball_no_progress_timeout(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
  timeout_steps: int = 100,
  min_progress: float = 0.1,
) -> torch.Tensor:
  """Terminate when the ball-to-goal distance hasn't decreased by
  ``min_progress`` over the last ``timeout_steps`` steps.

  Rolls a circular distance buffer; counts consecutive insufficient-progress
  steps.
  """
  ball: Entity = env.scene[ball_name]
  command = env.command_manager.get_term(command_name)
  cur_dist = torch.norm(
    ball.data.root_link_pos_w - command.goal_pos, dim=-1
  )

  hist = getattr(env, "_ball_distance_history", None)
  if hist is None or hist.shape[0] != env.num_envs or hist.shape[1] != timeout_steps:
    hist = torch.zeros((env.num_envs, timeout_steps), device=env.device)
    env._ball_distance_history = hist
    env._no_progress_counter = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  env._ball_distance_history = torch.roll(env._ball_distance_history, shifts=1, dims=1)
  env._ball_distance_history[:, 0] = cur_dist
  old_dist = env._ball_distance_history[:, -1]
  progress = old_dist - cur_dist

  insufficient = progress < min_progress
  env._no_progress_counter = torch.where(
    insufficient,
    env._no_progress_counter + 1,
    torch.zeros_like(env._no_progress_counter),
  )
  terminate = env._no_progress_counter >= timeout_steps
  env._no_progress_counter = torch.where(
    terminate,
    torch.zeros_like(env._no_progress_counter),
    env._no_progress_counter,
  )
  return terminate


def ball_passed_through_zone(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
  zone_radius: float = 1.5,
  min_redirect_speed: float = 1.0,
) -> torch.Tensor:
  """Passing-task success: ball is redirected back into the source zone.

  We latch ``_ball_redirected`` once the ball is observed moving toward
  the source zone above ``min_redirect_speed``; success fires when the
  ball is then within ``zone_radius`` of source.
  """
  ball: Entity = env.scene[ball_name]
  command = env.command_manager.get_term(command_name)

  ball_xy = ball.data.root_link_pos_w[:, :2]
  ball_vel_xy = ball.data.root_link_lin_vel_w[:, :2]
  src_xy = command.goal_pos[:, :2]  # ``goal_pos`` aliases ``source_pos``

  to_src = src_xy - ball_xy
  norm = torch.norm(to_src, dim=-1, keepdim=True).clamp(min=1e-6)
  to_src_dir = to_src / norm
  vel_toward_src = (ball_vel_xy * to_src_dir).sum(-1)
  dist = torch.norm(to_src, dim=-1)

  redirected = getattr(env, "_ball_redirected", None)
  if redirected is None or redirected.shape[0] != env.num_envs:
    redirected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._ball_redirected = redirected
  just_reset = env.episode_length_buf == 1
  env._ball_redirected = env._ball_redirected & ~just_reset
  env._ball_redirected = env._ball_redirected | (vel_toward_src >= min_redirect_speed)

  return env._ball_redirected & (dist < zone_radius)


def ball_stopped_after_kick(
  env: ManagerBasedRlEnv,
  ball_name: str,
  stop_speed: float = 0.1,
  stop_steps: int = 30,
  min_kick_speed: float = 1.5,
) -> torch.Tensor:
  """Kicking-task end condition: ball was kicked, then stopped for N steps.

  Slow pushes (speed never exceeds ``min_kick_speed``) never set the
  kicked-flag, so the episode runs to time-out if the robot never delivers
  a real kick.
  """
  ball: Entity = env.scene[ball_name]
  speed = torch.norm(ball.data.root_link_lin_vel_w, dim=-1)

  kicked = getattr(env, "_kick_was_kicked", None)
  if kicked is None or kicked.shape[0] != env.num_envs:
    env._kick_was_kicked = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    env._kick_stop_counter = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  just_reset = env.episode_length_buf == 1
  env._kick_was_kicked = env._kick_was_kicked & ~just_reset
  env._kick_stop_counter = torch.where(
    just_reset,
    torch.zeros_like(env._kick_stop_counter),
    env._kick_stop_counter,
  )

  env._kick_was_kicked = env._kick_was_kicked | (speed >= min_kick_speed)
  ball_slow = speed < stop_speed
  should_count = env._kick_was_kicked & ball_slow
  env._kick_stop_counter = torch.where(
    should_count,
    env._kick_stop_counter + 1,
    torch.zeros_like(env._kick_stop_counter),
  )
  return env._kick_stop_counter >= stop_steps
