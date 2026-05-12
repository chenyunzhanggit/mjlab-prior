"""Dribbling goal command — random target point on the ground.

Port of ``motionprior/.../mdp/commands_dribbling.py`` onto the mjlab
``CommandTerm`` API.

The goal is sampled in the env-local frame from
``cfg.goal_ranges.{x,y}`` with a minimum distance constraint, then
shifted by the robot's current world position so it remains relative to
each env's robot at reset time. Held fixed within an episode unless
``resampling_time_range`` rolls over (kept at ~episode length for the
downstream tasks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class DribblingGoalCommand(CommandTerm):
  """Generates a random target point on the ground for the dribbling task."""

  cfg: DribblingGoalCommandCfg

  def __init__(self, cfg: DribblingGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    # World-frame goal position (z=0 by default; reset event may override).
    self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["distance_to_goal"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.goal_pos

  @property
  def goal_pos_2d(self) -> torch.Tensor:
    return self.goal_pos[:, :2]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Sample a goal in env-local frame, offset by current robot pos.

    Rejection-samples until each env's offset >= ``min_distance``.
    Capped at 100 attempts to avoid infinite loops for degenerate ranges.
    """
    r = len(env_ids)
    if r == 0:
      return

    ranges = self.cfg.goal_ranges
    robot_pos = self.robot.data.root_link_pos_w[env_ids]

    goal_x = torch.zeros(r, device=self.device)
    goal_y = torch.zeros(r, device=self.device)
    valid = torch.zeros(r, dtype=torch.bool, device=self.device)

    for _ in range(100):
      remaining = ~valid
      if not remaining.any():
        break
      n = int(remaining.sum().item())
      new_x = torch.rand(n, device=self.device) * (ranges.x[1] - ranges.x[0]) + ranges.x[0]
      new_y = torch.rand(n, device=self.device) * (ranges.y[1] - ranges.y[0]) + ranges.y[0]
      ok = torch.sqrt(new_x**2 + new_y**2) >= ranges.min_distance
      # Absolute coords for the satisfied subset.
      abs_x = robot_pos[remaining, 0] + new_x
      abs_y = robot_pos[remaining, 1] + new_y
      goal_x[remaining] = torch.where(ok, abs_x, goal_x[remaining])
      goal_y[remaining] = torch.where(ok, abs_y, goal_y[remaining])
      valid[remaining] = valid[remaining] | ok

    self.goal_pos[env_ids, 0] = goal_x
    self.goal_pos[env_ids, 1] = goal_y
    self.goal_pos[env_ids, 2] = 0.0

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    robot_xy = self.robot.data.root_link_pos_w[:, :2]
    goal_xy = self.goal_pos[:, :2]
    self.metrics["distance_to_goal"] = torch.norm(goal_xy - robot_xy, dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for idx in env_indices:
      pos = self.goal_pos[idx].cpu().numpy()
      visualizer.add_sphere(
        center=pos,
        radius=0.3,
        color=(0.0, 1.0, 0.0, 0.45),
        label=f"dribbling_goal_{idx}",
      )


@dataclass(kw_only=True)
class DribblingGoalCommandCfg(CommandTermCfg):
  """Configuration for :class:`DribblingGoalCommand`."""

  asset_name: str = "robot"
  ball_name: str = "soccer_ball"

  @dataclass
  class Ranges:
    x: tuple[float, float] = (10.0, 20.0)
    """x-offset from robot's current position [m]."""
    y: tuple[float, float] = (0.0, 0.0)
    """y-offset from robot's current position [m]."""
    min_distance: float = 10.0
    """Reject samples closer than this to the robot [m]."""

  goal_ranges: Ranges = field(default_factory=Ranges)

  def build(self, env: ManagerBasedRlEnv) -> DribblingGoalCommand:
    return DribblingGoalCommand(self, env)
