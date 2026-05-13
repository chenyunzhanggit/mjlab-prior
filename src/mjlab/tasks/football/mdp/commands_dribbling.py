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
    """No-op. Goal placement is owned by ``reset_ball_along_line_dribbling``
    (an EventTerm). mjlab's reset order runs ``event_manager.apply`` BEFORE
    ``command_manager.reset``, so if this method actually wrote goal_pos it
    would always read ``robot.data.root_link_pos_w`` BEFORE sim.forward()
    propagates the event's new pose — stale value, wrong goal.

    Letting the event own both robot pose and goal_pos avoids that race
    and matches how the reference ``G1KickingEnv._reset_scene_along_line``
    handles it.
    """
    # Timer is reset by the base ``CommandTerm._resample`` wrapper; no
    # additional work needed here.

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
