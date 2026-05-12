"""Passing command — one-touch redirect target ("source zone")."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class PassingCommand(CommandTerm):
  """Tracks the source position from which the ball is launched."""

  cfg: PassingCommandCfg

  def __init__(self, cfg: PassingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.asset_name]
    self.ball: Entity = env.scene[cfg.ball_name]
    self.source_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.metrics["dist_ball_to_source"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.source_pos

  @property
  def goal_pos(self) -> torch.Tensor:
    """Alias so dribbling/kicking reward helpers can reuse this command."""
    return self.source_pos

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Source placement is owned by reset_ball_along_line_passing;
    # this resets only the resampling timer.
    pass

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    ball_xy = self.ball.data.root_link_pos_w[:, :2]
    src_xy = self.source_pos[:, :2]
    self.metrics["dist_ball_to_source"] = torch.norm(ball_xy - src_xy, dim=-1)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for idx in env_indices:
      pos = self.source_pos[idx].cpu().numpy().copy()
      pos[2] = 0.01
      visualizer.add_sphere(
        center=pos,
        radius=self.cfg.zone_radius,
        color=(0.0, 1.0, 0.2, 0.35),
        label=f"passing_zone_{idx}",
      )


@dataclass(kw_only=True)
class PassingCommandCfg(CommandTermCfg):
  """Configuration for :class:`PassingCommand`."""

  asset_name: str = "robot"
  ball_name: str = "soccer_ball"

  zone_radius: float = 1.5
  """Target zone radius [m] — ball passing through counts as success."""

  source_distance_range: tuple[float, float] = (3.0, 6.0)
  """Distance from robot to ball source position [m]."""

  source_lateral_range: tuple[float, float] = (-0.3, 0.3)
  """Lateral offset of source along robot's right vector [m]."""

  ball_speed_range: tuple[float, float] = (5.0, 9.0)
  """Incoming ball speed range (toward robot) [m/s]."""

  def build(self, env: ManagerBasedRlEnv) -> PassingCommand:
    return PassingCommand(self, env)
