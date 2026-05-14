"""Kicking goal command — soccer-goal target with frame visualization.

Inherits :class:`DribblingGoalCommand` for the buffer mechanics. The
actual goal_pos is written by ``reset_ball_along_line_kicking`` at every
reset (since the kicking layout couples robot / ball / goal along a
common direction), so ``_resample_command`` only manages the
resampling timer here. Compared to reference, debug visualization is
intentionally minimal — the renderer doesn't support per-env cuboid
markers in the same way as the IsaacLab marker manager, so we just draw
a sphere at the goal centre (left/right posts are sketched as side spheres).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .commands_dribbling import DribblingGoalCommand, DribblingGoalCommandCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class KickingGoalCommand(DribblingGoalCommand):
  """Goal command for the kicking task. Goal frame size from cfg."""

  cfg: KickingGoalCommandCfg

  def __init__(self, cfg: KickingGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    # Ball position recorded at reset; used by some reward terms.
    self.ball_init_pos = torch.zeros(self.num_envs, 3, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Robot/ball/goal placement is owned by reset_ball_along_line_kicking;
    # this just resets the resampling timer (handled by CommandTerm.reset).
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    half_w = self.cfg.goal_width / 2.0
    goal_h = self.cfg.goal_height
    for idx in env_indices:
      gx, gy, _gz = self.goal_pos[idx].cpu().numpy()
      # Centre of crossbar.
      visualizer.add_sphere(
        center=np.array([gx, gy, goal_h]),
        radius=0.12,
        color=(1.0, 1.0, 1.0, 0.9),
        label=f"kicking_goal_centre_{idx}",
      )
      # Posts as vertical "sphere stacks" — DebugVisualizer doesn't expose
      # cuboid; one sphere per post is enough for debugging.
      visualizer.add_sphere(
        center=np.array([gx, gy - half_w, goal_h / 2.0]),
        radius=0.12,
        color=(1.0, 1.0, 1.0, 0.9),
        label=f"kicking_goal_left_{idx}",
      )
      visualizer.add_sphere(
        center=np.array([gx, gy + half_w, goal_h / 2.0]),
        radius=0.12,
        color=(1.0, 1.0, 1.0, 0.9),
        label=f"kicking_goal_right_{idx}",
      )


@dataclass(kw_only=True)
class KickingGoalCommandCfg(DribblingGoalCommandCfg):
  """Configuration for :class:`KickingGoalCommand`."""

  goal_width: float = 3.66
  """Goal width in metres (FIFA 5-a-side standard)."""

  goal_height: float = 2.44
  """Goal height in metres."""

  def build(self, env: ManagerBasedRlEnv) -> KickingGoalCommand:
    return KickingGoalCommand(self, env)
