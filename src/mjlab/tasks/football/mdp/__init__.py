"""MDP terms for football downstream tasks.

Self-contained (does not re-export ``tracking.mdp``) because the football
env doesn't run any motion-tracking command — just velocity-style reward
shaping driven by ball / goal state.
"""

from .commands_dribbling import (
  DribblingGoalCommand,
  DribblingGoalCommandCfg,
)
from .commands_kicking import (
  KickingGoalCommand,
  KickingGoalCommandCfg,
)
from .commands_passing import (
  PassingCommand,
  PassingCommandCfg,
)
from .events import (
  reset_ball_along_line_dribbling,
  reset_ball_along_line_kicking,
  reset_ball_along_line_passing,
)
from .observations import (
  ball_absolute_position,
  ball_relative_position,
  ball_to_goal_vector,
  ball_velocity,
  depth_image,
  dribbling_goal_position,
  passing_source_position,
)
from .rewards import (
  align_to_kick_reward,
  approach_ball_reward,
  ball_distance_penalty,
  ball_impact_reward,
  ball_reach_goal_reward,
  ball_scored_goal_reward,
  ball_to_goal_progress,
  ball_too_far_penalty,
  foot_ball_contact_reward,
  foot_ball_proximity_reward,
  kick_ball_toward_goal,
)
from .terminations import (
  ball_no_progress_timeout,
  ball_passed_through_zone,
  ball_reach_goal_termination,
  ball_stopped_after_kick,
  ball_too_far_termination,
)

__all__ = [
  "DribblingGoalCommand",
  "DribblingGoalCommandCfg",
  "KickingGoalCommand",
  "KickingGoalCommandCfg",
  "PassingCommand",
  "PassingCommandCfg",
  "reset_ball_along_line_dribbling",
  "reset_ball_along_line_kicking",
  "reset_ball_along_line_passing",
  "ball_absolute_position",
  "ball_relative_position",
  "ball_to_goal_vector",
  "ball_velocity",
  "depth_image",
  "dribbling_goal_position",
  "passing_source_position",
  "align_to_kick_reward",
  "approach_ball_reward",
  "ball_distance_penalty",
  "ball_impact_reward",
  "ball_reach_goal_reward",
  "ball_scored_goal_reward",
  "ball_to_goal_progress",
  "ball_too_far_penalty",
  "foot_ball_contact_reward",
  "foot_ball_proximity_reward",
  "kick_ball_toward_goal",
  "ball_no_progress_timeout",
  "ball_passed_through_zone",
  "ball_reach_goal_termination",
  "ball_stopped_after_kick",
  "ball_too_far_termination",
]
