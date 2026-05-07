"""MDP terms for multi-motion tracking.

Re-exports the standard tracking MDP (commands, rewards, terminations,
observations, metrics) and overlays the multi-motion-specific observation
helpers.

The :class:`~mjlab.tasks.tracking.mdp.multi_commands.MultiMotionCommandCfg`
is also re-exported so callers can build the env without reaching across
package boundaries.
"""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403
from mjlab.tasks.tracking.mdp.multi_commands import (
  MultiMotionCommand as MultiMotionCommand,
)
from mjlab.tasks.tracking.mdp.multi_commands import (
  MultiMotionCommandCfg as MultiMotionCommandCfg,
)

from .observations import (
  anchor_height as anchor_height,
)
from .observations import (
  anchor_pos_error as anchor_pos_error,
)
from .observations import (
  motion_anchor_ang_vel as motion_anchor_ang_vel,
)
from .observations import (
  relative_body_orientation_error as relative_body_orientation_error,
)
from .observations import (
  relative_body_pos_error as relative_body_pos_error,
)
