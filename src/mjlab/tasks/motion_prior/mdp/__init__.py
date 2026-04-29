"""MDP terms for motion_prior tasks.

Re-exports the tracking task's MDP terms (commands, observations, rewards,
terminations) since motion_prior shares the motion-driven environment for
the teacher_a (Teleopit tracking) branch. Velocity-task terms are pulled in
on demand by the rough/teacher_b env_cfg.

Adds the reference-velocity obs needed by the Teleopit teacher_a actor
(``ref_base_lin_vel_b``, ``ref_base_ang_vel_b``, ``ref_projected_gravity_b``).
"""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403

from .observations import (
  ref_base_ang_vel_b as ref_base_ang_vel_b,
)
from .observations import (
  ref_base_lin_vel_b as ref_base_lin_vel_b,
)
from .observations import (
  ref_projected_gravity_b as ref_projected_gravity_b,
)
