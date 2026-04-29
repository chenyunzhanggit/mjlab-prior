"""Base motion-prior environment configuration.

The motion_prior task distills frozen TemporalCNN teacher(s) into a VAE /
VQ-VAE student. The environment is structurally identical to the tracking
task (same motion command, same scene, same dynamics) — only the
observation groups differ (student / teacher_a / teacher_b instead of
actor / critic).

This module currently defers to ``make_tracking_env_cfg``; observation
groups are replaced per-robot in ``config/<robot>/env_cfgs.py``.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def make_motion_prior_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base motion-prior environment configuration."""
  return make_tracking_env_cfg()
