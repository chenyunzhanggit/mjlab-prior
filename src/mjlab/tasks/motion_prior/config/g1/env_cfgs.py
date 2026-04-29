"""Unitree G1 motion-prior environment configurations.

Two distinct envs serve the dual-teacher distillation:

* **flat (teacher_a)** — inherits ``unitree_g1_flat_tracking_env_cfg``;
  motion command active; obs = ``student`` + ``teacher_a`` +
  ``teacher_a_history``. teacher_a is the Teleopit TemporalCNN.
* **rough (teacher_b)** — inherits ``unitree_g1_rough_env_cfg`` (velocity);
  twist command active with terrain raycast; obs = ``student`` +
  ``teacher_b``. teacher_b is the mjlab velocity MLP.

The ``student`` group has the **same schema in both envs** so a single
policy can be evaluated on either. Reward / termination trimming
(prior.md task #4) is applied per env to keep distillation rollouts focused
on the relevant signal.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.motion_prior import mdp
from mjlab.tasks.motion_prior.observations_cfg import (
  make_student_obs_group,
  make_teacher_a_obs_groups,
  make_teacher_b_obs_group,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg


def unitree_g1_flat_motion_prior_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat motion-prior env (teacher_a / Teleopit tracking branch)."""
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)

  enable_corruption = not play
  teacher_a, teacher_a_history = make_teacher_a_obs_groups(
    command_name="motion",
    enable_corruption=enable_corruption,
  )
  cfg.observations = {
    "student": make_student_obs_group(enable_corruption=enable_corruption),
    "teacher_a": teacher_a,
    "teacher_a_history": teacher_a_history,
  }

  # Distillation drops PPO reward shaping; only motion-anchor tracking is
  # retained so episode logs / curriculum hooks have a meaningful signal
  # (loss is computed from teacher actions, not env reward). Mirrors
  # ``g1_motion_prior_cfg.RewardsCfg`` from the isaaclab reference.
  cfg.rewards = {
    "motion_global_anchor_pos": RewardTermCfg(
      func=mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
  }
  return cfg


def unitree_g1_rough_motion_prior_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Rough motion-prior env (teacher_b / velocity branch)."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  enable_corruption = not play
  cfg.observations = {
    "student": make_student_obs_group(enable_corruption=enable_corruption),
    "teacher_b": make_teacher_b_obs_group(
      twist_command_name="twist",
      height_scan_sensor_name="terrain_scan",
      enable_corruption=enable_corruption,
    ),
  }
  return cfg


def unitree_g1_flat_motion_prior_vq_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """VQ variant — same env as VAE flat; algorithm differs at runner level."""
  return unitree_g1_flat_motion_prior_env_cfg(play=play)
