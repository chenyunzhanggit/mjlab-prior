"""Unitree G1 AMP-augmented velocity environment configuration.

Builds on top of the standard G1 rough velocity env by appending an ``amp``
observation group consumed by the AMP discriminator. The actor and critic
observation groups are unchanged; AMP features live in their own group and
are read directly by the AMP runner (not via ``obs_groups`` routing).

Reset is intentionally **not** RSI'd from motion clips: the env falls back to
the base velocity env's ``reset_root_state_uniform`` + ``reset_joints_by_offset``
(default standing pose + tiny perturbation). Motion clips are used only as
the expert source for the AMP discriminator.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.amp_velocity.mdp import (
  robot_body_ang_vel_b,
  robot_body_lin_vel_b,
  robot_body_ori_b,
  robot_body_pos_b,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

# 14 G1 bodies tracked by the AMP discriminator, matching the AMP_mjlab
# reference layout. Order is meaningful: it's hashed into the obs feature
# concatenation, so changing it requires retraining.
AMP_TRACKED_BODIES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

# Anchor body: AMP features are expressed in this body's frame.
AMP_ANCHOR_BODY: str = "torso_link"


def _build_amp_obs_group() -> ObservationGroupCfg:
  """Per-step AMP feature vector for the simulated robot.

  Layout (concatenated, per env):
    body_pos_b   : 3 * N
    body_ori_b   : 6 * N   (rot mat first two columns)
    body_lin_vel : 3 * N   (each body's own local frame)
    body_ang_vel : 3 * N   (each body's own local frame)
  where N = len(AMP_TRACKED_BODIES).
  """
  anchor_cfg = SceneEntityCfg("robot", body_names=(AMP_ANCHOR_BODY,))
  body_cfg = SceneEntityCfg("robot", body_names=AMP_TRACKED_BODIES)
  params = {"anchor_cfg": anchor_cfg, "body_cfg": body_cfg}
  return ObservationGroupCfg(
    terms={
      "body_pos_b": ObservationTermCfg(func=robot_body_pos_b, params=params),
      "body_ori_b": ObservationTermCfg(func=robot_body_ori_b, params=params),
      "body_lin_vel_b": ObservationTermCfg(func=robot_body_lin_vel_b, params=params),
      "body_ang_vel_b": ObservationTermCfg(func=robot_body_ang_vel_b, params=params),
    },
    concatenate_terms=True,
    enable_corruption=False,  # AMP discriminator sees clean features
    history_length=1,  # discriminator works on (s, s') pairs only
  )


def g1_amp_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Unitree G1 rough-terrain velocity env augmented with an AMP obs group.

  Identical to ``unitree_g1_rough_env_cfg`` except:
    - adds a new ``amp`` observation group (see ``_build_amp_obs_group``)
  Reset, rewards, terminations, curriculum, and command spec are unchanged.
  """
  cfg = unitree_g1_rough_env_cfg(play=play)
  cfg.observations["amp"] = _build_amp_obs_group()
  return cfg
