"""Unitree G1 AMP-augmented velocity environment configuration.

Builds on top of the standard G1 rough velocity env by appending an ``amp``
observation group consumed by the AMP discriminator. The actor and critic
observation groups are unchanged; AMP features live in their own group and
are read directly by the AMP runner (not via ``obs_groups`` routing).

Two AMP feature variants are supported (selected via ``amp_variant``):

- ``"body"``  (default): AMP_mjlab-style per-body features in an anchor frame.
  Single-frame (s, s') discriminator input. See ``_build_amp_obs_group_body``.
- ``"joint"``: telebotM2-style joint-space features
  (base_lin/ang_vel + joint_pos/vel). Pairs with a K-frame stack discriminator
  in the runner. See ``_build_amp_obs_group_joint``.

When changing ``amp_variant``, also flip ``rl_cfg.amp["variant"]`` to match;
they must agree because the runner consults the rl-side cfg to pick the
matching ``AMPLoader`` / ``Discriminator`` pair.

Reset behaviour is selectable via the ``use_rsi`` flag on
``g1_amp_velocity_rough_env_cfg``:

- ``use_rsi=False`` (default): falls back to the base velocity env's
  ``reset_root_state_uniform`` + ``reset_joints_by_offset`` (default standing
  pose + tiny perturbation). Motion clips are used only as the expert source
  for the AMP discriminator.
- ``use_rsi=True``: Reference State Initialization. On each reset, a random
  expert motion frame is sampled and its root pose / root velocity / joint
  state are written into the sim, so the policy starts inside the expert
  manifold every episode. See ``_install_rsi_events`` and the
  ``mdp.rsi_events`` module for details. The matching task ID is
  ``Mjlab-AMP-Velocity-Rough-Unitree-G1-RSI``.
"""

from __future__ import annotations

import os
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.amp_velocity.mdp import (
  init_motion_for_rsi,
  reset_from_motion,
  robot_base_ang_vel_b,
  robot_base_lin_vel_b,
  robot_body_ang_vel_b,
  robot_body_lin_vel_b,
  robot_body_ori_b,
  robot_body_pos_b,
  robot_joint_pos,
  robot_joint_vel,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg

# 13 G1 bodies tracked by the AMP discriminator, matching the AMP_mjlab
# reference layout. Order is meaningful: it's hashed into the obs feature
# concatenation, so changing it requires retraining.
# NOTE: ``torso_link`` is intentionally excluded — it is the anchor body, so
# its pos_b would be 0, ori_b would be the identity's first two columns, and
# vel_b would be the anchor's own velocity rotated by itself. Including it
# adds near-constant feature dims that dilute the discriminator's signal.
AMP_TRACKED_BODIES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

# Anchor body: AMP features are expressed in this body's frame.
AMP_ANCHOR_BODY: str = "torso_link"

# G1 has 29 actuated joints. Listed in MuJoCo spec order so even if a future
# user drops preserve_order, env-side and loader-side indexing stay aligned.
AMP_G1_ALL_JOINTS: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# Pelvis body name for the joint variant (expert loader needs it to extract
# base linear/angular velocities from npz body_*_vel_w fields).
AMP_PELVIS_BODY: str = "pelvis"


def _build_amp_obs_group_body() -> ObservationGroupCfg:
  """Per-step body-space AMP feature vector.

  Layout (concatenated, per env):
    body_pos_b   : 3 * N
    body_ori_b   : 6 * N   (rot mat first two columns)
    body_lin_vel : 3 * N   (each body's own local frame)
    body_ang_vel : 3 * N   (each body's own local frame)
  where N = len(AMP_TRACKED_BODIES). Consumed by the (s, s') discriminator.
  """
  anchor_cfg = SceneEntityCfg("robot", body_names=(AMP_ANCHOR_BODY,))
  # preserve_order=True is critical: AMPLoader indexes its motion features in
  # the exact order of AMP_TRACKED_BODIES (via list.index), so the env side
  # must produce body_ids in that same order. Default False would sort by
  # spec order and silently desync env vs. expert features.
  body_cfg = SceneEntityCfg("robot", body_names=AMP_TRACKED_BODIES, preserve_order=True)
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


def _build_amp_obs_group_joint() -> ObservationGroupCfg:
  """Per-step joint-space AMP feature vector (telebotM2-style).

  Layout (concatenated, per env):
    base_lin_vel_b : 3
    base_ang_vel_b : 3
    joint_pos      : J   (absolute values, no default subtraction)
    joint_vel      : J
  where J = len(AMP_G1_ALL_JOINTS) = 29. Per-step dim = 6 + 2J = 64.

  The K-frame stack is built in the runner (not via obs history) so the
  ``ReplayBufferMulti`` + ``DiscriminatorMulti`` path stays self-contained.
  Hence ``history_length=1`` here.
  """
  # preserve_order=True so env-side joint_ids match AMP_G1_ALL_JOINTS exactly,
  # matching AMPJointLoader's expert-side indexing.
  asset_cfg = SceneEntityCfg(
    "robot", joint_names=AMP_G1_ALL_JOINTS, preserve_order=True
  )
  params = {"asset_cfg": asset_cfg}
  return ObservationGroupCfg(
    terms={
      "base_lin_vel_b": ObservationTermCfg(func=robot_base_lin_vel_b, params=params),
      "base_ang_vel_b": ObservationTermCfg(func=robot_base_ang_vel_b, params=params),
      "joint_pos": ObservationTermCfg(func=robot_joint_pos, params=params),
      "joint_vel": ObservationTermCfg(func=robot_joint_vel, params=params),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )


def _resolve_rsi_motion_dir() -> str:
  """Resolve the expert motion directory used for RSI reset.

  Kept inline (instead of imported from ``rl_cfg``) to avoid a circular import:
  ``rl_cfg`` already imports the body / joint name constants from this module.
  Mirrors ``rl_cfg._default_motion_dir`` so both sides resolve to the same
  path by default.
  """
  override = os.environ.get("MJLAB_AMP_MOTION_DIR")
  if override:
    return override
  here = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
    os.path.join(here, "..", "..", "..", "..", "motions", "g1", "amp", "WalkandRun")
  )


def _install_rsi_events(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace the default standing-pose reset with motion-frame RSI.

  Drops the default ``reset_base`` / ``reset_robot_joints`` / ``reset_yaw_*``
  events (so they don't overwrite the RSI state) and installs:

    - ``init_motion_for_rsi`` (startup): preload the motion buffer once.
    - ``reset_from_motion`` (reset): on each reset, sample one expert frame
      per env and write root + joint state from it.
  """
  motion_dir = _resolve_rsi_motion_dir()
  # Strip any prior reset events that would overwrite the RSI root/joint state.
  for k in ("reset_base", "reset_yaw_on_stairs", "reset_robot_joints"):
    cfg.events.pop(k, None)
  cfg.events["init_motion_for_rsi"] = EventTermCfg(
    func=init_motion_for_rsi,
    mode="startup",
    params={"motion_dir": motion_dir},
  )
  cfg.events["reset_base"] = EventTermCfg(
    func=reset_from_motion,
    mode="reset",
    params={
      "motion_dir": motion_dir,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )


def g1_amp_velocity_rough_env_cfg(
  play: bool = False,
  amp_variant: Literal["body", "joint"] = "body",
  use_rsi: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Unitree G1 rough-terrain velocity env augmented with an AMP obs group.

  Identical to ``unitree_g1_rough_env_cfg`` except for the ``amp`` observation
  group; ``amp_variant`` selects between body-space (default, AMP_mjlab-style)
  and joint-space (telebotM2-style) features.

  When ``use_rsi=True``, the default standing-pose reset is replaced with
  Reference State Initialization: on each reset, the robot is placed at a
  random frame from the expert motion clips. This keeps the policy starting
  inside the expert manifold so the discriminator's signal stays non-vacuous.

  When switching ``amp_variant``, the rl-side cfg
  (``rl_cfg.amp["variant"]``) must be flipped to the same value; otherwise the
  runner will instantiate a loader / discriminator that disagrees with what
  the env produces.
  """
  cfg = unitree_g1_rough_env_cfg(play=play)
  if amp_variant == "body":
    cfg.observations["amp"] = _build_amp_obs_group_body()
  elif amp_variant == "joint":
    cfg.observations["amp"] = _build_amp_obs_group_joint()
  else:
    raise ValueError(
      f"Unknown amp_variant {amp_variant!r}; expected 'body' or 'joint'."
    )
  if use_rsi:
    _install_rsi_events(cfg)
  return cfg
