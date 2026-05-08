"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg as SingleMotionCommandCfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def _unitree_g1_flat_tracking_env_cfg(
  motion_command_cfg_cls,
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration.

  Shared factory used by both the single-motion and multi-motion (bfm)
  variants. ``motion_command_cfg_cls`` selects which ``MotionCommandCfg``
  class is instantiated by ``make_tracking_env_cfg``.
  """
  cfg = make_tracking_env_cfg(motion_command_cfg_cls=motion_command_cfg_cls)

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, (SingleMotionCommandCfg, MultiMotionCommandCfg))
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
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

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    # Multi-motion command exposes a "start" sampling mode (frame-0 RSI).
    # Single-motion command has no such field, so guard the assignment.
    if hasattr(motion_cmd, "sampling_mode"):
      motion_cmd.sampling_mode = "start"

    cfg.terminations.pop("anchor_ori", None)
    cfg.terminations.pop("anchor_pos", None)
    cfg.terminations.pop("ee_body_pos", None)
  return cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the single-motion Unitree G1 flat terrain tracking configuration."""
  return _unitree_g1_flat_tracking_env_cfg(
    SingleMotionCommandCfg,
    has_state_estimation=has_state_estimation,
    play=play,
  )


def unitree_g1_flat_tracking_bfm_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the multi-motion Unitree G1 flat terrain tracking configuration."""
  return _unitree_g1_flat_tracking_env_cfg(
    MultiMotionCommandCfg,
    has_state_estimation=has_state_estimation,
    play=play,
  )
