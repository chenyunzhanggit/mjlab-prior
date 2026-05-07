"""Unitree G1 multi-motion tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.multi_motion_tracking.mdp import MultiMotionCommandCfg
from mjlab.tasks.multi_motion_tracking.multi_motion_tracking_env_cfg import (
  make_multi_motion_tracking_env_cfg,
)


def unitree_g1_flat_multi_motion_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat-terrain multi-motion tracking configuration.

  ``motion_path`` (or ``motion_files``) is left empty: the user supplies
  it via the standard CLI plumbing in ``mjlab.scripts.train`` /
  ``mjlab.scripts.play`` (e.g., ``--motion-path /data/Data10k``; the
  long-form ``--env.commands.motion.motion-path`` is also accepted).
  The directory glob runs lazily inside
  :class:`MultiMotionCommand.__init__`, so CLI overrides land before
  the filesystem is touched.
  """
  cfg = make_multi_motion_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Self-collision sensor used by the ``self_collisions`` reward — same
  # spec as the single-motion tracking task so the cost is comparable.
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
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
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

  if play:
    # Effectively infinite episode length so the user can scrub a clip.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization on play.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.start_from_zero_step = True

  return cfg
