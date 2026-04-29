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
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.tasks.motion_prior import mdp
from mjlab.tasks.motion_prior.observations_cfg import (
  make_student_height_scan_term,
  make_student_obs_group,
  make_teacher_a_obs_groups,
  make_teacher_b_obs_group,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommandCfg as SingleMotionCommandCfg,
)
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg


def _make_g1_terrain_scan_sensor() -> RayCastSensorCfg:
  """Mirror the rough env's ``terrain_scan`` raycast 1:1 (pelvis-framed).

  Same pattern (16x10 grid, 0.1 spacing), max_distance, alignment, and
  geom filter as ``make_velocity_env_cfg``'s terrain_scan, so flat/rough
  student obs share an identical 160-dim height-scan slice. Frame is
  pinned to pelvis upfront (rough env does this in a post-processing
  loop; flat env owns the only sensor on its scene, so we set it here).
  """
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=True,
  )


def unitree_g1_flat_motion_prior_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat motion-prior env (teacher_a / Teleopit tracking branch).

  Replaces the inherited single-motion ``MotionCommandCfg`` with the
  ``MultiMotionCommandCfg`` so distillation can iterate over a directory
  of motion clips. The tracking task itself stays unchanged — we only
  swap the cfg for *this* env via in-place mutation. CLI override of
  ``--env.commands.motion.motion-path`` lands before the command's
  ``__init__`` runs, so the directory glob happens at the right moment.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)

  # Inject a terrain_scan raycast that matches rough env's spec, so the
  # student obs has an identical height_scan slice on both envs (required
  # for dual-env distillation into a shared latent space). On the flat
  # plane every ray returns the same constant height, but the dimension
  # and code path are the ones we'll use on rough / future deploy.
  existing = tuple(cfg.scene.sensors or ())
  if not any(s.name == "terrain_scan" for s in existing):
    cfg.scene.sensors = existing + (_make_g1_terrain_scan_sensor(),)

  # Swap single-motion -> multi-motion. Carry over every field so the
  # tracking-task defaults (resampling_time_range, pose/velocity ranges,
  # body lists, debug_vis, viz mode) survive the swap.
  old = cfg.commands["motion"]
  assert isinstance(old, SingleMotionCommandCfg)
  cfg.commands["motion"] = MultiMotionCommandCfg(
    resampling_time_range=old.resampling_time_range,
    debug_vis=old.debug_vis,
    entity_name=old.entity_name,
    anchor_body_name=old.anchor_body_name,
    body_names=old.body_names,
    pose_range=old.pose_range,
    velocity_range=old.velocity_range,
    joint_position_range=old.joint_position_range,
    motion_files=[],
    motion_path="",  # CLI injects via --env.commands.motion.motion-path
    motion_type="isaaclab",
    enable_adaptive_sampling=False,
    start_from_zero_step=False,
    if_log_metrics=True,
  )

  enable_corruption = not play
  teacher_a, teacher_a_history = make_teacher_a_obs_groups(
    command_name="motion",
    enable_corruption=enable_corruption,
  )
  cfg.observations = {
    "student": make_student_obs_group(
      enable_corruption=enable_corruption,
      extra_terms={"height_scan": make_student_height_scan_term("terrain_scan")},
    ),
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
    "student": make_student_obs_group(
      enable_corruption=enable_corruption,
      extra_terms={"height_scan": make_student_height_scan_term("terrain_scan")},
    ),
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
