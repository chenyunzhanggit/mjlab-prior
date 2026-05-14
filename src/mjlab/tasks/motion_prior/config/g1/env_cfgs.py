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

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ExtrinsicPerturbationCfg,
  GridPatternCfg,
  IntrinsicPerturbationCfg,
  NoisyGroupedRayCasterCameraCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
)
from mjlab.tasks.motion_prior import mdp
from mjlab.tasks.motion_prior.observations_cfg import (
  make_student_depth_obs_group,
  make_student_height_scan_term,
  make_student_obs_group,
  make_teacher_a_obs_groups,
  make_teacher_b_obs_groups,
)
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommandCfg as SingleMotionCommandCfg,
)
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.utils.noise import (
  CropAndResizeCfg,
  DepthDistanceGaussianNoiseCfg,
  DepthDropoutCfg,
  DepthNormalizationCfg,
)


def _make_g1_terrain_scan_sensor() -> RayCastSensorCfg:
  """Mirror the rough env's ``terrain_scan`` raycast 1:1 (pelvis-framed).

  Same pattern (1.6m x 1.0m, 0.1 spacing -> 17x11 grid), max_distance,
  alignment, and geom filter as ``make_velocity_env_cfg``'s terrain_scan,
  so flat/rough student obs share an identical height-scan slice. Frame is
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


def _make_g1_depth_camera_cfg() -> NoisyGroupedRayCasterCameraCfg:
  """Pelvis-mounted pinhole depth camera, pitched ~45 deg down.

  Geometry, noise pipeline, and per-episode extrinsic/intrinsic perturbations
  are copied 1:1 from ``mjlab-loco``'s ``_DEPTH_CAMERA_CFG`` so a policy
  trained here can later be deployed against the same camera spec. Raw
  resolution is 64x64 at fovy=57.9 deg; the noise pipeline crops 2px on each
  side and resizes to 60x60, normalizes to [0, 1], and applies
  distance-dependent Gaussian noise + 1%% pixel dropout (fill_value=-1).
  """
  return NoisyGroupedRayCasterCameraCfg(
    name="camera",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    pattern=PinholeCameraPatternCfg(
      height=64, width=64, fovy=57.9
    ),  # [TODO] fovy pending
    focal_length=1.0,
    horizontal_aperture=2 * math.tan(math.radians(89.04) / 2),  # [TODO] pending
    vertical_aperture=2 * math.tan(math.radians(57.9) / 2),  # [TODO] pending
    data_types=["distance_to_image_plane"],
    ray_alignment="base",
    include_geom_groups=(0, 2),
    min_distance=0.05,
    depth_clipping_behavior="max",
    update_period=1 / 10,
    offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
      pos=(0.0487988662332928, 0.01, 0.4378029937970051),
      rot=(
        0.9135367613482678,
        0.004363309284746571,
        0.4067366430758002,
        0.0,
      ),
      convention="world",
    ),
    noise_pipeline={
      "distance_gaussian": DepthDistanceGaussianNoiseCfg(
        depth_std=0.005, depth_std_multiplier=0.01
      ),
      "normalize": DepthNormalizationCfg(depth_range=(0.0, 2.0), normalize=True),
      "dropout": DepthDropoutCfg(drop_prob=0.01, fill_value=-1.0),
      "crop_resize": CropAndResizeCfg(crop_region=(2, 2, 2, 2), resize_shape=(60, 60)),
    },
    extrinsic_perturbation=ExtrinsicPerturbationCfg(
      pos_range=(0.02, 0.02, 0.02),
      roll_range=0.01745,
      pitch_range=0.08727,
      yaw_range=0.01745,
    ),
    intrinsic_perturbation=IntrinsicPerturbationCfg(
      fov_range=5.0,
      cx_range=1.0,
      cy_range=1.0,
    ),
  )


def unitree_g1_flat_motion_prior_env_cfg(
  play: bool = False, use_depth: bool = True
) -> ManagerBasedRlEnvCfg:
  """Flat motion-prior env (teacher_a / Teleopit tracking branch).

  Replaces the inherited single-motion ``MotionCommandCfg`` with the
  ``MultiMotionCommandCfg`` so distillation can iterate over a directory
  of motion clips. The tracking task itself stays unchanged — we only
  swap the cfg for *this* env via in-place mutation. CLI override of
  ``--env.commands.motion.motion-path`` lands before the command's
  ``__init__`` runs, so the directory glob happens at the right moment.

  ``use_depth=True`` (default) drops the legacy ``height_scan`` from the
  student obs and replaces it with a depth-image ``"depth"`` group fed by
  a pelvis-mounted pinhole camera. On the flat plane every depth pixel is
  the same constant, but the dimension and code path match the rough env
  + future deploy. Setting ``use_depth=False`` falls back to scandot.
  """
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=play)

  existing = tuple(cfg.scene.sensors or ())
  if use_depth:
    # Depth camera replaces the scandot perception input. We *also* register
    # a terrain_scan raycast: teacher_a never reads it, but downstream
    # critics may (privileged scandot in asymmetric actor-critic).
    if not any(s.name == "camera" for s in existing):
      cfg.scene.sensors = existing + (_make_g1_depth_camera_cfg(),)
      existing = cfg.scene.sensors
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
  if use_depth:
    student_extra: dict = {}
  else:
    student_extra = {"height_scan": make_student_height_scan_term("terrain_scan")}
  cfg.observations = {
    "student": make_student_obs_group(
      enable_corruption=enable_corruption,
      extra_terms=student_extra,
    ),
    "teacher_a": teacher_a,
    "teacher_a_history": teacher_a_history,
  }
  if use_depth:
    cfg.observations["depth"] = make_student_depth_obs_group()

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


def unitree_g1_rough_motion_prior_env_cfg(
  play: bool = False, use_depth: bool = True
) -> ManagerBasedRlEnvCfg:
  """Rough motion-prior env (teacher_b / velocity branch).

  ``use_depth=True`` adds a pelvis-mounted depth camera and a ``"depth"``
  obs group; the student's flat ``height_scan`` term is dropped because
  encoder_b / motion_prior / decoder will read depth instead. The
  ``terrain_scan`` raycast stays on the scene because teacher_b still
  needs scandot (its frozen MLP was trained with that 286-dim obs), and
  critics may use scandot as a privileged signal.
  """
  cfg = unitree_g1_rough_env_cfg(play=play)

  if use_depth:
    existing = tuple(cfg.scene.sensors or ())
    if not any(s.name == "camera" for s in existing):
      cfg.scene.sensors = existing + (_make_g1_depth_camera_cfg(),)

  enable_corruption = not play
  if use_depth:
    student_extra: dict = {}
  else:
    student_extra = {"height_scan": make_student_height_scan_term("terrain_scan")}
  teacher_b_prop, teacher_b_height = make_teacher_b_obs_groups(
    twist_command_name="twist",
    height_scan_sensor_name="terrain_scan",
    enable_corruption=enable_corruption,
  )
  cfg.observations = {
    "student": make_student_obs_group(
      enable_corruption=enable_corruption,
      extra_terms=student_extra,
    ),
    "teacher_b": teacher_b_prop,
    "teacher_b_height": teacher_b_height,
  }
  if use_depth:
    cfg.observations["depth"] = make_student_depth_obs_group()
  return cfg


def unitree_g1_flat_motion_prior_vq_env_cfg(
  play: bool = False, use_depth: bool = True
) -> ManagerBasedRlEnvCfg:
  """VQ variant — same env as VAE flat; algorithm differs at runner level."""
  return unitree_g1_flat_motion_prior_env_cfg(play=play, use_depth=use_depth)
