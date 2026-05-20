"""Unitree G1 football env configurations (dribbling / kicking / passing).

Each task inherits ``unitree_g1_flat_env_cfg`` (the same plane-terrain
G1 cfg used by downstream velocity tracking), then surgically:

  * Injects the soccer ball entity + ball contact sensor.
  * Replaces ``commands`` with the task-specific goal command (no twist).
  * Replaces ``observations`` with the 3-group downstream schema
    (motion_prior_obs / policy / critic). The motion_prior_obs group
    matches the trained VQ motion-prior backbone (with height_scan, 559-dim
    by default).
  * Replaces ``rewards`` / ``terminations`` with the per-task minimal set.
  * Drops the standard ``reset_base`` and adds a ``reset_ball_along_line``
    event that places robot + ball + goal along a random direction.
  * Increases ``env_spacing`` so the long-range dribbling / kicking layouts
    don't collide across envs.

The terrain_scan raycast (height_scan) is re-injected for parity with
the downstream velocity env so the frozen prior's expected
``motion_prior_obs`` dim (559) is preserved. Pass
``with_height_scan=False`` via the helper functions when re-training
without height_scan.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ExtrinsicPerturbationCfg,
  GridPatternCfg,
  IntrinsicPerturbationCfg,
  NoisyGroupedRayCasterCameraCfg,
  ObjRef,
  PinholeCameraPatternCfg,
  RayCastSensorCfg,
)
from mjlab.utils.noise import (
  CropAndResizeCfg,
  DepthDistanceGaussianNoiseCfg,
  DepthDropoutCfg,
  DepthNormalizationCfg,
  RectMaskCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.football import mdp as football_mdp
from mjlab.tasks.football.soccer_ball import (
  SOCCER_BALL_RADIUS,
  SoccerBallParams,
  soccer_ball_entity_cfg,
)
from mjlab.tasks.motion_prior.config.g1.downstream_env_cfgs import (
  _make_motion_prior_obs_group,
  _make_proprio_terms,
)
from mjlab.tasks.motion_prior.config.g1.env_cfgs import _make_g1_terrain_scan_sensor
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

# Soccer ball entity name (used by the ball entity, contact sensor secondary
# pattern, and every MDP term that reads the ball).
_BALL_ENTITY = "soccer_ball"
_BALL_CONTACT_SENSOR = "soccer_ball_contact"
_BALL_FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


def _make_ball_contact_sensor_cfg(robot_foot_geoms: str) -> ContactSensorCfg:
  """Net force on the ball from foot contacts.

  Configured with ``reduce="netforce"`` so ``ContactData.force`` is in the
  global frame and we can read a single (xy, z) wrench per env in
  ``foot_ball_contact_reward``.
  """
  return ContactSensorCfg(
    name=_BALL_CONTACT_SENSOR,
    primary=ContactMatch(
      mode="body",
      pattern="ball",
      entity=_BALL_ENTITY,
    ),
    secondary=ContactMatch(
      mode="geom",
      pattern=robot_foot_geoms,
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def _make_football_obs_groups(
  *,
  policy_extra_terms: dict[str, ObservationTermCfg],
  critic_extra_terms: dict[str, ObservationTermCfg],
  enable_corruption: bool,
  with_height_scan: bool = False,
) -> dict[str, ObservationGroupCfg]:
  """Build motion_prior_obs / policy / critic groups for one football task.

  * ``motion_prior_obs``: shared proprio (+ optional height_scan) — must
    match the frozen VQ motion-prior backbone's input dim.
  * ``policy``: task-specific terms first, then proprio history.
  * ``critic``: same as policy + privileged ``base_lin_vel`` + the caller's
    privileged extras (e.g. ball_absolute_position).

  All three groups carry ``nan_policy="warn"`` so that any degenerate
  obs term (zero-quat at the very first reset, raycast miss on a freshly
  spawned env, etc.) is logged once and replaced with zeros instead of
  crashing the downstream PPO with a ``normal expects std >= 0`` error.
  """
  motion_prior_obs = _make_motion_prior_obs_group(
    enable_corruption=enable_corruption,
    with_height_scan=with_height_scan,
  )
  # ``nan_policy`` is baked into the helper now; football's policy / critic
  # groups (built inline below) set it explicitly.

  policy_terms: dict[str, ObservationTermCfg] = {
    **policy_extra_terms,
    **_make_proprio_terms(),
  }
  policy = ObservationGroupCfg(
    terms=policy_terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  critic_terms: dict[str, ObservationTermCfg] = {
    **critic_extra_terms,
    **_make_proprio_terms(),
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }
  critic = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  return {
    "motion_prior_obs": motion_prior_obs,
    "policy": policy,
    "critic": critic,
  }


def _strip_velocity_extras(cfg: ManagerBasedRlEnvCfg) -> None:
  """Remove velocity-specific commands / events / curriculum.

  The football reward / termination sets replace all velocity-task
  rewards, but the velocity events (push_robot, foot_friction DR,
  base_com DR, encoder_bias DR) and the twist command would still run
  unless dropped — push_robot would knock the robot off the ball mid-
  episode, so we drop it; the DR events are harmless on a plane and we
  KEEP them so domain randomization parity with downstream velocity
  carries over.
  """
  cfg.commands = {}
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}


def _inject_terrain_scan(cfg: ManagerBasedRlEnvCfg) -> None:
  """Re-add the pelvis-framed terrain_scan raycast that ``unitree_g1_flat_env_cfg``
  strips (the motion-prior-obs height_scan term needs it)."""
  if not any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ())):
    cfg.scene.sensors = (*(cfg.scene.sensors or ()), _make_g1_terrain_scan_sensor())


def _add_soccer_ball(
  cfg: ManagerBasedRlEnvCfg,
  params: SoccerBallParams,
) -> None:
  """Add the soccer ball entity and its contact sensor to the scene.

  Damping is intentionally NOT applied at present:

  * Setting ``MjsJoint.damping`` on a freejoint at MjSpec build time hits
    a version-dependent shape contract in the mujoco python binding
    (some builds want an ``ndarray[(3, 1)]``, others a scalar), and
  * ``dr.dof_damping`` cannot target freejoints because mjlab's
    ``Entity.joint_names`` deliberately excludes the freejoint, so a
    ``joint_names=".*"`` ``SceneEntityCfg`` resolves to an empty list.

  Net effect: ``params.linear_damping`` / ``params.angular_damping`` are
  accepted for API parity but currently dropped. Ball deceleration is
  driven by ground friction + rolling resistance only. If the resulting
  ball behavior is too "slippery", we'll add a custom event function
  that writes ``mj_model.dof_damping`` at the freejoint's DOF addresses
  directly (the standard mjlab DR helpers can't express this today).
  ``params.radius`` / ``params.mass`` / ``params.rgba`` are still used
  for ball geometry / appearance.
  """
  cfg.scene.entities = {
    **cfg.scene.entities,
    _BALL_ENTITY: soccer_ball_entity_cfg(params),
  }
  # Reuse the foot geom regex from the flat-velocity env's foot_friction DR.
  robot_foot_geoms = r"(left|right)_foot[1-7]_collision"
  cfg.scene.sensors = (
    *(cfg.scene.sensors or ()),
    _make_ball_contact_sensor_cfg(robot_foot_geoms),
  )


# ---------------------------------------------------------------------------
# Dribbling
# ---------------------------------------------------------------------------


def unitree_g1_dribbling_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  _strip_velocity_extras(cfg)
  # ---- terrain_scan injection DISABLED (user request) ----
  # Re-enable together with ``with_height_scan=True`` on
  # ``_make_football_obs_groups`` to restore the 187-dim height_scan
  # slice in motion_prior_obs (needed for the legacy 559-dim ckpts).
  # _inject_terrain_scan(cfg)

  cfg.scene.env_spacing = 5.0
  cfg.episode_length_s = 50.0

  _add_soccer_ball(
    cfg,
    SoccerBallParams(linear_damping=0.8, angular_damping=0.0),
  )

  cfg.commands = {
    "dribbling_commands": football_mdp.DribblingGoalCommandCfg(
      resampling_time_range=(50.0, 50.0),
      debug_vis=True,
      asset_name="robot",
      ball_name=_BALL_ENTITY,
      goal_ranges=football_mdp.DribblingGoalCommandCfg.Ranges(
        x=(10.0, 20.0),
        y=(0.0, 0.0),
        min_distance=10.0,
      ),
    ),
  }

  enable_corruption = not play

  cfg.observations = _make_football_obs_groups(
    policy_extra_terms={
      "goal_position": ObservationTermCfg(
        func=football_mdp.dribbling_goal_position,
        params={"command_name": "dribbling_commands"},
      ),
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "ball_to_goal_vector": ObservationTermCfg(
        func=football_mdp.ball_to_goal_vector,
        params={"ball_name": _BALL_ENTITY, "command_name": "dribbling_commands"},
      ),
    },
    critic_extra_terms={
      "goal_position": ObservationTermCfg(
        func=football_mdp.dribbling_goal_position,
        params={"command_name": "dribbling_commands"},
      ),
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "ball_to_goal_vector": ObservationTermCfg(
        func=football_mdp.ball_to_goal_vector,
        params={"ball_name": _BALL_ENTITY, "command_name": "dribbling_commands"},
      ),
      "ball_absolute_position": ObservationTermCfg(
        func=football_mdp.ball_absolute_position,
        params={"ball_name": _BALL_ENTITY},
      ),
    },
    enable_corruption=enable_corruption,
  )

  cfg.rewards = {
    "ball_to_goal_progress": RewardTermCfg(
      func=football_mdp.ball_to_goal_progress,
      weight=20.0,
      params={"ball_name": _BALL_ENTITY, "command_name": "dribbling_commands"},
    ),
    "ball_distance_penalty": RewardTermCfg(
      func=football_mdp.ball_distance_penalty,
      weight=-3.0,
      params={
        "ball_name": _BALL_ENTITY,
        "max_distance": 1.5,
        "asset_name": "robot",
      },
    ),
    "approach_ball": RewardTermCfg(
      func=football_mdp.approach_ball_reward,
      weight=1.0,
      params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
    ),
    "foot_ball_contact": RewardTermCfg(
      func=football_mdp.foot_ball_contact_reward,
      weight=10.0,
      params={"ball_sensor_cfg": SceneEntityCfg(_BALL_CONTACT_SENSOR)},
    ),
    "ball_impact": RewardTermCfg(
      func=football_mdp.ball_impact_reward,
      weight=1.5,
      params={"ball_name": _BALL_ENTITY, "velocity_change_threshold": 0.3},
    ),
    "foot_ball_proximity": RewardTermCfg(
      func=football_mdp.foot_ball_proximity_reward,
      weight=2.0,
      params={
        "ball_name": _BALL_ENTITY,
        "foot_body_names": _BALL_FOOT_NAMES,
        "contact_distance": 0.15,
      },
    ),
    "reach_goal": RewardTermCfg(
      func=football_mdp.ball_reach_goal_reward,
      weight=10.0,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "dribbling_commands",
        "threshold": 0.3,
      },
    ),
    "joint_limit": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "ball_too_far": RewardTermCfg(
      func=football_mdp.ball_too_far_penalty,
      weight=-5.0,
      params={
        "ball_name": _BALL_ENTITY,
        "asset_name": "robot",
        "critical_distance": 1.5 * 1.5,
      },
    ),
  }

  cfg.terminations = {
    # Reset any env whose physics state goes NaN/Inf (e.g. due to a
    # pathological ball↔robot contact). Without this, ``bad_orientation``
    # sees NaN and returns False (NaN comparisons are False), and the env
    # stays a zombie until time_out — polluting the rollout with NaNs.
    # Must be first so the reset happens before reward / metric collection
    # reads NaN state.
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    ),
    "no_progress_timeout": TerminationTermCfg(
      func=football_mdp.ball_no_progress_timeout,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "dribbling_commands",
        "timeout_steps": 100,
        "min_progress": 0.5,
      },
    ),
    "ball_too_far": TerminationTermCfg(
      func=football_mdp.ball_too_far_termination,
      params={
        "ball_name": _BALL_ENTITY,
        "asset_name": "robot",
        "max_distance": 1.5 * 2.0,
      },
    ),
    "reach_goal": TerminationTermCfg(
      func=football_mdp.ball_reach_goal_termination,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "dribbling_commands",
        "threshold": 0.3,
      },
    ),
  }

  # Drop the velocity reset (we own the robot's reset pose), then add ours.
  cfg.events.pop("reset_base", None)
  cfg.events["reset_ball_along_line"] = EventTermCfg(
    func=football_mdp.reset_ball_along_line_dribbling,
    mode="reset",
    params={
      "ball_name": _BALL_ENTITY,
      "command_name": "dribbling_commands",
      "ball_radius": SOCCER_BALL_RADIUS,
    },
  )

  return cfg


# ---------------------------------------------------------------------------
# Kicking
# ---------------------------------------------------------------------------


def unitree_g1_kicking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  _strip_velocity_extras(cfg)
  # ---- terrain_scan injection DISABLED (user request) ----
  # Re-enable together with ``with_height_scan=True`` on
  # ``_make_football_obs_groups`` to restore the 187-dim height_scan
  # slice in motion_prior_obs (needed for the legacy 559-dim ckpts).
  # _inject_terrain_scan(cfg)

  cfg.scene.env_spacing = 6.0
  cfg.episode_length_s = 20.0

  _add_soccer_ball(
    cfg,
    SoccerBallParams(linear_damping=0.4, angular_damping=0.2),
  )

  cfg.commands = {
    "kicking_commands": football_mdp.KickingGoalCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      asset_name="robot",
      ball_name=_BALL_ENTITY,
      goal_width=3.66,
      goal_height=2.44,
      goal_ranges=football_mdp.KickingGoalCommandCfg.Ranges(
        x=(6.0, 12.0),
        y=(-1.5, 1.5),
        min_distance=6.0,
      ),
    ),
  }

  enable_corruption = not play

  cfg.observations = _make_football_obs_groups(
    policy_extra_terms={
      "goal_position": ObservationTermCfg(
        func=football_mdp.dribbling_goal_position,
        params={"command_name": "kicking_commands"},
      ),
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "ball_to_goal_vector": ObservationTermCfg(
        func=football_mdp.ball_to_goal_vector,
        params={"ball_name": _BALL_ENTITY, "command_name": "kicking_commands"},
      ),
    },
    critic_extra_terms={
      "goal_position": ObservationTermCfg(
        func=football_mdp.dribbling_goal_position,
        params={"command_name": "kicking_commands"},
      ),
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "ball_to_goal_vector": ObservationTermCfg(
        func=football_mdp.ball_to_goal_vector,
        params={"ball_name": _BALL_ENTITY, "command_name": "kicking_commands"},
      ),
      "ball_absolute_position": ObservationTermCfg(
        func=football_mdp.ball_absolute_position,
        params={"ball_name": _BALL_ENTITY},
      ),
    },
    enable_corruption=enable_corruption,
  )

  cfg.rewards = {
    "approach_ball": RewardTermCfg(
      func=football_mdp.approach_ball_reward,
      weight=1.5,
      params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
    ),
    "align_to_kick": RewardTermCfg(
      func=football_mdp.align_to_kick_reward,
      weight=5.0,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "kicking_commands",
        "asset_name": "robot",
        "kick_distance": 0.45,
      },
    ),
    "foot_ball_proximity": RewardTermCfg(
      func=football_mdp.foot_ball_proximity_reward,
      weight=1.0,
      params={
        "ball_name": _BALL_ENTITY,
        "foot_body_names": _BALL_FOOT_NAMES,
        "contact_distance": 0.15,
      },
    ),
    "kick_toward_goal": RewardTermCfg(
      func=football_mdp.kick_ball_toward_goal,
      weight=12.0,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "kicking_commands",
        "speed_threshold": 1.5,
      },
    ),
    "foot_ball_contact": RewardTermCfg(
      func=football_mdp.foot_ball_contact_reward,
      weight=5.0,
      params={"ball_sensor_cfg": SceneEntityCfg(_BALL_CONTACT_SENSOR)},
    ),
    "ball_scored_goal": RewardTermCfg(
      func=football_mdp.ball_scored_goal_reward,
      weight=50.0,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "kicking_commands",
        "goal_width": 3.66,
        "goal_height": 2.44,
        "ball_radius": SOCCER_BALL_RADIUS,
        "bonus": 1.0,
      },
    ),
    "ball_to_goal_dist": RewardTermCfg(
      func=football_mdp.ball_reach_goal_reward,
      weight=8.0,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "kicking_commands",
        "threshold": 1.0,
        "sigma": 4.0,
        "bonus_scale": 0.0,
      },
    ),
    "joint_limit": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Reference parity: ``undesired_contacts`` (-0.1) in motionprior's
    # ``G1KickingRewardsCfg``. Penalises self-contacts on the robot so it
    # doesn't learn to dribble/balance with non-foot body parts. Mirrors
    # the downstream-velocity env (which uses the same sensor + weight).
    "undesired_contacts": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision", "force_threshold": 1.0},
    ),
  }

  cfg.terminations = {
    # NaN sentinel — see dribbling_env_cfg for rationale.
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    ),
    "ball_stopped": TerminationTermCfg(
      func=football_mdp.ball_stopped_after_kick,
      params={
        "ball_name": _BALL_ENTITY,
        "stop_speed": 0.1,
        "stop_steps": 30,
        "min_kick_speed": 1.5,
      },
    ),
  }

  cfg.events.pop("reset_base", None)
  cfg.events["reset_ball_along_line"] = EventTermCfg(
    func=football_mdp.reset_ball_along_line_kicking,
    mode="reset",
    params={
      "ball_name": _BALL_ENTITY,
      "command_name": "kicking_commands",
      "ball_radius": SOCCER_BALL_RADIUS,
    },
  )

  # Reference parity: original motionprior ``g1_kicking_vq_cfg`` adds a
  # ``motion`` command with ``joint_position_range=(-0.05, 0.05)`` whose
  # only function on this task is RSI joint noise (init_from_motion=False,
  # so the motion clip is never read for the robot's joints). We achieve
  # the same effect by overriding the inherited velocity-env
  # ``reset_robot_joints`` event ranges (it defaults to (0.0, 0.0), which
  # gives **no** initial randomization and is the most likely cause of
  # the policy collapsing into "circle around the ball, never strike").
  # ``velocity_range=(-1.0, 1.0)`` matches the unitree_rl_lab joint
  # velocity RSI noise we use on the velocity-tracking downstream task.
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.05, 0.05)
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)

  return cfg


# ---------------------------------------------------------------------------
# Passing
# ---------------------------------------------------------------------------


def unitree_g1_passing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  _strip_velocity_extras(cfg)
  # ---- terrain_scan injection DISABLED (user request) ----
  # Re-enable together with ``with_height_scan=True`` on
  # ``_make_football_obs_groups`` to restore the 187-dim height_scan
  # slice in motion_prior_obs (needed for the legacy 559-dim ckpts).
  # _inject_terrain_scan(cfg)

  cfg.scene.env_spacing = 10.0
  cfg.episode_length_s = 20.0

  _add_soccer_ball(
    cfg,
    SoccerBallParams(linear_damping=0.3, angular_damping=0.2),
  )

  cfg.commands = {
    "passing_commands": football_mdp.PassingCommandCfg(
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      asset_name="robot",
      ball_name=_BALL_ENTITY,
      zone_radius=1.5,
      source_distance_range=(3.0, 6.0),
      source_lateral_range=(-0.3, 0.3),
      ball_speed_range=(5.0, 9.0),
    ),
  }

  enable_corruption = not play

  cfg.observations = _make_football_obs_groups(
    policy_extra_terms={
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "passing_source_position": ObservationTermCfg(
        func=football_mdp.passing_source_position,
        params={"command_name": "passing_commands"},
      ),
    },
    critic_extra_terms={
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "passing_source_position": ObservationTermCfg(
        func=football_mdp.passing_source_position,
        params={"command_name": "passing_commands"},
      ),
      "ball_absolute_position": ObservationTermCfg(
        func=football_mdp.ball_absolute_position,
        params={"ball_name": _BALL_ENTITY},
      ),
    },
    enable_corruption=enable_corruption,
  )

  cfg.rewards = {
    "foot_ball_proximity": RewardTermCfg(
      func=football_mdp.foot_ball_proximity_reward,
      weight=2.0,
      params={
        "ball_name": _BALL_ENTITY,
        "foot_body_names": _BALL_FOOT_NAMES,
        "contact_distance": 0.15,
      },
    ),
    "foot_ball_contact": RewardTermCfg(
      func=football_mdp.foot_ball_contact_reward,
      weight=6.0,
      params={"ball_sensor_cfg": SceneEntityCfg(_BALL_CONTACT_SENSOR)},
    ),
    "kick_toward_source": RewardTermCfg(
      func=football_mdp.kick_ball_toward_goal,
      weight=20.0,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "passing_commands",
        "speed_threshold": 1.5,
      },
    ),
    "ball_to_source_dist": RewardTermCfg(
      func=football_mdp.ball_reach_goal_reward,
      weight=3.0,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "passing_commands",
        "threshold": 1.5,
        "sigma": 5.0,
        "bonus_scale": 0.0,
      },
    ),
    "joint_limit": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  cfg.terminations = {
    # NaN sentinel — see dribbling_env_cfg for rationale. Passing is
    # the task most prone to NaN because the ball arrives at the robot
    # with 5–9 m/s and the impact can produce singular contact configs.
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    ),
    "ball_in_zone": TerminationTermCfg(
      func=football_mdp.ball_passed_through_zone,
      params={
        "ball_name": _BALL_ENTITY,
        "command_name": "passing_commands",
        "zone_radius": 1.5,
        "min_redirect_speed": 1.5,
      },
    ),
  }

  cfg.events.pop("reset_base", None)
  cfg.events["reset_ball_along_line"] = EventTermCfg(
    func=football_mdp.reset_ball_along_line_passing,
    mode="reset",
    params={
      "ball_name": _BALL_ENTITY,
      "command_name": "passing_commands",
      "ball_radius": SOCCER_BALL_RADIUS,
    },
  )

  return cfg


# ---------------------------------------------------------------------------
# Passing — Perception-only variant (mjlab-prior-main pipeline, D405 + 16:9)
# ---------------------------------------------------------------------------
#
# Sensor parameters calibrated against a real **Intel RealSense D405**
# (user's actual device, see uploaded ``d435 param.md`` despite filename):
#
#   Depth HFoV × VFoV  =  89.04° × 57.9°     (matches 480×270 / 640×360
#                                              / 1280×720 D405 depth modes)
#   Output aspect      =  16:9                (45×80 → 80/45 = 1.778)
#   Native frame rate  =  30 fps @ 720p       (we use 1/30s update_period)
#
# Architecture follows ``mjlab-prior-main/.../config/g1/env_cfgs.py`` and
# VisualMimic (Yin & Ze, 2025) §III-D-a:
#
#   Raw raycast (49×84) ──► NoisyCamera noise pipeline ──► (45×80) ──► policy
#                                         │
#                              distance_gaussian (σ = 5mm + 1.5%·d, fits D405 z-error)
#                              depth_normalize    (clip to 3m, scale to [0,1])
#                              dropout            (2% per-pixel, fill = -1)
#                              crop_resize        (crop 2px each side → 45×80, 16:9)
#                              rect_mask          (VisualMimic occluder)
#
# Per-episode extrinsic perturbation: pos ±3cm, roll/yaw ±1°, pitch ±5°.
# Per-episode intrinsic perturbation: fov ±5°, cx/cy ±1 px.
# History buffer: 4 frames at update_period = 1/30s (30 Hz, D405 native).

PERCEPTION_SENSOR_NAME = "camera"
"""Sensor name (must match the ``sensor_cfg`` passed to obs term)."""

PERCEPTION_RAW_HEIGHT = 49
PERCEPTION_RAW_WIDTH = 84
"""Raw pinhole pattern resolution (before crop+resize).
Picked so that ``crop_region=(2, 2, 2, 2)`` lands exactly at 45×80 (16:9)."""

PERCEPTION_IMAGE_HEIGHT = 45
PERCEPTION_IMAGE_WIDTH = 80
"""Resolved depth image shape after crop+resize pipeline.
``80 / 45 = 1.778 ≈ 16/9 = 1.778`` (matches D405 480×270 / 640×360 modes
which are also 16:9, and standard HDMI / wide-camera framing)."""

PERCEPTION_HISTORY_LEN = 4
"""History frames kept in the sensor's :class:`AsyncCircularBuffer`.
4 frames @ 30 Hz = 133 ms time window for ball-velocity inference."""

_PERCEPTION_FOVY = 57.9
"""Vertical FoV (deg). D405 measured value for 480×270 / 640×360 / 1280×720."""

_PERCEPTION_HFOV = 89.04
"""Horizontal FoV (deg). D405 measured value, matches the same modes as VFoV."""
_PERCEPTION_DEPTH_RANGE = (0.0, 3.0)
"""Depth clipping range (metres) before normalisation.
D405 nominal short-range is 0.07–0.5 m but it still returns usable
depth up to ~3 m with increased noise. Passing balls spawn at 3–6 m,
so we let the policy see up to 3 m and saturate (=1.0) beyond — the
``ball_to_source_dist`` reward + proprio history teach the policy to
move toward unseen balls.
Real-machine deploy will see the same saturation, sim2real consistent."""


def _make_g1_depth_camera_cfg() -> NoisyGroupedRayCasterCameraCfg:
  """Pelvis-mounted forward pinhole depth camera with VisualMimic-style
  sim2real augmentation.

  1:1 of mjlab-prior-main's ``_make_g1_depth_camera_cfg`` (originally from
  ``mjlab-loco``). The ``OffsetCfg.convention="world"`` puts forward=+X /
  up=+Z, then the ``rot=(w, x, y, z) = (0.914, 0.004, 0.407, 0)``
  quaternion adds ~45° pitch down so the camera sees both the ground and
  incoming ball.
  """
  return NoisyGroupedRayCasterCameraCfg(
    name=PERCEPTION_SENSOR_NAME,
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    # debug_vis=True makes viser auto-add a "Sensor debug viz → camera"
    # checkbox; GroupedRayCasterCamera.debug_vis then draws up to 512 red
    # ray-hit spheres + the camera coordinate frame each tick. Negligible
    # cost in play; ignored in headless training.
    debug_vis=True,
    pattern=PinholeCameraPatternCfg(
      height=PERCEPTION_RAW_HEIGHT,
      width=PERCEPTION_RAW_WIDTH,
      fovy=_PERCEPTION_FOVY,
    ),
    focal_length=1.0,
    # Apertures come straight from the D405 measured FoVs in
    # ``d435 param.md`` (HFoV=89.04°, VFoV=57.9°). With focal_length=1
    # these reduce to ``2*tan(FoV/2)``.
    horizontal_aperture=2 * math.tan(math.radians(_PERCEPTION_HFOV) / 2),
    vertical_aperture=2 * math.tan(math.radians(_PERCEPTION_FOVY) / 2),
    data_types=["distance_to_image_plane"],
    ray_alignment="base",
    include_geom_groups=(0, 2),  # 0=terrain default, 2=soccer ball geom
    min_distance=0.05,           # D405 datasheet MinZ ≈ 7 cm; we go 5 cm
                                  # to keep balls visible right at the foot.
    depth_clipping_behavior="max",
    update_period=1 / 30,        # 30 Hz, D405 native frame rate @ 720p / 480p
    offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
      pos=(0.0487988662332928, 0.01, 0.4378029937970051),  # chest height,
                                                            # ~5cm front, 1cm
                                                            # right, 44cm up
      rot=(
        0.9135367613482678,
        0.004363309284746571,
        0.4067366430758002,
        0.0,
      ),                          # ~45° pitch down — sees ground + incoming ball
      convention="world",         # forward=+X, up=+Z
    ),
    noise_pipeline={
      # Distance-dependent Gaussian. D405 stereo z-accuracy ≈ 0.5% × d² for
      # short range; in 0–3m a linear ``5mm + 1.5%·d`` approximation is
      # close enough.
      "distance_gaussian": DepthDistanceGaussianNoiseCfg(
        depth_std=0.005, depth_std_multiplier=0.015,
      ),
      # D405 effective range nominally 0.07–0.5 m but extends to 1.5–3 m
      # with noise. Pick 3 m so the ball (3–6 m at episode start) stays
      # visible long enough for the policy to learn the approach.
      "normalize": DepthNormalizationCfg(
        depth_range=_PERCEPTION_DEPTH_RANGE, normalize=True,
      ),
      # D405 typical invalid-pixel ratio: 1–3% in normal lighting.
      "dropout": DepthDropoutCfg(drop_prob=0.02, fill_value=-1.0),
      # Crop 2 px on each edge to drop the rectified-stereo border zone,
      # then resize is a no-op (49−4=45, 84−4=80).
      "crop_resize": CropAndResizeCfg(
        crop_region=(2, 2, 2, 2),
        resize_shape=(PERCEPTION_IMAGE_HEIGHT, PERCEPTION_IMAGE_WIDTH),
      ),
      # D2=b: VisualMimic-style random-rect mask as the LAST pipeline step.
      "rect_mask": RectMaskCfg(
        max_value=1.0,
        max_rects=6,
        prob_per_slot=0.10,
        max_h_frac=0.30,
        max_w_frac=0.30,
        bottom_left_prob=0.20,
        bottom_left_h_frac=0.40,
        bottom_left_w_frac=0.30,
      ),
    },
    # Real neck-mounted RealSense drifts ±5° pitch over a day of use;
    # roll/yaw and position drift are smaller because the mount is rigid
    # horizontally. Values copied from mjlab-prior-main / mjlab-loco.
    extrinsic_perturbation=ExtrinsicPerturbationCfg(
      pos_range=(0.03, 0.03, 0.03),
      roll_range=0.01745,   # ±1°
      pitch_range=0.08727,  # ±5°
      yaw_range=0.01745,    # ±1°
    ),
    intrinsic_perturbation=IntrinsicPerturbationCfg(
      fov_range=5.0,        # factory calibration drift ±5°
      cx_range=1.0,         # ±1 px principal-point drift
      cy_range=1.0,
    ),
    data_histories={"distance_to_image_plane_noised": PERCEPTION_HISTORY_LEN},
  )


def unitree_g1_passing_perception_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Perception-only passing task (VisualMimic-style depth-CNN pipeline).

  Same physics / rewards / terminations / reset events as
  :func:`unitree_g1_passing_env_cfg`, but the policy can no longer read
  ball state directly. Instead it sees:

  * ``passing_source_position`` — the task command ("kick the incoming
    ball back to *this* location"), in body frame. Lives in the 1-D
    ``policy`` group along with proprio.
  * ``depth_image`` — ``(B, 1, 60, 60)`` depth image from the pelvis
    pinhole camera, with sim2real noise pipeline + per-episode
    extrinsic / intrinsic perturbations + 4-frame history. Lives in its
    own ``depth`` group because 4-D tensors can't share a flat concat
    group with 1-D obs.

  Critic obs: ``ball_relative_position`` / ``ball_velocity`` /
  ``passing_source_position`` / ``ball_absolute_position`` (privileged) —
  the standard asymmetric actor-critic trick: the value function gets
  oracle ball state for stable training, the policy doesn't.

  D1=a, D2=b, D3=a, D4=a, D5=b, D6=b, D7=a, D8=a, D9=a, D10=a, D11=a.
  """
  cfg = unitree_g1_passing_env_cfg(play=play)

  # Attach the noisy depth camera to the existing sensor tuple.
  cfg.scene.sensors = (
    *(cfg.scene.sensors or ()),
    _make_g1_depth_camera_cfg(),
  )

  enable_corruption = not play

  # Reuse the shared motion-prior + proprio obs builders.
  motion_prior_obs = _make_motion_prior_obs_group(
    enable_corruption=enable_corruption, with_height_scan=False
  )

  # 1-D ``policy`` group: command + proprio. No depth.
  policy_terms: dict[str, ObservationTermCfg] = {
    "passing_source_position": ObservationTermCfg(
      func=football_mdp.passing_source_position,
      params={"command_name": "passing_commands"},
    ),
    **_make_proprio_terms(),
  }
  policy = ObservationGroupCfg(
    terms=policy_terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  # 4-D ``depth`` group. Sensor's noise pipeline already does normalisation /
  # dropout / crop+resize / rect_mask, so ``enable_corruption=False`` (no
  # obs-manager-side noise stacked on top).
  depth = ObservationGroupCfg(
    terms={
      "depth_image": ObservationTermCfg(
        func=envs_mdp.depth_image,
        params={
          "sensor_cfg": SceneEntityCfg(PERCEPTION_SENSOR_NAME),
          "data_type": "distance_to_image_plane_noised",
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  # Critic gets privileged ball state, NO depth — value function uses
  # oracle info. Adds base_lin_vel for parity with other downstream tasks.
  critic_terms: dict[str, ObservationTermCfg] = {
    "ball_relative_position": ObservationTermCfg(
      func=football_mdp.ball_relative_position,
      params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
    ),
    "ball_velocity": ObservationTermCfg(
      func=football_mdp.ball_velocity,
      params={"ball_name": _BALL_ENTITY},
    ),
    "passing_source_position": ObservationTermCfg(
      func=football_mdp.passing_source_position,
      params={"command_name": "passing_commands"},
    ),
    **_make_proprio_terms(),
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "ball_absolute_position": ObservationTermCfg(
      func=football_mdp.ball_absolute_position,
      params={"ball_name": _BALL_ENTITY},
    ),
  }
  critic = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  cfg.observations = {
    "motion_prior_obs": motion_prior_obs,
    "policy": policy,
    "depth": depth,
    "critic": critic,
  }

  return cfg


# ---------------------------------------------------------------------------
# Shared perception obs builder for kicking / dribbling perception variants
# ---------------------------------------------------------------------------
#
# NOTE: ``unitree_g1_passing_perception_env_cfg`` above builds its obs groups
# inline (kept as-is so its already-trained ckpts stay shape-compatible).
# The kicking / dribbling perception variants below share this helper, which
# follows the exact same 4-group layout:
#
#   motion_prior_obs : proprio×4 (frozen VQ backbone input)
#   policy           : task command(s) + proprio (NO direct ball state, NO depth)
#   depth            : (1, H, W) depth image from the pelvis camera
#   critic           : task command(s) + privileged ball state + proprio + base_lin_vel
#
# ``command_terms`` are the task goal observations that BOTH policy and
# critic see (e.g. ``goal_position`` — telling the robot where the net is,
# which is part of the task spec, not a "cheat" ball-state read).
# ``privileged_terms`` are the direct ball observations that ONLY the critic
# sees (asymmetric actor-critic).


def _make_perception_obs_groups(
  *,
  play: bool,
  command_terms: dict[str, ObservationTermCfg],
  privileged_terms: dict[str, ObservationTermCfg],
) -> dict[str, ObservationGroupCfg]:
  """Build the 4 obs groups (motion_prior_obs / policy / depth / critic) for
  a perception-only football task. See module-level note above."""
  enable_corruption = not play

  motion_prior_obs = _make_motion_prior_obs_group(
    enable_corruption=enable_corruption, with_height_scan=False
  )

  # policy: task command(s) + proprio. No ball state, no depth.
  policy = ObservationGroupCfg(
    terms={**command_terms, **_make_proprio_terms()},
    concatenate_terms=True,
    enable_corruption=enable_corruption,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  # depth: 4-D image group (sensor already did normalise/dropout/crop/mask).
  depth = ObservationGroupCfg(
    terms={
      "depth_image": ObservationTermCfg(
        func=envs_mdp.depth_image,
        params={
          "sensor_cfg": SceneEntityCfg(PERCEPTION_SENSOR_NAME),
          "data_type": "distance_to_image_plane_noised",
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  # critic: command(s) + privileged ball state + proprio + base_lin_vel.
  critic = ObservationGroupCfg(
    terms={
      **command_terms,
      **privileged_terms,
      **_make_proprio_terms(),
      "base_lin_vel": ObservationTermCfg(
        func=envs_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    nan_policy="warn",
    nan_check_per_term=True,
  )

  return {
    "motion_prior_obs": motion_prior_obs,
    "policy": policy,
    "depth": depth,
    "critic": critic,
  }


def unitree_g1_kicking_perception_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Perception-only kicking task.

  Same physics / rewards / terminations / reset as
  :func:`unitree_g1_kicking_env_cfg`, but the policy can't read ball state
  directly. It sees:

  * ``goal_position`` — net location relative to the robot (task command,
    kept; this is *where to kick to*, not a ball-state cheat).
  * ``depth_image`` — pelvis pinhole depth camera (same D405 16:9 setup
    as passing perception).

  Critic keeps privileged ``ball_relative_position`` / ``ball_velocity`` /
  ``ball_to_goal_vector`` / ``ball_absolute_position`` for asymmetric AC.
  """
  cfg = unitree_g1_kicking_env_cfg(play=play)
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), _make_g1_depth_camera_cfg())
  cfg.observations = _make_perception_obs_groups(
    play=play,
    command_terms={
      "goal_position": ObservationTermCfg(
        func=football_mdp.dribbling_goal_position,
        params={"command_name": "kicking_commands"},
      ),
    },
    privileged_terms={
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "ball_to_goal_vector": ObservationTermCfg(
        func=football_mdp.ball_to_goal_vector,
        params={"ball_name": _BALL_ENTITY, "command_name": "kicking_commands"},
      ),
      "ball_absolute_position": ObservationTermCfg(
        func=football_mdp.ball_absolute_position,
        params={"ball_name": _BALL_ENTITY},
      ),
    },
  )
  return cfg


def unitree_g1_dribbling_perception_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Perception-only dribbling task.

  Same physics / rewards / terminations / reset as
  :func:`unitree_g1_dribbling_env_cfg`, but the policy can't read ball state
  directly. It sees ``goal_position`` (task command) + ``depth_image``
  (pelvis camera). Critic keeps the privileged ball state.
  """
  cfg = unitree_g1_dribbling_env_cfg(play=play)
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), _make_g1_depth_camera_cfg())
  cfg.observations = _make_perception_obs_groups(
    play=play,
    command_terms={
      "goal_position": ObservationTermCfg(
        func=football_mdp.dribbling_goal_position,
        params={"command_name": "dribbling_commands"},
      ),
    },
    privileged_terms={
      "ball_relative_position": ObservationTermCfg(
        func=football_mdp.ball_relative_position,
        params={"ball_name": _BALL_ENTITY, "asset_name": "robot"},
      ),
      "ball_velocity": ObservationTermCfg(
        func=football_mdp.ball_velocity,
        params={"ball_name": _BALL_ENTITY},
      ),
      "ball_to_goal_vector": ObservationTermCfg(
        func=football_mdp.ball_to_goal_vector,
        params={"ball_name": _BALL_ENTITY, "command_name": "dribbling_commands"},
      ),
      "ball_absolute_position": ObservationTermCfg(
        func=football_mdp.ball_absolute_position,
        params={"ball_name": _BALL_ENTITY},
      ),
    },
  )
  return cfg
