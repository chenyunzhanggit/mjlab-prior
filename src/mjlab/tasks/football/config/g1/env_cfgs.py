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
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
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
# Passing — Perception-only variant
# ---------------------------------------------------------------------------

# Forward LiDAR sensor name. The policy reads its depth output via the
# ``depth_image`` MDP term; the critic still gets privileged ball state.
_PERCEPTION_SENSOR = "pelvis_forward_lidar"

# Geometry: 16 × 11 grid of parallel rays sweeping a 1.5 m × 1.0 m virtual
# scan plane in front of pelvis. resolution=0.1 → 176 rays (small enough
# to keep the policy input tractable, dense enough to detect the 22 cm
# soccer ball at 3–6 m).
_PERCEPTION_GRID_SIZE = (1.5, 1.0)
_PERCEPTION_GRID_RES = 0.1
_PERCEPTION_MAX_DISTANCE = 5.0
_PERCEPTION_NUM_RAYS = 176  # 16 × 11; sanity-checked in tests


def _make_g1_forward_lidar_sensor() -> RayCastSensorCfg:
  """Forward-facing LiDAR-style raycast on the G1 pelvis.

  Rays are parallel along pelvis local +X (forward), arranged on a 1.5 m ×
  1.0 m grid in the pelvis local YZ plane. With ``ray_alignment="base"``
  the whole array rotates with the pelvis, so as the robot turns the
  policy's depth image rotates with it — same semantics as a chest-mounted
  multi-line LiDAR or stereo depth camera.

  Why GridPattern instead of PinholeCamera: PinholeCameraPatternCfg
  generates rays along the MuJoCo camera convention (-Z forward), which
  would require adding a forward-rotated site to the G1 MJCF. GridPattern
  + ``direction=(1, 0, 0)`` avoids touching the robot spec and gives the
  same downstream-task signal (a "what's in front of me?" depth map).
  """
  return RayCastSensorCfg(
    name=_PERCEPTION_SENSOR,
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    pattern=GridPatternCfg(
      size=_PERCEPTION_GRID_SIZE,
      resolution=_PERCEPTION_GRID_RES,
      direction=(1.0, 0.0, 0.0),
    ),
    ray_alignment="base",
    max_distance=_PERCEPTION_MAX_DISTANCE,
    exclude_parent_body=True,
    # Include all geom groups (0=terrain, 1=collision, 2=ball etc.). The
    # ball geom is in the default group 0 of the soccer-ball spec, so it
    # registers as a hit.
    include_geom_groups=None,
    debug_vis=False,  # Set True locally to inspect rays in viser.
  )


def unitree_g1_passing_perception_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Perception-only passing task.

  Same physics / rewards / terminations / reset events as
  :func:`unitree_g1_passing_env_cfg`, but the policy can no longer read
  ball state directly. Instead it sees:

  * ``passing_source_position`` — the task command ("kick the incoming ball
    back to *this* location"), still in body frame. Necessary because the
    target zone is part of the goal, not something the robot has to find.
  * ``ball_depth_image`` — a flat 176-dim depth scan from a forward LiDAR
    on the pelvis. The policy has to *infer* where the ball is from the
    depth pattern, the same as a real robot would.

  Critic obs are unchanged: ``ball_relative_position`` / ``ball_velocity``
  / ``passing_source_position`` / ``ball_absolute_position`` (privileged).
  This is the standard PPO asymmetric-actor-critic trick: the value
  function gets oracle ball state for stable training, the policy
  doesn't, so deployment behavior matches what the policy learnt.
  """
  cfg = unitree_g1_passing_env_cfg(play=play)

  # Attach the forward LiDAR to the existing sensor tuple.
  cfg.scene.sensors = (*(cfg.scene.sensors or ()), _make_g1_forward_lidar_sensor())

  enable_corruption = not play

  # Rebuild observations: policy drops direct ball state, gains depth_image.
  # Critic keeps privileged ball state for value-function learning.
  cfg.observations = _make_football_obs_groups(
    policy_extra_terms={
      "passing_source_position": ObservationTermCfg(
        func=football_mdp.passing_source_position,
        params={"command_name": "passing_commands"},
      ),
      # 176-dim flat depth scan. ``scale=1/max_distance`` normalises into
      # roughly [0, 1] so it plays nicely with the actor MLP (the
      # downstream-VQ runner doesn't apply empirical normalisation).
      # ``noise`` mimics depth-sensor jitter for sim2real robustness.
      "ball_depth_image": ObservationTermCfg(
        func=football_mdp.depth_image,
        params={
          "sensor_name": _PERCEPTION_SENSOR,
          "scale": 1.0 / _PERCEPTION_MAX_DISTANCE,
        },
        noise=Unoise(n_min=-0.02, n_max=0.02),
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

  return cfg
