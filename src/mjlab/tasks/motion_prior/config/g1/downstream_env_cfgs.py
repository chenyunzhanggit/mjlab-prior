"""Unitree G1 downstream-task environment configurations.

Aligned with motionprior reference ``g1_downstream_vq_cfg.py`` per the
audit doc:

* **Terrain**: pure plane (inherit ``unitree_g1_flat_env_cfg``, which
  already strips ``terrain_scan`` + ``height_scan`` for us).
* **Obs groups**: ``motion_prior_obs`` / ``policy`` / ``critic`` —
  proprio-only (no height_scan). ``height_scan`` entries are present in
  the term builders **commented out**, so re-enabling later is a
  one-line uncomment + re-train.
* **Commands**: ``velocity_commands`` with the reference's actual
  training ranges (``lin_vel_x=(-1.0, 2.0)`` etc.). The zero-zero
  defaults in the reference file are debug placeholders — we use the
  commented-out training ranges directly.
* **RSI noise**: equivalent to reference's
  ``MotionCommandCfg(init_from_motion=False, pose_range=..., velocity_range=...,
  joint_position_range=(-0.1, 0.1))``. We don't carry the motion command
  itself (the downstream task doesn't consume motion data and our
  ``MultiMotionCommandCfg`` has no ``init_from_motion`` flag) — instead
  we override ``reset_base`` and ``reset_robot_joints`` events with the
  same noise ranges. Functionally identical for RSI purposes.
* **Rewards**: minimal — ``track_lin_vel_xy_exp`` (w=2), ``track_ang_vel_z_exp``
  (w=2), ``joint_limit`` (w=-10), ``undesired_contacts`` (w=-0.1).
  Standard velocity-env shaping (action_rate / dof_acc / dof_torque /
  feet_air_time / alive) is dropped to match reference, so the
  motion_prior backbone gets to express the gait without competing
  reward pressure.
* **Terminations**: ``time_out`` + ``bad_orientation(limit_angle=0.7)``.
* **Robot**: full-mesh G1 (mjlab default), not the cylinder collision
  variant the reference uses.

The frozen-backbone ckpt's ``motion_prior`` first-layer input dim must
match this env's ``motion_prior_obs`` dim. If your trained ckpt was
done with height_scan in student obs, re-train the motion_prior
*without* height_scan first, or re-enable the height_scan term below.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.motion_prior.config.g1.env_cfgs import _make_g1_terrain_scan_sensor
from mjlab.tasks.motion_prior.observations_cfg import (
  STUDENT_HISTORY_LENGTH,
  make_student_height_scan_term,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


# Mirrors reference's ``VELOCITY_RANGE`` in ``g1_downstream_vq_cfg.py``.
# Used for the RSI base-velocity randomization (replaces the empty
# ``reset_base.velocity_range`` of the flat velocity env).
_RSI_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def _make_proprio_terms(
  history_length: int = STUDENT_HISTORY_LENGTH,
) -> dict[str, ObservationTermCfg]:
  """5 proprio terms with history_length=4 (reference: PropObsCfg).

  ``height_scan`` is intentionally omitted — see commented line below;
  re-add via ``extra_terms`` if you re-introduce a height-scan-aware
  motion_prior ckpt.
  """
  return {
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=history_length,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=history_length,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=history_length,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.0, n_max=1.0),
      history_length=history_length,
    ),
    "actions": ObservationTermCfg(
      func=envs_mdp.last_action,
      history_length=history_length,
    ),
    # ---- Re-enable height_scan here to restore the height-scan path ----
    # Requires:
    #   (1) keeping the ``terrain_scan`` sensor on the scene (the flat env
    #       removes it; you'd need to re-inject it), and
    #   (2) a motion_prior ckpt trained with height_scan in student obs.
    # "height_scan": make_student_height_scan_term("terrain_scan"),
  }


def _make_motion_prior_obs_group(
  enable_corruption: bool = True,
  with_height_scan: bool = False,
) -> ObservationGroupCfg:
  """Frozen motion_prior backbone input.

  Defaults to **5 proprio × hist=4 = 372-dim** (proprio only). Used to
  default to 559-dim (proprio + height_scan); now matches the
  user-retrained motion_prior ckpt which drops the height_scan slice.

  Set ``with_height_scan=True`` to restore the 187-dim height_scan
  slice (559-dim total). Requires:
    1. The env scene has a ``terrain_scan`` raycast sensor (use
       :func:`_make_g1_terrain_scan_sensor` and add it to ``scene.sensors``).
    2. A motion_prior ckpt that was trained with height_scan in student
       obs (the disabled-by-default ``extra_terms`` line in
       :mod:`env_cfgs` controls the training side).
  """
  terms = _make_proprio_terms()
  if with_height_scan:
    terms["height_scan"] = make_student_height_scan_term("terrain_scan")
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
    # Auto-sanitize and log first occurrence per term. Catches edge cases
    # like a degenerate raycast hit or a missing sensor frame so a single
    # bad obs doesn't crash the downstream PPO with the std-NaN path
    # through ``Normal.sample``.
    nan_policy="warn",
    nan_check_per_term=True,
  )


def _make_policy_obs_group(
  twist_command_name: str = "twist",
  enable_corruption: bool = True,
) -> ObservationGroupCfg:
  """Actor obs: ``velocity_commands`` + 5 proprio × hist=4 (reference's
  ``PolicyObsCfg``).

  ``height_scan`` is commented out at the bottom — re-enable if you
  train a height-scan-aware downstream policy.
  """
  terms: dict[str, ObservationTermCfg] = {
    "velocity_commands": ObservationTermCfg(
      func=velocity_mdp.generated_commands,
      params={"command_name": twist_command_name},
    ),
    **_make_proprio_terms(),
    # ---- Re-enable height_scan in policy obs ----
    # "height_scan": make_student_height_scan_term("terrain_scan"),
  }
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
    nan_policy="warn",
    nan_check_per_term=True,
  )


def _make_critic_obs_group(
  twist_command_name: str = "twist",
  enable_corruption: bool = False,
) -> ObservationGroupCfg:
  """Critic obs = policy obs + ``base_lin_vel`` privileged term
  (reference's ``PrivilegedObsCfg``)."""
  terms: dict[str, ObservationTermCfg] = {
    "velocity_commands": ObservationTermCfg(
      func=velocity_mdp.generated_commands,
      params={"command_name": twist_command_name},
    ),
    **_make_proprio_terms(),
    # ---- Re-enable height_scan in critic obs ----
    # "height_scan": make_student_height_scan_term("terrain_scan"),
    "base_lin_vel": ObservationTermCfg(
      func=velocity_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
    nan_policy="warn",
    nan_check_per_term=True,
  )


def unitree_g1_downstream_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 velocity-tracking downstream env on a plane.

  Aligned with motionprior reference ``g1_downstream_vq_cfg.py``. See
  module docstring for the full diff against the original mjlab port.

  The ``motion_prior_obs`` group keeps ``height_scan`` (5 proprio × 4 +
  187 = 559) so the frozen motion_prior backbone trained with
  height-scan-aware student obs loads cleanly. ``policy`` / ``critic``
  groups stay clean (no height_scan) per user request — the actor
  doesn't need terrain info for plane-only velocity tracking.
  """
  cfg = unitree_g1_flat_env_cfg(play=play)

  # ---- terrain_scan injection DISABLED (user request) ----------------
  # Previously we re-injected the ``terrain_scan`` raycast that
  # ``unitree_g1_flat_env_cfg`` strips, because the frozen motion_prior
  # backbone read a 187-dim height_scan slice. Now we use a proprio-only
  # 372-dim motion_prior ckpt; the sensor isn't needed. To re-enable,
  # uncomment the block below AND flip ``with_height_scan=True`` on the
  # ``motion_prior_obs`` group construction.
  # if not any(s.name == "terrain_scan" for s in (cfg.scene.sensors or ())):
  #   cfg.scene.sensors = (*(cfg.scene.sensors or ()), _make_g1_terrain_scan_sensor())

  enable_corruption = not play

  # -------------------- Observations -------------------- #
  cfg.observations = {
    # Frozen motion_prior MLP input — must match the ckpt's prop_obs_dim.
    # Default is 372-dim (proprio only) matching the user-retrained
    # ``Mjlab-MotionPrior-Single-VQ-Trackingbfm-Unitree-G1`` ckpt.
    "motion_prior_obs": _make_motion_prior_obs_group(
      enable_corruption=enable_corruption,
      # with_height_scan=True,  # re-enable for the old 559-dim ckpts
    ),
    # Trainable actor: prepends velocity command to proprio (no height_scan).
    "policy": _make_policy_obs_group(
      twist_command_name="twist",
      enable_corruption=enable_corruption,
    ),
    # Critic gets privileged base_lin_vel on top (no height_scan).
    "critic": _make_critic_obs_group(
      twist_command_name="twist",
      enable_corruption=False,
    ),
  }

  # -------------------- Commands (velocity ranges) -------------------- #
  # Original training ranges (matching the reference cfg) — wide enough to
  # cover meaningful walking and turning speeds.
  twist_cmd = cfg.commands["twist"]
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-3.14, 3.14)

  if play:
    # Play: widen the joystick range by 1.5x so the operator can probe
    # mild out-of-distribution commands without having to first crank up
    # the Viser ``Max <axis>`` sub-slider (which goes up to 10.0).
    # ``rel_standing_envs=1.0`` plus zeroing heading/forward keeps the
    # auto-sampled command at (0, 0, 0) so the robot stands still by
    # default — the operator then ticks the joystick's "Enable"
    # checkbox and drags sliders to drive.
    play_scale = 1.5
    twist_cmd.ranges.lin_vel_x = (-1.0 * play_scale, 2.0 * play_scale)
    twist_cmd.ranges.lin_vel_y = (-1.0 * play_scale, 1.0 * play_scale)
    twist_cmd.ranges.ang_vel_z = (-3.14 * play_scale, 3.14 * play_scale)
    twist_cmd.rel_standing_envs = 1.0
    twist_cmd.rel_heading_envs = 0.0
    twist_cmd.rel_forward_envs = 0.0

  # -------------------- RSI noise (reference parity) -------------------- #
  # Equivalent to reference's
  #   ``MotionCommandCfg(init_from_motion=False, pose_range=..., velocity_range=...,
  #     joint_position_range=(-0.1, 0.1))``
  # — we tighten the flat-env's reset events to match reference's noise
  # ranges instead of carrying a no-op motion command.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (-0.01, 0.01),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
  }
  cfg.events["reset_base"].params["velocity_range"] = _RSI_VELOCITY_RANGE
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  # Joint velocity noise at reset — pushes the robot out of perfect
  # static rest each episode, breaking the "stand still" local
  # optimum during PPO exploration. This is the ONLY new addition vs
  # the prior reference-parity setup; mirrors unitree_rl_lab's
  # ``reset_robot_joints(velocity_range=(-1.0, 1.0))``.
  cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)

  # -------------------- Rewards (your minimal set + stability adds) ------- #
  # Restored: explicit minimal reward dict (overwrites velocity env's set).
  # On top of the 4 task-essential terms, we add the stability penalties
  # you tried earlier but rebalanced so the policy doesn't degenerate into
  # "stand still".
  #
  # Why this works without the "stand still" trap:
  #   * tracking weight stays dominant (2.0), penalties are kept SMALL
  #     so the gradient toward "track command" wins;
  #   * action_rate_l2 / joint_torques_l2 / joint_acc_l2 are tiny weights —
  #     they shape behavior to be smoother, not punish movement itself;
  #   * flat_orientation_l2 / lin_vel_z_l2 / ang_vel_xy_l2 / base_height_l2
  #     penalize ONLY off-task disturbances (bouncing / tilting / dropping
  #     height), not in-plane walking velocity (which is on the X/Y axes,
  #     not z; xy linear velocity is what tracking REWARDS).
  #
  # If you want to tone any term up/down without breaking walking:
  #   * If robot still doesn't walk: lower the four stability weights
  #     (flat_orientation_l2, lin_vel_z_l2, ang_vel_xy_l2, base_height_l2)
  #     by 2-5x.
  #   * If robot walks but is jerky / over-spins: raise action_rate_l2.
  #   * If robot bobs vertically: raise lin_vel_z_l2.
  cfg.rewards = {
    # -- Task -- #
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": 0.5},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": 0.5},
    ),
    # -- Joint-limit hard cap -- #
    "joint_limit": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # -- Self-collision soft cap -- #
    "undesired_contacts": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision", "force_threshold": 1.0},
    ),
    # -- Stability penalties (your additions, rebalanced) -- #
    # Weights ~10x smaller than the reference unitree_rl_lab values so the
    # tracking gradient still wins. Increase 2-5x if the policy walks but
    # is unstable; decrease 2-5x if it refuses to walk.
    "flat_orientation_l2": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-0.2,  # was -1.0 in your snippet
    ),
    "lin_vel_z_l2": RewardTermCfg(
      func=envs_mdp.lin_vel_z_l2,
      weight=-0.5,  # was -2.0
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=envs_mdp.ang_vel_xy_l2,
      weight=-0.02,  # was -0.05
    ),
    "base_height_l2": RewardTermCfg(
      func=envs_mdp.base_height_l2,
      weight=-0.2,  # was -1.0
      params={"target_height": 0.78},
    ),
    # -- Smoothness penalties (small weights, ok to keep) -- #
    "action_rate_l2": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.01,  # same as your snippet
    ),
    "joint_torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1.0e-5,
    ),
    "joint_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-2.5e-7,
    ),
  }

  # -------------------- Terminations (reference minimal set) -------------------- #
  # ``time_out`` and ``bad_orientation`` live in mjlab's shared
  # ``envs/mdp`` namespace (not the velocity-specific mdp).
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    ),
  }

  return cfg
