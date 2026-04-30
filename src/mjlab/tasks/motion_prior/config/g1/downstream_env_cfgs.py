"""Unitree G1 downstream-task environment configurations.

Builds on top of ``unitree_g1_rough_env_cfg`` (mjlab velocity-rough) and
swaps in three obs groups expected by ``DownStreamPolicy`` /
``DownStreamOnPolicyRunner``:

* ``policy``           — actor input: velocity_commands + proprio history + height_scan.
* ``motion_prior_obs`` — frozen ``motion_prior`` MLP input. **Schema MUST
                          match motion-prior training-time student obs**
                          (5 proprio × hist=4 + height_scan); we reuse
                          ``make_student_obs_group(extra_terms=...)`` so it's
                          the same builder. Mismatch here silently breaks
                          the frozen backbone (see audit doc §6.1).
* ``critic``           — privileged: policy + base_lin_vel.

Rewards / terminations / events / scene are inherited from velocity-rough
unchanged — that's the same task ``teacher_b`` (and therefore the trained
``motion_prior`` head) was distilled from, so the prior matches the
downstream task's reward structure out of the box.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.motion_prior.observations_cfg import (
  STUDENT_HISTORY_LENGTH,
  make_student_height_scan_term,
  make_student_obs_group,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def _make_proprio_terms(
  history_length: int = STUDENT_HISTORY_LENGTH,
) -> dict[str, ObservationTermCfg]:
  """Same 5 proprio terms as ``make_student_obs_group`` (without height_scan).

  Pulled out so ``policy`` / ``critic`` groups can prepend a velocity_commands
  term without re-listing each proprio entry.
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
  }


def _make_policy_obs_group(
  twist_command_name: str = "twist",
  height_scan_sensor_name: str = "terrain_scan",
  enable_corruption: bool = True,
) -> ObservationGroupCfg:
  """Actor obs: ``velocity_commands`` (3) + 5 proprio × hist=4 + height_scan."""
  terms: dict[str, ObservationTermCfg] = {
    "velocity_commands": ObservationTermCfg(
      func=velocity_mdp.generated_commands,
      params={"command_name": twist_command_name},
    ),
    **_make_proprio_terms(),
    "height_scan": make_student_height_scan_term(height_scan_sensor_name),
  }
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )


def _make_critic_obs_group(
  twist_command_name: str = "twist",
  height_scan_sensor_name: str = "terrain_scan",
  enable_corruption: bool = False,
) -> ObservationGroupCfg:
  """Critic obs = policy obs + ``base_lin_vel`` privileged term."""
  terms: dict[str, ObservationTermCfg] = {
    "velocity_commands": ObservationTermCfg(
      func=velocity_mdp.generated_commands,
      params={"command_name": twist_command_name},
    ),
    **_make_proprio_terms(),
    "height_scan": make_student_height_scan_term(height_scan_sensor_name),
    "base_lin_vel": ObservationTermCfg(
      func=velocity_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
  }
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )


def unitree_g1_downstream_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 velocity-tracking downstream env on rough terrain."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  enable_corruption = not play
  cfg.observations = {
    # Frozen ``motion_prior`` MLP input — schema fixed by motion-prior training.
    "motion_prior_obs": make_student_obs_group(
      enable_corruption=enable_corruption,
      extra_terms={"height_scan": make_student_height_scan_term("terrain_scan")},
    ),
    # Trainable actor sees the task command in addition.
    "policy": _make_policy_obs_group(
      twist_command_name="twist",
      height_scan_sensor_name="terrain_scan",
      enable_corruption=enable_corruption,
    ),
    # Critic gets privileged base_lin_vel on top.
    "critic": _make_critic_obs_group(
      twist_command_name="twist",
      height_scan_sensor_name="terrain_scan",
      enable_corruption=False,
    ),
  }
  return cfg
