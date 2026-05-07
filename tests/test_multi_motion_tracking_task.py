"""Tests specific to multi-motion tracking tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.multi_motion_tracking.mdp import MultiMotionCommandCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg


@pytest.fixture(scope="module")
def multi_motion_task_ids() -> list[str]:
  """Get all multi-motion tracking task IDs."""
  return [t for t in list_tasks() if "MultiMotionTracking" in t]


def test_g1_multi_motion_tracking_registered(
  multi_motion_task_ids: list[str],
) -> None:
  """The G1 multi-motion tracking task should be registered."""
  assert "Mjlab-MultiMotionTracking-Flat-Unitree-G1" in multi_motion_task_ids


def test_uses_multi_motion_command_cfg(multi_motion_task_ids: list[str]) -> None:
  """All multi-motion tasks should expose a MultiMotionCommandCfg."""
  for task_id in multi_motion_task_ids:
    cfg = load_env_cfg(task_id)
    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"
    assert isinstance(cfg.commands["motion"], MultiMotionCommandCfg), (
      f"Task {task_id} motion command is not MultiMotionCommandCfg"
    )


def test_motion_files_empty_by_default(multi_motion_task_ids: list[str]) -> None:
  """``motion_files`` / ``motion_path`` are CLI-injected; cfg defaults are empty."""
  for task_id in multi_motion_task_ids:
    cfg = load_env_cfg(task_id)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MultiMotionCommandCfg)
    assert motion_cmd.motion_files == [], (
      f"Task {task_id} ships with non-empty motion_files: {motion_cmd.motion_files}"
    )
    assert motion_cmd.motion_path == "", (
      f"Task {task_id} ships with non-empty motion_path: {motion_cmd.motion_path}"
    )


def test_g1_anchor_and_body_names_set() -> None:
  """G1 task fills in the robot-specific anchor + body name list."""
  cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1")
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  assert motion_cmd.anchor_body_name == "torso_link"
  assert "pelvis" in motion_cmd.body_names
  assert "torso_link" in motion_cmd.body_names


def test_g1_action_scale() -> None:
  """G1 multi-motion tracking should use G1_ACTION_SCALE."""
  cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1")
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert joint_pos_action.scale == G1_ACTION_SCALE


def test_self_collision_sensor_present(multi_motion_task_ids: list[str]) -> None:
  """Self-collision sensor is required by the ``self_collisions`` reward."""
  for task_id in multi_motion_task_ids:
    cfg = load_env_cfg(task_id)
    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"
    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_actor_observation_terms_match_motionprior() -> None:
  """Actor obs schema mirrors motionprior's TeacherObservationsCfg.PolicyCfg."""
  cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1")
  actor_terms = set(cfg.observations["actor"].terms.keys())
  expected = {
    "command",
    "projected_gravity",
    "motion_ref_ang_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "base_lin_vel",
    "anchor_pos_diff",
    "key_body_pos_diff",
    "key_body_rot_diff",
    "anchor_height",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
  }
  assert actor_terms == expected, (
    f"Actor obs term set mismatch — extra={actor_terms - expected}, "
    f"missing={expected - actor_terms}"
  )


def test_play_disables_corruption_and_push_robot() -> None:
  """Play mode strips obs corruption and the push_robot event."""
  cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1", play=True)
  assert cfg.observations["actor"].enable_corruption is False
  assert "push_robot" not in cfg.events


def test_play_disables_rsi() -> None:
  """Play mode disables RSI by clearing pose/velocity ranges."""
  cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1", play=True)
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MultiMotionCommandCfg)
  assert motion_cmd.pose_range == {}
  assert motion_cmd.velocity_range == {}
  assert motion_cmd.start_from_zero_step is True


def test_play_episode_length_infinite() -> None:
  """Play mode runs effectively forever."""
  cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1", play=True)
  assert cfg.episode_length_s >= 1e9


def test_train_and_play_obs_terms_match() -> None:
  """Training and play obs term sets should match (only knobs differ)."""
  train_cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1")
  play_cfg = load_env_cfg("Mjlab-MultiMotionTracking-Flat-Unitree-G1", play=True)
  for group_name in train_cfg.observations:
    train_terms = set(train_cfg.observations[group_name].terms.keys())
    play_terms = set(play_cfg.observations[group_name].terms.keys())
    assert train_terms == play_terms, (
      f"Obs terms mismatch in group '{group_name}'"
    )
