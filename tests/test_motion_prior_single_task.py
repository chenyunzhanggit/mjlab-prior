"""Configuration-level tests for single-encoder motion-prior tasks."""

import pytest

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.motion_prior.rl_cfg import (
  RslRlMotionPriorSingleRunnerCfg,
  RslRlMotionPriorSingleVQRunnerCfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp.multi_commands import MultiMotionCommandCfg


@pytest.fixture(scope="module")
def single_task_ids() -> list[str]:
  """Get all single-encoder motion-prior task IDs."""
  return [t for t in list_tasks() if "MotionPriorSingle" in t]


def test_single_tasks_registered(single_task_ids: list[str]) -> None:
  """Both VAE and VQ single-encoder tasks should be registered."""
  assert "Mjlab-MotionPriorSingle-Flat-Unitree-G1" in single_task_ids
  assert "Mjlab-MotionPriorSingle-VQ-Flat-Unitree-G1" in single_task_ids


def test_single_uses_multi_motion_command(single_task_ids: list[str]) -> None:
  """Single-encoder tasks reuse the multi-motion command (teacher trained on it)."""
  for task_id in single_task_ids:
    cfg = load_env_cfg(task_id)
    assert "motion" in cfg.commands
    assert isinstance(cfg.commands["motion"], MultiMotionCommandCfg), (
      f"Task {task_id} motion command is not MultiMotionCommandCfg"
    )


def test_single_obs_groups_present(single_task_ids: list[str]) -> None:
  """Both tasks expose ``student`` and ``teacher_tracking`` obs groups."""
  for task_id in single_task_ids:
    cfg = load_env_cfg(task_id)
    assert "student" in cfg.observations
    assert "teacher_tracking" in cfg.observations
    # Single-encoder envs do NOT expose dual-teacher groups.
    assert "teacher_a" not in cfg.observations
    assert "teacher_b" not in cfg.observations


def test_single_teacher_obs_term_set() -> None:
  """``teacher_tracking`` mirrors the multi-motion tracking actor schema."""
  cfg = load_env_cfg("Mjlab-MotionPriorSingle-Flat-Unitree-G1")
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
  got = set(cfg.observations["teacher_tracking"].terms.keys())
  assert got == expected, (
    f"teacher_tracking schema diverged from multi-motion tracking actor: "
    f"extra={got - expected}, missing={expected - got}"
  )


def test_single_rewards_trimmed(single_task_ids: list[str]) -> None:
  """Distillation tasks keep only the anchor-pos reward (other PPO shaping dropped)."""
  for task_id in single_task_ids:
    cfg = load_env_cfg(task_id)
    assert set(cfg.rewards.keys()) == {"motion_global_anchor_pos"}, (
      f"Task {task_id} unexpected rewards: {set(cfg.rewards.keys())}"
    )


def test_single_rl_cfg_types() -> None:
  """Each task ships with its matching typed RL cfg."""
  vae_cfg = load_rl_cfg("Mjlab-MotionPriorSingle-Flat-Unitree-G1")
  assert isinstance(vae_cfg, RslRlMotionPriorSingleRunnerCfg)
  assert vae_cfg.class_name == "MotionPriorSingleOnPolicyRunner"
  assert vae_cfg.algorithm.class_name == "DistillationMotionPriorSingle"
  assert vae_cfg.policy.class_name == "MotionPriorSinglePolicy"
  # Defaults must require user-supplied teacher path (no hard-coded path).
  assert vae_cfg.teacher_policy_path == ""

  vq_cfg = load_rl_cfg("Mjlab-MotionPriorSingle-VQ-Flat-Unitree-G1")
  assert isinstance(vq_cfg, RslRlMotionPriorSingleVQRunnerCfg)
  assert vq_cfg.class_name == "MotionPriorSingleVQOnPolicyRunner"
  assert vq_cfg.algorithm.class_name == "DistillationMotionPriorSingleVQ"
  assert vq_cfg.policy.class_name == "MotionPriorSingleVQPolicy"
  assert vq_cfg.teacher_policy_path == ""


def test_single_g1_action_scale(single_task_ids: list[str]) -> None:
  """G1 single-encoder tasks should use G1_ACTION_SCALE."""
  for task_id in single_task_ids:
    cfg = load_env_cfg(task_id)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    assert joint_pos_action.scale == G1_ACTION_SCALE


def test_single_play_disables_corruption_and_push() -> None:
  """Play mode: corruption off, push_robot stripped, episode infinite."""
  for task_id in [
    "Mjlab-MotionPriorSingle-Flat-Unitree-G1",
    "Mjlab-MotionPriorSingle-VQ-Flat-Unitree-G1",
  ]:
    cfg = load_env_cfg(task_id, play=True)
    assert cfg.observations["student"].enable_corruption is False
    assert cfg.observations["teacher_tracking"].enable_corruption is False
    assert "push_robot" not in cfg.events
    assert cfg.episode_length_s >= 1e9


def test_single_train_play_term_match() -> None:
  """Train and play obs term sets must match (only knobs differ)."""
  for task_id in [
    "Mjlab-MotionPriorSingle-Flat-Unitree-G1",
    "Mjlab-MotionPriorSingle-VQ-Flat-Unitree-G1",
  ]:
    train_cfg = load_env_cfg(task_id)
    play_cfg = load_env_cfg(task_id, play=True)
    for group in train_cfg.observations:
      assert set(train_cfg.observations[group].terms.keys()) == set(
        play_cfg.observations[group].terms.keys()
      ), f"Term mismatch in {task_id} group '{group}'"
