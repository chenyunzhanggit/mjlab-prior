"""Smoke test for ``MotionPriorOnPolicyRunner`` using stub vec envs.

A real mjlab env requires a motion file, GPU, and minutes of setup. The
runner's wiring (build secondary env → policy → algorithm → learn loop →
save/load) can be exercised on CPU with a tiny fake env that quacks like
``RslRlVecEnvWrapper``. Real-env coverage lands in prior.md task #10.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.motion_prior.rl import runner as runner_mod
from mjlab.tasks.motion_prior.rl.runner import MotionPriorOnPolicyRunner
from mjlab.tasks.motion_prior.teacher import (
  TELEOPIT_TEACHER_CFG,
  VELOCITY_TEACHER_CFG,
)

from _motion_prior_helpers import DEFAULT_DEPTH_SHAPE, teacher_ckpts_or_skip

PROP_OBS_DIM = (3 + 3 + 29 + 29 + 29) * 4
NUM_ACTIONS = 29


def _ckpts_or_skip():
  return teacher_ckpts_or_skip()


class _FakeVecEnv:
  """Minimal stand-in for ``RslRlVecEnvWrapper``."""

  def __init__(self, num_envs: int, group_shapes: dict[str, tuple[int, ...]]) -> None:
    self.num_envs = num_envs
    self.num_actions = NUM_ACTIONS
    self._group_shapes = group_shapes
    self.device = torch.device("cpu")
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self._step_count = 0

  def _td(self) -> TensorDict:
    data = {
      key: torch.randn(self.num_envs, *shape)
      for key, shape in self._group_shapes.items()
    }
    return TensorDict(data, batch_size=[self.num_envs])

  def get_observations(self) -> TensorDict:
    return self._td()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    self._step_count += 1
    self.episode_length_buf += 1
    rew = torch.zeros(self.num_envs)
    # Trigger an occasional reset so episode-stat tracking exercises both
    # branches of ``_track_episode_stats``.
    done = (self.episode_length_buf % 5 == 0).long()
    self.episode_length_buf[done.bool()] = 0
    return self._td(), rew, done, {}


def _make_train_cfg(secondary_task_id: str = "_unused_") -> dict:
  ta, tb = _ckpts_or_skip()
  return {
    "num_steps_per_env": 4,
    "save_interval": 1,
    "secondary_task_id": secondary_task_id,
    "secondary_num_envs": 8,
    "teacher_a_policy_path": str(ta),
    "teacher_b_policy_path": str(tb),
    "policy": {
      "encoder_hidden_dims": (64, 32),
      "decoder_hidden_dims": (64, 32),
      "motion_prior_hidden_dims": (64, 32),
      "latent_z_dims": 16,
      "activation": "elu",
    },
    "algorithm": {
      "num_learning_epochs": 2,
      "learning_rate": 1e-3,
      "max_grad_norm": 1.0,
      "kl_loss_coeff_max": 0.01,
      "kl_loss_coeff_min": 0.001,
      "anneal_start_iter": 2500,
      "anneal_end_iter": 5000,
    },
    "logger": "none",  # disable tensorboard / wandb in tests
  }


@pytest.fixture
def fake_envs(monkeypatch) -> tuple[_FakeVecEnv, _FakeVecEnv]:
  flat = _FakeVecEnv(
    num_envs=8,
    group_shapes={
      "student": (PROP_OBS_DIM,),
      "teacher_a": (TELEOPIT_TEACHER_CFG.actor_obs_dim,),
      "teacher_a_history": (
        TELEOPIT_TEACHER_CFG.actor_history_length,
        TELEOPIT_TEACHER_CFG.actor_history_obs_dim,
      ),
      "depth": DEFAULT_DEPTH_SHAPE,
    },
  )
  rough = _FakeVecEnv(
    num_envs=8,
    group_shapes={
      "student": (PROP_OBS_DIM,),
      "teacher_b": (VELOCITY_TEACHER_CFG.actor_obs_dim,),
      "teacher_b_height": VELOCITY_TEACHER_CFG.height_obs_dim,
      "depth": DEFAULT_DEPTH_SHAPE,
    },
  )

  def _stub_secondary(*_args, **_kwargs) -> _FakeVecEnv:
    return rough

  monkeypatch.setattr(runner_mod, "_build_secondary_env", _stub_secondary)
  return flat, rough


def test_runner_constructs_and_runs_learn(fake_envs, tmp_path: Path) -> None:
  flat, rough = fake_envs
  cfg = _make_train_cfg()
  runner = MotionPriorOnPolicyRunner(
    env=flat,  # type: ignore[arg-type]
    train_cfg=cfg,
    log_dir=str(tmp_path),
    device="cpu",
  )
  assert runner.env is flat
  assert runner.env_b is rough
  assert runner.policy.latent_z_dims == 16

  runner.learn(num_learning_iterations=2)
  assert runner.current_learning_iteration == 1  # last iter index in range(0, 2)


def test_runner_save_load_roundtrip(fake_envs, tmp_path: Path) -> None:
  flat, _ = fake_envs
  cfg = _make_train_cfg()
  runner = MotionPriorOnPolicyRunner(
    env=flat,  # type: ignore[arg-type]
    train_cfg=cfg,
    log_dir=str(tmp_path),
    device="cpu",
  )
  runner.current_learning_iteration = 7

  path = tmp_path / "ckpt.pt"
  runner.save(str(path))
  assert path.is_file()

  # Mutate trainable weights, reload, weights should match the saved snapshot.
  before = {
    k: v.detach().clone() for k, v in runner.policy.encoder_a.state_dict().items()
  }
  with torch.no_grad():
    for p in runner.policy.encoder_a.parameters():
      p.add_(1.0)

  infos = runner.load(str(path), strict=True)
  after = runner.policy.encoder_a.state_dict()
  for k, v in before.items():
    assert torch.allclose(v, after[k]), f"encoder_a[{k}] not restored"
  assert runner.current_learning_iteration == 7
  assert infos == {}


def test_runner_add_git_repo_to_log_is_noop(fake_envs, tmp_path: Path) -> None:
  flat, _ = fake_envs
  cfg = _make_train_cfg()
  runner = MotionPriorOnPolicyRunner(
    env=flat,  # type: ignore[arg-type]
    train_cfg=cfg,
    log_dir=str(tmp_path),
    device="cpu",
  )
  # Must not raise — train.py calls this unconditionally.
  runner.add_git_repo_to_log("/tmp/whatever.py")


def test_runner_accepts_typed_cfg_via_asdict(fake_envs, tmp_path: Path) -> None:
  """Round-trip the typed dataclass through ``asdict`` (mirrors train.py)."""
  from dataclasses import asdict

  from mjlab.tasks.motion_prior.rl_cfg import (
    RslRlMotionPriorAlgoCfg,
    RslRlMotionPriorPolicyCfg,
    RslRlMotionPriorRunnerCfg,
  )

  ta, tb = _ckpts_or_skip()
  flat, _ = fake_envs
  typed = RslRlMotionPriorRunnerCfg(
    experiment_name="g1_motion_prior_test",
    num_steps_per_env=4,
    save_interval=1,
    secondary_num_envs=8,
    teacher_a_policy_path=str(ta),
    teacher_b_policy_path=str(tb),
    policy=RslRlMotionPriorPolicyCfg(
      encoder_hidden_dims=(64, 32),
      decoder_hidden_dims=(64, 32),
      motion_prior_hidden_dims=(64, 32),
      latent_z_dims=16,
    ),
    algorithm=RslRlMotionPriorAlgoCfg(num_learning_epochs=1, learning_rate=1e-3),
  )
  cfg_dict = asdict(typed)
  runner = MotionPriorOnPolicyRunner(
    env=flat,  # type: ignore[arg-type]
    train_cfg=cfg_dict,
    log_dir=str(tmp_path),
    device="cpu",
  )
  assert runner.policy.latent_z_dims == 16
  runner.learn(num_learning_iterations=1)
