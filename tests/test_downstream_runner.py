"""Smoke tests for ``DownStreamOnPolicyRunner`` using a stub vec env.

Real env spin-up needs GPU + a velocity-rough scene. Wiring (env →
policy → PPO → learn loop → save/load) is exercised on CPU with a fake
env that quacks like ``RslRlVecEnvWrapper``. End-to-end on GPU lands in
``test_downstream_e2e.py``.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.motion_prior.rl.downstream_runner import DownStreamOnPolicyRunner

PROP = 559
LATENT = 32
NUM_ACTIONS = 29
NUM_OBS = PROP + 3
NUM_PRIV = NUM_OBS + 3
NUM_ENVS = 8
NUM_STEPS = 4


def _find_motion_prior_ckpt() -> Path | None:
  pattern = str(
    Path("~/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/*/model_*.pt").expanduser()
  )
  matches = sorted(glob.glob(pattern))
  return Path(matches[-1]) if matches else None


@pytest.fixture(scope="module")
def ckpt_path() -> Path:
  p = _find_motion_prior_ckpt()
  if p is None:
    pytest.skip("no motion_prior ckpt available")
  return p


class _FakeVecEnv:
  """Minimal stand-in for ``RslRlVecEnvWrapper`` used by downstream tests."""

  def __init__(self) -> None:
    self.num_envs = NUM_ENVS
    self.num_actions = NUM_ACTIONS
    self.device = torch.device("cpu")
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
    self._step_count = 0

  def _td(self) -> TensorDict:
    return TensorDict(
      {
        "policy": torch.randn(self.num_envs, NUM_OBS),
        "motion_prior_obs": torch.randn(self.num_envs, PROP),
        "critic": torch.randn(self.num_envs, NUM_PRIV),
      },
      batch_size=[self.num_envs],
    )

  def get_observations(self) -> TensorDict:
    return self._td()

  def step(
    self, actions: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    self._step_count += 1
    self.episode_length_buf += 1
    rew = torch.zeros(self.num_envs)
    # Trigger occasional dones so the episode-stat tracker hits both branches.
    done = (self.episode_length_buf % 3 == 0).long()
    self.episode_length_buf[done.bool()] = 0
    return self._td(), rew, done, {}


def _make_train_cfg(ckpt: Path) -> dict:
  return {
    "num_steps_per_env": NUM_STEPS,
    "save_interval": 100,
    "motion_prior_ckpt_path": str(ckpt),
    "policy": {
      "latent_z_dims": LATENT,
      "actor_hidden_dims": (64, 32),
      "critic_hidden_dims": (64, 32),
      "init_noise_std": 1.0,
    },
    "algorithm": {
      "num_learning_epochs": 2,
      "num_mini_batches": 2,
      "learning_rate": 1e-3,
      "max_grad_norm": 1.0,
    },
    "logger": "none",
  }


def test_runner_constructs_and_runs_learn(ckpt_path: Path, tmp_path: Path) -> None:
  env = _FakeVecEnv()
  runner = DownStreamOnPolicyRunner(
    env=env,  # type: ignore[arg-type]
    train_cfg=_make_train_cfg(ckpt_path),
    log_dir=str(tmp_path),
    device="cpu",
  )
  assert runner.env is env
  runner.learn(num_learning_iterations=2)
  assert runner.current_learning_iteration == 1


def test_runner_save_load_only_persists_trainable(
  ckpt_path: Path, tmp_path: Path
) -> None:
  env = _FakeVecEnv()
  runner = DownStreamOnPolicyRunner(
    env=env,  # type: ignore[arg-type]
    train_cfg=_make_train_cfg(ckpt_path),
    log_dir=str(tmp_path),
    device="cpu",
  )
  runner.current_learning_iteration = 11

  ckpt_out = tmp_path / "ckpt.pt"
  runner.save(str(ckpt_out))
  state = torch.load(ckpt_out, weights_only=False)

  # Frozen backbone keys must NOT appear in saved state.
  assert "motion_prior" not in state
  assert "decoder" not in state
  assert "mp_mu" not in state
  # Trainable parts must be present.
  assert {"actor", "critic", "std", "optimizer", "iter"} <= set(state.keys())
  assert state["iter"] == 11

  # Mutate then reload — actor weights restored.
  before = {k: v.detach().clone() for k, v in runner.policy.actor.state_dict().items()}
  with torch.no_grad():
    for p in runner.policy.actor.parameters():
      p.add_(1.0)
  runner.load(str(ckpt_out))
  for k, v in runner.policy.actor.state_dict().items():
    assert torch.allclose(before[k], v.detach())
  assert runner.current_learning_iteration == 11


def test_get_inference_policy_runs_path_through_actor(
  ckpt_path: Path, tmp_path: Path
) -> None:
  env = _FakeVecEnv()
  runner = DownStreamOnPolicyRunner(
    env=env,  # type: ignore[arg-type]
    train_cfg=_make_train_cfg(ckpt_path),
    log_dir=str(tmp_path),
    device="cpu",
  )
  policy_fn = runner.get_inference_policy()
  obs = env.get_observations()
  action = policy_fn(obs)
  assert action.shape == (NUM_ENVS, NUM_ACTIONS)
  # No teacher obs in env — but downstream doesn't care; only policy +
  # motion_prior_obs are consumed.
  assert torch.isfinite(action).all()


def test_runner_add_git_repo_to_log_is_noop(ckpt_path: Path, tmp_path: Path) -> None:
  env = _FakeVecEnv()
  runner = DownStreamOnPolicyRunner(
    env=env,  # type: ignore[arg-type]
    train_cfg=_make_train_cfg(ckpt_path),
    log_dir=str(tmp_path),
    device="cpu",
  )
  runner.add_git_repo_to_log("/tmp/whatever.py")  # must not raise
