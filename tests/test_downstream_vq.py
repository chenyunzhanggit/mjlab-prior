"""Smoke tests for the VQ downstream stack.

Generates a synthetic VQ motion-prior checkpoint on the fly (no real VQ
training needed) so this works on a clean clone. The fake ckpt has
exactly the keys ``MotionPriorVQOnPolicyRunner.save`` writes:
``decoder``, ``motion_prior``, ``quantizer``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.tasks.motion_prior.onnx import (
  DownStreamVQCombinedDeployModel,
  export_downstream_to_onnx,
)
from mjlab.tasks.motion_prior.rl.policies import DownStreamVQPolicy
from mjlab.tasks.motion_prior.rl.policies.quantizer import EMAQuantizer

PROP = 559
NUM_CODE = 64
CODE_DIM = 16
NUM_ACTIONS = 29
NUM_OBS = PROP + 3
NUM_PRIV = NUM_OBS + 3
B = 4


@pytest.fixture(scope="module")
def fake_vq_ckpt(tmp_path_factory: pytest.TempPathFactory) -> Path:
  """Build a fresh VQ motion-prior backbone, save it in our ckpt format."""
  ckpt_dir = tmp_path_factory.mktemp("vq_ckpt")
  ckpt_path = ckpt_dir / "fake_vq.pt"

  motion_prior = MLP(PROP, CODE_DIM, hidden_dims=(64, 32), activation="elu")
  decoder = MLP(PROP + CODE_DIM, NUM_ACTIONS, hidden_dims=(64, 32), activation="elu")
  quantizer = EMAQuantizer(num_code=NUM_CODE, code_dim=CODE_DIM)

  torch.save(
    {
      "motion_prior": motion_prior.state_dict(),
      "decoder": decoder.state_dict(),
      "quantizer": quantizer.state_dict(),
      "iter": 0,
      "infos": {},
    },
    ckpt_path,
  )
  return ckpt_path


@pytest.fixture(scope="module")
def vq_policy(fake_vq_ckpt: Path) -> DownStreamVQPolicy:
  return DownStreamVQPolicy(
    num_obs=NUM_OBS,
    num_actions=NUM_ACTIONS,
    num_privileged_obs=NUM_PRIV,
    prop_obs_dim=PROP,
    motion_prior_ckpt_path=fake_vq_ckpt,
    num_code=NUM_CODE,
    code_dim=CODE_DIM,
    motion_prior_hidden_dims=(64, 32),
    decoder_hidden_dims=(64, 32),
    actor_hidden_dims=(64, 32),
    critic_hidden_dims=(64, 32),
    device="cpu",
  )


# ---------------------------------------------------------------------------
# Policy structure
# ---------------------------------------------------------------------------


def test_vq_frozen_modules_are_frozen(vq_policy: DownStreamVQPolicy) -> None:
  for m in (vq_policy.motion_prior, vq_policy.quantizer, vq_policy.decoder):
    for name, p in m.named_parameters():
      assert not p.requires_grad, f"{m.__class__.__name__}.{name} should be frozen"


def test_vq_trainable_modules_are_trainable(vq_policy: DownStreamVQPolicy) -> None:
  for m in (vq_policy.actor, vq_policy.critic):
    for p in m.parameters():
      assert p.requires_grad
  assert vq_policy.std.requires_grad


def test_vq_act_returns_pair(vq_policy: DownStreamVQPolicy) -> None:
  po = torch.randn(B, NUM_OBS)
  pr = torch.randn(B, PROP)
  recons, raw = vq_policy.act(po, pr)
  assert recons.shape == (B, NUM_ACTIONS)
  assert raw.shape == (B, CODE_DIM)


def test_vq_grad_only_flows_to_trainable(vq_policy: DownStreamVQPolicy) -> None:
  vq_policy.zero_grad()
  po = torch.randn(B, NUM_OBS)
  pr = torch.randn(B, PROP)
  cr = torch.randn(B, NUM_PRIV)
  _, raw = vq_policy.act(po, pr)
  loss = vq_policy.get_actions_log_prob(raw).sum() + vq_policy.evaluate(cr).sum()
  loss.backward()

  def has_grad(m: torch.nn.Module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

  assert not has_grad(vq_policy.motion_prior)
  assert not has_grad(vq_policy.quantizer)
  assert not has_grad(vq_policy.decoder)
  assert has_grad(vq_policy.actor)
  assert has_grad(vq_policy.critic)


def test_vq_lab_toggles_clipping(fake_vq_ckpt: Path) -> None:
  """``use_lab=False`` ⇒ raw additive residual; ``True`` ⇒ tanh-clipped."""
  kwargs = dict(
    num_obs=NUM_OBS,
    num_actions=NUM_ACTIONS,
    num_privileged_obs=NUM_PRIV,
    prop_obs_dim=PROP,
    motion_prior_ckpt_path=fake_vq_ckpt,
    num_code=NUM_CODE,
    code_dim=CODE_DIM,
    motion_prior_hidden_dims=(64, 32),
    decoder_hidden_dims=(64, 32),
    actor_hidden_dims=(64, 32),
    critic_hidden_dims=(64, 32),
    device="cpu",
  )
  pol_no_lab = DownStreamVQPolicy(use_lab=False, **kwargs)
  pol_lab = DownStreamVQPolicy(use_lab=True, lab_lambda=3.0, **kwargs)

  prior = torch.randn(B, CODE_DIM)
  raw = torch.randn(B, CODE_DIM) * 100  # blow it up so tanh saturates
  no_lab_z = pol_no_lab._combine(prior, raw)
  lab_z = pol_lab._combine(prior, raw)
  assert torch.allclose(no_lab_z, prior + raw)
  # tanh saturates → |z - prior| ≤ λ + epsilon.
  assert (lab_z - prior).abs().max() <= 3.0 + 1e-5


# ---------------------------------------------------------------------------
# VQ runner integration (fake env)
# ---------------------------------------------------------------------------


class _FakeEnv:
  num_envs = 4
  num_actions = NUM_ACTIONS
  device = torch.device("cpu")

  def __init__(self) -> None:
    self.episode_length_buf = torch.zeros(4, dtype=torch.long)

  def get_observations(self) -> TensorDict:
    return TensorDict(
      {
        "policy": torch.randn(4, NUM_OBS),
        "motion_prior_obs": torch.randn(4, PROP),
        "critic": torch.randn(4, NUM_PRIV),
      },
      batch_size=[4],
    )

  def step(
    self, _: torch.Tensor
  ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
    self.episode_length_buf += 1
    rew = torch.zeros(4)
    done = (self.episode_length_buf % 5 == 0).long()
    self.episode_length_buf[done.bool()] = 0
    return self.get_observations(), rew, done, {}


def test_vq_runner_constructs_and_learns(fake_vq_ckpt: Path, tmp_path: Path) -> None:
  from mjlab.tasks.motion_prior.rl.downstream_runner import (
    DownStreamVQOnPolicyRunner,
  )

  env = _FakeEnv()
  cfg = {
    "num_steps_per_env": 4,
    "save_interval": 100,
    "motion_prior_ckpt_path": str(fake_vq_ckpt),
    "policy": {
      "num_code": NUM_CODE,
      "code_dim": CODE_DIM,
      "motion_prior_hidden_dims": (64, 32),
      "decoder_hidden_dims": (64, 32),
      "actor_hidden_dims": (64, 32),
      "critic_hidden_dims": (64, 32),
      "use_lab": True,
      "lab_lambda": 3.0,
    },
    "algorithm": {
      "num_learning_epochs": 1,
      "num_mini_batches": 1,
      "learning_rate": 1e-3,
    },
    "logger": "none",
  }
  runner = DownStreamVQOnPolicyRunner(
    env=env,  # type: ignore[arg-type]
    train_cfg=cfg,
    log_dir=str(tmp_path),
    device="cpu",
  )
  runner.learn(num_learning_iterations=2)
  assert runner.current_learning_iteration == 1
  assert isinstance(runner.policy, DownStreamVQPolicy)


# ---------------------------------------------------------------------------
# VQ ONNX export parity
# ---------------------------------------------------------------------------


def test_vq_combined_onnx_matches_pytorch(
  vq_policy: DownStreamVQPolicy, tmp_path: Path
) -> None:
  out = tmp_path / "vq_combined.onnx"
  export_downstream_to_onnx(vq_policy, out, mode="combined")
  assert out.is_file()

  m = DownStreamVQCombinedDeployModel(vq_policy).to("cpu").eval()
  gen = torch.Generator().manual_seed(7)
  prop = torch.randn(B, PROP, generator=gen)
  pol = torch.randn(B, NUM_OBS, generator=gen)
  with torch.no_grad():
    torch_out = m(prop, pol).cpu().numpy()

  sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
  ort_out = sess.run(None, {"prop_obs": prop.numpy(), "policy_obs": pol.numpy()})[0]
  np.testing.assert_allclose(torch_out, ort_out, atol=1e-5, rtol=1e-5)
