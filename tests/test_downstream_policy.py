"""Smoke tests for ``DownStreamPolicy`` (frozen mp + decoder + trainable head).

Loads a real ``MotionPriorOnPolicyRunner`` checkpoint to validate the
backbone load path; tests skip cleanly if no ckpt is around.
"""

from __future__ import annotations

import glob
from pathlib import Path

import pytest
import torch

from mjlab.tasks.motion_prior.rl.policies import DownStreamPolicy
from mjlab.tasks.motion_prior.teacher import load_motion_prior_components

PROP = (
  559  # student obs dim from current motion_prior training (proprio + height_scan).
)
LATENT = 32
NUM_ACTIONS = 29
NUM_OBS = PROP + 3
NUM_PRIV = NUM_OBS + 3
B = 4


def _find_motion_prior_ckpt() -> Path | None:
  """Pick the most recent local motion-prior ckpt, if one exists."""
  pattern = str(
    Path("~/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/*/model_*.pt").expanduser()
  )
  matches = sorted(glob.glob(pattern))
  return Path(matches[-1]) if matches else None


@pytest.fixture(scope="module")
def ckpt_path() -> Path:
  p = _find_motion_prior_ckpt()
  if p is None:
    pytest.skip("no motion_prior ckpt under ~/zcy/mjlab-prior/logs/...")
  return p


@pytest.fixture(scope="module")
def policy(ckpt_path: Path) -> DownStreamPolicy:
  return DownStreamPolicy(
    num_obs=NUM_OBS,
    num_actions=NUM_ACTIONS,
    num_privileged_obs=NUM_PRIV,
    prop_obs_dim=PROP,
    motion_prior_ckpt_path=ckpt_path,
    latent_z_dims=LATENT,
    device="cpu",
  )


def test_ckpt_loader_returns_three_components(ckpt_path: Path) -> None:
  parts = load_motion_prior_components(ckpt_path)
  assert set(parts.keys()) == {"decoder", "motion_prior", "mp_mu"}
  for v in parts.values():
    assert isinstance(v, dict) and len(v) > 0


def test_frozen_modules_are_frozen(policy: DownStreamPolicy) -> None:
  for m in (policy.motion_prior, policy.mp_mu, policy.decoder):
    for name, p in m.named_parameters():
      assert not p.requires_grad, f"{m.__class__.__name__}.{name} should be frozen"
    assert m.training is False


def test_trainable_modules_are_trainable(policy: DownStreamPolicy) -> None:
  for m in (policy.actor, policy.critic):
    for name, p in m.named_parameters():
      assert p.requires_grad, f"{m.__class__.__name__}.{name} should be trainable"
  assert policy.std.requires_grad


def test_act_returns_action_pair(policy: DownStreamPolicy) -> None:
  po = torch.randn(B, NUM_OBS)
  pr = torch.randn(B, PROP)
  recons, raw = policy.act(po, pr)
  assert recons.shape == (B, NUM_ACTIONS)
  assert raw.shape == (B, LATENT)


def test_policy_inference_is_deterministic(policy: DownStreamPolicy) -> None:
  po = torch.randn(B, NUM_OBS)
  pr = torch.randn(B, PROP)
  out1 = policy.policy_inference(po, pr)
  out2 = policy.policy_inference(po, pr)
  assert torch.allclose(out1, out2)


def test_evaluate_returns_value(policy: DownStreamPolicy) -> None:
  cr = torch.randn(B, NUM_PRIV)
  v = policy.evaluate(cr)
  assert v.shape == (B, 1)


def test_grad_only_flows_to_trainable(policy: DownStreamPolicy) -> None:
  policy.zero_grad()
  po = torch.randn(B, NUM_OBS)
  pr = torch.randn(B, PROP)
  cr = torch.randn(B, NUM_PRIV)
  _, raw = policy.act(po, pr)
  log_prob = policy.get_actions_log_prob(raw)
  value = policy.evaluate(cr)
  (log_prob.sum() + value.sum()).backward()

  def has_grad(m: torch.nn.Module) -> bool:
    return any(
      par.grad is not None and par.grad.abs().sum() > 0 for par in m.parameters()
    )

  assert not has_grad(policy.motion_prior)
  assert not has_grad(policy.mp_mu)
  assert not has_grad(policy.decoder)
  assert has_grad(policy.actor)
  assert has_grad(policy.critic)
  assert policy.std.grad is not None and policy.std.grad.abs().sum() > 0


def test_unknown_ckpt_path_raises(tmp_path: Path) -> None:
  with pytest.raises(FileNotFoundError):
    DownStreamPolicy(
      num_obs=NUM_OBS,
      num_actions=NUM_ACTIONS,
      num_privileged_obs=NUM_PRIV,
      prop_obs_dim=PROP,
      motion_prior_ckpt_path=tmp_path / "does_not_exist.pt",
      device="cpu",
    )
