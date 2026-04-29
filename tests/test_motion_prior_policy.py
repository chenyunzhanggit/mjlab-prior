"""Smoke tests for ``MotionPriorPolicy`` (VAE dual-teacher).

These tests verify structural assembly, forward shapes, teacher freeze, and
gradient routing on CPU. The teacher checkpoints are required; tests skip
cleanly if either is missing so this file remains green on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.tasks.motion_prior.rl.policies import MotionPriorPolicy
from mjlab.tasks.motion_prior.teacher import (
  TELEOPIT_TEACHER_CFG,
  VELOCITY_TEACHER_CFG,
)

TEACHER_A_CKPT = Path("~/zcy/Teleopit/track.pt").expanduser()
TEACHER_B_CKPT = Path("~/zcy/mjlab-prior/logs/model_21000.pt").expanduser()

# Real student obs dim: 5 proprio terms × history_length=4.
PROP_OBS_DIM = (3 + 3 + 29 + 29 + 29) * 4
NUM_ACTIONS = 29
LATENT_Z_DIMS = 32
B = 4


def _ckpts_or_skip() -> tuple[Path, Path]:
  if not TEACHER_A_CKPT.is_file():
    pytest.skip(f"teacher_a checkpoint missing: {TEACHER_A_CKPT}")
  if not TEACHER_B_CKPT.is_file():
    pytest.skip(f"teacher_b checkpoint missing: {TEACHER_B_CKPT}")
  return TEACHER_A_CKPT, TEACHER_B_CKPT


@pytest.fixture(scope="module")
def policy() -> MotionPriorPolicy:
  a, b = _ckpts_or_skip()
  return MotionPriorPolicy(
    prop_obs_dim=PROP_OBS_DIM,
    num_actions=NUM_ACTIONS,
    teacher_a_policy_path=a,
    teacher_b_policy_path=b,
    latent_z_dims=LATENT_Z_DIMS,
    device="cpu",
  )


@pytest.fixture(scope="module")
def dummy_batch() -> dict[str, torch.Tensor]:
  gen = torch.Generator().manual_seed(7)
  return {
    "prop": torch.randn(B, PROP_OBS_DIM, generator=gen),
    "teacher_a": torch.randn(B, TELEOPIT_TEACHER_CFG.actor_obs_dim, generator=gen),
    "teacher_a_history": torch.randn(
      B,
      TELEOPIT_TEACHER_CFG.actor_history_length,
      TELEOPIT_TEACHER_CFG.actor_history_obs_dim,
      generator=gen,
    ),
    "teacher_b": torch.randn(B, VELOCITY_TEACHER_CFG.actor_obs_dim, generator=gen),
  }


def test_forward_a_shapes(policy, dummy_batch) -> None:
  enc_mu, enc_lv, z, act, mp_mu, mp_lv = policy.forward_a(
    dummy_batch["prop"], dummy_batch["teacher_a"]
  )
  for t in (enc_mu, enc_lv, z, mp_mu, mp_lv):
    assert t.shape == (B, LATENT_Z_DIMS)
  assert act.shape == (B, NUM_ACTIONS)


def test_forward_b_shapes(policy, dummy_batch) -> None:
  enc_mu, enc_lv, z, act, mp_mu, mp_lv = policy.forward_b(
    dummy_batch["prop"], dummy_batch["teacher_b"]
  )
  for t in (enc_mu, enc_lv, z, mp_mu, mp_lv):
    assert t.shape == (B, LATENT_Z_DIMS)
  assert act.shape == (B, NUM_ACTIONS)


def test_teachers_are_frozen(policy) -> None:
  for name, p in policy.teacher_a.named_parameters():
    assert not p.requires_grad, f"teacher_a param {name} not frozen"
  for name, p in policy.teacher_b.named_parameters():
    assert not p.requires_grad, f"teacher_b param {name} not frozen"
  assert policy.teacher_a.training is False
  assert policy.teacher_b.training is False


def test_evaluate_outputs_are_no_grad(policy, dummy_batch) -> None:
  act_a = policy.evaluate_a(dummy_batch["teacher_a"], dummy_batch["teacher_a_history"])
  act_b = policy.evaluate_b(dummy_batch["teacher_b"])
  assert act_a.shape == (B, NUM_ACTIONS)
  assert act_b.shape == (B, NUM_ACTIONS)
  assert act_a.requires_grad is False
  assert act_b.requires_grad is False


def test_inference_paths(policy, dummy_batch) -> None:
  out_a = policy.policy_inference_a(dummy_batch["prop"], dummy_batch["teacher_a"])
  out_b = policy.policy_inference_b(dummy_batch["prop"], dummy_batch["teacher_b"])
  mp_mu = policy.motion_prior_inference(dummy_batch["prop"])
  assert out_a.shape == out_b.shape == (B, NUM_ACTIONS)
  assert mp_mu.shape == (B, LATENT_Z_DIMS)


def test_backward_only_updates_trainable_modules(policy, dummy_batch) -> None:
  policy.zero_grad()
  _, _, _, sa_a, _, _ = policy.forward_a(dummy_batch["prop"], dummy_batch["teacher_a"])
  _, _, _, sa_b, _, _ = policy.forward_b(dummy_batch["prop"], dummy_batch["teacher_b"])
  target = torch.zeros_like(sa_a)
  loss = (sa_a - target).pow(2).mean() + (sa_b - target).pow(2).mean()
  loss.backward()

  # Frozen teachers must remain grad-free.
  for name, p in policy.teacher_a.named_parameters():
    assert p.grad is None, f"teacher_a {name} got grad"
  for name, p in policy.teacher_b.named_parameters():
    assert p.grad is None, f"teacher_b {name} got grad"

  # Encoders + decoder were on the path; expect non-zero grads.
  for name, p in policy.encoder_a.named_parameters():
    assert p.grad is not None, f"encoder_a {name} missing grad"
    assert p.grad.abs().sum() > 0, f"encoder_a {name} zero grad"
  for name, p in policy.encoder_b.named_parameters():
    assert p.grad is not None, f"encoder_b {name} missing grad"
    assert p.grad.abs().sum() > 0, f"encoder_b {name} zero grad"
  for name, p in policy.decoder.named_parameters():
    assert p.grad is not None, f"decoder {name} missing grad"
    assert p.grad.abs().sum() > 0, f"decoder {name} zero grad"


def test_motion_prior_head_grad_when_used(policy, dummy_batch) -> None:
  policy.zero_grad()
  mp_mu, mp_lv = policy.motion_prior_head(dummy_batch["prop"])
  loss = mp_mu.pow(2).sum() + mp_lv.pow(2).sum()
  loss.backward()
  for name, p in policy.motion_prior.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name
  for name, p in policy.mp_mu.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name
  for name, p in policy.mp_var.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_reparameterize_is_stochastic(policy) -> None:
  mu = torch.zeros(B, LATENT_Z_DIMS)
  log_var = torch.zeros(B, LATENT_Z_DIMS)  # std = 1
  z1 = policy.reparameterize(mu, log_var)
  z2 = policy.reparameterize(mu, log_var)
  assert not torch.allclose(z1, z2)
  # eval-mean: latent should be near 0 in expectation
  zs = torch.stack([policy.reparameterize(mu, log_var) for _ in range(64)], dim=0)
  assert zs.mean().abs() < 0.5
