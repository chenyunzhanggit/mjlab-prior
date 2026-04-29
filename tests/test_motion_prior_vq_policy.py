"""Smoke tests for ``MotionPriorVQPolicy`` and the ``EMAQuantizer``.

Quantizer tests run on synthetic data alone (no teacher ckpt needed).
Policy-level tests skip cleanly if the teacher checkpoints are missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.tasks.motion_prior.rl.policies import (
  EMAQuantizer,
  MotionPriorVQPolicy,
)
from mjlab.tasks.motion_prior.teacher import (
  TELEOPIT_TEACHER_CFG,
  VELOCITY_TEACHER_CFG,
)

TEACHER_A_CKPT = Path("~/zcy/Teleopit/track.pt").expanduser()
TEACHER_B_CKPT = Path("~/zcy/mjlab-prior/logs/model_21000.pt").expanduser()

PROP_OBS_DIM = (3 + 3 + 29 + 29 + 29) * 4
NUM_ACTIONS = 29
NUM_CODE = 64
CODE_DIM = 16
B = 8


def _ckpts_or_skip() -> tuple[Path, Path]:
  if not TEACHER_A_CKPT.is_file():
    pytest.skip(f"teacher_a checkpoint missing: {TEACHER_A_CKPT}")
  if not TEACHER_B_CKPT.is_file():
    pytest.skip(f"teacher_b checkpoint missing: {TEACHER_B_CKPT}")
  return TEACHER_A_CKPT, TEACHER_B_CKPT


# ---------------------------------------------------------------------------
# EMAQuantizer
# ---------------------------------------------------------------------------


def test_quantizer_init_shapes() -> None:
  q = EMAQuantizer(num_code=NUM_CODE, code_dim=CODE_DIM)
  assert q.codebook.shape == (NUM_CODE, CODE_DIM)
  assert q.code_sum.shape == (NUM_CODE, CODE_DIM)
  assert q.code_count.shape == (NUM_CODE,)
  # Buffers, not parameters — and roughly unit-norm sphere with 1/sqrt(D) scale.
  assert all(p.numel() == 0 or False for p in q.parameters())  # no Parameters
  norms = q.codebook.norm(dim=1)
  expected = 1.0 / (CODE_DIM**0.5)
  assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-5)


def test_quantizer_inference_returns_no_commit_loss() -> None:
  q = EMAQuantizer(num_code=NUM_CODE, code_dim=CODE_DIM)
  x = torch.randn(B, CODE_DIM)
  x_d, commit_loss, perplexity = q(x, training=False)
  assert x_d.shape == x.shape
  assert commit_loss is None
  assert perplexity > 0


def test_quantizer_training_returns_commit_loss_and_updates_codebook() -> None:
  q = EMAQuantizer(num_code=NUM_CODE, code_dim=CODE_DIM, ema_decay=0.5)
  before = q.codebook.detach().clone()
  x = torch.randn(B * 4, CODE_DIM)
  x_d, commit_loss, _ = q(x, training=True)
  assert commit_loss is not None and commit_loss.item() >= 0
  assert x_d.shape == x.shape
  # Codebook drifted after the EMA update.
  assert not torch.allclose(before, q.codebook)


def test_quantizer_straight_through_grad_flows_to_input() -> None:
  q = EMAQuantizer(num_code=NUM_CODE, code_dim=CODE_DIM)
  x = torch.randn(B, CODE_DIM, requires_grad=True)
  x_d, _, _ = q(x, training=True)
  loss = x_d.pow(2).sum()
  loss.backward()
  assert x.grad is not None
  assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# MotionPriorVQPolicy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vq_policy() -> MotionPriorVQPolicy:
  a, b = _ckpts_or_skip()
  return MotionPriorVQPolicy(
    prop_obs_dim=PROP_OBS_DIM,
    num_actions=NUM_ACTIONS,
    teacher_a_policy_path=a,
    teacher_b_policy_path=b,
    num_code=NUM_CODE,
    code_dim=CODE_DIM,
    encoder_hidden_dims=(64, 32),
    decoder_hidden_dims=(64, 32),
    motion_prior_hidden_dims=(64, 32),
    device="cpu",
  )


@pytest.fixture(scope="module")
def dummy_batch() -> dict[str, torch.Tensor]:
  gen = torch.Generator().manual_seed(11)
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


def test_vq_forward_a_shapes(vq_policy, dummy_batch) -> None:
  act, q, enc, mp_code, commit, ppl = vq_policy.forward_a(
    dummy_batch["prop"], dummy_batch["teacher_a"], training=True
  )
  assert act.shape == (B, NUM_ACTIONS)
  assert q.shape == (B, CODE_DIM)
  assert enc.shape == (B, CODE_DIM)
  assert mp_code.shape == (B, CODE_DIM)
  assert commit is not None
  assert ppl > 0


def test_vq_forward_b_shapes(vq_policy, dummy_batch) -> None:
  act, q, enc, mp_code, commit, ppl = vq_policy.forward_b(
    dummy_batch["prop"], dummy_batch["teacher_b"], training=True
  )
  assert act.shape == (B, NUM_ACTIONS)
  assert q.shape == (B, CODE_DIM)
  assert enc.shape == (B, CODE_DIM)
  assert mp_code.shape == (B, CODE_DIM)
  assert commit is not None
  assert ppl > 0


def test_vq_inference_paths_have_no_commit_loss(vq_policy, dummy_batch) -> None:
  act_a, _, _, _, ca, _ = vq_policy.forward_a(
    dummy_batch["prop"], dummy_batch["teacher_a"], training=False
  )
  act_b, _, _, _, cb, _ = vq_policy.forward_b(
    dummy_batch["prop"], dummy_batch["teacher_b"], training=False
  )
  assert act_a.shape == (B, NUM_ACTIONS)
  assert act_b.shape == (B, NUM_ACTIONS)
  assert ca is None and cb is None
  # policy_inference convenience wrappers must agree.
  pa = vq_policy.policy_inference_a(dummy_batch["prop"], dummy_batch["teacher_a"])
  pb = vq_policy.policy_inference_b(dummy_batch["prop"], dummy_batch["teacher_b"])
  assert pa.shape == pb.shape == (B, NUM_ACTIONS)


def test_vq_teachers_are_frozen(vq_policy) -> None:
  for name, p in vq_policy.teacher_a.named_parameters():
    assert not p.requires_grad, name
  for name, p in vq_policy.teacher_b.named_parameters():
    assert not p.requires_grad, name
  assert vq_policy.teacher_a.training is False
  assert vq_policy.teacher_b.training is False


def test_vq_evaluate_outputs_no_grad(vq_policy, dummy_batch) -> None:
  act_a = vq_policy.evaluate_a(
    dummy_batch["teacher_a"], dummy_batch["teacher_a_history"]
  )
  act_b = vq_policy.evaluate_b(dummy_batch["teacher_b"])
  assert act_a.shape == (B, NUM_ACTIONS)
  assert act_b.shape == (B, NUM_ACTIONS)
  assert not act_a.requires_grad
  assert not act_b.requires_grad


def test_vq_backward_only_updates_trainable_modules(vq_policy, dummy_batch) -> None:
  vq_policy.zero_grad()
  act_a, _, _, _, _, _ = vq_policy.forward_a(
    dummy_batch["prop"], dummy_batch["teacher_a"], training=True
  )
  act_b, _, _, _, _, _ = vq_policy.forward_b(
    dummy_batch["prop"], dummy_batch["teacher_b"], training=True
  )
  target = torch.zeros_like(act_a)
  ((act_a - target).pow(2).mean() + (act_b - target).pow(2).mean()).backward()
  for name, p in vq_policy.teacher_a.named_parameters():
    assert p.grad is None, f"teacher_a {name} got grad"
  for name, p in vq_policy.teacher_b.named_parameters():
    assert p.grad is None, f"teacher_b {name} got grad"
  for name, p in vq_policy.encoder_a.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name
  for name, p in vq_policy.encoder_b.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name
  for name, p in vq_policy.decoder.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_vq_codebook_in_state_dict(vq_policy) -> None:
  """Buffer registration must persist codebook + EMA stats in state_dict."""
  sd = vq_policy.state_dict()
  assert "quantizer.codebook" in sd
  assert "quantizer.code_sum" in sd
  assert "quantizer.code_count" in sd
  assert sd["quantizer.codebook"].shape == (NUM_CODE, CODE_DIM)
