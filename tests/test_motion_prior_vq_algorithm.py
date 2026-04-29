"""Smoke tests for ``DistillationMotionPriorVQ``.

Mirrors ``test_motion_prior_algorithm.py`` for the VQ branch:

  * loss components are finite and non-negative where expected,
  * frozen teachers stay frozen across optimizer steps,
  * behavior + commit + mp loss decreases when re-fed the same batch,
  * commit_loss is MSE between encoder output and detached quantized lookup,
  * mp_loss is MSE between motion_prior output and detached quantized code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.tasks.motion_prior.rl.algorithms import DistillationMotionPriorVQ
from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior_vq import (
  DistillationVQLossCfg,
)
from mjlab.tasks.motion_prior.rl.policies import MotionPriorVQPolicy
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
NUM_ENVS = 8
SEQ_LEN = 6
B = NUM_ENVS * SEQ_LEN


def _ckpts_or_skip() -> tuple[Path, Path]:
  if not TEACHER_A_CKPT.is_file():
    pytest.skip(f"teacher_a checkpoint missing: {TEACHER_A_CKPT}")
  if not TEACHER_B_CKPT.is_file():
    pytest.skip(f"teacher_b checkpoint missing: {TEACHER_B_CKPT}")
  return TEACHER_A_CKPT, TEACHER_B_CKPT


@pytest.fixture(scope="module")
def algo() -> DistillationMotionPriorVQ:
  a, b = _ckpts_or_skip()
  policy = MotionPriorVQPolicy(
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
  return DistillationMotionPriorVQ(policy, learning_rate=1e-3, device="cpu")


def _make_batch(algo: DistillationMotionPriorVQ, seed: int = 0) -> dict:
  gen = torch.Generator().manual_seed(seed)
  prop_a = torch.randn(B, PROP_OBS_DIM, generator=gen)
  prop_b = torch.randn(B, PROP_OBS_DIM, generator=gen)
  ta_obs = torch.randn(B, TELEOPIT_TEACHER_CFG.actor_obs_dim, generator=gen)
  ta_hist = torch.randn(
    B,
    TELEOPIT_TEACHER_CFG.actor_history_length,
    TELEOPIT_TEACHER_CFG.actor_history_obs_dim,
    generator=gen,
  )
  tb_obs = torch.randn(B, VELOCITY_TEACHER_CFG.actor_obs_dim, generator=gen)

  sa_a, q_a, enc_a, mp_a, commit_a, ppl_a = algo.policy.forward_a(
    prop_a, ta_obs, training=True
  )
  sa_b, q_b, enc_b, mp_b, commit_b, ppl_b = algo.policy.forward_b(
    prop_b, tb_obs, training=True
  )
  with torch.no_grad():
    teacher_a_act = algo.policy.evaluate_a(ta_obs, ta_hist)
    teacher_b_act = algo.policy.evaluate_b(tb_obs)

  enc_a_time = enc_a.view(NUM_ENVS, SEQ_LEN, CODE_DIM)
  enc_b_time = enc_b.view(NUM_ENVS, SEQ_LEN, CODE_DIM)

  return dict(
    actions_teacher_a=teacher_a_act,
    actions_student_a=sa_a,
    enc_a=enc_a,
    q_a=q_a,
    mp_code_a=mp_a,
    commit_a=commit_a,
    perplexity_a=ppl_a,
    enc_a_time_stack=enc_a_time,
    progress_buf_a=None,
    actions_teacher_b=teacher_b_act,
    actions_student_b=sa_b,
    enc_b=enc_b,
    q_b=q_b,
    mp_code_b=mp_b,
    commit_b=commit_b,
    perplexity_b=ppl_b,
    enc_b_time_stack=enc_b_time,
    progress_buf_b=None,
  )


def test_loss_dict_finite(algo) -> None:
  batch = _make_batch(algo, seed=1)
  out = algo.compute_loss_one_batch(**batch)
  for k, v in out.items():
    assert torch.isfinite(torch.tensor(v)), f"{k} = {v} is not finite"
  assert out["loss/behavior_a"] >= 0
  assert out["loss/behavior_b"] >= 0
  assert out["loss/commit_a"] >= 0
  assert out["loss/commit_b"] >= 0
  assert out["loss/mp"] >= 0
  assert out["perplexity_a"] > 0
  assert out["perplexity_b"] > 0


def test_teachers_remain_frozen_after_step(algo) -> None:
  batch = _make_batch(algo, seed=2)
  algo.compute_loss_one_batch(**batch)
  for name, p in algo.policy.teacher_a.named_parameters():
    assert p.grad is None or p.grad.abs().sum() == 0, f"teacher_a {name} got grad"
    assert not p.requires_grad
  for name, p in algo.policy.teacher_b.named_parameters():
    assert p.grad is None or p.grad.abs().sum() == 0, f"teacher_b {name} got grad"
    assert not p.requires_grad


def test_total_loss_decreases_on_repeat_batch(algo) -> None:
  losses = []
  for _ in range(8):
    batch = _make_batch(algo, seed=3)
    out = algo.compute_loss_one_batch(**batch)
    losses.append(out["loss/total"])
  assert losses[-1] < losses[0], (
    f"total loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
  )


def test_commit_loss_matches_encoder_to_quantized_mse(algo) -> None:
  """``commit_loss`` from the quantizer must equal MSE(enc, q.detach())."""
  batch = _make_batch(algo, seed=4)
  manual = torch.nn.functional.mse_loss(batch["enc_a"], batch["q_a"].detach())
  assert torch.allclose(batch["commit_a"], manual, atol=1e-6)
  manual = torch.nn.functional.mse_loss(batch["enc_b"], batch["q_b"].detach())
  assert torch.allclose(batch["commit_b"], manual, atol=1e-6)


def test_mp_loss_target_is_detached_quantized_code(algo) -> None:
  """When mp_loss_coeff is the only nonzero term, only motion_prior gets grad."""
  algo_iso = DistillationMotionPriorVQ(
    algo.policy,
    learning_rate=1e-3,
    loss_cfg=DistillationVQLossCfg(
      behavior_weight_a=0.0,
      behavior_weight_b=0.0,
      mu_regu_loss_coeff=0.0,
      commit_loss_coeff=0.0,
      mp_loss_coeff=1.0,
    ),
    device="cpu",
  )
  algo.policy.zero_grad()
  batch = _make_batch(algo, seed=5)
  algo_iso.compute_loss_one_batch(**batch)
  # motion_prior received grad, encoders did not.
  for name, p in algo.policy.motion_prior.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name
  for name, p in algo.policy.encoder_a.named_parameters():
    assert p.grad is None or p.grad.abs().sum() == 0, name


def test_align_loss_field_absent() -> None:
  """Sanity: VQ algorithm has no ``align_loss`` knob (codebook handles it)."""
  cfg = DistillationVQLossCfg()
  assert not hasattr(cfg, "align_loss_coeff")
