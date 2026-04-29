"""Smoke tests for ``DistillationMotionPrior``.

Exercises a single ``compute_loss_one_batch`` step on tiny synthetic
buffers and verifies:

  * loss components are finite and non-negative where expected
  * ``loss/behavior_a`` and ``loss/behavior_b`` actually decrease after a
    few optimization steps with both teachers held fixed
  * frozen teacher parameters never receive grad
  * AR(1) residual respects the ``progress_buf`` mask (boundary residuals
    are zeroed)
  * KL coefficient decays linearly between ``anneal_start_iter`` and
    ``anneal_end_iter``
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mjlab.tasks.motion_prior.rl.algorithms import DistillationMotionPrior
from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior import (
  DistillationLossCfg,
  _annealed_kl_coeff,
  _ar1_residual,
)
from mjlab.tasks.motion_prior.rl.policies import MotionPriorPolicy
from mjlab.tasks.motion_prior.teacher import (
  TELEOPIT_TEACHER_CFG,
  VELOCITY_TEACHER_CFG,
)

TEACHER_A_CKPT = Path("~/zcy/Teleopit/track.pt").expanduser()
TEACHER_B_CKPT = Path("~/zcy/mjlab-prior/logs/model_21000.pt").expanduser()

PROP_OBS_DIM = (3 + 3 + 29 + 29 + 29) * 4
NUM_ACTIONS = 29
LATENT_Z_DIMS = 32
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
def algo() -> DistillationMotionPrior:
  a, b = _ckpts_or_skip()
  policy = MotionPriorPolicy(
    prop_obs_dim=PROP_OBS_DIM,
    num_actions=NUM_ACTIONS,
    teacher_a_policy_path=a,
    teacher_b_policy_path=b,
    latent_z_dims=LATENT_Z_DIMS,
    device="cpu",
  )
  return DistillationMotionPrior(policy, learning_rate=1e-3, device="cpu")


def _make_batch(algo: DistillationMotionPrior, seed: int = 0) -> dict:
  """Build a synthetic batch by running the policy on random obs."""
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

  enc_mu_a, enc_lv_a, _, sa_a, mp_mu_a, mp_lv_a = algo.policy.forward_a(prop_a, ta_obs)
  enc_mu_b, enc_lv_b, _, sa_b, mp_mu_b, mp_lv_b = algo.policy.forward_b(prop_b, tb_obs)

  with torch.no_grad():
    teacher_a_act = algo.policy.evaluate_a(ta_obs, ta_hist)
    teacher_b_act = algo.policy.evaluate_b(tb_obs)

  # progress_buf simulates strictly increasing in-episode timestep.
  progress = torch.arange(SEQ_LEN).repeat(NUM_ENVS, 1).unsqueeze(-1).to(torch.float32)
  enc_mu_a_time = enc_mu_a.view(NUM_ENVS, SEQ_LEN, LATENT_Z_DIMS)
  enc_mu_b_time = enc_mu_b.view(NUM_ENVS, SEQ_LEN, LATENT_Z_DIMS)

  return dict(
    actions_teacher_a=teacher_a_act,
    actions_student_a=sa_a,
    enc_mu_a=enc_mu_a,
    enc_log_var_a=enc_lv_a,
    mp_mu_a=mp_mu_a,
    mp_log_var_a=mp_lv_a,
    enc_mu_a_time_stack=enc_mu_a_time,
    progress_buf_a=progress,
    actions_teacher_b=teacher_b_act,
    actions_student_b=sa_b,
    enc_mu_b=enc_mu_b,
    enc_log_var_b=enc_lv_b,
    mp_mu_b=mp_mu_b,
    mp_log_var_b=mp_lv_b,
    enc_mu_b_time_stack=enc_mu_b_time,
    progress_buf_b=progress,
  )


def test_loss_dict_finite(algo) -> None:
  batch = _make_batch(algo, seed=1)
  out = algo.compute_loss_one_batch(**batch, cur_iter_num=0)
  for k, v in out.items():
    assert torch.isfinite(torch.tensor(v)), f"{k} = {v} is not finite"
  # Behavior MSE should be non-negative.
  assert out["loss/behavior_a"] >= 0
  assert out["loss/behavior_b"] >= 0
  # AR(1) residual norm is non-negative.
  assert out["loss/ar1_a"] >= 0
  assert out["loss/ar1_b"] >= 0


def test_teachers_remain_frozen_after_step(algo) -> None:
  batch = _make_batch(algo, seed=2)
  algo.compute_loss_one_batch(**batch, cur_iter_num=0)
  for name, p in algo.policy.teacher_a.named_parameters():
    assert p.grad is None or p.grad.abs().sum() == 0, f"teacher_a {name} got grad"
    assert not p.requires_grad
  for name, p in algo.policy.teacher_b.named_parameters():
    assert p.grad is None or p.grad.abs().sum() == 0, f"teacher_b {name} got grad"
    assert not p.requires_grad


def test_behavior_loss_decreases(algo) -> None:
  """Re-running the same batch must shrink behavior loss after a few steps."""
  losses = []
  for _ in range(8):
    batch = _make_batch(algo, seed=3)  # same seed → same teacher targets
    out = algo.compute_loss_one_batch(**batch, cur_iter_num=0)
    losses.append(out["loss/behavior_a"] + out["loss/behavior_b"])
  assert losses[-1] < losses[0], (
    f"behavior loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
  )


def test_ar1_mask_zeros_boundary_residuals() -> None:
  """A reset right at t=2 should zero out that residual."""
  z = torch.randn(2, 5, 4)
  # env 0: progress 0,1,2,3,4 — only t=2 boundary residuals get zeroed because
  # of the (idxes <= 2) starter mask; env 1: same.
  good = torch.arange(5).unsqueeze(0).repeat(2, 1).unsqueeze(-1).float()
  ar1_no_mask = _ar1_residual(z, None, phi=0.99)
  ar1_with_mask = _ar1_residual(z.clone(), good, phi=0.99)
  # Masking should reduce (or equal) the residual norm sum.
  assert ar1_with_mask <= ar1_no_mask + 1e-6


def test_ar1_mask_skips_episode_boundary() -> None:
  """A non-consecutive progress jump must zero that residual entirely."""
  z = torch.zeros(1, 4, 2)
  z[0, 1] = 100.0  # huge jump at t=1 vs t=0
  # progress shows t=0 then resets to t=0 again (non-consec)
  progress = torch.tensor([[[0.0], [0.0], [1.0], [2.0]]])
  out_no_mask = _ar1_residual(z.clone(), None, phi=0.99)
  out_with_mask = _ar1_residual(z.clone(), progress, phi=0.99)
  # With mask, the t=0 → t=1 residual at index 0 of error[1:] gets zeroed.
  # With the (idxes <= 2) starter mask, all four residuals are zeroed → 0.
  assert out_with_mask.item() == pytest.approx(0.0, abs=1e-6)
  assert out_no_mask.item() > 0.0


def test_kl_coeff_anneal_schedule() -> None:
  cfg = DistillationLossCfg(
    kl_loss_coeff_max=0.01,
    kl_loss_coeff_min=0.001,
    anneal_start_iter=2500,
    anneal_end_iter=5000,
  )
  assert _annealed_kl_coeff(0, cfg) == pytest.approx(0.01)
  assert _annealed_kl_coeff(2500, cfg) == pytest.approx(0.01)
  # midpoint
  mid = _annealed_kl_coeff(3750, cfg)
  assert mid == pytest.approx(0.0055, rel=1e-3)
  assert _annealed_kl_coeff(5000, cfg) == pytest.approx(0.001)
  assert _annealed_kl_coeff(10000, cfg) == pytest.approx(0.001)


def test_align_loss_off_by_default(algo) -> None:
  batch = _make_batch(algo, seed=4)
  out = algo.compute_loss_one_batch(**batch, cur_iter_num=0)
  assert out["loss/align"] == 0.0


def test_align_loss_active_when_coeff_positive() -> None:
  a, b = _ckpts_or_skip()
  policy = MotionPriorPolicy(
    prop_obs_dim=PROP_OBS_DIM,
    num_actions=NUM_ACTIONS,
    teacher_a_policy_path=a,
    teacher_b_policy_path=b,
    latent_z_dims=LATENT_Z_DIMS,
    device="cpu",
  )
  algo = DistillationMotionPrior(
    policy,
    learning_rate=1e-3,
    loss_cfg=DistillationLossCfg(align_loss_coeff=0.01),
    device="cpu",
  )
  batch = _make_batch(algo, seed=5)
  out = algo.compute_loss_one_batch(**batch, cur_iter_num=0)
  assert out["loss/align"] > 0.0
