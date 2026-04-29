"""Dual-teacher VAE distillation algorithm.

Generalizes ``motionprior/.../distillation_motion_prior.py`` to two
teachers: each contributes its own behavior MSE, AR(1) latent smoothing,
and KL-to-motion-prior term. An optional ``align_loss`` (off by default)
nudges the two encoders toward matching first-moment statistics so the
motion_prior head sees a coherent latent space.

Loss layout::

  behavior = w_a · MSE(student_a, teacher_a) + w_b · MSE(student_b, teacher_b)
  ar1      = mu_regu_loss_coeff · (||Δz_a||  +  ||Δz_b||)
  kl       = kl_coeff · ( KL(N(μ_a,σ_a²) ‖ N(μ_mp_a,σ_mp_a²))
                        + KL(N(μ_b,σ_b²) ‖ N(μ_mp_b,σ_mp_b²)) )
  align    = align_loss_coeff · ( ‖mean(μ_a) - mean(μ_b)‖²
                                + ‖mean(logσ_a²) - mean(logσ_b²)‖² )
  total    = behavior + ar1 + kl + align

Per-submodule gradient clipping is applied to encoders, reparameterization
heads, the shared decoder, and the motion_prior head — matching the
reference implementation. Frozen teachers are skipped automatically since
they have no grad.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.tasks.motion_prior.rl.policies.motion_prior_policy import MotionPriorPolicy


@dataclass
class DistillationLossCfg:
  """Coefficients and schedule for distillation loss components."""

  loss_type: str = "mse"  # {"mse", "huber"}
  behavior_weight_a: float = 1.0
  behavior_weight_b: float = 1.0
  mu_regu_loss_coeff: float = 0.01
  ar1_phi: float = 0.99
  kl_loss_coeff_max: float = 0.01
  kl_loss_coeff_min: float = 0.001
  anneal_start_iter: int = 2500
  anneal_end_iter: int = 5000
  align_loss_coeff: float = 0.0


def _kl_two_gaussians(
  enc_mu: torch.Tensor,
  enc_log_var: torch.Tensor,
  mp_mu: torch.Tensor,
  mp_log_var: torch.Tensor,
) -> torch.Tensor:
  """Mean elementwise KL(N(enc) ‖ N(mp)). Same form as the reference impl."""
  return 0.5 * torch.mean(
    mp_log_var
    - enc_log_var
    + (torch.exp(enc_log_var) + (enc_mu - mp_mu) ** 2) / torch.exp(mp_log_var)
    - 1
  )


def _ar1_residual(
  z_time_stack: torch.Tensor,
  progress_buf: torch.Tensor | None,
  phi: float,
) -> torch.Tensor:
  """AR(1) residual ``z_t - φ z_{t-1}`` flat-norm mean.

  When ``progress_buf`` is supplied, residuals across episode boundaries
  (non-consecutive progress, or progress ≤ 2 right after a reset) are
  zeroed so the loss only enforces smoothness *within* an episode.
  """
  num_envs, horizon, _ = z_time_stack.shape
  error = z_time_stack[:, 1:] - z_time_stack[:, :-1] * phi
  if progress_buf is not None:
    idxes = progress_buf.view(num_envs, horizon, -1)
    not_consecs = ((idxes[:, 1:] - idxes[:, :-1]) != 1).view(-1)
    error = error.reshape(-1, error.shape[-1])
    error[not_consecs] = 0
    starteres = ((idxes <= 2)[:, 1:] + (idxes <= 2)[:, :-1]).view(-1)
    error[starteres] = 0
  else:
    error = error.reshape(-1, error.shape[-1])
  return torch.norm(error, dim=-1).mean()


def _annealed_kl_coeff(cur_iter: int, cfg: DistillationLossCfg) -> float:
  if cur_iter <= cfg.anneal_start_iter:
    return cfg.kl_loss_coeff_max
  span = max(cfg.anneal_end_iter - cfg.anneal_start_iter, 1)
  decayed = (cfg.kl_loss_coeff_max - cfg.kl_loss_coeff_min) * max(
    (cfg.anneal_end_iter - cur_iter) / span, 0.0
  ) + cfg.kl_loss_coeff_min
  return decayed


class DistillationMotionPrior:
  """VAE distillation algorithm with two frozen teachers and a shared decoder."""

  def __init__(
    self,
    policy: MotionPriorPolicy,
    *,
    learning_rate: float = 5e-4,
    max_grad_norm: float | None = 1.0,
    loss_cfg: DistillationLossCfg | None = None,
    device: str | torch.device = "cpu",
  ) -> None:
    self.policy = policy
    self.policy.to(device)
    self.device = device
    self.max_grad_norm = max_grad_norm
    self.loss_cfg = loss_cfg or DistillationLossCfg()

    if self.loss_cfg.loss_type == "mse":
      self._behavior_loss_fn = nn.functional.mse_loss
    elif self.loss_cfg.loss_type == "huber":
      self._behavior_loss_fn = nn.functional.huber_loss
    else:
      raise ValueError(f"Unknown loss_type: {self.loss_cfg.loss_type}")

    # Optimizer iterates only over parameters that require grad — the
    # frozen teachers contribute none.
    trainable = [p for p in policy.parameters() if p.requires_grad]
    self.optimizer = optim.Adam(trainable, lr=learning_rate)

  # ------------------------------------------------------------------ #
  # Submodules subject to per-module grad clipping.                    #
  # ------------------------------------------------------------------ #

  def _clipped_modules(self) -> list[nn.Module]:
    p = self.policy
    return [
      p.encoder_a,
      p.encoder_b,
      p.es_a_mu,
      p.es_a_var,
      p.es_b_mu,
      p.es_b_var,
      p.decoder,
      p.motion_prior,
      p.mp_mu,
      p.mp_var,
    ]

  # ------------------------------------------------------------------ #
  # Loss assembly                                                      #
  # ------------------------------------------------------------------ #

  def compute_loss_one_batch(
    self,
    *,
    actions_teacher_a: torch.Tensor,
    actions_student_a: torch.Tensor,
    enc_mu_a: torch.Tensor,
    enc_log_var_a: torch.Tensor,
    mp_mu_a: torch.Tensor,
    mp_log_var_a: torch.Tensor,
    enc_mu_a_time_stack: torch.Tensor,
    progress_buf_a: torch.Tensor | None,
    actions_teacher_b: torch.Tensor,
    actions_student_b: torch.Tensor,
    enc_mu_b: torch.Tensor,
    enc_log_var_b: torch.Tensor,
    mp_mu_b: torch.Tensor,
    mp_log_var_b: torch.Tensor,
    enc_mu_b_time_stack: torch.Tensor,
    progress_buf_b: torch.Tensor | None,
    cur_iter_num: int = 0,
  ) -> dict[str, float]:
    """Compute combined loss, backprop, clip, step. Returns scalar log dict."""
    cfg = self.loss_cfg

    # 1) behavior loss (dual teacher, weighted)
    behavior_a = self._behavior_loss_fn(actions_student_a, actions_teacher_a)
    behavior_b = self._behavior_loss_fn(actions_student_b, actions_teacher_b)
    behavior_loss = (
      cfg.behavior_weight_a * behavior_a + cfg.behavior_weight_b * behavior_b
    )

    # 2) AR(1) smoothing prior (one per encoder)
    ar1_a = _ar1_residual(enc_mu_a_time_stack, progress_buf_a, cfg.ar1_phi)
    ar1_b = _ar1_residual(enc_mu_b_time_stack, progress_buf_b, cfg.ar1_phi)
    ar1_prior = ar1_a + ar1_b

    # 3) KL to motion_prior (one per encoder)
    kl_a = _kl_two_gaussians(enc_mu_a, enc_log_var_a, mp_mu_a, mp_log_var_a)
    kl_b = _kl_two_gaussians(enc_mu_b, enc_log_var_b, mp_mu_b, mp_log_var_b)
    kl_loss = kl_a + kl_b
    kl_coeff = _annealed_kl_coeff(cur_iter_num, cfg)

    # 4) optional alignment loss (first-moment matching)
    if cfg.align_loss_coeff > 0:
      align_loss = (enc_mu_a.mean(dim=0) - enc_mu_b.mean(dim=0)).pow(2).sum() + (
        enc_log_var_a.mean(dim=0) - enc_log_var_b.mean(dim=0)
      ).pow(2).sum()
    else:
      align_loss = torch.zeros((), device=behavior_loss.device)

    total = (
      behavior_loss
      + cfg.mu_regu_loss_coeff * ar1_prior
      + kl_coeff * kl_loss
      + cfg.align_loss_coeff * align_loss
    )

    self.optimizer.zero_grad()
    total.backward()
    if self.max_grad_norm:
      for module in self._clipped_modules():
        nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)
    self.optimizer.step()

    return {
      "loss/total": total.item(),
      "loss/behavior_a": behavior_a.item(),
      "loss/behavior_b": behavior_b.item(),
      "loss/ar1_a": ar1_a.item(),
      "loss/ar1_b": ar1_b.item(),
      "loss/kl_a": kl_a.item(),
      "loss/kl_b": kl_b.item(),
      "loss/align": float(align_loss.item()) if cfg.align_loss_coeff > 0 else 0.0,
      "schedule/kl_coeff": kl_coeff,
    }
