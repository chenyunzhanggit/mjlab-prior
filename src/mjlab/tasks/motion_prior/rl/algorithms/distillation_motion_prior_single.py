"""Single-teacher VAE distillation algorithm.

Mirror of :class:`DistillationMotionPrior` with the ``_b`` branch dropped:
one teacher, one encoder. The shared motion_prior head and the deploy path
remain the same as the dual variant so checkpoints are interchangeable on
the proprio-only inference path.

Loss layout::

  behavior = w · MSE(student, teacher)
  ar1      = mu_regu_loss_coeff · ||Δz||
  kl       = kl_coeff · KL(N(μ_enc, σ²_enc) ‖ N(μ_mp, σ²_mp))
  total    = behavior + ar1 + kl

No alignment loss in single-encoder mode (only one latent path).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior import (
  _annealed_kl_coeff,
  _ar1_residual,
  _kl_two_gaussians,
)
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_encoder_policy import (
  MotionPriorSingleEncoderPolicy,
)


@dataclass
class DistillationSingleLossCfg:
  """Coefficients and schedule for single-encoder distillation loss components."""

  loss_type: str = "mse"  # {"mse", "huber"}
  behavior_weight: float = 1.0
  mu_regu_loss_coeff: float = 0.01
  ar1_phi: float = 0.99
  kl_loss_coeff_max: float = 0.01
  kl_loss_coeff_min: float = 0.001
  anneal_start_iter: int = 2500
  anneal_end_iter: int = 5000


# Reuse the dual-encoder annealing helper by aliasing through a thin
# adapter that exposes the field names the helper looks up.
def _annealed_kl_coeff_single(
  cur_iter: int, cfg: DistillationSingleLossCfg
) -> float:
  if cur_iter <= cfg.anneal_start_iter:
    return cfg.kl_loss_coeff_max
  span = max(cfg.anneal_end_iter - cfg.anneal_start_iter, 1)
  decayed = (cfg.kl_loss_coeff_max - cfg.kl_loss_coeff_min) * max(
    (cfg.anneal_end_iter - cur_iter) / span, 0.0
  ) + cfg.kl_loss_coeff_min
  return decayed


# Sanity hook: keep the imports above active so future refactors notice.
_ = _annealed_kl_coeff


class DistillationMotionPriorSingle:
  """VAE distillation algorithm with one frozen teacher and a shared decoder."""

  def __init__(
    self,
    policy: MotionPriorSingleEncoderPolicy,
    *,
    learning_rate: float = 5e-4,
    max_grad_norm: float | None = 1.0,
    loss_cfg: DistillationSingleLossCfg | None = None,
    device: str | torch.device = "cpu",
  ) -> None:
    self.policy = policy
    self.policy.to(device)
    self.device = device
    self.max_grad_norm = max_grad_norm
    self.loss_cfg = loss_cfg or DistillationSingleLossCfg()

    if self.loss_cfg.loss_type == "mse":
      self._behavior_loss_fn = nn.functional.mse_loss
    elif self.loss_cfg.loss_type == "huber":
      self._behavior_loss_fn = nn.functional.huber_loss
    else:
      raise ValueError(f"Unknown loss_type: {self.loss_cfg.loss_type}")

    # Optimizer iterates only over parameters that require grad — the
    # frozen teacher contributes none.
    trainable = [p for p in policy.parameters() if p.requires_grad]
    self.optimizer = optim.Adam(trainable, lr=learning_rate)

  # ------------------------------------------------------------------ #
  # Submodules subject to per-module grad clipping.                    #
  # ------------------------------------------------------------------ #

  def _clipped_modules(self) -> list[nn.Module]:
    p = self.policy
    return [
      p.encoder,
      p.es_mu,
      p.es_var,
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
    actions_teacher: torch.Tensor,
    actions_student: torch.Tensor,
    enc_mu: torch.Tensor,
    enc_log_var: torch.Tensor,
    mp_mu: torch.Tensor,
    mp_log_var: torch.Tensor,
    enc_mu_time_stack: torch.Tensor,
    progress_buf: torch.Tensor | None,
    cur_iter_num: int = 0,
  ) -> dict[str, float]:
    """Compute combined loss, backprop, clip, step. Returns scalar log dict."""
    cfg = self.loss_cfg

    # 1) behavior loss
    behavior = self._behavior_loss_fn(actions_student, actions_teacher)
    behavior_loss = cfg.behavior_weight * behavior

    # 2) AR(1) smoothing prior
    ar1 = _ar1_residual(enc_mu_time_stack, progress_buf, cfg.ar1_phi)

    # 3) KL to motion_prior
    kl = _kl_two_gaussians(enc_mu, enc_log_var, mp_mu, mp_log_var)
    kl_coeff = _annealed_kl_coeff_single(cur_iter_num, cfg)

    total = behavior_loss + cfg.mu_regu_loss_coeff * ar1 + kl_coeff * kl

    self.optimizer.zero_grad()
    total.backward()
    if self.max_grad_norm:
      for module in self._clipped_modules():
        nn.utils.clip_grad_norm_(module.parameters(), self.max_grad_norm)
    self.optimizer.step()

    return {
      "loss/total": total.item(),
      "loss/behavior": behavior.item(),
      "loss/ar1": ar1.item(),
      "loss/kl": kl.item(),
      "schedule/kl_coeff": kl_coeff,
    }
