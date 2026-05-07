"""Single-teacher VAE distillation algorithm.

Counterpart of :class:`DistillationMotionPrior` (dual teacher) for a
single-encoder student. Mirrors the upstream
``motionprior/.../distillation_motion_prior.py`` reference math directly
(no ``align_loss`` since there's only one encoder).

Loss layout::

  behavior = MSE(student, teacher)
  ar1      = mu_regu_loss_coeff · ||Δμ||                # episode-boundary masked
  kl       = kl_coeff · KL(N(μ, σ²) ‖ N(μ_mp, σ_mp²))
  total    = behavior + ar1 + kl

``kl_coeff`` linearly anneals from ``kl_loss_coeff_max`` to
``kl_loss_coeff_min`` between iters ``anneal_start_iter`` and
``anneal_end_iter`` — same schedule as the dual algorithm.
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
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_policy import (
  MotionPriorSinglePolicy,
)


@dataclass
class DistillationSingleLossCfg:
  """Coefficients and KL annealing schedule for the single-teacher VAE."""

  loss_type: str = "mse"  # {"mse", "huber"}
  behavior_weight: float = 1.0
  mu_regu_loss_coeff: float = 0.01
  ar1_phi: float = 0.99
  kl_loss_coeff_max: float = 0.01
  kl_loss_coeff_min: float = 0.001
  anneal_start_iter: int = 2500
  anneal_end_iter: int = 5000


# --------------------------------------------------------------------------- #
# Adapter so we can re-use ``_annealed_kl_coeff`` (which expects fields named
# ``kl_loss_coeff_*`` and ``anneal_*_iter``) without duplicating the helper.
# ``DistillationSingleLossCfg`` already has those names, so a direct call
# works — kept here as a marker for future divergence.
# --------------------------------------------------------------------------- #


class DistillationMotionPriorSingle:
  """VAE distillation algorithm with one frozen tracking teacher."""

  def __init__(
    self,
    policy: MotionPriorSinglePolicy,
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

    trainable = [p for p in policy.parameters() if p.requires_grad]
    self.optimizer = optim.Adam(trainable, lr=learning_rate)

  # ------------------------------------------------------------------ #
  # Per-submodule grad clipping (mirrors dual algorithm's submodule list)
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
    """Combined loss + backward + clip + step. Returns scalar log dict."""
    cfg = self.loss_cfg

    behavior = self._behavior_loss_fn(actions_student, actions_teacher)
    ar1 = _ar1_residual(enc_mu_time_stack, progress_buf, cfg.ar1_phi)
    kl = _kl_two_gaussians(enc_mu, enc_log_var, mp_mu, mp_log_var)

    # ``_annealed_kl_coeff`` reads coeff fields from ``DistillationLossCfg``;
    # ``DistillationSingleLossCfg`` exposes the same field names so it
    # plugs in directly.
    kl_coeff = _annealed_kl_coeff(cur_iter_num, cfg)  # type: ignore[arg-type]

    total = (
      cfg.behavior_weight * behavior + cfg.mu_regu_loss_coeff * ar1 + kl_coeff * kl
    )

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
