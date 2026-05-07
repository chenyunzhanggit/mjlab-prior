"""Single-teacher VQ-VAE distillation algorithm.

Counterpart of :class:`DistillationMotionPriorVQ` (dual teacher) for the
single-encoder VQ student. Same loss recipe, dropped to one teacher::

  behavior = MSE(student, teacher)
  ar1      = mu_regu_loss_coeff · ||Δenc||           # raw encoder output
  commit   = commit_loss_coeff · commit              # straight from EMAQuantizer
  mp       = mp_loss_coeff · MSE(mp_code, q.detach())
  total    = behavior + ar1 + commit + mp

No KL term (the codebook itself pins the latent dictionary), no
``align_loss`` (single encoder, nothing to align across).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior import (
  _ar1_residual,
)
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_vq_policy import (
  MotionPriorSingleVQPolicy,
)


@dataclass
class DistillationSingleVQLossCfg:
  """Coefficients for single-teacher VQ distillation."""

  loss_type: str = "mse"  # {"mse", "huber"}
  behavior_weight: float = 1.0

  mu_regu_loss_coeff: float = 0.0
  """AR(1) on raw encoder output. ``0.0`` matches the upstream reference's
  default (no AR(1) on VQ); set e.g. ``0.01`` to re-enable."""

  ar1_phi: float = 0.99

  commit_loss_coeff: float = 1.0
  """VQ commit-loss weight (β). Upstream uses an implicit 1.0; the
  textbook VQ-VAE β=0.25 can be set explicitly."""

  mp_loss_coeff: float = 1.0
  """Weight on the motion_prior code-prediction MSE."""


class DistillationMotionPriorSingleVQ:
  """VQ-VAE distillation with one frozen tracking teacher."""

  def __init__(
    self,
    policy: MotionPriorSingleVQPolicy,
    *,
    learning_rate: float = 1e-3,
    max_grad_norm: float | None = None,
    loss_cfg: DistillationSingleVQLossCfg | None = None,
    device: str | torch.device = "cpu",
  ) -> None:
    self.policy = policy
    self.policy.to(device)
    self.device = device
    self.max_grad_norm = max_grad_norm
    self.loss_cfg = loss_cfg or DistillationSingleVQLossCfg()

    if self.loss_cfg.loss_type == "mse":
      self._behavior_loss_fn = nn.functional.mse_loss
    elif self.loss_cfg.loss_type == "huber":
      self._behavior_loss_fn = nn.functional.huber_loss
    else:
      raise ValueError(f"Unknown loss_type: {self.loss_cfg.loss_type}")

    trainable = [p for p in policy.parameters() if p.requires_grad]
    self.optimizer = optim.Adam(trainable, lr=learning_rate)

  def _clipped_modules(self) -> list[nn.Module]:
    p = self.policy
    return [p.encoder, p.decoder, p.motion_prior]

  def compute_loss_one_batch(
    self,
    *,
    actions_teacher: torch.Tensor,
    actions_student: torch.Tensor,
    enc: torch.Tensor,
    q: torch.Tensor,
    mp_code: torch.Tensor,
    commit: torch.Tensor,
    perplexity: torch.Tensor,
    enc_time_stack: torch.Tensor,
    progress_buf: torch.Tensor | None,
    cur_iter_num: int = 0,
  ) -> dict[str, float]:
    """Combined loss + backward + clip + step. Returns scalar log dict."""
    del cur_iter_num  # VQ has no annealing; accept for runner-API parity.
    cfg = self.loss_cfg

    behavior = self._behavior_loss_fn(actions_student, actions_teacher)
    if cfg.mu_regu_loss_coeff > 0:
      ar1 = _ar1_residual(enc_time_stack, progress_buf, cfg.ar1_phi)
    else:
      ar1 = torch.zeros((), device=behavior.device)

    mp_loss = self._behavior_loss_fn(mp_code, q.detach())

    total = (
      cfg.behavior_weight * behavior
      + cfg.mu_regu_loss_coeff * ar1
      + cfg.commit_loss_coeff * commit
      + cfg.mp_loss_coeff * mp_loss
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
      "loss/commit": commit.item(),
      "loss/mp": mp_loss.item(),
      "perplexity": perplexity.item(),
    }
