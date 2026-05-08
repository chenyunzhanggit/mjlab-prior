"""Single-teacher VQ-VAE distillation algorithm.

Mirror of :class:`DistillationMotionPriorVQ` with the ``_b`` branch
dropped: one teacher, one encoder, one shared codebook.

Loss layout::

  behavior = w · MSE(student, teacher)
  ar1      = mu_regu_loss_coeff · ||Δenc||
  commit   = commit_loss_coeff · commit
  mp       = mp_loss_coeff · MSE(mp_code, q.detach())
  total    = behavior + ar1 + commit + mp
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior import (
  _ar1_residual,
)
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_encoder_vq_policy import (
  MotionPriorSingleEncoderVQPolicy,
)


@dataclass
class DistillationSingleVQLossCfg:
  """Coefficients for single-encoder VQ distillation loss components."""

  loss_type: str = "mse"  # {"mse", "huber"}
  behavior_weight: float = 1.0

  mu_regu_loss_coeff: float = 0.0
  """AR(1) coefficient on raw encoder outputs. 0.0 = disabled (matches
  upstream VQ default); set e.g. 0.01 to re-enable temporal smoothing."""
  ar1_phi: float = 0.99

  commit_loss_coeff: float = 1.0
  """VQ commitment-loss weight (β). Upstream uses an implicit 1.0; the
  textbook VQ-VAE β=0.25 can be set explicitly if desired."""

  mp_loss_coeff: float = 1.0
  """Weight on the motion_prior code-prediction MSE."""


class DistillationMotionPriorSingleVQ:
  """VQ-VAE distillation algorithm with one frozen teacher and a shared codebook."""

  def __init__(
    self,
    policy: MotionPriorSingleEncoderVQPolicy,
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
    del cur_iter_num  # VQ has no annealing schedule; accept for API parity.
    cfg = self.loss_cfg

    # 1) behavior loss
    behavior = self._behavior_loss_fn(actions_student, actions_teacher)
    behavior_loss = cfg.behavior_weight * behavior

    # 2) AR(1) smoothing on raw (pre-quantization) encoder outputs.
    ar1 = _ar1_residual(enc_time_stack, progress_buf, cfg.ar1_phi)

    # 3) commit loss — already MSE(enc, q.detach()) from EMAQuantizer.
    commit_loss = commit

    # 4) motion_prior code-prediction loss — predict the quantized code
    # from prop_obs alone. Detach q so encoder/quantizer aren't pulled by
    # this regression.
    mp_loss = nn.functional.mse_loss(mp_code, q.detach())

    total = (
      behavior_loss
      + cfg.mu_regu_loss_coeff * ar1
      + cfg.commit_loss_coeff * commit_loss
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
      "loss/commit": commit_loss.item(),
      "loss/mp": mp_loss.item(),
      "perplexity": perplexity.item(),
    }
