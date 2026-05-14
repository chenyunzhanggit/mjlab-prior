"""Dual-teacher VQ-VAE distillation algorithm.

VQ counterpart of ``DistillationMotionPrior``. The VAE's KL term is
replaced by:

  * ``commit_loss`` — VQ commitment loss (MSE between encoder output and
    its quantized lookup), one per teacher, both supplied directly by
    ``EMAQuantizer.forward(training=True)``.
  * ``mp_loss`` — supervises the motion_prior head to predict the
    quantized code from prop_obs alone (MSE on detached q). At deployment
    no teacher is available, so this is the path the student would take.

``align_loss`` is dropped: prior.md task #6 notes that the shared codebook
already pins both encoders into the same discrete latent dictionary.

Loss layout::

  behavior = w_a · MSE(student_a, teacher_a) + w_b · MSE(student_b, teacher_b)
  ar1      = mu_regu_loss_coeff · (||Δenc_a|| + ||Δenc_b||)
  commit   = commit_loss_coeff · (commit_a + commit_b)
  mp       = mp_loss_coeff · ( MSE(mp_code_a, q_a.detach())
                             + MSE(mp_code_b, q_b.detach()) )
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
from mjlab.tasks.motion_prior.rl.policies.motion_prior_vq_policy import (
  MotionPriorVQPolicy,
)


@dataclass
class DistillationVQLossCfg:
  """Coefficients for VQ distillation loss components."""

  loss_type: str = "mse"  # {"mse", "huber"}
  behavior_weight_a: float = 1.0
  behavior_weight_b: float = 1.0

  mu_regu_loss_coeff: float = 0.01
  ar1_phi: float = 0.99

  commit_loss_coeff: float = 0.25
  """Standard VQ-VAE β commitment-loss weight."""

  mp_loss_coeff: float = 0.1
  """Weight on the motion_prior code-prediction MSE."""


class DistillationMotionPriorVQ:
  """VQ-VAE distillation algorithm with two frozen teachers and a shared codebook."""

  def __init__(
    self,
    policy: MotionPriorVQPolicy,
    *,
    learning_rate: float = 5e-4,
    max_grad_norm: float | None = 1.0,
    loss_cfg: DistillationVQLossCfg | None = None,
    device: str | torch.device = "cpu",
    multi_gpu_cfg: dict | None = None,
  ) -> None:
    self.policy = policy
    self.policy.to(device)
    self.device = device
    self.max_grad_norm = max_grad_norm
    self.loss_cfg = loss_cfg or DistillationVQLossCfg()

    self._multi_gpu_cfg = multi_gpu_cfg
    self._world_size = (
      int(multi_gpu_cfg["world_size"]) if multi_gpu_cfg is not None else 1
    )

    if self.loss_cfg.loss_type == "mse":
      self._behavior_loss_fn = nn.functional.mse_loss
    elif self.loss_cfg.loss_type == "huber":
      self._behavior_loss_fn = nn.functional.huber_loss
    else:
      raise ValueError(f"Unknown loss_type: {self.loss_cfg.loss_type}")

    self._trainable_params: list[torch.nn.Parameter] = [
      p for p in policy.parameters() if p.requires_grad
    ]
    self.optimizer = optim.Adam(self._trainable_params, lr=learning_rate)

    # Wire codebook EMA all-reduce: the EMAQuantizer's _update_codebook needs
    # to know the world size + a callable to all-reduce its per-batch stats
    # so each rank's EMA sees the global batch (otherwise codebooks diverge).
    self._wire_codebook_sync()

  def _wire_codebook_sync(self) -> None:
    """Attach world_size to the policy's quantizer so its EMA update can
    all-reduce ``code_sum_batch`` / ``code_count_batch`` across ranks.
    No-op when single-GPU."""
    q = getattr(self.policy, "quantizer", None)
    if q is None:
      return
    q.set_distributed_world_size(self._world_size)

  def _sync_gradients(self) -> None:
    """Average ``param.grad`` across ranks. No-op when world_size == 1."""
    if self._world_size <= 1:
      return
    grads = [p.grad for p in self._trainable_params if p.grad is not None]
    if not grads:
      return
    flat = torch._utils._flatten_dense_tensors(grads)
    torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
    flat.div_(self._world_size)
    for buf, synced in zip(
      grads, torch._utils._unflatten_dense_tensors(flat, grads), strict=True
    ):
      buf.copy_(synced)

  def _clipped_modules(self) -> list[nn.Module]:
    p = self.policy
    return [p.encoder_a, p.encoder_b, p.decoder, p.motion_prior]

  def compute_loss_one_batch(
    self,
    *,
    actions_teacher_a: torch.Tensor,
    actions_student_a: torch.Tensor,
    enc_a: torch.Tensor,
    q_a: torch.Tensor,
    mp_code_a: torch.Tensor,
    commit_a: torch.Tensor,
    perplexity_a: torch.Tensor,
    enc_a_time_stack: torch.Tensor,
    progress_buf_a: torch.Tensor | None,
    actions_teacher_b: torch.Tensor,
    actions_student_b: torch.Tensor,
    enc_b: torch.Tensor,
    q_b: torch.Tensor,
    mp_code_b: torch.Tensor,
    commit_b: torch.Tensor,
    perplexity_b: torch.Tensor,
    enc_b_time_stack: torch.Tensor,
    progress_buf_b: torch.Tensor | None,
    cur_iter_num: int = 0,
  ) -> dict[str, float]:
    """Combined loss + backward + clip + step. Returns scalar log dict."""
    del cur_iter_num  # VQ has no annealing schedule (yet); accept for API parity.
    cfg = self.loss_cfg

    # 1) behavior loss (dual teacher, weighted)
    behavior_a = self._behavior_loss_fn(actions_student_a, actions_teacher_a)
    behavior_b = self._behavior_loss_fn(actions_student_b, actions_teacher_b)
    behavior_loss = (
      cfg.behavior_weight_a * behavior_a + cfg.behavior_weight_b * behavior_b
    )

    # 2) AR(1) smoothing on raw (pre-quantization) encoder outputs.
    # Quantized codes are discrete, so smoothing them would penalize
    # legitimate code switches; encoder outputs are continuous and do
    # capture per-step drift.
    ar1_a = _ar1_residual(enc_a_time_stack, progress_buf_a, cfg.ar1_phi)
    ar1_b = _ar1_residual(enc_b_time_stack, progress_buf_b, cfg.ar1_phi)
    ar1_prior = ar1_a + ar1_b

    # 3) commitment loss — already MSE(enc, q.detach()), summed across teachers.
    commit_loss = commit_a + commit_b

    # 4) motion_prior code-prediction loss — predict the quantized code
    # from prop_obs alone. Detach q so encoder/quantizer aren't pulled by
    # this regression.
    mp_loss = nn.functional.mse_loss(mp_code_a, q_a.detach()) + nn.functional.mse_loss(
      mp_code_b, q_b.detach()
    )

    total = (
      behavior_loss
      + cfg.mu_regu_loss_coeff * ar1_prior
      + cfg.commit_loss_coeff * commit_loss
      + cfg.mp_loss_coeff * mp_loss
    )

    self.optimizer.zero_grad()
    total.backward()
    self._sync_gradients()
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
      "loss/commit_a": commit_a.item(),
      "loss/commit_b": commit_b.item(),
      "loss/mp": mp_loss.item(),
      "perplexity_a": perplexity_a.item(),
      "perplexity_b": perplexity_b.item(),
    }
