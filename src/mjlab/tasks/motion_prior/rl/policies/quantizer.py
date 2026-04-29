"""EMA-update vector quantizer for the motion-prior VQ-VAE student.

Port of ``motionprior/.../vq_policy/quantize_cnn.py:QuantizeEMAReset`` with
two corrections:

  * ``code_sum`` / ``code_count`` are registered as buffers so they move
    with ``.to(device)`` and persist in ``state_dict``. The upstream
    version stores them as plain attributes, which silently breaks
    multi-GPU training and resume.
  * Initialization no longer hardcodes ``.cuda()``. The codebook starts on
    CPU and follows ``policy.to(device)`` like every other parameter.

Forward contract::

  forward(x, training: bool) -> (x_d, commit_loss, perplexity)

  * ``training=False``: nearest-neighbor lookup only; ``commit_loss=None``.
  * ``training=True``: also runs an EMA codebook update and returns
    ``commit_loss = MSE(x, x_d.detach())``. ``x_d`` carries the
    straight-through estimator ``x + (x_d - x).detach()`` so gradients
    flow back into the encoder unchanged.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAQuantizer(nn.Module):
  """Vector-quantization bottleneck with exponential-moving-average codebook update."""

  def __init__(self, num_code: int, code_dim: int, ema_decay: float = 0.99) -> None:
    super().__init__()
    if not 0.0 < ema_decay < 1.0:
      raise ValueError(f"ema_decay must be in (0, 1); got {ema_decay}")
    self.num_code = num_code
    self.code_dim = code_dim
    self.ema_decay = ema_decay

    # Initialize on a unit sphere, scaled by 1/sqrt(code_dim) so the code
    # magnitude matches what an unbiased linear layer typically outputs.
    init = F.normalize(torch.randn(num_code, code_dim), p=2, dim=1)
    init = init / math.sqrt(code_dim)

    self.register_buffer("codebook", init.clone())
    self.register_buffer("code_sum", init.clone())
    self.register_buffer("code_count", torch.ones(num_code))

  # ``register_buffer`` makes ``self.codebook`` etc. typed as
  # ``Tensor | Module`` from the static type checker's view; these
  # narrowing accessors keep the rest of the file Tensor-typed.
  @property
  def _codebook(self) -> torch.Tensor:
    return cast(torch.Tensor, self.codebook)

  @property
  def _code_sum(self) -> torch.Tensor:
    return cast(torch.Tensor, self.code_sum)

  @property
  def _code_count(self) -> torch.Tensor:
    return cast(torch.Tensor, self.code_count)

  # ------------------------------------------------------------------ #
  # Lookups                                                            #
  # ------------------------------------------------------------------ #

  def quantize(self, x: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor lookup, returning code indices of shape (B,)."""
    # ‖x - c‖² = ‖x‖² - 2 x·c + ‖c‖² ; argmin over codes.
    k_w = self._codebook.t()  # (D, K)
    distance = (
      torch.sum(x**2, dim=-1, keepdim=True)
      - 2 * torch.matmul(x, k_w)
      + torch.sum(k_w**2, dim=0, keepdim=True)
    )
    return distance.argmin(dim=-1)

  def dequantize(self, code_idx: torch.Tensor) -> torch.Tensor:
    return F.embedding(code_idx, self._codebook)

  # ------------------------------------------------------------------ #
  # EMA update + perplexity                                            #
  # ------------------------------------------------------------------ #

  @torch.no_grad()
  def _update_codebook(self, x: torch.Tensor, code_idx: torch.Tensor) -> torch.Tensor:
    """In-place EMA refresh of ``codebook`` from a batch ``x`` and its assigned codes."""
    code_onehot = torch.zeros(self.num_code, x.shape[0], device=x.device)
    code_onehot.scatter_(0, code_idx.view(1, -1), 1)

    code_sum_batch = code_onehot @ x  # (K, D)
    code_count_batch = code_onehot.sum(dim=-1)  # (K,)

    # Replacement vectors for codes that didn't get used this batch.
    code_rand = self._tile(x)[: self.num_code]

    self._code_sum.mul_(self.ema_decay).add_(code_sum_batch, alpha=1 - self.ema_decay)
    self._code_count.mul_(self.ema_decay).add_(
      code_count_batch, alpha=1 - self.ema_decay
    )
    used = (self._code_count >= 1.0).float().view(-1, 1)
    refreshed = self._code_sum / self._code_count.view(-1, 1).clamp_min(1e-8)
    self._codebook.copy_(used * refreshed + (1 - used) * code_rand)

    prob = code_count_batch / code_count_batch.sum().clamp_min(1e-8)
    return torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

  @torch.no_grad()
  def _compute_perplexity(self, code_idx: torch.Tensor) -> torch.Tensor:
    code_onehot = torch.zeros(self.num_code, code_idx.shape[0], device=code_idx.device)
    code_onehot.scatter_(0, code_idx.view(1, -1), 1)
    code_count = code_onehot.sum(dim=-1)
    prob = code_count / code_count.sum().clamp_min(1e-8)
    return torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

  def _tile(self, x: torch.Tensor) -> torch.Tensor:
    """Repeat ``x`` (with small noise) to at least ``num_code`` rows."""
    n_rows = x.shape[0]
    if n_rows >= self.num_code:
      return x
    n_repeats = (self.num_code + n_rows - 1) // n_rows
    std = 0.01 / math.sqrt(self.code_dim)
    return x.repeat(n_repeats, 1) + torch.randn_like(x.repeat(n_repeats, 1)) * std

  # ------------------------------------------------------------------ #
  # Forward                                                            #
  # ------------------------------------------------------------------ #

  def forward(
    self, x: torch.Tensor, training: bool
  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    code_idx = self.quantize(x)
    x_d = self.dequantize(code_idx)

    if training:
      perplexity = self._update_codebook(x, code_idx)
      commit_loss = F.mse_loss(x, x_d.detach())
      x_d = x + (x_d - x).detach()  # straight-through estimator
      return x_d, commit_loss, perplexity

    perplexity = self._compute_perplexity(code_idx)
    return x_d, None, perplexity
