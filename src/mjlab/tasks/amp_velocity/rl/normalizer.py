"""Running mean/std normalizer for AMP discriminator inputs.

Adapted from AMP_mjlab/rsl_rl/utils/utils.py (BSD-3-Clause).
Tracks statistics in numpy and provides a torch-side normalize helper.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
  """Streaming mean/var via Welford's parallel algorithm."""

  def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
    self.mean = np.zeros(shape, np.float64)
    self.var = np.ones(shape, np.float64)
    self.count = epsilon

  def update(self, arr: np.ndarray) -> None:
    batch_mean = np.mean(arr, axis=0)
    batch_var = np.var(arr, axis=0)
    batch_count = arr.shape[0]
    self._update_from_moments(batch_mean, batch_var, batch_count)

  def _update_from_moments(
    self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
  ) -> None:
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
    new_var = m_2 / tot_count

    self.mean = new_mean
    self.var = new_var
    self.count = tot_count


class Normalizer(RunningMeanStd):
  """Clipped running-mean/std normalizer used to whiten AMP obs.

  Operates on numpy for `update` and torch for `normalize_torch`. Stats are
  kept on CPU; `normalize_torch` lazily moves a copy to the input device on
  each call (cheap relative to the discriminator forward).
  """

  def __init__(self, input_dim: int, epsilon: float = 1e-4, clip_obs: float = 10.0):
    super().__init__(shape=(input_dim,))
    self.epsilon = epsilon
    self.clip_obs = clip_obs

  def normalize_torch(
    self, x: torch.Tensor, device: str | torch.device
  ) -> torch.Tensor:
    mean = torch.tensor(self.mean, device=device, dtype=torch.float32)
    std = torch.sqrt(
      torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32)
    )
    return torch.clamp((x - mean) / std, -self.clip_obs, self.clip_obs)
