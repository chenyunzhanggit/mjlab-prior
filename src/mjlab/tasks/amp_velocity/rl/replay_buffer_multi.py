"""Ring-buffer of K-frame AMP stacks for the multi-frame discriminator path.

Stores ``(K, d)`` per slot rather than the ``(d, d)`` pair stored by
:class:`ReplayBuffer`. Insert/sample interface is otherwise identical so the
runner / AMPPPO can swap variants by isinstance check.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBufferMulti:
  """Fixed-size ring buffer holding AMP K-frame stacks from the policy."""

  def __init__(
    self,
    state_dim: int,
    num_frames: int,
    buffer_size: int,
    device: str | torch.device,
  ) -> None:
    self.states = torch.zeros(buffer_size, num_frames, state_dim, device=device)
    self.state_dim = state_dim
    self.num_frames = num_frames
    self.buffer_size = buffer_size
    self.device = device

    self.step = 0
    self.num_samples = 0

  def insert(self, states: torch.Tensor) -> None:
    """Append a batch of stacks to the buffer.

    Args:
      states: ``(B, K, d)`` tensor of policy-side stacks.
    """
    assert states.dim() == 3 and states.shape[1] == self.num_frames, (
      f"ReplayBufferMulti.insert expects (B, K={self.num_frames}, d), "
      f"got {tuple(states.shape)}"
    )
    num_states = states.shape[0]
    start_idx = self.step
    end_idx = self.step + num_states
    if end_idx > self.buffer_size:
      split = self.buffer_size - self.step
      self.states[self.step : self.buffer_size] = states[:split]
      self.states[: end_idx - self.buffer_size] = states[split:]
    else:
      self.states[start_idx:end_idx] = states

    self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
    self.step = (self.step + num_states) % self.buffer_size

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    """Yield ``num_mini_batch`` random ``(B, K, d)`` minibatches."""
    if self.num_samples == 0:
      raise RuntimeError(
        "AMP ReplayBufferMulti is empty; insert policy stacks before generating."
      )
    for _ in range(num_mini_batch):
      sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
      yield self.states[sample_idxs]
