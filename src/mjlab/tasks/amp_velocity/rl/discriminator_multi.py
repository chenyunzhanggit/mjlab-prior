"""K-frame stack AMP discriminator (telebotM2-style).

Companion to :class:`Discriminator` (the (s, s') pair variant). This module
takes a single stack of ``K`` consecutive frames per sample, flattens it into
a ``(B, K*d)`` vector, and outputs scalar logits with spectral-norm Linear
layers throughout. The reward path mirrors the LSGAN clamp used in
``Discriminator.predict_amp_reward`` so the runner sees a comparable signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd
from torch.nn.utils import spectral_norm


class DiscriminatorMulti(nn.Module):
  """LSGAN-style discriminator over K-frame stacks with spectral-norm.

  Args:
    state_dim: Per-step feature dim ``d``.
    num_frames: ``K`` — number of stacked frames in a single sample.
    amp_reward_coef: Scaling applied to the discriminator-derived reward.
    hidden_layer_sizes: Sizes of the MLP trunk.
    device: Module device.
    task_reward_lerp: 0 → pure AMP reward, 1 → pure task reward; linear blend.
  """

  def __init__(
    self,
    state_dim: int,
    num_frames: int,
    amp_reward_coef: float,
    hidden_layer_sizes: list[int] | tuple[int, ...],
    device: str | torch.device,
    task_reward_lerp: float = 0.0,
  ) -> None:
    super().__init__()
    self.device = device
    self.state_dim = state_dim
    self.num_frames = num_frames
    self.input_dim = state_dim * num_frames
    self.amp_reward_coef = amp_reward_coef
    self.task_reward_lerp = task_reward_lerp

    layers: list[nn.Module] = []
    curr_in = self.input_dim
    for h in hidden_layer_sizes:
      layers.append(spectral_norm(nn.Linear(curr_in, h)))
      layers.append(nn.ReLU())
      curr_in = h
    self.trunk = nn.Sequential(*layers).to(device)
    self.amp_linear = spectral_norm(nn.Linear(hidden_layer_sizes[-1], 1)).to(device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward over an already-flattened input ``(B, K*d)``."""
    return self.amp_linear(self.trunk(x))

  def compute_grad_pen(
    self,
    expert_states: torch.Tensor,
    lambda_: float = 10.0,
  ) -> torch.Tensor:
    """LSGAN gradient penalty on expert data.

    Args:
      expert_states: ``(B, K, d)`` stack of expert frames.
      lambda_: Penalty multiplier.
    """
    expert_data = expert_states.flatten(1)
    expert_data.requires_grad = True

    disc = self.amp_linear(self.trunk(expert_data))
    ones = torch.ones(disc.size(), device=disc.device)
    grad = autograd.grad(
      outputs=disc,
      inputs=expert_data,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()

  def predict_amp_reward(
    self,
    states: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer=None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the AMP reward over a K-frame stack.

    Args:
      states: ``(B, K, d)`` stack.
      task_reward: ``(B,)`` env task reward used for lerp when enabled.
      normalizer: Optional running-stats normalizer applied to ``states``.

    Returns:
      reward: ``(B,)`` shaped AMP (or lerp'd) reward.
      d: ``(B, 1)`` raw discriminator logits.
    """
    was_training = self.training
    self.eval()
    try:
      with torch.no_grad():
        if normalizer is not None:
          states = normalizer.normalize_torch(states, self.device)

        state_cat = states.flatten(1)  # (B, K*d)
        d = self.amp_linear(self.trunk(state_cat))
        reward = self.amp_reward_coef * torch.clamp(
          1 - 0.25 * torch.square(d - 1), min=0
        )
        if self.task_reward_lerp > 0:
          reward = (
            1.0 - self.task_reward_lerp
          ) * reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
        reward = reward.squeeze(-1)
    finally:
      if was_training:
        self.train()
    return reward, d
