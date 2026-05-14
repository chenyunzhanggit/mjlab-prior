"""AMP discriminator MLP.

Adapted from AMP_mjlab/rsl_rl/modules/discriminator.py (BSD-3-Clause).
Outputs scalar logits; least-squares GAN targets +1 (expert) / -1 (policy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import autograd


class Discriminator(nn.Module):
  """LSGAN-style discriminator with gradient penalty + AMP reward head.

  Args:
    input_dim: Dimension of concatenated (state, next_state) input.
    amp_reward_coef: Scaling applied to the discriminator-derived reward.
    hidden_layer_sizes: Sizes of the MLP trunk.
    device: Module device.
    task_reward_lerp: 0 → pure AMP reward, 1 → pure task reward; linear blend.
  """

  def __init__(
    self,
    input_dim: int,
    amp_reward_coef: float,
    hidden_layer_sizes: list[int] | tuple[int, ...],
    device: str | torch.device,
    task_reward_lerp: float = 0.0,
  ) -> None:
    super().__init__()
    self.device = device
    self.input_dim = input_dim
    self.amp_reward_coef = amp_reward_coef
    self.task_reward_lerp = task_reward_lerp

    layers: list[nn.Module] = []
    curr_in = input_dim
    for h in hidden_layer_sizes:
      layers.append(nn.Linear(curr_in, h))
      layers.append(nn.ReLU())
      curr_in = h
    self.trunk = nn.Sequential(*layers).to(device)
    self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.amp_linear(self.trunk(x))

  def compute_grad_pen(
    self,
    expert_state: torch.Tensor,
    expert_next_state: torch.Tensor,
    lambda_: float = 10.0,
  ) -> torch.Tensor:
    """Penalize ||∂D/∂x|| → 0 on expert data (LSGAN gradient penalty)."""
    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
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
    state: torch.Tensor,
    next_state: torch.Tensor,
    task_reward: torch.Tensor,
    normalizer=None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the AMP reward (optionally lerp'd with task reward).

    Args:
      state: ``(B, input_dim // 2)`` AMP features at step t.
      next_state: ``(B, input_dim // 2)`` AMP features at step t+1.
      task_reward: ``(B,)`` task reward used for lerp when enabled.
      normalizer: Optional running-stats normalizer applied to (state, next).

    Returns:
      reward: ``(B,)`` shaped AMP (or lerp'd) reward.
      d: ``(B, 1)`` raw discriminator logits.
    """
    was_training = self.training
    self.eval()
    try:
      with torch.no_grad():
        if normalizer is not None:
          state = normalizer.normalize_torch(state, self.device)
          next_state = normalizer.normalize_torch(next_state, self.device)

        d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
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
