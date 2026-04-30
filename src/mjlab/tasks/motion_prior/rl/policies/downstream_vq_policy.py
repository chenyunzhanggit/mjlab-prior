"""VQ-VAE downstream-task RL policy.

Same actor / critic / std PPO interface as :class:`DownStreamPolicy`, but
the frozen backbone is the VQ flavor:

  prop_obs ──► motion_prior MLP ──► z_prior (code_dim)
  policy_obs ──► actor MLP ──► raw_action (code_dim)
                                    │
                                    ▼
                              z = z_prior + raw_action
                                    │
                                    ▼  (optional: tanh limit + λ scale)
                                quantizer ──► q_z (code_dim, from codebook)
                                    │
       cat([prop_obs, q_z]) ──► decoder MLP ──► action (num_actions)

Compared to VAE downstream:

* No ``mp_mu`` Linear (motion_prior outputs ``code_dim`` directly).
* The combined latent goes through the quantizer's nearest-neighbor
  lookup before decoding.
* Optional Latent Action Barrier (``use_lab=True``): clip the actor's
  raw residual through ``λ * tanh(.)`` before adding to the prior. This
  bounds how far the residual can push the latent away from the prior,
  which improves training stability when the codebook is sparse.

The codebook (and EMA stats) stay frozen at deploy — quantizer is read-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP
from torch.distributions import Normal

from mjlab.tasks.motion_prior.rl.policies.quantizer import EMAQuantizer
from mjlab.tasks.motion_prior.teacher.downstream_ckpt_loader import (
  load_motion_prior_vq_components,
)


class DownStreamVQPolicy(nn.Module):
  """RL actor-critic on top of a frozen motion_prior + quantizer + decoder."""

  is_recurrent = False

  def __init__(
    self,
    num_obs: int,
    num_actions: int,
    num_privileged_obs: int,
    *,
    prop_obs_dim: int,
    motion_prior_ckpt_path: str | Path,
    num_code: int = 2048,
    code_dim: int = 64,
    motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128),
    decoder_hidden_dims: tuple[int, ...] = (512, 256, 128),
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
    activation: str = "elu",
    init_noise_std: float = 1.0,
    use_lab: bool = True,
    lab_lambda: float = 3.0,
    device: str | torch.device = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(
        f"DownStreamVQPolicy got unexpected kwargs (ignored): {list(kwargs.keys())}"
      )
    super().__init__()
    self.num_obs = num_obs
    self.num_privileged_obs = num_privileged_obs
    self.num_actions = num_actions
    self.prop_obs_dim = prop_obs_dim
    self.num_code = num_code
    self.code_dim = code_dim
    self.use_lab = use_lab
    self.lab_lambda = lab_lambda
    self.latent_dim = code_dim  # rsl_rl runner reads this name

    # ----- frozen VQ backbone ----- #
    components = load_motion_prior_vq_components(motion_prior_ckpt_path, device=device)

    self.motion_prior = MLP(
      prop_obs_dim,
      code_dim,
      hidden_dims=motion_prior_hidden_dims,
      activation=activation,
    )
    self.motion_prior.load_state_dict(components["motion_prior"], strict=True)

    # Build with the same hyperparams used at training time. ema_decay is
    # not in the ckpt; default of 0.99 matches MotionPriorVQPolicy default.
    self.quantizer = EMAQuantizer(num_code=num_code, code_dim=code_dim)
    self.quantizer.load_state_dict(components["quantizer"], strict=True)

    self.decoder = MLP(
      prop_obs_dim + code_dim,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )
    self.decoder.load_state_dict(components["decoder"], strict=True)

    for module in (self.motion_prior, self.quantizer, self.decoder):
      for p in module.parameters():
        p.requires_grad = False
      module.eval()

    # ----- trainable actor / critic / std ----- #
    self.actor = MLP(
      num_obs,
      code_dim,
      hidden_dims=actor_hidden_dims,
      activation=activation,
    )
    self.critic = MLP(
      num_privileged_obs,
      1,
      hidden_dims=critic_hidden_dims,
      activation=activation,
    )
    self.std = nn.Parameter(init_noise_std * torch.ones(code_dim))

    self.distribution: Normal | None = None
    Normal.set_default_validate_args(False)

    self.to(device)

  # --------------------------- rsl_rl ActorCritic interface --------------------------- #

  def reset(self, dones: torch.Tensor | None = None, hidden_states: Any = None) -> None:
    pass

  def get_hidden_states(self) -> None:
    return None

  def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
    pass

  def forward(self) -> torch.Tensor:
    raise NotImplementedError("Use act() / policy_inference() instead.")

  @property
  def action_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def action_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy().sum(dim=-1)

  def update_distribution(self, policy_obs: torch.Tensor) -> None:
    actor_mean = self.actor(policy_obs)
    std = self.std.expand_as(actor_mean)
    self.distribution = Normal(actor_mean, std)

  def _combine(
    self, prior_latent: torch.Tensor, raw_action: torch.Tensor
  ) -> torch.Tensor:
    """``prior + λ·tanh(raw)`` if LAB enabled, else ``prior + raw``."""
    if self.use_lab:
      return prior_latent + self.lab_lambda * torch.tanh(raw_action)
    return prior_latent + raw_action

  def act(
    self, policy_obs: torch.Tensor, prop_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample latent residual, combine with prior, quantize, decode."""
    self.update_distribution(policy_obs)
    raw_action = self.distribution.sample()  # type: ignore[union-attr]
    with torch.no_grad():
      prior_latent = self.motion_prior(prop_obs)
    z = self._combine(prior_latent, raw_action)
    with torch.no_grad():
      # Quantize at inference; we never update the codebook from the
      # downstream actor (codebook is frozen).
      q_z, _, _ = self.quantizer(z, training=False)
    recons_actions = self.decoder(torch.cat([prop_obs, q_z], dim=-1))
    return recons_actions, raw_action

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.critic(critic_obs)

  # --------------------------- inference --------------------------- #

  def policy_inference(
    self, policy_obs: torch.Tensor, prop_obs: torch.Tensor
  ) -> torch.Tensor:
    """Deterministic forward (uses actor mean, no Normal sampling)."""
    raw_action = self.actor(policy_obs)
    with torch.no_grad():
      prior_latent = self.motion_prior(prop_obs)
    z = self._combine(prior_latent, raw_action)
    with torch.no_grad():
      q_z, _, _ = self.quantizer(z, training=False)
    return self.decoder(torch.cat([prop_obs, q_z], dim=-1))
