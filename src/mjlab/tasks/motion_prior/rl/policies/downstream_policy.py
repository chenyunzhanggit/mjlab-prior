"""Downstream-task RL policy that builds on a frozen motion-prior backbone.

Ports ``~/zcy/motionprior/.../my_modules/downstream_task_policy.py:DownStreamPolicy``
onto mjlab. Architecture (per downstream_migration_audit.md §2):

  prop_obs (D_prop)  ──► motion_prior MLP ──► z_prior (latent_z_dims)
                                              │
                                              ▼
                                            mp_mu Linear ──► z_prior_mu (latent_z_dims)
                                                                │
  policy_obs (D_pol) ──► actor MLP ──► μ_actor (latent_z_dims)   │
                                          │                     │
                                          └─ sample (Normal) ──► raw_action (latent_z_dims)
                                                                │   ┌────────┘
                                                                ▼   ▼
                                                            z = z_prior_mu + raw_action
                                                                │
                                  cat([prop_obs, z]) ──► decoder MLP ──► action (num_actions)

  critic_obs (D_crit) ──► critic MLP ──► value (1)

PPO trains on ``raw_action`` (the latent residual). The env steps with
``recons_action`` (the decoded joint command). Frozen backbone = motion_prior
+ mp_mu + decoder, loaded strict from a ``MotionPriorOnPolicyRunner`` ckpt
via ``load_motion_prior_components``.

Deviation from reference: reference hardcodes a 64-d intermediate dim between
``motion_prior`` and ``mp_mu``; we use ``latent_z_dims`` directly to match
how ``MotionPriorPolicy`` was actually trained. Functionally equivalent.
See audit doc §3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP
from torch.distributions import Normal

from mjlab.tasks.motion_prior.teacher.downstream_ckpt_loader import (
  load_motion_prior_components,
)


class DownStreamPolicy(nn.Module):
  """RL actor-critic with a frozen motion-prior + decoder backbone.

  Inputs at construction:
    * ``prop_obs_dim`` — should match the env's ``motion_prior_obs`` group
      width AND the trained motion-prior's ``prop_obs_dim`` (decoder input
      dim minus ``latent_z_dims``).
    * ``num_obs`` — width of the env's ``policy`` obs group (actor input).
    * ``num_privileged_obs`` — width of the env's ``critic`` obs group.
    * ``num_actions`` — joint-command width (29 for G1).
    * ``motion_prior_ckpt_path`` — path to a ``MotionPriorOnPolicyRunner``
      checkpoint.

  ``hidden_dims`` for the frozen modules MUST match the trained ckpt;
  defaults align with mjlab ``MotionPriorPolicy`` defaults.
  """

  is_recurrent = False

  def __init__(
    self,
    num_obs: int,
    num_actions: int,
    num_privileged_obs: int,
    *,
    prop_obs_dim: int,
    motion_prior_ckpt_path: str | Path,
    latent_z_dims: int = 32,
    motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128),
    decoder_hidden_dims: tuple[int, ...] = (512, 256, 128),
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
    activation: str = "elu",
    init_noise_std: float = 1.0,
    device: str | torch.device = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(f"DownStreamPolicy got unexpected kwargs (ignored): {list(kwargs.keys())}")
    super().__init__()
    self.num_obs = num_obs
    self.num_privileged_obs = num_privileged_obs
    self.num_actions = num_actions
    self.prop_obs_dim = prop_obs_dim
    self.latent_dim = latent_z_dims  # rsl_rl runner reads this name

    # ----- frozen motion-prior backbone (load + strict + freeze + eval) ----
    components = load_motion_prior_components(motion_prior_ckpt_path, device=device)

    self.motion_prior = MLP(
      prop_obs_dim,
      latent_z_dims,
      hidden_dims=motion_prior_hidden_dims,
      activation=activation,
    )
    self.motion_prior.load_state_dict(components["motion_prior"], strict=True)

    self.mp_mu = nn.Linear(latent_z_dims, latent_z_dims)
    self.mp_mu.load_state_dict(components["mp_mu"], strict=True)

    self.decoder = MLP(
      prop_obs_dim + latent_z_dims,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )
    self.decoder.load_state_dict(components["decoder"], strict=True)

    for module in (self.motion_prior, self.mp_mu, self.decoder):
      for p in module.parameters():
        p.requires_grad = False
      module.eval()

    # ----- trainable actor (policy_obs -> latent_z residual) -----
    self.actor = MLP(
      num_obs,
      latent_z_dims,
      hidden_dims=actor_hidden_dims,
      activation=activation,
    )

    # ----- trainable critic (privileged_obs -> value) -----
    self.critic = MLP(
      num_privileged_obs,
      1,
      hidden_dims=critic_hidden_dims,
      activation=activation,
    )

    # ----- learnable Gaussian std on the latent residual -----
    self.std = nn.Parameter(init_noise_std * torch.ones(latent_z_dims))
    self.distribution: Normal | None = None
    Normal.set_default_validate_args(False)

    self.to(device)

  # --------------------------------------------------------------------- #
  # rsl_rl ActorCritic interface                                          #
  # --------------------------------------------------------------------- #

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

  def act(
    self, policy_obs: torch.Tensor, prop_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample latent residual, decode through frozen backbone.

    Returns:
      ``(recons_actions, raw_action)``.
        * ``recons_actions``: (B, num_actions) — joint command for env.step
        * ``raw_action``: (B, latent_z_dims) — what PPO trains on
    """
    self.update_distribution(policy_obs)
    raw_action = self.distribution.sample()  # type: ignore[union-attr]
    with torch.no_grad():
      mp_h = self.motion_prior(prop_obs)
      z_prior_mu = self.mp_mu(mp_h)
    z = z_prior_mu + raw_action
    recons_actions = self.decoder(torch.cat([prop_obs, z], dim=-1))
    return recons_actions, raw_action

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.critic(critic_obs)

  # --------------------------------------------------------------------- #
  # Deterministic inference paths                                         #
  # --------------------------------------------------------------------- #

  def policy_inference(
    self, policy_obs: torch.Tensor, prop_obs: torch.Tensor
  ) -> torch.Tensor:
    """Deterministic forward (uses actor mean, no sampling)."""
    raw_action = self.actor(policy_obs)
    with torch.no_grad():
      mp_h = self.motion_prior(prop_obs)
      z_prior_mu = self.mp_mu(mp_h)
    z = z_prior_mu + raw_action
    return self.decoder(torch.cat([prop_obs, z], dim=-1))
