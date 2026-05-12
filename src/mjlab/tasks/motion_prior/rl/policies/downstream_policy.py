"""Downstream-task RL policy that builds on a frozen motion-prior backbone.

Architecture::

  depth_image ──► depth_cnn ──► depth_latent (depth_latent_dim)
                                  │
                                  ├──────────────────────────────────────┐
                                  │                                       │
  prop_obs    ──┬──cat──► motion_prior MLP ──► h ──► mp_mu ──► z_prior_mu │
                │                                                         │
  policy_obs ─┬─┴─cat──► actor MLP ──► μ_actor ──► sample ──► raw_action  │
              │ (with depth_latent)                          │            │
              │                                              ▼            ▼
              │                                          z = z_prior_mu + raw_action
              │                                              │
              └──► cat([prop_obs, z]) ──► decoder MLP ──► action

  critic_obs ──► critic MLP ──► value

PPO trains on ``raw_action`` (the latent residual). The env steps with
``recons_action`` (the decoded joint command). Frozen backbone =
``depth_cnn`` + ``motion_prior`` + ``mp_mu`` + ``decoder``, loaded strict
from a ``MotionPriorOnPolicyRunner`` ckpt via ``load_motion_prior_components``.

Both the frozen motion_prior path AND the trainable actor consume the
``depth_latent`` produced by the shared frozen ``depth_cnn``. ``depth_cnn``
is run once per transition in ``act()`` and the resulting latent is fed to
both branches (and stored in the rollout buffer so ``update()`` can rebuild
the actor distribution without re-rendering depth).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP
from torch.distributions import Normal

from mjlab.rl.cnn_proj import CNNWithProjection
from mjlab.tasks.motion_prior.rl.policies.motion_prior_policy import (
  _DEFAULT_DEPTH_CNN_CFG,
  _DEFAULT_DEPTH_LATENT_DIM,
  _DEFAULT_DEPTH_SHAPE,
)
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
    depth_shape: tuple[int, int, int] = _DEFAULT_DEPTH_SHAPE,
    depth_latent_dim: int = _DEFAULT_DEPTH_LATENT_DIM,
    depth_cnn_cfg: dict[str, Any] | None = None,
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
    self.depth_shape = depth_shape
    self.depth_latent_dim = depth_latent_dim

    # ----- frozen motion-prior backbone (load + strict + freeze + eval) ----
    components = load_motion_prior_components(motion_prior_ckpt_path, device=device)

    cnn_channels, cnn_height, cnn_width = depth_shape
    cnn_kwargs = dict(depth_cnn_cfg or _DEFAULT_DEPTH_CNN_CFG)
    self.depth_cnn = CNNWithProjection(
      input_dim=(cnn_height, cnn_width),
      input_channels=cnn_channels,
      proj_dim=depth_latent_dim,
      proj_activation=activation,
      **cnn_kwargs,
    )
    self.depth_cnn.load_state_dict(components["depth_cnn"], strict=True)

    self.motion_prior = MLP(
      prop_obs_dim + depth_latent_dim,
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

    for module in (self.depth_cnn, self.motion_prior, self.mp_mu, self.decoder):
      for p in module.parameters():
        p.requires_grad = False
      module.eval()

    # ----- trainable actor ([policy_obs, depth_latent] -> latent_z residual) -----
    # Actor consumes the frozen depth_latent so it can condition its
    # residual on terrain. depth_cnn is frozen, so gradients only flow
    # into the actor's input layer (not into the CNN).
    self.actor = MLP(
      num_obs + depth_latent_dim,
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

  def update_distribution(
    self, policy_obs: torch.Tensor, depth_latent: torch.Tensor
  ) -> None:
    """Re-build the actor distribution from ``[policy_obs, depth_latent]``.

    ``depth_latent`` is the output of ``self.depth_cnn``; the caller is
    responsible for providing it (either freshly computed from a raw depth
    image, or replayed from the rollout buffer during PPO ``update()``).
    """
    actor_mean = self.actor(torch.cat([policy_obs, depth_latent], dim=-1))
    std = self.std.expand_as(actor_mean)
    self.distribution = Normal(actor_mean, std)

  def encode_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
    """CNN-encode depth to a latent vector (frozen weights, no grad)."""
    with torch.no_grad():
      return self.depth_cnn(depth_image)

  def _z_prior_mu(
    self, prop_obs: torch.Tensor, depth_latent: torch.Tensor
  ) -> torch.Tensor:
    """Frozen-backbone forward: motion_prior + mp_mu (given depth_latent)."""
    with torch.no_grad():
      mp_h = self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))
      return self.mp_mu(mp_h)

  def act(
    self,
    policy_obs: torch.Tensor,
    prop_obs: torch.Tensor,
    depth_image: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample latent residual, decode through frozen backbone.

    Returns:
      ``(recons_actions, raw_action, depth_latent)``.
        * ``recons_actions``: (B, num_actions) — joint command for env.step
        * ``raw_action``: (B, latent_z_dims) — what PPO trains on
        * ``depth_latent``: (B, depth_latent_dim) — frozen CNN output,
          surfaced so the caller can cache it in the rollout buffer and
          replay it during ``update()`` without re-running the CNN.
    """
    depth_latent = self.encode_depth(depth_image)
    self.update_distribution(policy_obs, depth_latent)
    raw_action = self.distribution.sample()  # type: ignore[union-attr]
    z_prior_mu = self._z_prior_mu(prop_obs, depth_latent)
    z = z_prior_mu + raw_action
    recons_actions = self.decoder(torch.cat([prop_obs, z], dim=-1))
    return recons_actions, raw_action, depth_latent

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
    return self.critic(critic_obs)

  # --------------------------------------------------------------------- #
  # Deterministic inference paths                                         #
  # --------------------------------------------------------------------- #

  def policy_inference(
    self,
    policy_obs: torch.Tensor,
    prop_obs: torch.Tensor,
    depth_image: torch.Tensor,
  ) -> torch.Tensor:
    """Deterministic forward (uses actor mean, no sampling)."""
    depth_latent = self.encode_depth(depth_image)
    raw_action = self.actor(torch.cat([policy_obs, depth_latent], dim=-1))
    z_prior_mu = self._z_prior_mu(prop_obs, depth_latent)
    z = z_prior_mu + raw_action
    return self.decoder(torch.cat([prop_obs, z], dim=-1))
