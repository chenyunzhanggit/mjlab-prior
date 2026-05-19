"""Vision-augmented downstream VQ policy (Phase 2 of depth pipeline migration).

Identical to :class:`DownStreamVQPolicy` except the actor accepts a
separate ``depth_image`` tensor (shape ``(B, C, H, W)``) on top of the
1-D ``policy_obs``. Internally:

  ┌─ depth_image (B, C, H, W) ──► DepthCNN ──► depth_emb (B, depth_emb_dim)
  │                                                       │
  │  policy_obs (B, num_obs_1d) ──────────► concat ◄──────┘
  │                                              │
  │                                              ▼
  │                                          MLP head
  │                                              │
  │                                              ▼
  │                                       raw_action (B, code_dim)
  ...

The frozen VQ backbone (motion_prior / decoder / quantizer) and the
critic are inherited unchanged from :class:`DownStreamVQPolicy`. Only
``self.actor`` is swapped for a CNN+MLP module, and the ``act`` /
``update_distribution`` / ``policy_inference`` methods are overridden
to forward the extra ``depth_image`` argument.

The depth tensor's history dimension (if any) is folded into the
channel axis: a sensor with ``data_histories={"...":4}`` yields
``depth_image`` shape ``(B, 4, H, W)``, which the CNN happily eats as
a 4-channel single-frame image. Set ``depth_channels=(c1, c2, c3)``'s
first Conv ``in_channels`` to whatever ``C`` ends up being.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP
from torch.distributions import Normal

from mjlab.tasks.motion_prior.rl.policies.downstream_vq_policy import (
  DownStreamVQPolicy,
)


class DepthCNNEncoder(nn.Module):
  """Small ConvNet: ``(B, C_in, H, W) → (B, embedding_dim)``.

  Three strided convs (downsample 2× each → ~8× total downsample) + a
  flatten + Linear projection. Output dim is configurable. The input
  channel count is configurable so a 4-frame history can be channel-stacked.
  """

  def __init__(
    self,
    image_height: int,
    image_width: int,
    in_channels: int = 1,
    embedding_dim: int = 64,
    channels: tuple[int, int, int] = (16, 32, 32),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

    c1, c2, c3 = channels
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2),
      act_cls(),
      nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
      act_cls(),
      nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
      act_cls(),
    )

    # Probe conv output size with a dummy forward.
    with torch.no_grad():
      dummy = torch.zeros(1, in_channels, image_height, image_width)
      conv_out = self.conv(dummy)
      flat_dim = int(conv_out.numel())
    self.flatten = nn.Flatten()
    self.proj = nn.Sequential(
      nn.Linear(flat_dim, embedding_dim),
      act_cls(),
    )

    self.image_height = image_height
    self.image_width = image_width
    self.in_channels = in_channels
    self.embedding_dim = embedding_dim

  def forward(self, depth: torch.Tensor) -> torch.Tensor:
    """``depth`` shape: ``(B, C, H, W)``. Returns ``(B, embedding_dim)``."""
    h = self.conv(depth)
    h = self.flatten(h)
    return self.proj(h)


class _VisionActor(nn.Module):
  """CNN-branch (depth) + 1-D branch (policy_obs) → concat → MLP → code_dim.

  Drop-in replacement for the vanilla ``MLP(num_obs → code_dim)`` actor
  in :class:`DownStreamVQPolicy`, but takes **two** tensors:

    forward(policy_obs: (B, num_obs), depth: (B, C, H, W)) → (B, code_dim)
  """

  def __init__(
    self,
    *,
    num_obs: int,
    image_height: int,
    image_width: int,
    image_channels: int = 1,
    code_dim: int,
    depth_embedding_dim: int = 64,
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
    depth_channels: tuple[int, int, int] = (16, 32, 32),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self._image_height = image_height
    self._image_width = image_width
    self._image_channels = image_channels

    self.depth_encoder = DepthCNNEncoder(
      image_height=image_height,
      image_width=image_width,
      in_channels=image_channels,
      embedding_dim=depth_embedding_dim,
      channels=depth_channels,
      activation=activation,
    )
    self.mlp = MLP(
      num_obs + depth_embedding_dim,
      code_dim,
      hidden_dims=actor_hidden_dims,
      activation=activation,
    )

  def forward(
    self, policy_obs: torch.Tensor, depth: torch.Tensor
  ) -> torch.Tensor:
    """``policy_obs``: ``(B, num_obs)``. ``depth``: ``(B, C, H, W)``."""
    depth_emb = self.depth_encoder(depth)
    return self.mlp(torch.cat([policy_obs, depth_emb], dim=-1))


class DownStreamVQVisionPolicy(DownStreamVQPolicy):
  """Vision-aware downstream VQ policy.

  Same VQ backbone / critic / std as :class:`DownStreamVQPolicy`. The
  actor is a CNN + MLP module that takes two tensors:

    - ``policy_obs``: ``(B, num_obs)`` — 1-D command + proprio
    - ``depth_image``: ``(B, image_channels, H, W)`` — single or multi-frame

  ``act`` / ``policy_inference`` are overridden to forward the depth
  tensor; the rsl_rl PPO runner needs to call them with three args
  (``policy_obs, prop_obs, depth_image``).
  """

  def __init__(
    self,
    num_obs: int,
    num_actions: int,
    num_privileged_obs: int,
    *,
    prop_obs_dim: int,
    motion_prior_ckpt_path: str | Path,
    image_height: int,
    image_width: int,
    image_channels: int = 1,
    depth_embedding_dim: int = 64,
    depth_channels: tuple[int, int, int] = (16, 32, 32),
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
    super().__init__(
      num_obs=num_obs,
      num_actions=num_actions,
      num_privileged_obs=num_privileged_obs,
      prop_obs_dim=prop_obs_dim,
      motion_prior_ckpt_path=motion_prior_ckpt_path,
      num_code=num_code,
      code_dim=code_dim,
      motion_prior_hidden_dims=motion_prior_hidden_dims,
      decoder_hidden_dims=decoder_hidden_dims,
      actor_hidden_dims=actor_hidden_dims,
      critic_hidden_dims=critic_hidden_dims,
      activation=activation,
      init_noise_std=init_noise_std,
      use_lab=use_lab,
      lab_lambda=lab_lambda,
      device=device,
      **kwargs,
    )

    # Swap the parent's actor (plain MLP) with the vision-aware actor.
    self.actor = _VisionActor(
      num_obs=num_obs,
      image_height=image_height,
      image_width=image_width,
      image_channels=image_channels,
      code_dim=code_dim,
      depth_embedding_dim=depth_embedding_dim,
      actor_hidden_dims=actor_hidden_dims,
      depth_channels=depth_channels,
      activation=activation,
    )
    self.image_height = image_height
    self.image_width = image_width
    self.image_channels = image_channels
    self.to(device)

  # ----------------------------------------------------------------- #
  # Overrides — actor takes (policy_obs, depth) instead of just policy_obs
  # ----------------------------------------------------------------- #

  def update_distribution(  # type: ignore[override]
    self, policy_obs: torch.Tensor, depth_image: torch.Tensor
  ) -> None:
    actor_mean = self.actor(policy_obs, depth_image)
    actor_mean = torch.nan_to_num(
      actor_mean, nan=0.0, posinf=1.0e3, neginf=-1.0e3
    )
    std = self.std.expand_as(actor_mean).clamp(min=1.0e-6)
    self.distribution = Normal(actor_mean, std)

  def act(  # type: ignore[override]
    self,
    policy_obs: torch.Tensor,
    prop_obs: torch.Tensor,
    depth_image: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample latent residual, combine with prior, quantize, decode.

    Three required inputs (vs the parent's two), because the depth
    image is now a separate obs group rather than a slice of policy_obs.
    """
    self.update_distribution(policy_obs, depth_image)
    raw_action = self.distribution.sample()  # type: ignore[union-attr]
    with torch.no_grad():
      prior_latent = self.motion_prior(prop_obs)
    z = self._combine(prior_latent, raw_action)
    with torch.no_grad():
      q_z, _, _ = self.quantizer(z, training=False)
    recons_actions = self.decoder(torch.cat([prop_obs, q_z], dim=-1))
    return recons_actions, raw_action

  def policy_inference(  # type: ignore[override]
    self,
    policy_obs: torch.Tensor,
    prop_obs: torch.Tensor,
    depth_image: torch.Tensor,
  ) -> torch.Tensor:
    """Deterministic forward (uses actor mean, no Normal sampling)."""
    raw_action = self.actor(policy_obs, depth_image)
    with torch.no_grad():
      prior_latent = self.motion_prior(prop_obs)
    z = self._combine(prior_latent, raw_action)
    with torch.no_grad():
      q_z, _, _ = self.quantizer(z, training=False)
    return self.decoder(torch.cat([prop_obs, q_z], dim=-1))
