"""Vision-augmented downstream VQ policy.

Identical to :class:`DownStreamVQPolicy` except the actor is replaced
with a two-branch network:

  obs := [ non-vision (passing_src, proprio history, ...) | depth pixels ]
                                                                │
                                                                ▼
                                                       reshape (B,1,H,W)
                                                                │
                                                                ▼
                                                          DepthCNN
                                                                │
                                                                ▼
   ┌─── non-vision ────────────────┐               depth embedding (B, depth_emb_dim)
   │                                │                            │
   └────────────────►   concat   ◄──┴────────────────────────────┘
                            │
                            ▼
                  ActorMLP → raw_action (code_dim)

This matches the "depth → CNN encoder → concat with proprio → MLP"
recipe from VisualMimic (Yin & Ze, 2025, §III-B-b / §III-D-a). The
frozen VQ backbone (motion_prior / decoder / quantizer) is shared with
:class:`DownStreamVQPolicy` so the same single-VQ ckpt loads cleanly.

Convention: depth pixels are assumed to occupy the **last**
``image_height * image_width`` slots of ``policy_obs``. The env_cfg
should therefore put any non-vision policy terms before
``ball_depth_image`` so the slice ``obs[:, -H*W:]`` recovers the
flattened depth map row-major (H rows, W cols).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP

from mjlab.tasks.motion_prior.rl.policies.downstream_vq_policy import (
  DownStreamVQPolicy,
)


class DepthCNNEncoder(nn.Module):
  """Small ConvNet that maps a (B, 1, H, W) depth image to a (B, D)
  embedding. Mirrors the encoder topology used in
  VisualMimic / legged-vision works: 3 strided convs, then a
  flatten + Linear projection."""

  def __init__(
    self,
    image_height: int,
    image_width: int,
    embedding_dim: int = 64,
    channels: tuple[int, int, int] = (16, 32, 32),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

    c1, c2, c3 = channels
    self.conv = nn.Sequential(
      nn.Conv2d(1, c1, kernel_size=5, stride=2, padding=2),
      act_cls(),
      nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
      act_cls(),
      nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
      act_cls(),
    )

    # Probe conv output size with a dummy forward.
    with torch.no_grad():
      dummy = torch.zeros(1, 1, image_height, image_width)
      conv_out = self.conv(dummy)
      flat_dim = int(conv_out.numel())
    self.flatten = nn.Flatten()
    self.proj = nn.Sequential(
      nn.Linear(flat_dim, embedding_dim),
      act_cls(),
    )

    self.image_height = image_height
    self.image_width = image_width
    self.embedding_dim = embedding_dim

  def forward(self, depth_flat: torch.Tensor) -> torch.Tensor:
    """``depth_flat``: (B, H*W) or (B, 1, H, W). Returns (B, embedding_dim)."""
    if depth_flat.dim() == 2:
      depth = depth_flat.view(-1, 1, self.image_height, self.image_width)
    else:
      depth = depth_flat
    h = self.conv(depth)
    h = self.flatten(h)
    return self.proj(h)


class _VisionActor(nn.Module):
  """CNN-branch + non-vision concat → MLP. Drop-in replacement for the
  vanilla ``MLP(num_obs → code_dim)`` in :class:`DownStreamVQPolicy`."""

  def __init__(
    self,
    *,
    num_obs: int,
    image_height: int,
    image_width: int,
    code_dim: int,
    depth_embedding_dim: int = 64,
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
    depth_channels: tuple[int, int, int] = (16, 32, 32),
    activation: str = "elu",
  ) -> None:
    super().__init__()
    self._image_size = image_height * image_width
    if self._image_size > num_obs:
      raise ValueError(
        f"image_size={self._image_size} > num_obs={num_obs}; "
        "depth pixels should be a contiguous suffix of policy_obs."
      )
    self._non_vision_dim = num_obs - self._image_size
    self._image_height = image_height
    self._image_width = image_width

    self.depth_encoder = DepthCNNEncoder(
      image_height=image_height,
      image_width=image_width,
      embedding_dim=depth_embedding_dim,
      channels=depth_channels,
      activation=activation,
    )
    self.mlp = MLP(
      self._non_vision_dim + depth_embedding_dim,
      code_dim,
      hidden_dims=actor_hidden_dims,
      activation=activation,
    )

  def forward(self, policy_obs: torch.Tensor) -> torch.Tensor:
    """``policy_obs`` shape: ``(B, num_obs)`` where the **last**
    ``image_height*image_width`` columns are the flat depth image."""
    non_vision = policy_obs[:, : self._non_vision_dim]
    depth = policy_obs[:, self._non_vision_dim :]
    depth_emb = self.depth_encoder(depth)
    return self.mlp(torch.cat([non_vision, depth_emb], dim=-1))


class DownStreamVQVisionPolicy(DownStreamVQPolicy):
  """Vision-aware downstream VQ policy.

  All forward logic / VQ backbone / critic are inherited unchanged from
  :class:`DownStreamVQPolicy`. The only thing we override is ``self.actor``:
  instead of a plain MLP it's a CNN-branch + concat + MLP module.
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

    # Swap the parent's ``self.actor`` (plain MLP) with the vision actor.
    self.actor = _VisionActor(
      num_obs=num_obs,
      image_height=image_height,
      image_width=image_width,
      code_dim=code_dim,
      depth_embedding_dim=depth_embedding_dim,
      actor_hidden_dims=actor_hidden_dims,
      depth_channels=depth_channels,
      activation=activation,
    )
    self.image_height = image_height
    self.image_width = image_width
    self.to(device)
