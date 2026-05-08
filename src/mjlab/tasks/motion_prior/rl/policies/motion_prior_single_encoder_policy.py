"""Single-teacher VAE motion-prior policy.

Single-encoder counterpart to :class:`MotionPriorPolicy`: one frozen teacher,
one encoder, shared decoder + motion_prior head. Mirrors the dual-encoder
structure with the ``_b`` branch removed::

  teacher_t (Trackingbfm MLP, frozen)        produces target actions
  encoder_t(teacher_t_obs)                   → (μ, logσ²) → z
  motion_prior(prop_obs)                     → (μ_mp, logσ²_mp)
  decoder([prop_obs, z])                     → student_action

Encoder input is the **1-D** trackingbfm actor obs (no history axis); the
trackingbfm teacher itself is also a plain MLP, so no Conv1D path is
needed on either side.

Design parity with :class:`MotionPriorPolicy`:

* The motion_prior head is retained (KL prior over the proprio-only
  deploy path), so the saved checkpoint can drive the same proprio-only
  inference path that the dual-encoder policy supports.
* Reparameterization heads (``es_mu`` / ``es_var``) live as separate
  ``nn.Linear`` layers on top of the encoder backbone for state-dict
  compatibility with the dual-encoder layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.tasks.motion_prior.teacher import (
  TrackingbfmTeacherCfg,
  load_trackingbfm_teacher,
)


class MotionPriorSingleEncoderPolicy(nn.Module):
  """Single-teacher VAE motion-prior policy with a frozen MLP teacher."""

  is_recurrent = False

  def __init__(
    self,
    prop_obs_dim: int,
    num_actions: int,
    teacher_obs_dim: int,
    teacher_policy_path: str | Path,
    *,
    teacher_hidden_dims: tuple[int, ...] = (
      2048,
      2048,
      1024,
      1024,
      512,
      256,
      128,
    ),
    teacher_activation: str = "elu",
    teacher_obs_normalization: bool = True,
    encoder_hidden_dims: tuple[int, ...] = (512, 256, 128),
    decoder_hidden_dims: tuple[int, ...] = (512, 256, 128),
    motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128),
    latent_z_dims: int = 32,
    activation: str = "elu",
    device: str | torch.device = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(
        "MotionPriorSingleEncoderPolicy.__init__ got unexpected arguments, "
        f"ignoring: {list(kwargs.keys())}"
      )
    super().__init__()

    self.prop_obs_dim = prop_obs_dim
    self.num_actions = num_actions
    self.teacher_obs_dim = teacher_obs_dim
    self.latent_z_dims = latent_z_dims

    # Frozen teacher — load straight from ckpt path so the policy is
    # self-contained at construction time.
    teacher_cfg = TrackingbfmTeacherCfg(
      actor_obs_dim=teacher_obs_dim,
      num_actions=num_actions,
      hidden_dims=teacher_hidden_dims,
      activation=teacher_activation,
      obs_normalization=teacher_obs_normalization,
    )
    self.teacher_cfg = teacher_cfg
    self.teacher: MLPModel = load_trackingbfm_teacher(
      teacher_policy_path, cfg=teacher_cfg, device=device, freeze=True
    )

    # Encoder (single).
    self.encoder = MLP(
      teacher_obs_dim,
      latent_z_dims,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    # Reparameterization heads.
    self.es_mu = nn.Linear(latent_z_dims, latent_z_dims)
    self.es_var = nn.Linear(latent_z_dims, latent_z_dims)

    # Shared decoder maps [prop_obs, latent_z] → action.
    self.decoder = MLP(
      prop_obs_dim + latent_z_dims,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )

    # Motion-prior head: prop_obs → latent prior (μ_mp, logσ²_mp).
    self.motion_prior = MLP(
      prop_obs_dim,
      latent_z_dims,
      hidden_dims=motion_prior_hidden_dims,
      activation=activation,
    )
    self.mp_mu = nn.Linear(latent_z_dims, latent_z_dims)
    self.mp_var = nn.Linear(latent_z_dims, latent_z_dims)

    self.to(device)

  # ------------------------------------------------------------------ #
  # Frozen-teacher inference                                           #
  # ------------------------------------------------------------------ #

  @torch.no_grad()
  def evaluate(self, teacher_obs: torch.Tensor) -> torch.Tensor:
    """Run the trackingbfm teacher deterministically."""
    td = TensorDict({"actor": teacher_obs}, batch_size=[teacher_obs.shape[0]])
    return self.teacher(td)

  # ------------------------------------------------------------------ #
  # Encoder / motion-prior / decoder primitives                        #
  # ------------------------------------------------------------------ #

  def encode(self, teacher_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.encoder(teacher_obs)
    return self.es_mu(h), self.es_var(h)

  def motion_prior_head(
    self, prop_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.motion_prior(prop_obs)
    return self.mp_mu(h), self.mp_var(h)

  def decode(self, prop_obs: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
    return self.decoder(torch.cat([prop_obs, latent_z], dim=-1))

  @staticmethod
  def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

  # ------------------------------------------------------------------ #
  # Training-time forward                                              #
  # ------------------------------------------------------------------ #

  def forward(
    self, prop_obs: torch.Tensor, teacher_obs: torch.Tensor
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Returns ``(enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var)``."""
    enc_mu, enc_log_var = self.encode(teacher_obs)
    z = self.reparameterize(enc_mu, enc_log_var)
    student_act = self.decode(prop_obs, z)
    mp_mu, mp_log_var = self.motion_prior_head(prop_obs)
    return enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var

  # ------------------------------------------------------------------ #
  # Inference paths (deterministic: latent_z = enc_mu)                 #
  # ------------------------------------------------------------------ #

  def policy_inference(
    self, prop_obs: torch.Tensor, teacher_obs: torch.Tensor
  ) -> torch.Tensor:
    enc_mu, _ = self.encode(teacher_obs)
    return self.decode(prop_obs, enc_mu)

  def encoder_inference(
    self, teacher_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.encode(teacher_obs)

  def motion_prior_inference(self, prop_obs: torch.Tensor) -> torch.Tensor:
    """Deployment path: μ_mp from prop_obs alone (no teacher available)."""
    mp_mu, _ = self.motion_prior_head(prop_obs)
    return mp_mu

  def decoder_inference(
    self, prop_obs: torch.Tensor, latent_z: torch.Tensor
  ) -> torch.Tensor:
    return self.decode(prop_obs, latent_z)

  # ------------------------------------------------------------------ #
  # rsl_rl runner hooks                                                #
  # ------------------------------------------------------------------ #

  def reset(self, dones: torch.Tensor | None = None, hidden_states: Any = None) -> None:
    pass

  def get_hidden_states(self) -> None:
    return None

  def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
    pass
