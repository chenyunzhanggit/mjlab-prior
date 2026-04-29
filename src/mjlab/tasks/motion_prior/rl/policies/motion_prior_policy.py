"""Dual-teacher VAE motion-prior policy.

Mirrors the structure of the isaaclab reference
``my_modules/motion_prior_policy.py`` but with two frozen teachers and a
shared decoder / motion_prior head:

  teacher_a (Teleopit TemporalCNN, frozen)  ┐
  teacher_b (mjlab Velocity MLP, frozen)    ┘  produce target actions
  encoder_a(teacher_a_obs)                  → (μ_a, logσ²_a) → z_a
  encoder_b(teacher_b_obs)                  → (μ_b, logσ²_b) → z_b
  motion_prior(prop_obs)                    → (μ_mp, logσ²_mp)   (shared)
  decoder([prop_obs, z])                    → student_action     (shared)

Encoder inputs are the **1-D** teacher obs only (per prior.md task #5: "用
MLP 简单起步,因为 encoder 的目标是把 teacher_obs 压成 latent z,不需要
时序卷积"). The 3-D ``actor_history`` is consumed by teacher_a alone via
its TemporalCNN path.
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
  TELEOPIT_TEACHER_CFG,
  VELOCITY_TEACHER_CFG,
  TeleopitTeacherCfg,
  TemporalCNNModel,
  VelocityTeacherCfg,
  load_teleopit_teacher,
  load_velocity_teacher,
)


class MotionPriorPolicy(nn.Module):
  """VAE motion-prior policy with two frozen teachers and a shared decoder."""

  is_recurrent = False

  def __init__(
    self,
    prop_obs_dim: int,
    num_actions: int,
    teacher_a_policy_path: str | Path,
    teacher_b_policy_path: str | Path,
    *,
    teacher_a_cfg: TeleopitTeacherCfg = TELEOPIT_TEACHER_CFG,
    teacher_b_cfg: VelocityTeacherCfg = VELOCITY_TEACHER_CFG,
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
        "MotionPriorPolicy.__init__ got unexpected arguments, ignoring: "
        f"{list(kwargs.keys())}"
      )
    super().__init__()

    self.prop_obs_dim = prop_obs_dim
    self.num_actions = num_actions
    self.latent_z_dims = latent_z_dims
    self.teacher_a_cfg = teacher_a_cfg
    self.teacher_b_cfg = teacher_b_cfg

    # Frozen teachers — load straight from ckpt paths so the policy is
    # self-contained at construction time.
    self.teacher_a: TemporalCNNModel = load_teleopit_teacher(
      teacher_a_policy_path, cfg=teacher_a_cfg, device=device, freeze=True
    )
    self.teacher_b: MLPModel = load_velocity_teacher(
      teacher_b_policy_path, cfg=teacher_b_cfg, device=device, freeze=True
    )

    # Encoders (one per teacher, shared latent_z_dims).
    self.encoder_a = MLP(
      teacher_a_cfg.actor_obs_dim,
      latent_z_dims,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )
    self.encoder_b = MLP(
      teacher_b_cfg.actor_obs_dim,
      latent_z_dims,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    # Reparameterization heads (one pair per encoder).
    self.es_a_mu = nn.Linear(latent_z_dims, latent_z_dims)
    self.es_a_var = nn.Linear(latent_z_dims, latent_z_dims)
    self.es_b_mu = nn.Linear(latent_z_dims, latent_z_dims)
    self.es_b_var = nn.Linear(latent_z_dims, latent_z_dims)

    # Shared decoder maps [prop_obs, latent_z] → action.
    self.decoder = MLP(
      prop_obs_dim + latent_z_dims,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )

    # Shared motion_prior head: prop_obs → latent prior (μ_mp, logσ²_mp).
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
  def evaluate_a(
    self, actor_obs: torch.Tensor, actor_history: torch.Tensor
  ) -> torch.Tensor:
    """Run teacher_a (Teleopit TemporalCNN) deterministically."""
    td = TensorDict(
      {"actor": actor_obs, "actor_history": actor_history},
      batch_size=[actor_obs.shape[0]],
    )
    return self.teacher_a(td)

  @torch.no_grad()
  def evaluate_b(self, teacher_b_obs: torch.Tensor) -> torch.Tensor:
    """Run teacher_b (mjlab velocity MLP) deterministically."""
    td = TensorDict({"actor": teacher_b_obs}, batch_size=[teacher_b_obs.shape[0]])
    return self.teacher_b(td)

  # ------------------------------------------------------------------ #
  # Encoder / motion-prior / decoder primitives                        #
  # ------------------------------------------------------------------ #

  def encode_a(self, teacher_a_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.encoder_a(teacher_a_obs)
    return self.es_a_mu(h), self.es_a_var(h)

  def encode_b(self, teacher_b_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.encoder_b(teacher_b_obs)
    return self.es_b_mu(h), self.es_b_var(h)

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
  # Training-time forward (one path per teacher)                       #
  # ------------------------------------------------------------------ #

  def forward_a(
    self, prop_obs: torch.Tensor, teacher_a_obs: torch.Tensor
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Encoder_a path. Returns (enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var)."""
    enc_mu, enc_log_var = self.encode_a(teacher_a_obs)
    z = self.reparameterize(enc_mu, enc_log_var)
    student_act = self.decode(prop_obs, z)
    mp_mu, mp_log_var = self.motion_prior_head(prop_obs)
    return enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var

  def forward_b(
    self, prop_obs: torch.Tensor, teacher_b_obs: torch.Tensor
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Encoder_b path. Returns (enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var)."""
    enc_mu, enc_log_var = self.encode_b(teacher_b_obs)
    z = self.reparameterize(enc_mu, enc_log_var)
    student_act = self.decode(prop_obs, z)
    mp_mu, mp_log_var = self.motion_prior_head(prop_obs)
    return enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var

  # ------------------------------------------------------------------ #
  # Inference paths (deterministic: latent_z = enc_mu)                 #
  # ------------------------------------------------------------------ #

  def policy_inference_a(
    self, prop_obs: torch.Tensor, teacher_a_obs: torch.Tensor
  ) -> torch.Tensor:
    enc_mu, _ = self.encode_a(teacher_a_obs)
    return self.decode(prop_obs, enc_mu)

  def policy_inference_b(
    self, prop_obs: torch.Tensor, teacher_b_obs: torch.Tensor
  ) -> torch.Tensor:
    enc_mu, _ = self.encode_b(teacher_b_obs)
    return self.decode(prop_obs, enc_mu)

  def encoder_inference_a(
    self, teacher_a_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.encode_a(teacher_a_obs)

  def encoder_inference_b(
    self, teacher_b_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.encode_b(teacher_b_obs)

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
