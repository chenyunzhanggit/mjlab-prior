"""Dual-teacher VQ-VAE motion-prior policy.

VQ counterpart of ``MotionPriorPolicy``. Replaces the per-encoder
reparameterization heads with a **shared** EMA-updated codebook so both
teachers' encoder outputs land in the same discrete latent space (per
prior.md task #6: "VQ 下推荐方案 2 + 共享 codebook").

  teacher_a (Teleopit TemporalCNN, frozen)  ┐
  teacher_b (mjlab Velocity MLP, frozen)    ┘  produce target actions
  encoder_a(teacher_a_obs)  → code_dim raw vector ─┐
  encoder_b(teacher_b_obs)  → code_dim raw vector ─┴─► quantizer (shared)
                                                    → q_code_dim
  motion_prior(prop_obs)   → code_dim regression target          (shared)
  decoder([prop_obs, q])   → student_action                       (shared)

The quantizer is the single source of truth for the discrete latent;
``training=True`` triggers EMA updates and returns ``commit_loss`` for the
straight-through estimator. Frozen-teacher load and freeze logic are
identical to ``MotionPriorPolicy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.tasks.motion_prior.rl.policies.quantizer import EMAQuantizer
from mjlab.tasks.motion_prior.teacher import (
  TELEOPIT_TEACHER_CFG,
  VELOCITY_TEACHER_CFG,
  TeleopitTeacherCfg,
  TemporalCNNModel,
  VelocityTeacherCfg,
  load_teleopit_teacher,
  load_velocity_teacher,
)


class MotionPriorVQPolicy(nn.Module):
  """VQ-VAE motion-prior policy with two frozen teachers and a shared codebook."""

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
    num_code: int = 2048,
    code_dim: int = 64,
    ema_decay: float = 0.99,
    activation: str = "elu",
    device: str | torch.device = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(
        "MotionPriorVQPolicy.__init__ got unexpected arguments, ignoring: "
        f"{list(kwargs.keys())}"
      )
    super().__init__()

    self.prop_obs_dim = prop_obs_dim
    self.num_actions = num_actions
    self.num_code = num_code
    self.code_dim = code_dim
    self.teacher_a_cfg = teacher_a_cfg
    self.teacher_b_cfg = teacher_b_cfg

    # Frozen teachers — same loaders as the VAE policy.
    self.teacher_a: TemporalCNNModel = load_teleopit_teacher(
      teacher_a_policy_path, cfg=teacher_a_cfg, device=device, freeze=True
    )
    self.teacher_b: MLPModel = load_velocity_teacher(
      teacher_b_policy_path, cfg=teacher_b_cfg, device=device, freeze=True
    )

    # Encoders project teacher obs to ``code_dim`` (no separate μ/σ heads).
    self.encoder_a = MLP(
      teacher_a_cfg.actor_obs_dim,
      code_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )
    self.encoder_b = MLP(
      teacher_b_cfg.actor_obs_dim,
      code_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    # Single shared codebook so both teachers compete for the same discrete
    # latent dictionary.
    self.quantizer = EMAQuantizer(
      num_code=num_code, code_dim=code_dim, ema_decay=ema_decay
    )

    # Shared decoder maps [prop_obs, code] → action.
    self.decoder = MLP(
      prop_obs_dim + code_dim,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )

    # Shared motion_prior head: prop_obs → code_dim regression target.
    self.motion_prior = MLP(
      prop_obs_dim,
      code_dim,
      hidden_dims=motion_prior_hidden_dims,
      activation=activation,
    )

    self.to(device)

  # ------------------------------------------------------------------ #
  # Frozen-teacher inference                                           #
  # ------------------------------------------------------------------ #

  @torch.no_grad()
  def evaluate_a(
    self, actor_obs: torch.Tensor, actor_history: torch.Tensor
  ) -> torch.Tensor:
    td = TensorDict(
      {"actor": actor_obs, "actor_history": actor_history},
      batch_size=[actor_obs.shape[0]],
    )
    return self.teacher_a(td)

  @torch.no_grad()
  def evaluate_b(self, teacher_b_obs: torch.Tensor) -> torch.Tensor:
    td = TensorDict({"actor": teacher_b_obs}, batch_size=[teacher_b_obs.shape[0]])
    return self.teacher_b(td)

  # ------------------------------------------------------------------ #
  # Encoder / quantizer / motion-prior / decoder primitives            #
  # ------------------------------------------------------------------ #

  def encode_a(self, teacher_a_obs: torch.Tensor) -> torch.Tensor:
    return self.encoder_a(teacher_a_obs)

  def encode_b(self, teacher_b_obs: torch.Tensor) -> torch.Tensor:
    return self.encoder_b(teacher_b_obs)

  def quantize(
    self, encoder_output: torch.Tensor, training: bool
  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the shared quantizer; returns ``(q, commit_loss, perplexity)``."""
    return self.quantizer(encoder_output, training=training)

  def motion_prior_head(self, prop_obs: torch.Tensor) -> torch.Tensor:
    return self.motion_prior(prop_obs)

  def decode(self, prop_obs: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
    return self.decoder(torch.cat([prop_obs, latent_z], dim=-1))

  # ------------------------------------------------------------------ #
  # Training-time forward (one path per teacher)                       #
  # ------------------------------------------------------------------ #

  def forward_a(
    self,
    prop_obs: torch.Tensor,
    teacher_a_obs: torch.Tensor,
    training: bool = True,
  ) -> tuple[
    torch.Tensor,  # student_action
    torch.Tensor,  # quantized code (ST-estimator path)
    torch.Tensor,  # raw encoder output
    torch.Tensor,  # mp_code
    torch.Tensor | None,  # commit_loss (None when training=False)
    torch.Tensor,  # perplexity
  ]:
    enc = self.encode_a(teacher_a_obs)
    q, commit_loss, perplexity = self.quantize(enc, training=training)
    student_act = self.decode(prop_obs, q)
    mp_code = self.motion_prior_head(prop_obs)
    return student_act, q, enc, mp_code, commit_loss, perplexity

  def forward_b(
    self,
    prop_obs: torch.Tensor,
    teacher_b_obs: torch.Tensor,
    training: bool = True,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
  ]:
    enc = self.encode_b(teacher_b_obs)
    q, commit_loss, perplexity = self.quantize(enc, training=training)
    student_act = self.decode(prop_obs, q)
    mp_code = self.motion_prior_head(prop_obs)
    return student_act, q, enc, mp_code, commit_loss, perplexity

  # ------------------------------------------------------------------ #
  # Inference paths (no codebook update, no commit loss)               #
  # ------------------------------------------------------------------ #

  def policy_inference_a(
    self, prop_obs: torch.Tensor, teacher_a_obs: torch.Tensor
  ) -> torch.Tensor:
    enc = self.encode_a(teacher_a_obs)
    q, _, _ = self.quantize(enc, training=False)
    return self.decode(prop_obs, q)

  def policy_inference_b(
    self, prop_obs: torch.Tensor, teacher_b_obs: torch.Tensor
  ) -> torch.Tensor:
    enc = self.encode_b(teacher_b_obs)
    q, _, _ = self.quantize(enc, training=False)
    return self.decode(prop_obs, q)

  def encoder_inference_a(self, teacher_a_obs: torch.Tensor) -> torch.Tensor:
    return self.encode_a(teacher_a_obs)

  def encoder_inference_b(self, teacher_b_obs: torch.Tensor) -> torch.Tensor:
    return self.encode_b(teacher_b_obs)

  def quantizer_inference(self, encoder_output: torch.Tensor) -> torch.Tensor:
    q, _, _ = self.quantize(encoder_output, training=False)
    return q

  def motion_prior_inference(self, prop_obs: torch.Tensor) -> torch.Tensor:
    """Deployment path: predicted code from prop_obs (no teacher available)."""
    return self.motion_prior_head(prop_obs)

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
