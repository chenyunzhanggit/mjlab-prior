"""Single-teacher VQ-VAE motion-prior policy.

VQ counterpart of :class:`MotionPriorSinglePolicy`. The continuous μ/σ
heads are replaced by an EMA-updated codebook (shared with the
dual-teacher VQ implementation via :class:`EMAQuantizer`)::

  teacher (mjlab MultiMotionTracking actor, frozen)  ──► action_target
  encoder(teacher_obs)  ──► code_dim raw vector ──► quantizer ──► q
  motion_prior(prop)    ──► code_dim regression target  (deploy path)
  decoder([prop, q])    ──► student_action

``training=True`` triggers EMA updates and yields ``commit_loss`` for the
straight-through estimator; ``training=False`` reuses the codebook
deterministically (no updates) and skips ``commit_loss``.
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
  TrackingTeacherCfg,
  load_tracking_teacher,
)


class MotionPriorSingleVQPolicy(nn.Module):
  """VQ-VAE motion-prior policy with one frozen tracking teacher."""

  is_recurrent = False

  def __init__(
    self,
    prop_obs_dim: int,
    teacher_obs_dim: int,
    num_actions: int,
    teacher_policy_path: str | Path,
    *,
    encoder_hidden_dims: tuple[int, ...] = (512, 256, 128),
    decoder_hidden_dims: tuple[int, ...] = (512, 256, 128),
    motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128),
    teacher_hidden_dims: tuple[int, ...] = (512, 256, 128),
    teacher_activation: str = "elu",
    num_code: int = 2048,
    code_dim: int = 64,
    ema_decay: float = 0.99,
    activation: str = "elu",
    device: str | torch.device = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(
        "MotionPriorSingleVQPolicy.__init__ got unexpected arguments, ignoring: "
        f"{list(kwargs.keys())}"
      )
    super().__init__()

    self.prop_obs_dim = prop_obs_dim
    self.teacher_obs_dim = teacher_obs_dim
    self.num_actions = num_actions
    self.num_code = num_code
    self.code_dim = code_dim

    teacher_cfg = TrackingTeacherCfg(
      actor_obs_dim=teacher_obs_dim,
      num_actions=num_actions,
      hidden_dims=teacher_hidden_dims,
      activation=teacher_activation,
    )
    self.teacher_cfg = teacher_cfg
    self.teacher: MLPModel = load_tracking_teacher(
      teacher_policy_path, cfg=teacher_cfg, device=device, freeze=True
    )

    self.encoder = MLP(
      teacher_obs_dim,
      code_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    self.quantizer = EMAQuantizer(
      num_code=num_code, code_dim=code_dim, ema_decay=ema_decay
    )

    self.decoder = MLP(
      prop_obs_dim + code_dim,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )

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
  def evaluate(self, teacher_obs: torch.Tensor) -> torch.Tensor:
    td = TensorDict({"actor": teacher_obs}, batch_size=[teacher_obs.shape[0]])
    return self.teacher(td)

  # ------------------------------------------------------------------ #
  # Encoder / quantizer / motion-prior / decoder primitives            #
  # ------------------------------------------------------------------ #

  def encode(self, teacher_obs: torch.Tensor) -> torch.Tensor:
    return self.encoder(teacher_obs)

  def quantize(
    self, encoder_output: torch.Tensor, training: bool
  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the codebook; returns ``(q, commit_loss, perplexity)``."""
    return self.quantizer(encoder_output, training=training)

  def motion_prior_head(self, prop_obs: torch.Tensor) -> torch.Tensor:
    return self.motion_prior(prop_obs)

  def decode(self, prop_obs: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
    return self.decoder(torch.cat([prop_obs, latent_z], dim=-1))

  # ------------------------------------------------------------------ #
  # Training-time forward                                              #
  # ------------------------------------------------------------------ #

  def forward(
    self,
    prop_obs: torch.Tensor,
    teacher_obs: torch.Tensor,
    training: bool = True,
  ) -> tuple[
    torch.Tensor,  # student_action
    torch.Tensor,  # quantized code (ST-estimator path)
    torch.Tensor,  # raw encoder output
    torch.Tensor,  # mp_code
    torch.Tensor | None,  # commit_loss (None when training=False)
    torch.Tensor,  # perplexity
  ]:
    enc = self.encode(teacher_obs)
    q, commit_loss, perplexity = self.quantize(enc, training=training)
    student_act = self.decode(prop_obs, q)
    mp_code = self.motion_prior_head(prop_obs)
    return student_act, q, enc, mp_code, commit_loss, perplexity

  # ------------------------------------------------------------------ #
  # Inference paths (no codebook update, no commit loss)               #
  # ------------------------------------------------------------------ #

  def policy_inference(
    self, prop_obs: torch.Tensor, teacher_obs: torch.Tensor
  ) -> torch.Tensor:
    enc = self.encode(teacher_obs)
    q, _, _ = self.quantize(enc, training=False)
    return self.decode(prop_obs, q)

  def encoder_inference(self, teacher_obs: torch.Tensor) -> torch.Tensor:
    return self.encode(teacher_obs)

  def quantizer_inference(self, encoder_output: torch.Tensor) -> torch.Tensor:
    q, _, _ = self.quantize(encoder_output, training=False)
    return q

  def motion_prior_inference(self, prop_obs: torch.Tensor) -> torch.Tensor:
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
