"""Single-teacher VQ-VAE motion-prior policy.

Single-encoder counterpart to :class:`MotionPriorVQPolicy`: one frozen
trackingbfm teacher, one encoder, one shared codebook, shared decoder +
motion_prior head. The discrete latent layout matches the dual-encoder VQ
variant so a single-encoder ckpt can in principle be re-mapped onto a
dual codebook (kept symmetric so future swaps don't require codebook
re-init).

  teacher_t (Trackingbfm MLP, frozen)         produces target actions
  encoder(teacher_t_obs)   → code_dim raw vector
                            → quantizer (EMA)  → q_code_dim
  motion_prior(prop_obs)   → code_dim regression target
  decoder([prop_obs, q])   → student_action

Encoder input is the **1-D** trackingbfm actor obs (no history axis); the
trackingbfm teacher is itself a plain MLP, so no Conv1D path is needed.
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
  TrackingbfmTeacherCfg,
  load_trackingbfm_teacher,
)


class MotionPriorSingleEncoderVQPolicy(nn.Module):
  """VQ-VAE motion-prior policy with one frozen teacher and a shared codebook."""

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
    num_code: int = 2048,
    code_dim: int = 64,
    ema_decay: float = 0.99,
    activation: str = "elu",
    device: str | torch.device = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(
        "MotionPriorSingleEncoderVQPolicy.__init__ got unexpected arguments, "
        f"ignoring: {list(kwargs.keys())}"
      )
    super().__init__()

    self.prop_obs_dim = prop_obs_dim
    self.num_actions = num_actions
    self.teacher_obs_dim = teacher_obs_dim
    self.num_code = num_code
    self.code_dim = code_dim

    # Frozen teacher.
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

    # Encoder projects teacher obs to ``code_dim`` (no μ/σ heads).
    self.encoder = MLP(
      teacher_obs_dim,
      code_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    # Shared codebook (single-encoder still benefits from quantization +
    # EMA bookkeeping, and keeping the layout matched to the dual variant
    # makes ckpt cross-loading viable later).
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

    # Motion_prior head: prop_obs → code_dim regression target (deploy path).
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
    """Run the frozen trackingbfm teacher deterministically."""
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
    """Run the shared quantizer; returns ``(q, commit_loss, perplexity)``."""
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
