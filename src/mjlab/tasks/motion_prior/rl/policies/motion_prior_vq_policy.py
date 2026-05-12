"""Dual-teacher VQ-VAE motion-prior policy.

VQ counterpart of ``MotionPriorPolicy``. Replaces the per-encoder
reparameterization heads with a **shared** EMA-updated codebook so both
teachers' encoder outputs land in the same discrete latent space (per
prior.md task #6).

  teacher_a (Teleopit TemporalCNN, frozen)  ┐
  teacher_b (mjlab Velocity MLP, frozen)    ┘  produce target actions
  encoder_a(teacher_a_obs)             → code_dim raw vector ─┐
  encoder_b([prop, depth_latent])      → code_dim raw vector ─┴─► quantizer
                                                                  → q
  motion_prior([prop, depth_latent])   → code_dim regression target (shared)
  decoder([prop_obs, q])               → student_action            (shared)

Same depth pipeline as :class:`MotionPriorPolicy`: a shared CNN encoder
turns the 2D depth image into a fixed-size latent that feeds encoder_b and
the motion_prior head. encoder_a is unchanged (teacher_a is depth-blind).
The decoder still consumes only ``[prop, q]`` — terrain info is already
baked into the quantized code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.rl.cnn_proj import CNNProjModel, CNNWithProjection
from mjlab.tasks.motion_prior.rl.policies.motion_prior_policy import (
  _DEFAULT_DEPTH_CNN_CFG,
  _DEFAULT_DEPTH_LATENT_DIM,
  _DEFAULT_DEPTH_SHAPE,
)
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
    depth_shape: tuple[int, int, int] = _DEFAULT_DEPTH_SHAPE,
    depth_latent_dim: int = _DEFAULT_DEPTH_LATENT_DIM,
    depth_cnn_cfg: dict[str, Any] | None = None,
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
    self.depth_shape = depth_shape
    self.depth_latent_dim = depth_latent_dim

    # Frozen teachers — same loaders as the VAE policy.
    self.teacher_a: TemporalCNNModel = load_teleopit_teacher(
      teacher_a_policy_path, cfg=teacher_a_cfg, device=device, freeze=True
    )
    self.teacher_b: CNNProjModel = load_velocity_teacher(
      teacher_b_policy_path, cfg=teacher_b_cfg, device=device, freeze=True
    )

    # Shared depth CNN, same architecture as MotionPriorPolicy. encoder_b
    # and motion_prior_head both read the resulting depth_latent.
    cnn_channels, cnn_height, cnn_width = depth_shape
    cnn_kwargs = dict(depth_cnn_cfg or _DEFAULT_DEPTH_CNN_CFG)
    self.depth_cnn = CNNWithProjection(
      input_dim=(cnn_height, cnn_width),
      input_channels=cnn_channels,
      proj_dim=depth_latent_dim,
      proj_activation=activation,
      **cnn_kwargs,
    )

    # encoder_a: unchanged (teacher_a is depth-blind).
    self.encoder_a = MLP(
      teacher_a_cfg.actor_obs_dim,
      code_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )
    # encoder_b: now reads [prop, depth_latent] instead of teacher_b's
    # 286-dim scandot obs.
    self.encoder_b = MLP(
      prop_obs_dim + depth_latent_dim,
      code_dim,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    # Single shared codebook so both teachers compete for the same discrete
    # latent dictionary.
    self.quantizer = EMAQuantizer(
      num_code=num_code, code_dim=code_dim, ema_decay=ema_decay
    )

    # Shared decoder maps [prop_obs, code] -> action.
    self.decoder = MLP(
      prop_obs_dim + code_dim,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )

    # Shared motion_prior head: [prop_obs, depth_latent] -> code_dim
    # regression target. Same terrain-aware input as encoder_b so the
    # deploy-time prior matches training-time encoder distribution.
    self.motion_prior = MLP(
      prop_obs_dim + depth_latent_dim,
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
  def evaluate_b(
    self,
    teacher_b_obs: torch.Tensor,
    teacher_b_height: torch.Tensor,
  ) -> torch.Tensor:
    """teacher_b (CNNProjModel) needs both 1D proprio and 2D height inputs."""
    td = TensorDict(
      {"actor": teacher_b_obs, "height": teacher_b_height},
      batch_size=[teacher_b_obs.shape[0]],
    )
    return self.teacher_b(td)

  # ------------------------------------------------------------------ #
  # Encoder / quantizer / motion-prior / decoder primitives            #
  # ------------------------------------------------------------------ #

  def encode_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
    """CNN-encode a ``[B, C, H, W]`` depth image to a ``[B, depth_latent_dim]`` vector."""
    return self.depth_cnn(depth_image)

  def encode_a(self, teacher_a_obs: torch.Tensor) -> torch.Tensor:
    return self.encoder_a(teacher_a_obs)

  def encode_b(
    self, prop_obs: torch.Tensor, depth_latent: torch.Tensor
  ) -> torch.Tensor:
    return self.encoder_b(torch.cat([prop_obs, depth_latent], dim=-1))

  def quantize(
    self, encoder_output: torch.Tensor, training: bool
  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the shared quantizer; returns ``(q, commit_loss, perplexity)``."""
    return self.quantizer(encoder_output, training=training)

  def motion_prior_head(
    self, prop_obs: torch.Tensor, depth_latent: torch.Tensor
  ) -> torch.Tensor:
    return self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))

  def decode(self, prop_obs: torch.Tensor, latent_z: torch.Tensor) -> torch.Tensor:
    return self.decoder(torch.cat([prop_obs, latent_z], dim=-1))

  # ------------------------------------------------------------------ #
  # Training-time forward (one path per teacher)                       #
  # ------------------------------------------------------------------ #

  def forward_a(
    self,
    prop_obs: torch.Tensor,
    teacher_a_obs: torch.Tensor,
    depth_image: torch.Tensor,
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
    depth_latent = self.encode_depth(depth_image)
    mp_code = self.motion_prior_head(prop_obs, depth_latent)
    return student_act, q, enc, mp_code, commit_loss, perplexity

  def forward_b(
    self,
    prop_obs: torch.Tensor,
    teacher_b_obs: torch.Tensor,
    depth_image: torch.Tensor,
    training: bool = True,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
  ]:
    del teacher_b_obs  # only used by the frozen teacher via evaluate_b
    depth_latent = self.encode_depth(depth_image)
    enc = self.encode_b(prop_obs, depth_latent)
    q, commit_loss, perplexity = self.quantize(enc, training=training)
    student_act = self.decode(prop_obs, q)
    mp_code = self.motion_prior_head(prop_obs, depth_latent)
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
    self, prop_obs: torch.Tensor, depth_image: torch.Tensor
  ) -> torch.Tensor:
    depth_latent = self.encode_depth(depth_image)
    enc = self.encode_b(prop_obs, depth_latent)
    q, _, _ = self.quantize(enc, training=False)
    return self.decode(prop_obs, q)

  def encoder_inference_a(self, teacher_a_obs: torch.Tensor) -> torch.Tensor:
    return self.encode_a(teacher_a_obs)

  def encoder_inference_b(
    self, prop_obs: torch.Tensor, depth_image: torch.Tensor
  ) -> torch.Tensor:
    depth_latent = self.encode_depth(depth_image)
    return self.encode_b(prop_obs, depth_latent)

  def quantizer_inference(self, encoder_output: torch.Tensor) -> torch.Tensor:
    q, _, _ = self.quantize(encoder_output, training=False)
    return q

  def motion_prior_inference(
    self, prop_obs: torch.Tensor, depth_image: torch.Tensor
  ) -> torch.Tensor:
    """Deployment path: predicted code from ``[prop_obs, depth_latent]``."""
    depth_latent = self.encode_depth(depth_image)
    return self.motion_prior_head(prop_obs, depth_latent)

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
