"""Dual-teacher VAE motion-prior policy.

Mirrors the structure of the isaaclab reference
``my_modules/motion_prior_policy.py`` but with two frozen teachers and a
shared decoder / motion_prior head:

  teacher_a (Teleopit TemporalCNN, frozen)  ┐
  teacher_b (mjlab Velocity MLP, frozen)    ┘  produce target actions
  encoder_a(teacher_a_obs)                  → (μ_a, logσ²_a) → z_a
  encoder_b([prop_obs, depth_latent])       → (μ_b, logσ²_b) → z_b
  motion_prior([prop_obs, depth_latent])    → (μ_mp, logσ²_mp)   (shared)
  decoder([prop_obs, z])                    → student_action     (shared)

Teachers are frozen ckpts (286-dim MLP and 166-dim TemporalCNN) and keep
their original scandot / proprio inputs verbatim. On the student side the
scandot input has been replaced with a depth image (see Phase 1): a shared
CNN encoder compresses ``[B, 1, H, W]`` to a ``depth_latent_dim``-d vector,
which is concatenated with the proprio obs before encoder_b and the
motion-prior head. encoder_a is unchanged because the flat-tracking teacher
never reads terrain. Decoder still consumes ``[prop, z]`` — depth info
already lives inside z.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.rl.cnn_proj import CNNProjModel, CNNWithProjection
from mjlab.tasks.motion_prior.teacher import (
  TELEOPIT_TEACHER_CFG,
  VELOCITY_TEACHER_CFG,
  TeleopitTeacherCfg,
  TemporalCNNModel,
  VelocityTeacherCfg,
  load_teleopit_teacher,
  load_velocity_teacher,
)

# Depth CNN config matching mjlab-loco's _DEPTH_CAMERA_CFG output (60x60 after
# crop+resize): Conv(1->32, k=5, s=2) -> Conv(32->64, k=3, s=2) -> Conv(64->128,
# k=3, s=2) -> global avg-pool -> Linear(128, depth_latent_dim).
_DEFAULT_DEPTH_SHAPE: tuple[int, int, int] = (1, 60, 60)
_DEFAULT_DEPTH_LATENT_DIM = 128
_DEFAULT_DEPTH_CNN_CFG: dict[str, Any] = {
  "output_channels": [32, 64, 128],
  "kernel_size": [5, 3, 3],
  "stride": [2, 2, 2],
  "padding": "none",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "avg",
}


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
    depth_shape: tuple[int, int, int] = _DEFAULT_DEPTH_SHAPE,
    depth_latent_dim: int = _DEFAULT_DEPTH_LATENT_DIM,
    depth_cnn_cfg: dict[str, Any] | None = None,
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
    self.depth_shape = depth_shape
    self.depth_latent_dim = depth_latent_dim

    # Frozen teachers — load straight from ckpt paths so the policy is
    # self-contained at construction time. Teacher inputs are unchanged.
    self.teacher_a: TemporalCNNModel = load_teleopit_teacher(
      teacher_a_policy_path, cfg=teacher_a_cfg, device=device, freeze=True
    )
    self.teacher_b: CNNProjModel = load_velocity_teacher(
      teacher_b_policy_path, cfg=teacher_b_cfg, device=device, freeze=True
    )

    # Shared depth-image CNN encoder. encoder_b and motion_prior_head both
    # consume the same depth_latent (single camera, single frame) so we use
    # one instance rather than separate copies — matches the
    # share_cnn_encoders=True convention used in the velocity actor/critic.
    cnn_channels, cnn_height, cnn_width = depth_shape
    cnn_kwargs = dict(depth_cnn_cfg or _DEFAULT_DEPTH_CNN_CFG)
    self.depth_cnn = CNNWithProjection(
      input_dim=(cnn_height, cnn_width),
      input_channels=cnn_channels,
      proj_dim=depth_latent_dim,
      proj_activation=activation,
      **cnn_kwargs,
    )

    # encoder_a: unchanged (flat-tracking teacher never sees terrain).
    self.encoder_a = MLP(
      teacher_a_cfg.actor_obs_dim,
      latent_z_dims,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )
    # encoder_b: now reads [prop, depth_latent] instead of teacher_b's
    # 286-dim scandot obs. teacher_b stays frozen on its original obs;
    # encoder_b is the trainable student-side mapping.
    self.encoder_b = MLP(
      prop_obs_dim + depth_latent_dim,
      latent_z_dims,
      hidden_dims=encoder_hidden_dims,
      activation=activation,
    )

    # Reparameterization heads (one pair per encoder).
    self.es_a_mu = nn.Linear(latent_z_dims, latent_z_dims)
    self.es_a_var = nn.Linear(latent_z_dims, latent_z_dims)
    self.es_b_mu = nn.Linear(latent_z_dims, latent_z_dims)
    self.es_b_var = nn.Linear(latent_z_dims, latent_z_dims)

    # Shared decoder maps [prop_obs, latent_z] → action. Depth info is
    # already absorbed into z by the encoder; not redundantly fed here.
    self.decoder = MLP(
      prop_obs_dim + latent_z_dims,
      num_actions,
      hidden_dims=decoder_hidden_dims,
      activation=activation,
    )

    # Shared motion_prior head: [prop_obs, depth_latent] -> latent prior
    # (μ_mp, logσ²_mp). At deploy time the prior runs without a teacher,
    # so it needs the same terrain-aware input as encoder_b to track the
    # encoder distribution.
    self.motion_prior = MLP(
      prop_obs_dim + depth_latent_dim,
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
  def evaluate_b(
    self,
    teacher_b_obs: torch.Tensor,
    teacher_b_height: torch.Tensor,
  ) -> torch.Tensor:
    """Run teacher_b (mjlab velocity CNN+MLP) deterministically.

    teacher_b is a :class:`CNNProjModel`, so its forward expects a
    TensorDict with both the 1D proprio key ``"actor"`` and the 2D
    height key ``"height"``. The motion-prior runner reads these from the
    env's ``"teacher_b"`` and ``"teacher_b_height"`` obs groups.
    """
    td = TensorDict(
      {"actor": teacher_b_obs, "height": teacher_b_height},
      batch_size=[teacher_b_obs.shape[0]],
    )
    return self.teacher_b(td)

  # ------------------------------------------------------------------ #
  # Encoder / motion-prior / decoder primitives                        #
  # ------------------------------------------------------------------ #

  def encode_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
    """CNN-encode a ``[B, C, H, W]`` depth image to a ``[B, depth_latent_dim]`` vector."""
    return self.depth_cnn(depth_image)

  def encode_a(self, teacher_a_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.encoder_a(teacher_a_obs)
    return self.es_a_mu(h), self.es_a_var(h)

  def encode_b(
    self, prop_obs: torch.Tensor, depth_latent: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """encoder_b consumes ``[prop_obs, depth_latent]`` (no teacher_b obs)."""
    h = self.encoder_b(torch.cat([prop_obs, depth_latent], dim=-1))
    return self.es_b_mu(h), self.es_b_var(h)

  def motion_prior_head(
    self, prop_obs: torch.Tensor, depth_latent: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """motion_prior consumes ``[prop_obs, depth_latent]`` so the deploy-time
    prior sees the same terrain signal that encoder_b sees during training."""
    h = self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))
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
    self,
    prop_obs: torch.Tensor,
    teacher_a_obs: torch.Tensor,
    depth_image: torch.Tensor,
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Encoder_a path. Returns (enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var).

    encoder_a is depth-blind (flat-tracking teacher never reads terrain),
    but the shared motion_prior head still receives ``depth_latent`` so it
    learns one terrain-aware prior used across both branches.
    """
    enc_mu, enc_log_var = self.encode_a(teacher_a_obs)
    z = self.reparameterize(enc_mu, enc_log_var)
    student_act = self.decode(prop_obs, z)
    depth_latent = self.encode_depth(depth_image)
    mp_mu, mp_log_var = self.motion_prior_head(prop_obs, depth_latent)
    return enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var

  def forward_b(
    self,
    prop_obs: torch.Tensor,
    teacher_b_obs: torch.Tensor,
    depth_image: torch.Tensor,
  ) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
  ]:
    """Encoder_b path. Returns (enc_mu, enc_log_var, z, student_act, mp_mu, mp_log_var).

    ``teacher_b_obs`` is *not* fed to encoder_b anymore — it is only used
    by the frozen teacher_b at action-supervision time (via :meth:`evaluate_b`,
    called outside the policy). encoder_b now reads ``[prop, depth_latent]``.
    The argument stays in the signature so the runner can pass it through
    without a separate plumbing path.
    """
    del teacher_b_obs  # only used by the frozen teacher via evaluate_b
    depth_latent = self.encode_depth(depth_image)
    enc_mu, enc_log_var = self.encode_b(prop_obs, depth_latent)
    z = self.reparameterize(enc_mu, enc_log_var)
    student_act = self.decode(prop_obs, z)
    mp_mu, mp_log_var = self.motion_prior_head(prop_obs, depth_latent)
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
    self, prop_obs: torch.Tensor, depth_image: torch.Tensor
  ) -> torch.Tensor:
    depth_latent = self.encode_depth(depth_image)
    enc_mu, _ = self.encode_b(prop_obs, depth_latent)
    return self.decode(prop_obs, enc_mu)

  def encoder_inference_a(
    self, teacher_a_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.encode_a(teacher_a_obs)

  def encoder_inference_b(
    self, prop_obs: torch.Tensor, depth_image: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    depth_latent = self.encode_depth(depth_image)
    return self.encode_b(prop_obs, depth_latent)

  def motion_prior_inference(
    self, prop_obs: torch.Tensor, depth_image: torch.Tensor
  ) -> torch.Tensor:
    """Deployment path: μ_mp from ``[prop_obs, depth_latent]`` (no teacher)."""
    depth_latent = self.encode_depth(depth_image)
    mp_mu, _ = self.motion_prior_head(prop_obs, depth_latent)
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
