"""ONNX deploy wrappers for the motion-prior student.

At deployment the robot has no ``teacher_obs`` and no motion-tracking
command — only proprioception **and a depth image**. So the exported
model bundles **Path 3** of the trained policy::

  (prop_obs, depth)
       │
       │     depth_latent = depth_cnn(depth)
       │     h = motion_prior([prop, depth_latent])
       ▼
       z (= mp_mu(h) for VAE / h directly for VQ)
       │
       ▼     decoder([prop, z]) ──► action

For VAE policies the prior head adds an ``mp_mu`` Linear after the MLP
(``motion_prior_inference`` returns ``mp_mu``). For VQ policies the MLP
outputs ``code_dim`` directly. Both share the same Phase-2 ``depth_cnn``.

Frozen teachers and per-teacher encoders are intentionally **not**
exported; they would require obs the deployment environment does not
have.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn

from mjlab.tasks.motion_prior.rl.policies.downstream_policy import DownStreamPolicy
from mjlab.tasks.motion_prior.rl.policies.downstream_vq_policy import (
  DownStreamVQPolicy,
)
from mjlab.tasks.motion_prior.rl.policies.motion_prior_policy import MotionPriorPolicy
from mjlab.tasks.motion_prior.rl.policies.motion_prior_vq_policy import (
  MotionPriorVQPolicy,
)


class MotionPriorVAEDeployModel(nn.Module):
  """Deployable Path 3 head for the VAE motion-prior student."""

  is_recurrent: bool = False
  input_names = ("prop_obs", "depth")
  output_names = ("actions",)

  def __init__(self, policy: MotionPriorPolicy) -> None:
    super().__init__()
    self.depth_cnn = copy.deepcopy(policy.depth_cnn)
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.mp_mu = copy.deepcopy(policy.mp_mu)
    self.decoder = copy.deepcopy(policy.decoder)
    self.input_size = policy.prop_obs_dim
    self.depth_shape = policy.depth_shape
    self.output_size = policy.num_actions
    self.eval()

  def forward(self, prop_obs: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    depth_latent = self.depth_cnn(depth)
    h = self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))
    z = self.mp_mu(h)  # μ_mp from the VAE prior; no sampling at deploy
    return self.decoder(torch.cat([prop_obs, z], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.input_size),
      torch.zeros(1, *self.depth_shape),
    )


class MotionPriorVQDeployModel(nn.Module):
  """Deployable Path 3 head for the VQ-VAE motion-prior student."""

  is_recurrent: bool = False
  input_names = ("prop_obs", "depth")
  output_names = ("actions",)

  def __init__(self, policy: MotionPriorVQPolicy) -> None:
    super().__init__()
    self.depth_cnn = copy.deepcopy(policy.depth_cnn)
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.decoder = copy.deepcopy(policy.decoder)
    self.input_size = policy.prop_obs_dim
    self.depth_shape = policy.depth_shape
    self.output_size = policy.num_actions
    self.eval()

  def forward(self, prop_obs: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    depth_latent = self.depth_cnn(depth)
    z = self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))  # code_dim
    return self.decoder(torch.cat([prop_obs, z], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.input_size),
      torch.zeros(1, *self.depth_shape),
    )


class MotionPriorVQDualBranchDeployModel(nn.Module):
  """Dual-branch deploy wrapper for the VQ motion-prior student.

  Exposes both teacher paths in a single ONNX file. A runtime ``mode`` flag
  picks between tracking (encoder_a) and locomotion (motion_prior head)::

      mode == 1 (tracking):
          z = encoder_a(teacher_a_obs)
      mode == 0 (locomotion):
          z = motion_prior([prop, depth_cnn(depth)])

      q = quantizer.dequantize(quantizer.quantize(z))    # shared codebook
      action = decoder([prop, q])

  Both paths route through the same shared codebook + decoder, matching the
  training-time forward_a / forward_b semantics. ``teacher_a_obs`` must be
  the same 1D vector teacher_a consumes during distillation (e.g. 166-dim
  for the G1 Teleopit teacher: proprio + motion-tracking command frames).
  """

  is_recurrent: bool = False
  input_names = ("prop_obs", "depth", "teacher_a_obs", "mode")
  output_names = ("actions",)

  def __init__(self, policy: MotionPriorVQPolicy) -> None:
    super().__init__()
    self.depth_cnn = copy.deepcopy(policy.depth_cnn)
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.encoder_a = copy.deepcopy(policy.encoder_a)
    self.quantizer = copy.deepcopy(policy.quantizer)
    self.decoder = copy.deepcopy(policy.decoder)
    self.prop_obs_dim = policy.prop_obs_dim
    self.depth_shape = policy.depth_shape
    self.teacher_a_obs_dim = policy.teacher_a_cfg.actor_obs_dim
    self.num_actions = policy.num_actions
    self.eval()

  def forward(
    self,
    prop_obs: torch.Tensor,
    depth: torch.Tensor,
    teacher_a_obs: torch.Tensor,
    mode: torch.Tensor,
  ) -> torch.Tensor:
    depth_latent = self.depth_cnn(depth)
    z_loco = self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))
    z_track = self.encoder_a(teacher_a_obs)
    # mode is broadcast as [B, 1] float in {0., 1.}; tracking when >= 0.5.
    use_track = (mode >= 0.5).to(z_loco.dtype)
    z = use_track * z_track + (1.0 - use_track) * z_loco
    code_idx = self.quantizer.quantize(z)
    q = self.quantizer.dequantize(code_idx)
    return self.decoder(torch.cat([prop_obs, q], dim=-1))

  def get_dummy_inputs(
    self,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.prop_obs_dim),
      torch.zeros(1, *self.depth_shape),
      torch.zeros(1, self.teacher_a_obs_dim),
      torch.zeros(1, 1),
    )


def build_deploy_model(
  policy: MotionPriorPolicy | MotionPriorVQPolicy,
  *,
  dual_branch: bool = False,
) -> nn.Module:
  """Pick the right deploy wrapper for the given policy.

  ``dual_branch=True`` exposes both tracking (encoder_a) and locomotion
  (motion_prior) paths with a runtime ``mode`` flag. Only supported for
  VQ policies — they share a single codebook so both paths can reuse the
  same decoder cleanly.
  """
  if isinstance(policy, MotionPriorVQPolicy):
    if dual_branch:
      return MotionPriorVQDualBranchDeployModel(policy)
    return MotionPriorVQDeployModel(policy)
  if isinstance(policy, MotionPriorPolicy):
    if dual_branch:
      raise NotImplementedError(
        "Dual-branch deploy is only implemented for VQ motion-prior policies."
      )
    return MotionPriorVAEDeployModel(policy)
  raise TypeError(f"Unsupported policy type for deploy export: {type(policy).__name__}")


def export_motion_prior_to_onnx(
  policy: MotionPriorPolicy | MotionPriorVQPolicy,
  output_path: str | Path,
  *,
  verbose: bool = False,
  dual_branch: bool = False,
) -> None:
  """Export the deployable motion-prior student to ONNX.

  Single-branch (default): inputs are ``prop_obs`` and ``depth``; only
  the motion_prior (locomotion) head is exported.

  ``dual_branch=True`` (VQ only): inputs are ``prop_obs``, ``depth``,
  ``teacher_a_obs``, ``mode``. The deploy-time ``mode`` flag picks between
  tracking (``mode>=0.5``, routes through ``encoder_a(teacher_a_obs)``)
  and locomotion (``mode<0.5``, routes through ``motion_prior``); both
  paths go through the shared codebook and decoder.
  """
  output_path = Path(output_path)
  os.makedirs(output_path.parent, exist_ok=True)
  model = build_deploy_model(policy, dual_branch=dual_branch).to("cpu")
  model.eval()
  input_names: tuple[str, ...] = model.input_names  # type: ignore[assignment]
  output_names: tuple[str, ...] = model.output_names  # type: ignore[assignment]
  dynamic_axes: dict[str, dict[int, str]] = {
    name: {0: "batch"} for name in (*input_names, *output_names)
  }
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),  # type: ignore[operator]
    str(output_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=list(input_names),
    output_names=list(output_names),
    dynamic_axes=dynamic_axes,
    dynamo=False,
  )


# ===========================================================================
# Downstream-task deploy wrappers (frozen mp + decoder + trainable actor)
# ===========================================================================


class DownStreamCombinedDeployModel(nn.Module):
  """Full ``DownStreamPolicy`` deploy chain in one ONNX file.

    (prop_obs, depth, policy_obs) ──►
        depth_latent = depth_cnn(depth)
        z_prior_mu = mp_mu(motion_prior([prop_obs, depth_latent]))
        raw_action = actor(policy_obs)
        z = z_prior_mu + raw_action
        action = decoder(cat([prop_obs, z]))

  This is the **recommended deploy path**: a single self-contained file
  the on-board controller can run without composing multiple ONNXes.
  """

  is_recurrent: bool = False
  input_names = ("prop_obs", "depth", "policy_obs")
  output_names = ("actions",)

  def __init__(self, policy: DownStreamPolicy) -> None:
    super().__init__()
    self.depth_cnn = copy.deepcopy(policy.depth_cnn)
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.mp_mu = copy.deepcopy(policy.mp_mu)
    self.decoder = copy.deepcopy(policy.decoder)
    self.actor = copy.deepcopy(policy.actor)
    self.prop_obs_dim = policy.prop_obs_dim
    self.depth_shape = policy.depth_shape
    self.num_obs = policy.num_obs
    self.num_actions = policy.num_actions
    self.eval()

  def forward(
    self,
    prop_obs: torch.Tensor,
    depth: torch.Tensor,
    policy_obs: torch.Tensor,
  ) -> torch.Tensor:
    depth_latent = self.depth_cnn(depth)
    z_prior_mu = self.mp_mu(
      self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))
    )
    raw_action = self.actor(torch.cat([policy_obs, depth_latent], dim=-1))
    z = z_prior_mu + raw_action
    return self.decoder(torch.cat([prop_obs, z], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.prop_obs_dim),
      torch.zeros(1, *self.depth_shape),
      torch.zeros(1, self.num_obs),
    )


class DownStreamActorDeployModel(nn.Module):
  """Trainable-only path: ``[policy_obs, depth_latent] → raw_latent``.

  Provided for users who want to keep the frozen backbone separate (e.g.
  share one motion_prior backbone across multiple downstream actors). The
  caller is responsible for running ``depth_cnn`` externally and supplying
  the resulting ``depth_latent`` here — the actor now reads it alongside
  ``policy_obs`` so it can condition the residual on terrain. For most
  cases use ``DownStreamCombinedDeployModel`` instead.
  """

  is_recurrent: bool = False
  input_names = ("policy_obs", "depth_latent")
  output_names = ("raw_action",)

  def __init__(self, policy: DownStreamPolicy | DownStreamVQPolicy) -> None:
    super().__init__()
    self.actor = copy.deepcopy(policy.actor)
    self.num_obs = policy.num_obs
    self.depth_latent_dim = policy.depth_latent_dim
    self.latent_dim = policy.latent_dim
    self.eval()

  def forward(
    self, policy_obs: torch.Tensor, depth_latent: torch.Tensor
  ) -> torch.Tensor:
    return self.actor(torch.cat([policy_obs, depth_latent], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.num_obs),
      torch.zeros(1, self.depth_latent_dim),
    )


class DownStreamVQCombinedDeployModel(nn.Module):
  """Full ``DownStreamVQPolicy`` deploy chain in one ONNX file.

    (prop_obs, depth, policy_obs) ──►
        depth_latent = depth_cnn(depth)
        prior_latent = motion_prior([prop_obs, depth_latent])  # frozen
        raw_action   = actor(policy_obs)                       # trainable
        z = prior_latent + λ·tanh(raw_action)                  # LAB on by default
        q = quantizer.dequantize(quantizer.quantize(z))        # frozen codebook
        action = decoder(cat([prop_obs, q]))                   # frozen

  The codebook nearest-neighbor lookup uses ``argmin`` which exports cleanly
  to ONNX opset 18. EMA updates are off here (we never update at deploy).
  """

  is_recurrent: bool = False
  input_names = ("prop_obs", "depth", "policy_obs")
  output_names = ("actions",)

  def __init__(self, policy: DownStreamVQPolicy) -> None:
    super().__init__()
    self.depth_cnn = copy.deepcopy(policy.depth_cnn)
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.quantizer = copy.deepcopy(policy.quantizer)
    self.decoder = copy.deepcopy(policy.decoder)
    self.actor = copy.deepcopy(policy.actor)
    self.prop_obs_dim = policy.prop_obs_dim
    self.depth_shape = policy.depth_shape
    self.num_obs = policy.num_obs
    self.num_actions = policy.num_actions
    self.use_lab = policy.use_lab
    self.lab_lambda = policy.lab_lambda
    self.eval()

  def forward(
    self,
    prop_obs: torch.Tensor,
    depth: torch.Tensor,
    policy_obs: torch.Tensor,
  ) -> torch.Tensor:
    depth_latent = self.depth_cnn(depth)
    prior_latent = self.motion_prior(torch.cat([prop_obs, depth_latent], dim=-1))
    raw_action = self.actor(torch.cat([policy_obs, depth_latent], dim=-1))
    if self.use_lab:
      z = prior_latent + self.lab_lambda * torch.tanh(raw_action)
    else:
      z = prior_latent + raw_action
    code_idx = self.quantizer.quantize(z)
    q = self.quantizer.dequantize(code_idx)
    return self.decoder(torch.cat([prop_obs, q], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.prop_obs_dim),
      torch.zeros(1, *self.depth_shape),
      torch.zeros(1, self.num_obs),
    )


def export_downstream_to_onnx(
  policy: DownStreamPolicy | DownStreamVQPolicy,
  output_path: str | Path,
  *,
  mode: str = "combined",
  verbose: bool = False,
) -> None:
  """Export a downstream policy to ONNX.

  ``mode='combined'`` (default) writes the full deploy pipeline (auto-
  selects VAE or VQ variant based on ``policy`` type); ``'actor'`` writes
  just the trainable actor (raw latent residual, identical for VAE/VQ).
  """
  output_path = Path(output_path)
  os.makedirs(output_path.parent, exist_ok=True)

  if mode == "combined":
    if isinstance(policy, DownStreamVQPolicy):
      model: nn.Module = DownStreamVQCombinedDeployModel(policy)
    else:
      model = DownStreamCombinedDeployModel(policy)
    dynamic_axes = {
      "prop_obs": {0: "batch"},
      "depth": {0: "batch"},
      "policy_obs": {0: "batch"},
      "actions": {0: "batch"},
    }
  elif mode == "actor":
    model = DownStreamActorDeployModel(policy)
    dynamic_axes = {
      "policy_obs": {0: "batch"},
      "depth_latent": {0: "batch"},
      "raw_action": {0: "batch"},
    }
  else:
    raise ValueError(f"Unknown export mode: {mode!r}; use 'combined' or 'actor'.")

  model = model.to("cpu").eval()
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    str(output_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=list(model.input_names),
    output_names=list(model.output_names),
    dynamic_axes=dynamic_axes,
    dynamo=False,
  )
