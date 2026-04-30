"""ONNX deploy wrappers for the motion-prior student.

At deployment the robot has no ``teacher_obs`` and no motion-tracking
command — only proprioception. So the exported model bundles **Path 3**
of the trained policy (per prior.md task #13)::

  prop_obs ──► motion_prior_head ──► z ──► decoder ──► action

For VAE policies that means ``motion_prior MLP → mp_mu Linear → decoder``
since ``motion_prior_inference`` returns ``mp_mu``. For VQ policies the
motion_prior MLP already outputs ``code_dim`` directly.

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
  input_names = ("prop_obs",)
  output_names = ("actions",)

  def __init__(self, policy: MotionPriorPolicy) -> None:
    super().__init__()
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.mp_mu = copy.deepcopy(policy.mp_mu)
    self.decoder = copy.deepcopy(policy.decoder)
    self.input_size = policy.prop_obs_dim
    self.output_size = policy.num_actions
    self.eval()

  def forward(self, prop_obs: torch.Tensor) -> torch.Tensor:
    h = self.motion_prior(prop_obs)
    z = self.mp_mu(h)  # μ_mp from the VAE prior; no sampling at deploy
    return self.decoder(torch.cat([prop_obs, z], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)


class MotionPriorVQDeployModel(nn.Module):
  """Deployable Path 3 head for the VQ-VAE motion-prior student."""

  is_recurrent: bool = False
  input_names = ("prop_obs",)
  output_names = ("actions",)

  def __init__(self, policy: MotionPriorVQPolicy) -> None:
    super().__init__()
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.decoder = copy.deepcopy(policy.decoder)
    self.input_size = policy.prop_obs_dim
    self.output_size = policy.num_actions
    self.eval()

  def forward(self, prop_obs: torch.Tensor) -> torch.Tensor:
    z = self.motion_prior(prop_obs)  # already code_dim
    return self.decoder(torch.cat([prop_obs, z], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)


def build_deploy_model(
  policy: MotionPriorPolicy | MotionPriorVQPolicy,
) -> nn.Module:
  """Pick the right deploy wrapper for the given policy."""
  if isinstance(policy, MotionPriorVQPolicy):
    return MotionPriorVQDeployModel(policy)
  if isinstance(policy, MotionPriorPolicy):
    return MotionPriorVAEDeployModel(policy)
  raise TypeError(f"Unsupported policy type for deploy export: {type(policy).__name__}")


def export_motion_prior_to_onnx(
  policy: MotionPriorPolicy | MotionPriorVQPolicy,
  output_path: str | Path,
  *,
  verbose: bool = False,
) -> None:
  """Export the deployable Path 3 student to ONNX (input ``prop_obs`` only)."""
  output_path = Path(output_path)
  os.makedirs(output_path.parent, exist_ok=True)
  model = build_deploy_model(policy).to("cpu")
  model.eval()
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),  # type: ignore[operator]
    str(output_path),
    export_params=True,
    opset_version=18,
    verbose=verbose,
    input_names=list(model.input_names),  # type: ignore[arg-type]
    output_names=list(model.output_names),  # type: ignore[arg-type]
    dynamic_axes={"prop_obs": {0: "batch"}, "actions": {0: "batch"}},
    dynamo=False,
  )


# ===========================================================================
# Downstream-task deploy wrappers (frozen mp + decoder + trainable actor)
# ===========================================================================


class DownStreamCombinedDeployModel(nn.Module):
  """Full ``DownStreamPolicy`` deploy chain in one ONNX file.

    (prop_obs, policy_obs) ──►
        z_prior_mu = mp_mu(motion_prior(prop_obs))
        raw_action = actor(policy_obs)
        z = z_prior_mu + raw_action
        action = decoder(cat([prop_obs, z]))

  This is the **recommended deploy path**: a single self-contained file
  the on-board controller can run without composing multiple ONNXes.
  """

  is_recurrent: bool = False
  input_names = ("prop_obs", "policy_obs")
  output_names = ("actions",)

  def __init__(self, policy: DownStreamPolicy) -> None:
    super().__init__()
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.mp_mu = copy.deepcopy(policy.mp_mu)
    self.decoder = copy.deepcopy(policy.decoder)
    self.actor = copy.deepcopy(policy.actor)
    self.prop_obs_dim = policy.prop_obs_dim
    self.num_obs = policy.num_obs
    self.num_actions = policy.num_actions
    self.eval()

  def forward(self, prop_obs: torch.Tensor, policy_obs: torch.Tensor) -> torch.Tensor:
    z_prior_mu = self.mp_mu(self.motion_prior(prop_obs))
    raw_action = self.actor(policy_obs)
    z = z_prior_mu + raw_action
    return self.decoder(torch.cat([prop_obs, z], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.prop_obs_dim),
      torch.zeros(1, self.num_obs),
    )


class DownStreamActorDeployModel(nn.Module):
  """Trainable-only path: ``policy_obs → raw_latent``.

  Provided for users who want to keep the frozen backbone separate (e.g.
  share one motion_prior backbone across multiple downstream actors). The
  on-board controller would then run this actor + the existing motion_prior
  pipeline + a thin "add residual & decode" layer in C++/runtime. For most
  cases use ``DownStreamCombinedDeployModel`` instead.
  """

  is_recurrent: bool = False
  input_names = ("policy_obs",)
  output_names = ("raw_action",)

  def __init__(self, policy: DownStreamPolicy | DownStreamVQPolicy) -> None:
    super().__init__()
    self.actor = copy.deepcopy(policy.actor)
    self.num_obs = policy.num_obs
    self.latent_dim = policy.latent_dim
    self.eval()

  def forward(self, policy_obs: torch.Tensor) -> torch.Tensor:
    return self.actor(policy_obs)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.num_obs),)


class DownStreamVQCombinedDeployModel(nn.Module):
  """Full ``DownStreamVQPolicy`` deploy chain in one ONNX file.

    (prop_obs, policy_obs) ──►
        prior_latent = motion_prior(prop_obs)            # frozen
        raw_action   = actor(policy_obs)                 # trainable
        z = prior_latent + λ·tanh(raw_action)            # LAB on by default
        q = quantizer.dequantize(quantizer.quantize(z))  # frozen codebook lookup
        action = decoder(cat([prop_obs, q]))             # frozen

  The codebook nearest-neighbor lookup uses ``argmin`` which exports cleanly
  to ONNX opset 18. EMA updates are off here (we never update at deploy).
  """

  is_recurrent: bool = False
  input_names = ("prop_obs", "policy_obs")
  output_names = ("actions",)

  def __init__(self, policy: DownStreamVQPolicy) -> None:
    super().__init__()
    self.motion_prior = copy.deepcopy(policy.motion_prior)
    self.quantizer = copy.deepcopy(policy.quantizer)
    self.decoder = copy.deepcopy(policy.decoder)
    self.actor = copy.deepcopy(policy.actor)
    self.prop_obs_dim = policy.prop_obs_dim
    self.num_obs = policy.num_obs
    self.num_actions = policy.num_actions
    self.use_lab = policy.use_lab
    self.lab_lambda = policy.lab_lambda
    self.eval()

  def forward(self, prop_obs: torch.Tensor, policy_obs: torch.Tensor) -> torch.Tensor:
    prior_latent = self.motion_prior(prop_obs)
    raw_action = self.actor(policy_obs)
    if self.use_lab:
      z = prior_latent + self.lab_lambda * torch.tanh(raw_action)
    else:
      z = prior_latent + raw_action
    code_idx = self.quantizer.quantize(z)
    q = self.quantizer.dequantize(code_idx)
    return self.decoder(torch.cat([prop_obs, q], dim=-1))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (
      torch.zeros(1, self.prop_obs_dim),
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
      "policy_obs": {0: "batch"},
      "actions": {0: "batch"},
    }
  elif mode == "actor":
    model = DownStreamActorDeployModel(policy)
    dynamic_axes = {"policy_obs": {0: "batch"}, "raw_action": {0: "batch"}}
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
