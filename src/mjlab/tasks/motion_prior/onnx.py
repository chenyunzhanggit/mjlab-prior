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
