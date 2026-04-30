"""Typed runner / policy / PPO config for downstream-task training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class RslRlDownstreamPolicyCfg:
  """Policy hyperparams; frozen-backbone hidden_dims must match training ckpt."""

  motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128)
  decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  latent_z_dims: int = 32
  activation: str = "elu"
  init_noise_std: float = 1.0
  class_name: str = "DownStreamPolicy"


@dataclass
class RslRlDownstreamVQPolicyCfg:
  """VQ-flavor policy hyperparams. ``num_code`` / ``code_dim`` MUST match
  whatever the motion-prior VQ training used."""

  motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128)
  decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
  critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
  num_code: int = 2048
  code_dim: int = 64
  activation: str = "elu"
  init_noise_std: float = 1.0
  use_lab: bool = True
  lab_lambda: float = 3.0
  class_name: str = "DownStreamVQPolicy"


@dataclass
class RslRlDownstreamPpoCfg:
  """PPO hyperparams for the downstream task. Standard knobs only."""

  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.005
  learning_rate: float = 1.0e-3
  max_grad_norm: float = 1.0
  use_clipped_value_loss: bool = True
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  desired_kl: float = 0.01
  class_name: str = "DownStreamPPO"


@dataclass
class RslRlDownstreamRunnerCfg(RslRlBaseRunnerCfg):
  """Top-level config for ``DownStreamOnPolicyRunner``."""

  class_name: str = "DownStreamOnPolicyRunner"

  motion_prior_ckpt_path: str = ""
  """Path to a ``MotionPriorOnPolicyRunner`` checkpoint. Must be set
  before running training (CLI override or edited in cfg)."""

  policy: RslRlDownstreamPolicyCfg = field(default_factory=RslRlDownstreamPolicyCfg)
  algorithm: RslRlDownstreamPpoCfg = field(default_factory=RslRlDownstreamPpoCfg)


@dataclass
class RslRlDownstreamVQRunnerCfg(RslRlBaseRunnerCfg):
  """Top-level config for ``DownStreamVQOnPolicyRunner``.

  ``motion_prior_ckpt_path`` should point at a VQ motion-prior checkpoint
  (saved by ``MotionPriorVQOnPolicyRunner.save``), which carries
  ``decoder``, ``motion_prior``, and ``quantizer`` top-level keys.
  """

  class_name: str = "DownStreamVQOnPolicyRunner"

  motion_prior_ckpt_path: str = ""

  policy: RslRlDownstreamVQPolicyCfg = field(default_factory=RslRlDownstreamVQPolicyCfg)
  algorithm: RslRlDownstreamPpoCfg = field(default_factory=RslRlDownstreamPpoCfg)
