"""Typed runner / algorithm / policy config for motion-prior distillation.

``RslRlMotionPriorRunnerCfg`` extends mjlab's ``RslRlBaseRunnerCfg`` so it
plugs into ``register_mjlab_task`` and the standard ``train.py`` /
``play.py`` flow. ``train.py`` calls ``dataclasses.asdict`` on the agent
cfg, and ``MotionPriorOnPolicyRunner.__init__`` reads from that dict via
``cfg["policy"]["latent_z_dims"]`` etc. — keep field names in sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class RslRlMotionPriorPolicyCfg:
  """Hidden-dim / latent shape spec for the dual-encoder VAE policy."""

  encoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128)
  latent_z_dims: int = 32
  activation: str = "elu"
  class_name: str = "MotionPriorPolicy"


@dataclass
class RslRlMotionPriorAlgoCfg:
  """Loss weights, KL annealing schedule, and optimizer knobs."""

  loss_type: Literal["mse", "huber"] = "mse"
  learning_rate: float = 5.0e-4
  max_grad_norm: float = 1.0
  num_learning_epochs: int = 5

  behavior_weight_a: float = 1.0
  behavior_weight_b: float = 1.0

  mu_regu_loss_coeff: float = 0.01
  ar1_phi: float = 0.99

  kl_loss_coeff_max: float = 0.01
  kl_loss_coeff_min: float = 0.001
  anneal_start_iter: int = 2500
  anneal_end_iter: int = 5000

  align_loss_coeff: float = 0.0
  class_name: str = "DistillationMotionPrior"


@dataclass
class RslRlMotionPriorRunnerCfg(RslRlBaseRunnerCfg):
  """Top-level config for the dual-env motion-prior runner."""

  class_name: str = "MotionPriorOnPolicyRunner"

  # Secondary (rough / velocity) env spun up internally by the runner.
  secondary_task_id: str = "Mjlab-MotionPrior-Rough-Unitree-G1"
  """Task ID of the rough env (teacher_b's training distribution)."""
  secondary_num_envs: int = 1
  """Number of envs for the rough env. Defaults match a typical primary."""

  # Frozen teacher checkpoints (paths can use ``~`` for $HOME).
  teacher_a_policy_path: str = "~/zcy/Teleopit/track.pt"
  teacher_b_policy_path: str = "~/zcy/mjlab-prior/logs/model_21000.pt"

  policy: RslRlMotionPriorPolicyCfg = field(default_factory=RslRlMotionPriorPolicyCfg)
  algorithm: RslRlMotionPriorAlgoCfg = field(default_factory=RslRlMotionPriorAlgoCfg)


@dataclass
class RslRlMotionPriorVQPolicyCfg:
  """Hidden-dim / codebook spec for the dual-encoder VQ-VAE policy."""

  encoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128)
  num_code: int = 2048
  code_dim: int = 64
  ema_decay: float = 0.99
  activation: str = "elu"
  class_name: str = "MotionPriorVQPolicy"


@dataclass
class RslRlMotionPriorVQAlgoCfg:
  """Loss weights and optimizer knobs for the VQ-VAE distillation.

  Defaults aligned to the upstream ``motionprior`` reference's VQ algorithm:
  unit weights on commit / mp losses, AR(1) disabled (upstream's live
  ``update`` path never applied AR(1)), Adam ``lr=1e-3``, no gradient
  clipping. AR(1) can be re-enabled by setting ``mu_regu_loss_coeff > 0``.
  """

  loss_type: Literal["mse", "huber"] = "mse"
  learning_rate: float = 1.0e-3
  max_grad_norm: float | None = None
  num_learning_epochs: int = 5

  behavior_weight_a: float = 1.0
  behavior_weight_b: float = 1.0

  mu_regu_loss_coeff: float = 0.0
  """AR(1) coefficient on raw encoder outputs. 0.0 = disabled (upstream
  parity); set e.g. 0.01 to re-enable temporal smoothing."""
  ar1_phi: float = 0.99

  commit_loss_coeff: float = 1.0
  """VQ commitment-loss weight (β). Upstream uses an implicit 1.0; the
  textbook VQ-VAE β=0.25 can be set explicitly if desired."""

  mp_loss_coeff: float = 1.0
  """Weight on the motion_prior code-prediction MSE. Upstream uses
  implicit 1.0."""

  class_name: str = "DistillationMotionPriorVQ"


@dataclass
class RslRlMotionPriorVQRunnerCfg(RslRlBaseRunnerCfg):
  """Top-level config for the dual-env VQ-VAE motion-prior runner."""

  class_name: str = "MotionPriorVQOnPolicyRunner"

  secondary_task_id: str = "Mjlab-MotionPrior-Rough-Unitree-G1"
  """Task ID of the rough env (teacher_b's training distribution)."""
  secondary_num_envs: int = 1
  """Number of envs for the rough env. Defaults match a typical primary."""

  teacher_a_policy_path: str = "~/zcy/Teleopit/track.pt"
  teacher_b_policy_path: str = "~/zcy/mjlab-prior/logs/model_21000.pt"

  policy: RslRlMotionPriorVQPolicyCfg = field(
    default_factory=RslRlMotionPriorVQPolicyCfg
  )
  algorithm: RslRlMotionPriorVQAlgoCfg = field(
    default_factory=RslRlMotionPriorVQAlgoCfg
  )


# --------------------------------------------------------------------------- #
# Single-encoder (one teacher) variants                                       #
# --------------------------------------------------------------------------- #


@dataclass
class RslRlMotionPriorSinglePolicyCfg:
  """Hidden-dim / latent shape for the single-encoder VAE policy."""

  encoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128)
  teacher_hidden_dims: tuple[int, ...] = (512, 256, 128)
  teacher_activation: str = "elu"
  latent_z_dims: int = 32
  activation: str = "elu"
  class_name: str = "MotionPriorSinglePolicy"


@dataclass
class RslRlMotionPriorSingleAlgoCfg:
  """Loss weights and optimizer knobs for single-encoder VAE distillation."""

  loss_type: Literal["mse", "huber"] = "mse"
  learning_rate: float = 5.0e-4
  max_grad_norm: float = 1.0
  num_learning_epochs: int = 5

  behavior_weight: float = 1.0
  mu_regu_loss_coeff: float = 0.01
  ar1_phi: float = 0.99

  kl_loss_coeff_max: float = 0.01
  kl_loss_coeff_min: float = 0.001
  anneal_start_iter: int = 2500
  anneal_end_iter: int = 5000

  class_name: str = "DistillationMotionPriorSingle"


@dataclass
class RslRlMotionPriorSingleRunnerCfg(RslRlBaseRunnerCfg):
  """Top-level config for the single-env, single-teacher VAE runner."""

  class_name: str = "MotionPriorSingleOnPolicyRunner"

  teacher_policy_path: str = ""
  """Path to the frozen multi-motion tracking actor checkpoint
  (``actor_state_dict`` produced by mjlab's PPO runner). Required."""

  policy: RslRlMotionPriorSinglePolicyCfg = field(
    default_factory=RslRlMotionPriorSinglePolicyCfg
  )
  algorithm: RslRlMotionPriorSingleAlgoCfg = field(
    default_factory=RslRlMotionPriorSingleAlgoCfg
  )


@dataclass
class RslRlMotionPriorSingleVQPolicyCfg:
  """Hidden-dim / codebook spec for the single-encoder VQ policy."""

  encoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  motion_prior_hidden_dims: tuple[int, ...] = (512, 256, 128)
  teacher_hidden_dims: tuple[int, ...] = (512, 256, 128)
  teacher_activation: str = "elu"
  num_code: int = 2048
  code_dim: int = 64
  ema_decay: float = 0.99
  activation: str = "elu"
  class_name: str = "MotionPriorSingleVQPolicy"


@dataclass
class RslRlMotionPriorSingleVQAlgoCfg:
  """Loss weights for single-encoder VQ-VAE distillation.

  Defaults follow the upstream reference (AR(1) off, commit/mp coeffs at
  1.0, no gradient clipping).
  """

  loss_type: Literal["mse", "huber"] = "mse"
  learning_rate: float = 1.0e-3
  max_grad_norm: float | None = None
  num_learning_epochs: int = 5

  behavior_weight: float = 1.0
  mu_regu_loss_coeff: float = 0.0
  ar1_phi: float = 0.99
  commit_loss_coeff: float = 1.0
  mp_loss_coeff: float = 1.0

  class_name: str = "DistillationMotionPriorSingleVQ"


@dataclass
class RslRlMotionPriorSingleVQRunnerCfg(RslRlBaseRunnerCfg):
  """Top-level config for the single-env, single-teacher VQ runner."""

  class_name: str = "MotionPriorSingleVQOnPolicyRunner"

  teacher_policy_path: str = ""
  """Path to the frozen multi-motion tracking actor checkpoint."""

  policy: RslRlMotionPriorSingleVQPolicyCfg = field(
    default_factory=RslRlMotionPriorSingleVQPolicyCfg
  )
  algorithm: RslRlMotionPriorSingleVQAlgoCfg = field(
    default_factory=RslRlMotionPriorSingleVQAlgoCfg
  )
