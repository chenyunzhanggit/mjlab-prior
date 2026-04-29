"""Load a frozen mjlab Velocity-Rough actor teacher (teacher_b).

The velocity G1 actor is a plain ``rsl_rl.models.MLPModel`` (no temporal
path). The checkpoint stores the actor under ``actor_state_dict``:

  - ``obs_normalizer._mean / _var / _std`` shape ``(1, 286)``
  - ``obs_normalizer.count`` scalar
  - ``mlp.{0,2,4,6}.{weight,bias}`` for hidden_dims=(512, 256, 128) → 29
  - ``distribution.std_param`` shape ``(29,)`` (scalar Gaussian)

Architecture constants below mirror
``mjlab/tasks/velocity/config/g1/rl_cfg.py`` and were verified against
``logs/model_21000.pt`` (April 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict


@dataclass(frozen=True)
class VelocityTeacherCfg:
  """Architectural hyperparameters of the mjlab velocity-rough teacher."""

  actor_obs_dim: int = 286
  num_actions: int = 29

  hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = True

  distribution_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )


VELOCITY_TEACHER_CFG = VelocityTeacherCfg()


def _make_dummy_obs(
  cfg: VelocityTeacherCfg = VELOCITY_TEACHER_CFG, batch_size: int = 1
) -> TensorDict:
  """Build a TensorDict with the exact shape ``MLPModel`` expects."""
  return TensorDict(
    {"actor": torch.zeros(batch_size, cfg.actor_obs_dim)},
    batch_size=[batch_size],
  )


def build_velocity_teacher(
  cfg: VelocityTeacherCfg = VELOCITY_TEACHER_CFG,
  *,
  device: str | torch.device = "cpu",
) -> MLPModel:
  """Instantiate the velocity teacher with random weights (no ckpt yet)."""
  obs = _make_dummy_obs(cfg)
  obs_groups = {"actor": ["actor"]}
  model = MLPModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=cfg.num_actions,
    hidden_dims=cfg.hidden_dims,
    activation=cfg.activation,
    obs_normalization=cfg.obs_normalization,
    distribution_cfg=dict(cfg.distribution_cfg),
  )
  return model.to(device)


def load_velocity_teacher(
  ckpt_path: str | Path,
  *,
  cfg: VelocityTeacherCfg = VELOCITY_TEACHER_CFG,
  device: str | torch.device = "cpu",
  freeze: bool = True,
  strict: bool = True,
) -> MLPModel:
  """Load a frozen mjlab velocity teacher from ``ckpt_path``."""
  ckpt_path = Path(ckpt_path).expanduser()
  if not ckpt_path.is_file():
    raise FileNotFoundError(f"Velocity teacher checkpoint not found: {ckpt_path}")

  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  if "actor_state_dict" not in ckpt:
    raise KeyError(
      f"Expected 'actor_state_dict' in {ckpt_path}, got top-level keys "
      f"{list(ckpt.keys())}."
    )

  model = build_velocity_teacher(cfg, device=device)
  missing, unexpected = model.load_state_dict(ckpt["actor_state_dict"], strict=strict)
  if missing or unexpected:
    print(f"[load_velocity_teacher] missing={missing}\nunexpected={unexpected}")

  if freeze:
    for p in model.parameters():
      p.requires_grad = False
    model.eval()

  return model
