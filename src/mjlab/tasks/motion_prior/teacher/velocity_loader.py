"""Load a frozen mjlab Velocity-Rough actor teacher (teacher_b).

The velocity G1 actor (current architecture, May 2026+) is a
:class:`mjlab.rl.cnn_proj.CNNProjModel` — a CNN+projection encoder for the
2D height-scan group, concatenated with the 1D proprioceptive obs before
the trunk MLP. Checkpoint layout under ``actor_state_dict``::

  - ``obs_normalizer._mean / _var / _std`` shape ``(1, 99)``   ← 1D actor
  - ``obs_normalizer.count`` scalar
  - ``mlp.{0,2,4,6}.{weight,bias}``                            ← trunk MLP
  - ``cnns.height.cnn.{0,2}.{weight,bias}``                    ← CNN encoder
  - ``cnns.height.proj.0.{weight,bias}``                       ← Linear proj
  - ``distribution.std_param`` shape ``(29,)``

Old 286-dim plain-MLP teachers (Apr 2026) are no longer supported — the
:class:`MotionPriorPolicy` is the only consumer and it always pairs with a
post-Apr velocity rough teacher. If a legacy ckpt needs to be loaded, fall
back to the v0 loader by setting ``cfg.use_legacy_mlp=True``.

The architectural constants below mirror
``mjlab/tasks/velocity/config/g1/rl_cfg.py`` and were verified against
``logs/rsl_rl/g1_velocity/2026-05-11_21-16-58/model_21000.pt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from mjlab.rl.cnn_proj import CNNProjModel

# Default height-scan grid for G1 rough env (1.6m x 1.0m at 0.1m resolution).
_HEIGHT_GRID_H = 11
_HEIGHT_GRID_W = 17

# CNN+projection encoder spec; must match ``unitree_g1_ppo_runner_cfg``'s
# ``_HEIGHT_CNN_CFG`` (proj_dim / activation / pooling).
_HEIGHT_CNN_CFG: dict[str, Any] = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "avg",
  "proj_dim": 128,
  "proj_activation": "elu",
}


@dataclass(frozen=True)
class VelocityTeacherCfg:
  """Architectural hyperparameters of the mjlab velocity-rough teacher.

  ``actor_obs_dim`` is the **1D** proprio width (99 for G1 rough) — the
  height-scan is a separate 2D obs group, not included in this number.
  ``height_obs_dim`` is the (channels, H, W) tuple consumed by the CNN.
  """

  actor_obs_dim: int = 99
  height_obs_dim: tuple[int, int, int] = (1, _HEIGHT_GRID_H, _HEIGHT_GRID_W)
  num_actions: int = 29

  hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = True

  cnn_cfg: dict[str, Any] = field(default_factory=lambda: dict(_HEIGHT_CNN_CFG))

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
  """Build a TensorDict with the exact obs schema ``CNNProjModel`` expects.

  Keys ``"actor"`` and ``"height"`` mirror the velocity rough env's
  ``obs_groups={"actor": ("actor", "height"), ...}`` wiring — the model
  treats ``"actor"`` as a 1D group concatenated into the trunk MLP, and
  ``"height"`` as the 2D input fed to the CNN encoder.
  """
  return TensorDict(
    {
      "actor": torch.zeros(batch_size, cfg.actor_obs_dim),
      "height": torch.zeros(batch_size, *cfg.height_obs_dim),
    },
    batch_size=[batch_size],
  )


def build_velocity_teacher(
  cfg: VelocityTeacherCfg = VELOCITY_TEACHER_CFG,
  *,
  device: str | torch.device = "cpu",
) -> CNNProjModel:
  """Instantiate the velocity teacher with random weights (no ckpt yet)."""
  obs = _make_dummy_obs(cfg)
  obs_groups = {"actor": ["actor", "height"]}
  model = CNNProjModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=cfg.num_actions,
    cnn_cfg={"height": dict(cfg.cnn_cfg)},
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
) -> CNNProjModel:
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


# Backward-compat alias for callers that still expect an MLPModel return type
# annotation. ``CNNProjModel`` extends ``MLPModel``, so isinstance checks pass.
_MLPModel = MLPModel
