"""Load a frozen mjlab MultiMotionTracking actor as a single-teacher.

The :mod:`mjlab.tasks.multi_motion_tracking` task trains a plain
``rsl_rl.models.MLPModel`` (same family as :func:`load_velocity_teacher`),
producing a checkpoint laid out as::

  obs_normalizer._mean / _var / _std    shape ``(1, actor_obs_dim)``
  obs_normalizer.count                  scalar
  mlp.{0,2,4,6}.{weight,bias}           hidden_dims=(512, 256, 128) → 29
  distribution.std_param                shape ``(29,)`` (scalar Gaussian)

The single-encoder motion-prior task uses this teacher as its only target
policy. Unlike :class:`VelocityTeacherCfg` we do **not** hard-code
``actor_obs_dim`` — the multi-motion tracking actor obs depends on the
robot's body-name list (``key_body_pos_diff`` / ``key_body_rot_diff``
scale linearly with ``len(body_names)``), so callers pass the dimension
inferred from the env's ``teacher_tracking`` obs group at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict


@dataclass(frozen=True)
class TrackingTeacherCfg:
  """Architectural hyperparameters of the multi-motion tracking teacher.

  The defaults mirror
  :func:`mjlab.tasks.multi_motion_tracking.config.g1.rl_cfg.unitree_g1_multi_motion_tracking_ppo_runner_cfg`;
  ``actor_obs_dim`` is required because the obs schema width depends on
  ``len(body_names)`` and is not a portable constant.
  """

  actor_obs_dim: int
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


def _make_dummy_obs(cfg: TrackingTeacherCfg, batch_size: int = 1) -> TensorDict:
  """Build a TensorDict with the exact shape ``MLPModel`` expects."""
  return TensorDict(
    {"actor": torch.zeros(batch_size, cfg.actor_obs_dim)},
    batch_size=[batch_size],
  )


def build_tracking_teacher(
  cfg: TrackingTeacherCfg,
  *,
  device: str | torch.device = "cpu",
) -> MLPModel:
  """Instantiate the tracking teacher with random weights (no ckpt yet)."""
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


def load_tracking_teacher(
  ckpt_path: str | Path,
  *,
  cfg: TrackingTeacherCfg,
  device: str | torch.device = "cpu",
  freeze: bool = True,
  strict: bool = True,
) -> MLPModel:
  """Load a frozen multi-motion tracking teacher from ``ckpt_path``."""
  ckpt_path = Path(ckpt_path).expanduser()
  if not ckpt_path.is_file():
    raise FileNotFoundError(f"Tracking teacher checkpoint not found: {ckpt_path}")

  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  if "actor_state_dict" not in ckpt:
    raise KeyError(
      f"Expected 'actor_state_dict' in {ckpt_path}, got top-level keys "
      f"{list(ckpt.keys())}."
    )

  model = build_tracking_teacher(cfg, device=device)
  missing, unexpected = model.load_state_dict(ckpt["actor_state_dict"], strict=strict)
  if missing or unexpected:
    print(f"[load_tracking_teacher] missing={missing}\nunexpected={unexpected}")

  if freeze:
    for p in model.parameters():
      p.requires_grad = False
    model.eval()

  return model
