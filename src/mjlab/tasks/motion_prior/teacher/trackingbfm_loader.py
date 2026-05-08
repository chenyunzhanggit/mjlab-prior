"""Load a frozen Trackingbfm multi-motion tracking actor (single-encoder teacher).

The trackingbfm task (``Mjlab-Trackingbfm-Flat-Unitree-G1``) trains a plain
``rsl_rl.models.MLPModel`` actor on the tracking env's ``teacher_actor_terms``
observation. The checkpoint stores the actor under ``actor_state_dict``:

  - ``obs_normalizer.{_mean,_var,_std,count}`` shape ``(1, actor_obs_dim)``
  - ``mlp.{0,2,4,6,8,10,12}.{weight,bias}`` for the 7-layer MLP
  - ``distribution.std_param`` shape ``(29,)`` (scalar Gaussian)

Architectural constants below mirror
``mjlab/tasks/tracking/config/g1/rl_cfg.py:unitree_g1_trackingbfm_ppo_runner_cfg``
(``hidden_dims=(2048,2048,1024,1024,512,256,128)``, ELU, scalar Gaussian).

Unlike ``TeleopitTeacherCfg`` / ``VelocityTeacherCfg`` we leave
``actor_obs_dim`` un-set in the dataclass: the dim is determined at
construction time by whichever motion command + ``history_steps`` /
``future_steps`` the host env was registered with, so the runner discovers it
from the env's ``teacher_t`` obs group and passes it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict


@dataclass(frozen=True)
class TrackingbfmTeacherCfg:
  """Architectural hyperparameters of the trackingbfm tracking teacher."""

  actor_obs_dim: int
  """Concatenated dim of the trackingbfm actor obs (env-dependent)."""

  num_actions: int = 29
  hidden_dims: tuple[int, ...] = (2048, 2048, 1024, 1024, 512, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = True

  distribution_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )


def _make_dummy_obs(cfg: TrackingbfmTeacherCfg, batch_size: int = 1) -> TensorDict:
  """Build a TensorDict with the exact shape ``MLPModel`` expects."""
  return TensorDict(
    {"actor": torch.zeros(batch_size, cfg.actor_obs_dim)},
    batch_size=[batch_size],
  )


def build_trackingbfm_teacher(
  cfg: TrackingbfmTeacherCfg,
  *,
  device: str | torch.device = "cpu",
) -> MLPModel:
  """Instantiate the trackingbfm teacher with random weights (no ckpt yet)."""
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


def load_trackingbfm_teacher(
  ckpt_path: str | Path,
  *,
  cfg: TrackingbfmTeacherCfg,
  device: str | torch.device = "cpu",
  freeze: bool = True,
  strict: bool = True,
) -> MLPModel:
  """Load a frozen trackingbfm teacher policy from ``ckpt_path``.

  Args:
    ckpt_path: Path to the trackingbfm PPO checkpoint (``model_xxx.pt``).
      The file must contain a top-level ``actor_state_dict`` produced by
      ``MjlabOnPolicyRunner.save``.
    cfg: Architecture spec; ``actor_obs_dim`` MUST match the dim of the
      env's trackingbfm-style actor obs at training time (history/future
      step settings affect this).
    device: Device to place the model on.
    freeze: If True, set ``requires_grad=False`` on every parameter and
      switch the model to ``.eval()`` mode.
    strict: Forwarded to ``load_state_dict``.

  Returns:
    The fully-loaded ``MLPModel``. The teacher is **not** wrapped in
    ``torch.no_grad`` — callers should wrap inference themselves to keep
    autograd graphs clean.
  """
  ckpt_path = Path(ckpt_path).expanduser()
  if not ckpt_path.is_file():
    raise FileNotFoundError(f"Trackingbfm teacher checkpoint not found: {ckpt_path}")

  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  if "actor_state_dict" not in ckpt:
    raise KeyError(
      f"Expected 'actor_state_dict' in {ckpt_path}, got top-level keys "
      f"{list(ckpt.keys())}."
    )

  model = build_trackingbfm_teacher(cfg, device=device)
  missing, unexpected = model.load_state_dict(ckpt["actor_state_dict"], strict=strict)
  if missing or unexpected:
    print(f"[load_trackingbfm_teacher] missing={missing}\nunexpected={unexpected}")

  if freeze:
    for p in model.parameters():
      p.requires_grad = False
    model.eval()

  return model
