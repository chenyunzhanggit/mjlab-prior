"""Load a frozen Teleopit ``TemporalCNNModel`` teacher from a saved checkpoint.

The Teleopit checkpoint (e.g. ``~/project/Teleopit/track.pt``) stores the
actor's state dict under the top-level key ``actor_state_dict``. The shapes
reveal the fixed architectural choices used to train it (verified by
``scripts/inspect_teleopit_ckpt.py``):

  - ``actor`` (1-D current-frame obs): D = 166
  - ``actor_history`` (3-D obs): T = 10, D = 166
  - MLP: in=198 (=166+32), hidden=(1024,512,256,256,128), out=29
  - Conv1D: 166 → 128 → 64 → 32, kernel_size=3, ELU, AdaptiveAvgPool1d(1)
  - GaussianDistribution: scalar std, std_param shape (29,)

These constants are baked in here as ``TELEOPIT_TEACHER_CFG`` so that the
loader can reconstruct the model without needing live access to the Teleopit
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from mjlab.tasks.motion_prior.teacher.temporal_cnn_model import TemporalCNNModel

# ---------------------------------------------------------------------------
# Architectural constants — must match Teleopit/train_mimic/tasks/tracking/
# config/rl.py (verified against track.pt shapes April 2026).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeleopitTeacherCfg:
  """Architectural hyperparameters of the Teleopit tracking teacher."""

  actor_obs_dim: int = 166
  actor_history_length: int = 10
  actor_history_obs_dim: int = 166
  num_actions: int = 29

  hidden_dims: tuple[int, ...] = (1024, 512, 256, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = True

  cnn_output_channels: tuple[int, ...] = (128, 64, 32)
  cnn_kernel_size: int = 3
  cnn_global_pool: str = "avg"

  distribution_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )


TELEOPIT_TEACHER_CFG = TeleopitTeacherCfg()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dummy_obs(
  cfg: TeleopitTeacherCfg = TELEOPIT_TEACHER_CFG, batch_size: int = 1
) -> TensorDict:
  """Build a TensorDict with the exact shapes ``TemporalCNNModel`` expects.

  The model's ``__init__`` only uses ``obs[g].shape`` to infer dimensions, so
  zeros are fine.
  """
  return TensorDict(
    {
      "actor": torch.zeros(batch_size, cfg.actor_obs_dim),
      "actor_history": torch.zeros(
        batch_size, cfg.actor_history_length, cfg.actor_history_obs_dim
      ),
    },
    batch_size=[batch_size],
  )


def build_teleopit_teacher(
  cfg: TeleopitTeacherCfg = TELEOPIT_TEACHER_CFG,
  *,
  device: str | torch.device = "cpu",
) -> TemporalCNNModel:
  """Instantiate the teacher network with random weights (no ckpt yet)."""
  obs = make_dummy_obs(cfg)
  obs_groups = {"actor": ["actor", "actor_history"]}

  cnn_cfg = {
    "output_channels": cfg.cnn_output_channels,
    "kernel_size": cfg.cnn_kernel_size,
    "activation": cfg.activation,
    "global_pool": cfg.cnn_global_pool,
  }

  # rsl_rl's MLPModel mutates ``distribution_cfg`` (calls .pop on it),
  # so we always pass a fresh copy to keep the cached cfg intact.
  model = TemporalCNNModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=cfg.num_actions,
    hidden_dims=cfg.hidden_dims,
    activation=cfg.activation,
    obs_normalization=cfg.obs_normalization,
    distribution_cfg=dict(cfg.distribution_cfg),
    cnn_cfg=cnn_cfg,
  )
  return model.to(device)


def load_teleopit_teacher(
  ckpt_path: str | Path,
  *,
  cfg: TeleopitTeacherCfg = TELEOPIT_TEACHER_CFG,
  device: str | torch.device = "cpu",
  freeze: bool = True,
  strict: bool = True,
) -> TemporalCNNModel:
  """Load a frozen Teleopit teacher policy from ``ckpt_path``.

  Args:
    ckpt_path: Path to ``track.pt``-style checkpoint with top-level
      ``actor_state_dict`` key.
    cfg: Architecture spec; defaults match Teleopit's tracking task.
    device: Device to place the model on.
    freeze: If True, set ``requires_grad=False`` on every parameter and
      switch the model to ``.eval()`` mode.
    strict: Forwarded to ``load_state_dict``.

  Returns:
    The fully-loaded ``TemporalCNNModel``. The teacher is **not** wrapped
    in ``torch.no_grad`` — callers should wrap inference themselves to keep
    autograd graphs clean.
  """
  ckpt_path = Path(ckpt_path).expanduser()
  if not ckpt_path.is_file():
    raise FileNotFoundError(f"Teleopit teacher checkpoint not found: {ckpt_path}")

  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  if "actor_state_dict" not in ckpt:
    raise KeyError(
      f"Expected 'actor_state_dict' in {ckpt_path}, got top-level keys "
      f"{list(ckpt.keys())}."
    )

  model = build_teleopit_teacher(cfg, device=device)
  missing, unexpected = model.load_state_dict(ckpt["actor_state_dict"], strict=strict)
  if missing or unexpected:
    # With strict=True this branch is unreachable (load_state_dict raises),
    # but keep a hook for strict=False debugging.
    print(f"[load_teleopit_teacher] missing={missing}\nunexpected={unexpected}")

  if freeze:
    for p in model.parameters():
      p.requires_grad = False
    model.eval()

  return model
