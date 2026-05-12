from dataclasses import dataclass, field

from mjlab.utils.noise import ImageNoiseCfg, NoiseCfg


@dataclass(kw_only=True)
class ExtrinsicPerturbationCfg:
  """Per-episode extrinsic (pose) perturbation sampled at env reset.

  All ranges are symmetric: the actual perturbation is drawn from Uniform(-x, +x).
  Perturbations are applied on top of the nominal offset configured in the camera cfg.
  """

  pos_range: tuple[float, float, float] = (0.02, 0.02, 0.02)
  """Max position offset (meters) for x, y, z axes."""

  roll_range: float = 0.01745  # 1 deg in radians
  """Max roll perturbation (radians)."""

  pitch_range: float = 0.08727  # 5 deg in radians
  """Max pitch perturbation (radians)."""

  yaw_range: float = 0.01745  # 1 deg in radians
  """Max yaw perturbation (radians)."""


@dataclass(kw_only=True)
class IntrinsicPerturbationCfg:
  """Per-episode intrinsic (camera matrix) perturbation sampled at env reset.

  All ranges are symmetric: the actual perturbation is drawn from Uniform(-x, +x).
  """

  fov_range: float = 5.0
  """Max FOV perturbation (degrees). Scales both f_x and f_y uniformly."""

  cx_range: float = 1.0
  """Max principal-point cx perturbation (pixels)."""

  cy_range: float = 1.0
  """Max principal-point cy perturbation (pixels)."""


@dataclass(kw_only=True)
class NoisyCameraCfgMixin:
  """Mixin config that adds noise pipeline and history settings to a camera cfg."""

  noise_pipeline: dict[str, ImageNoiseCfg | NoiseCfg] = field(default_factory=dict)
  """Named noise/transform stages applied in insertion order.
    All data_types listed in the camera cfg will pass through this pipeline.
    """

  data_histories: dict[str, int] = field(default_factory=dict)
  """Map of data_type -> history length.
    Stacked history is stored under ``sensor.data.output[f"{data_type}_history"]``.
    """

  extrinsic_perturbation: ExtrinsicPerturbationCfg | None = None
  """If set, camera pose (position + orientation) is randomly perturbed at each
    env reset. Simulates mounting errors. Perturbation is fixed per episode."""

  intrinsic_perturbation: IntrinsicPerturbationCfg | None = None
  """If set, camera intrinsics (FOV, principal point) are randomly perturbed at
    each env reset. Simulates calibration errors. Perturbation is fixed per episode."""
