from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import torch
from typing_extensions import override

from mjlab.utils.noise import noise_model

# Type alias for noise parameters: scalar or per-dimension values.
NoiseParam = float | tuple[float, ...]


@dataclass(kw_only=True)
class NoiseCfg(abc.ABC):
  """Base configuration for a noise term."""

  operation: Literal["add", "scale", "abs"] = "add"

  # Cache for converted tensors, keyed by device string.
  _tensor_cache: dict[str, dict[str, torch.Tensor]] = field(
    default_factory=dict, init=False, repr=False
  )

  def _get_cached_tensor(
    self, name: str, value: NoiseParam, device: torch.device
  ) -> torch.Tensor:
    """Get a cached tensor for the given parameter on the specified device."""
    device_key = str(device)
    if device_key not in self._tensor_cache:
      self._tensor_cache[device_key] = {}
    if name not in self._tensor_cache[device_key]:
      self._tensor_cache[device_key][name] = torch.tensor(value, device=device)
    return self._tensor_cache[device_key][name]

  @abc.abstractmethod
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to the input data."""


@dataclass
class ConstantNoiseCfg(NoiseCfg):
  bias: NoiseParam = 0.0

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    bias = self._get_cached_tensor("bias", self.bias, data.device)

    if self.operation == "add":
      return data + bias
    elif self.operation == "scale":
      return data * bias
    elif self.operation == "abs":
      return torch.zeros_like(data) + bias
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class UniformNoiseCfg(NoiseCfg):
  n_min: NoiseParam = -1.0
  n_max: NoiseParam = 1.0

  def __post_init__(self):
    if isinstance(self.n_min, float) and isinstance(self.n_max, float):
      if self.n_min >= self.n_max:
        raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    n_min = self._get_cached_tensor("n_min", self.n_min, data.device)
    n_max = self._get_cached_tensor("n_max", self.n_max, data.device)

    # Generate uniform noise in [0, 1) and scale to [n_min, n_max).
    noise = torch.rand_like(data) * (n_max - n_min) + n_min

    if self.operation == "add":
      return data + noise
    elif self.operation == "scale":
      return data * noise
    elif self.operation == "abs":
      return noise
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


@dataclass
class GaussianNoiseCfg(NoiseCfg):
  mean: NoiseParam = 0.0
  std: NoiseParam = 1.0

  def __post_init__(self):
    if isinstance(self.std, float) and self.std <= 0:
      raise ValueError(f"std ({self.std}) must be positive")

  @override
  def apply(self, data: torch.Tensor) -> torch.Tensor:
    mean = self._get_cached_tensor("mean", self.mean, data.device)
    std = self._get_cached_tensor("std", self.std, data.device)

    # Generate standard normal noise and scale.
    noise = mean + std * torch.randn_like(data)

    if self.operation == "add":
      return data + noise
    elif self.operation == "scale":
      return data * noise
    elif self.operation == "abs":
      return noise
    else:
      raise ValueError(f"Unsupported noise operation: {self.operation}")


##
# Noise models.
##


@dataclass(kw_only=True)
class NoiseModelCfg:
  """Configuration for a noise model."""

  noise_cfg: NoiseCfg

  class_type: ClassVar[type[noise_model.NoiseModel]] = noise_model.NoiseModel

  def __init_subclass__(cls, class_type: type[noise_model.NoiseModel]):
    cls.class_type = class_type


@dataclass(kw_only=True)
class NoiseModelWithAdditiveBiasCfg(
  NoiseModelCfg, class_type=noise_model.NoiseModelWithAdditiveBias
):
  """Configuration for an additive Gaussian noise with bias model."""

  bias_noise_cfg: NoiseCfg | None = None
  sample_bias_per_component: bool = True

  def __post_init__(self):
    if self.bias_noise_cfg is None:
      raise ValueError(
        "bias_noise_cfg must be specified for NoiseModelWithAdditiveBiasCfg"
      )


# ============================================================================
# Image noise pipeline (ported from mjlab-prior-main:utils/noise/noise_cfg.py).
#
# Each ImageNoiseCfg subclass describes one stage in a depth-image noise
# pipeline that lives **inside** the camera sensor (NoisyCameraMixin runs
# them in dict-insertion order on the sensor's output buffer). The signature
# of every ``func`` is:
#
#     func(data: torch.Tensor, cfg: ImageNoiseCfg, env_ids) -> torch.Tensor
#
# ``data`` shape is ``(N, H, W, C)`` (channels-last, mjlab/IsaacLab convention).
# ============================================================================


@dataclass(kw_only=True)
class ImageNoiseCfg:
  """Base configuration for image noise / transform terms."""

  func: (
    Callable[
      [torch.Tensor, "ImageNoiseCfg", torch.Tensor | Sequence[int]], torch.Tensor
    ]
    | type[noise_model.ImageNoiseModel]
  ) = noise_model.ImageNoiseModel
  """Callable that applies the transform. Signature: (data, cfg, env_ids) -> data."""

  device: str | torch.device = "cpu"


@dataclass(kw_only=True)
class DepthNormalizationCfg(ImageNoiseCfg):
  """Clip depth to a range and optionally normalize to [0, 1]."""

  depth_range: tuple[float, float] = (0.0, 10.0)
  normalize: bool = True
  output_range: tuple[float, float] = (0.0, 1.0)
  func: Callable[
    [torch.Tensor, "DepthNormalizationCfg", torch.Tensor | Sequence[int]], torch.Tensor
  ] = noise_model.depth_normalization


@dataclass(kw_only=True)
class CropAndResizeCfg(ImageNoiseCfg):
  """Crop border pixels and resize the image."""

  crop_region: tuple[int, int, int, int] = (0, 0, 0, 0)
  """(top, bottom, left, right) pixels to crop from each edge."""

  resize_shape: tuple[int, int] | None = None
  """Output (height, width) after resize. None = no resize."""

  func: Callable[
    [torch.Tensor, "CropAndResizeCfg", torch.Tensor | Sequence[int]], torch.Tensor
  ] = noise_model.crop_and_resize


@dataclass(kw_only=True)
class DepthDistanceGaussianNoiseCfg(ImageNoiseCfg):
  """Distance-dependent Gaussian noise.

  ``sigma = depth_std + depth * depth_std_multiplier``

  Applied in meter space (before normalization). Matches real depth camera
  behaviour where noise grows with distance::

    sigma = 0.005 + 0.01 * d  →  at 2 m: sigma ≈ 0.025 m
  """

  depth_std: float = 0.005
  """Base noise std (meters). Independent of distance."""

  depth_std_multiplier: float = 0.01
  """Per-meter noise growth. Total std = depth_std + depth * depth_std_multiplier."""

  func: Callable[
    [torch.Tensor, "DepthDistanceGaussianNoiseCfg", torch.Tensor | Sequence[int]],
    torch.Tensor,
  ] = noise_model.depth_distance_gaussian_noise


@dataclass(kw_only=True)
class DepthDropoutCfg(ImageNoiseCfg):
  """Randomly invalidate depth pixels to simulate missing returns.

  Each pixel is independently dropped with probability ``drop_prob``.
  Applied after normalization; dropped pixels receive ``fill_value``
  (default -1.0) which lies outside [0, 1] so the policy can distinguish
  them from real returns.
  """

  drop_prob: float = 0.01
  """Probability in [0, 1) that a pixel is dropped."""

  fill_value: float = -1.0
  """Value for dropped pixels. -1.0 is outside normalized [0, 1] range."""

  func: Callable[
    [torch.Tensor, "DepthDropoutCfg", torch.Tensor | Sequence[int]], torch.Tensor
  ] = noise_model.depth_dropout


@dataclass(kw_only=True)
class RectMaskCfg(ImageNoiseCfg):
  """VisualMimic-style random-rectangle mask augmentation (D2=b).

  On each call, with independent Bernoulli for every env, stamps up to
  ``max_rects`` axis-aligned rectangles onto the depth image. Each rect's
  fill is uniformly drawn from {white = ``max_value``, black = 0,
  gray ~ U(0, ``max_value``)} to mimic three real-camera failure modes
  (saturated white, dead pixels, smoothed-out blur). Optionally also
  stamps a fixed bottom-left rectangle to simulate the robot's chin /
  shoulder occluder seen on a chest-mounted RealSense (VisualMimic
  §III-D-a).

  Applied **after** normalization (so ``max_value`` is the normalized
  high end, typically 1.0).
  """

  max_value: float = 1.0
  """Fill value for "white" mask and upper bound of gray. Should match
  the upper end of the depth pipeline's normalized range."""

  max_rects: int = 6
  """Maximum number of random rectangles per env."""

  prob_per_slot: float = 0.10
  """Per-rect Bernoulli probability."""

  max_h_frac: float = 0.30
  """Each rect's max height as a fraction of image H."""

  max_w_frac: float = 0.30
  """Each rect's max width as a fraction of image W."""

  bottom_left_prob: float = 0.20
  """Probability of stamping the fixed bottom-left occluder."""

  bottom_left_h_frac: float = 0.40
  """Bottom-left occluder height fraction."""

  bottom_left_w_frac: float = 0.30
  """Bottom-left occluder width fraction."""

  func: Callable[
    [torch.Tensor, "RectMaskCfg", torch.Tensor | Sequence[int]], torch.Tensor
  ] = noise_model.rect_mask
