from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import torch.nn.functional as F
from typing_extensions import override

if TYPE_CHECKING:
  from mjlab.utils.noise import noise_cfg


class NoiseModel:
  """Base class for noise models."""

  def __init__(
    self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str
  ):
    self._noise_model_cfg = noise_model_cfg
    self._num_envs = num_envs
    self._device = device

    # Validate configuration.
    if not hasattr(noise_model_cfg, "noise_cfg") or noise_model_cfg.noise_cfg is None:
      raise ValueError("NoiseModelCfg must have a valid noise_cfg")

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset noise model state. Override in subclasses if needed."""

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise to input data."""
    assert self._noise_model_cfg.noise_cfg is not None
    return self._noise_model_cfg.noise_cfg.apply(data)


class NoiseModelWithAdditiveBias(NoiseModel):
  """Noise model with additional additive bias that is constant for the duration
  of the entire episode."""

  def __init__(
    self,
    noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg,
    num_envs: int,
    device: str,
  ):
    super().__init__(noise_model_cfg, num_envs, device)

    # Validate bias configuration.
    if (
      not hasattr(noise_model_cfg, "bias_noise_cfg")
      or noise_model_cfg.bias_noise_cfg is None
    ):
      raise ValueError("NoiseModelWithAdditiveBiasCfg must have a valid bias_noise_cfg")

    self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
    self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    # Initialize bias tensor.
    self._bias = torch.zeros((num_envs, 1), device=self._device)
    self._num_components: int | None = None
    self._bias_initialized = False

  @override
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset bias values for specified environments."""
    indices = slice(None) if env_ids is None else env_ids
    # Sample new bias values.
    self._bias[indices] = self._bias_noise_cfg.apply(self._bias[indices])

  def _initialize_bias_shape(self, data_shape: torch.Size) -> None:
    """Initialize bias tensor shape based on data and configuration."""
    if self._sample_bias_per_component and not self._bias_initialized:
      *_, self._num_components = data_shape
      # Expand bias to match number of components.
      self._bias = self._bias.repeat(1, self._num_components)
      self._bias_initialized = True
      # Resample bias with new shape.
      self.reset()

  @override
  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    """Apply noise and additive bias to input data."""
    self._initialize_bias_shape(data.shape)
    noisy_data = super().__call__(data)
    return noisy_data + self._bias


##
# Image noise / transform functions.
##


class ImageNoiseModel:
  """Base image-noise model (no-op by default)."""

  def __init__(
    self,
    cfg: noise_cfg.ImageNoiseCfg,
    num_envs: int = 1,
    device: str | torch.device = "cpu",
  ):
    self.cfg = cfg
    self.num_envs = num_envs
    self.device = device

  def __call__(
    self,
    data: torch.Tensor,
    cfg: noise_cfg.ImageNoiseCfg,
    env_ids: torch.Tensor | Sequence[int],
  ) -> torch.Tensor:
    return data

  def reset(self, env_ids: Sequence[int] | None = None):
    pass


def depth_normalization(
  data: torch.Tensor,
  cfg: noise_cfg.DepthNormalizationCfg,
  env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
  """Clip depth to cfg.depth_range and optionally normalize to cfg.output_range."""
  if data.dim() == 4 and data.shape[-1] == 1:
    data = data.permute(0, 3, 1, 2)

  min_depth, max_depth = cfg.depth_range
  data = data.clip(min_depth, max_depth)

  if cfg.normalize:
    data = (data - min_depth) / (max_depth - min_depth)
    data = data * (cfg.output_range[1] - cfg.output_range[0]) + cfg.output_range[0]

  if data.dim() == 4 and data.shape[1] == 1:
    data = data.permute(0, 2, 3, 1)

  return data


def crop_and_resize(
  data: torch.Tensor,
  cfg: noise_cfg.CropAndResizeCfg,
  env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
  """Crop border pixels then resize.  data shape: (N, H, W, C)."""
  top, bottom, left, right = cfg.crop_region
  h, w = data.shape[1], data.shape[2]
  cropped = data[:, top : h - bottom, left : w - right, :]

  if cfg.resize_shape is None:
    return cropped

  cropped = cropped.permute(0, 3, 1, 2)  # (N, C, H, W)
  resized = F.interpolate(
    cropped, size=cfg.resize_shape, mode="bilinear", align_corners=False
  )
  return resized.permute(0, 2, 3, 1)  # (N, H, W, C)


def depth_distance_gaussian_noise(
  data: torch.Tensor,
  cfg: noise_cfg.DepthDistanceGaussianNoiseCfg,
  env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
  """Distance-dependent Gaussian noise: sigma = depth_std + depth * depth_std_multiplier.

  data shape: (N, H, W, C), values in meters (applied before normalization).
  """
  std_dev = cfg.depth_std + data * cfg.depth_std_multiplier
  return data + torch.randn_like(data) * std_dev


def depth_dropout(
  data: torch.Tensor,
  cfg: noise_cfg.DepthDropoutCfg,
  env_ids: torch.Tensor | Sequence[int],
) -> torch.Tensor:
  """Randomly invalidate depth pixels with probability cfg.drop_prob.

  data shape: (N, H, W, C). Applied after normalization so fill_value is in
  normalized space (default -1.0 lies outside [0, 1] so the policy can
  distinguish invalid pixels from real returns).
  """
  mask = torch.rand(data.shape[:-1], device=data.device) >= cfg.drop_prob
  return data * mask.unsqueeze(-1) + cfg.fill_value * (~mask).unsqueeze(-1).float()
