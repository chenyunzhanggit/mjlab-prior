from __future__ import annotations

import inspect
from collections.abc import Sequence

import torch

from mjlab.utils.lab_api.string import string_to_callable
from mjlab.utils.noise import ImageNoiseCfg


class NoisyCameraMixin:
  """Mixin that adds a configurable noise/transform pipeline to a camera sensor."""

  def build_noise_pipeline(self):
    """Build the noise pipeline from cfg.noise_pipeline."""
    self.noise_pipeline: Sequence[ImageNoiseCfg] | list[ImageNoiseCfg] = []

    for noise_name, noise_cfg in self.cfg.noise_pipeline.items():  # type: ignore[attr-defined]
      if not isinstance(noise_cfg, ImageNoiseCfg):
        raise ValueError(f"Invalid noise configuration for {noise_name}: {noise_cfg}")

      noise_cfg.device = self._device

      if isinstance(noise_cfg.func, str):
        noise_cfg.func = string_to_callable(noise_cfg.func)

      if inspect.isclass(noise_cfg.func):
        noise_cfg.func = noise_cfg.func(
          noise_cfg, num_envs=self._num_envs, device=self._device
        )

      self.noise_pipeline.append(noise_cfg)

    # Initialise noised output buffers.
    for data_type in self.cfg.data_types:  # type: ignore[attr-defined]
      self._camera_data.output[f"{data_type}_noised"] = self.apply_noise_pipeline(
        self._camera_data.output[data_type], env_ids=self._ALL_INDICES
      )

  def apply_noise_pipeline(
    self, data: torch.Tensor, env_ids: torch.Tensor | Sequence[int]
  ) -> torch.Tensor:
    """Apply the full noise pipeline to *data* (shape N_, H, W, C)."""
    if self.noise_pipeline is None:
      raise RuntimeError("Noise pipeline not built. Call build_noise_pipeline() first.")
    if len(self.noise_pipeline) == 0:
      return data.clone()
    for noise_cfg in self.noise_pipeline:
      data = noise_cfg.func(data, noise_cfg, env_ids)  # type: ignore[operator]
    return data

  def apply_noise_pipeline_to_all_data_types(
    self, env_ids: torch.Tensor | Sequence[int]
  ):
    """Apply the pipeline to every configured data type for the given envs."""
    for data_type in self.cfg.data_types:  # type: ignore[attr-defined]
      self._camera_data.output[f"{data_type}_noised"][env_ids] = (
        self.apply_noise_pipeline(
          self._camera_data.output[data_type][env_ids], env_ids=env_ids
        )
      )

  def reset_noise_pipeline(self, env_ids: Sequence[int] | None = None):
    """Reset stateful noise functions (e.g. latency buffer)."""
    if self.noise_pipeline is None:
      raise RuntimeError("Noise pipeline not built. Call build_noise_pipeline() first.")
    for noise_cfg in self.noise_pipeline:
      if hasattr(noise_cfg.func, "reset"):
        noise_cfg.func.reset(env_ids)

  # ------------------------------------------------------------------
  # History buffers (optional — only allocated when data_histories != {})
  # ------------------------------------------------------------------

  def build_history_buffers(self):
    """Build history ring-buffers for each data type listed in cfg.data_histories."""
    from mjlab.utils.buffers.async_circular_buffer import AsyncCircularBuffer

    self.output_history_buffers: dict[str, AsyncCircularBuffer] = {}

    for data_type, history_length in self.cfg.data_histories.items():  # type: ignore[attr-defined]
      self.output_history_buffers[data_type] = AsyncCircularBuffer(
        history_length, self._num_envs, self._device
      )
      data_shape = self._camera_data.output[data_type].shape
      self._camera_data.output[f"{data_type}_history"] = torch.zeros(
        (data_shape[0], history_length, *data_shape[1:]), device=self._device
      )

  def update_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
    """Append current output to history buffers."""
    for data_type in self.cfg.data_histories.keys():  # type: ignore[attr-defined]
      self.output_history_buffers[data_type].append(
        self._camera_data.output[data_type][env_ids], env_ids
      )
      self._camera_data.output[f"{data_type}_history"][env_ids] = (
        self.output_history_buffers[data_type].__getitem__(batch_ids=env_ids)
      )

  def reset_history_buffers(self, env_ids: torch.Tensor | Sequence[int] | None):
    """Reset history ring-buffers."""
    for data_type in self.cfg.data_histories.keys():  # type: ignore[attr-defined]
      self.output_history_buffers[data_type].reset(env_ids)
