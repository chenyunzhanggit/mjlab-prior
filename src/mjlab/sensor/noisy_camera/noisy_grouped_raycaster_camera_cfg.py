from dataclasses import dataclass

from mjlab.sensor.grouped_ray_caster import GroupedRayCasterCameraCfg

from .noisy_camera_cfg import NoisyCameraCfgMixin
from .noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera


@dataclass(kw_only=True)
class NoisyGroupedRayCasterCameraCfg(NoisyCameraCfgMixin, GroupedRayCasterCameraCfg):
  """Configuration for NoisyGroupedRayCasterCamera (noise pipeline + optional history)."""

  class_type: type = NoisyGroupedRayCasterCamera

  def build(self) -> NoisyGroupedRayCasterCamera:
    return NoisyGroupedRayCasterCamera(self)
