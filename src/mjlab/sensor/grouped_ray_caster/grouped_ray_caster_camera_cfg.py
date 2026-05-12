"""Configuration for the grouped-ray-cast camera sensor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mjlab.sensor.raycast_sensor import PinholeCameraPatternCfg

from .grouped_ray_caster_camera import GroupedRayCasterCamera
from .grouped_ray_caster_cfg import GroupedRayCasterCfg


@dataclass(kw_only=True)
class GroupedRayCasterCameraCfg(GroupedRayCasterCfg):
  """Configuration for the grouped-ray-cast camera sensor."""

  class_type: type = GroupedRayCasterCamera

  @dataclass(kw_only=True)
  class OffsetCfg:
    """The offset pose of the sensor's frame from the sensor's parent frame."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""

    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion rotation (w, x, y, z) w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    convention: Literal["opengl", "ros", "world"] = "ros"
    """The convention in which the frame offset is applied. Defaults to "ros".

        - ``"opengl"`` - forward axis: ``-Z`` - up axis: ``+Y``
        - ``"ros"``    - forward axis: ``+Z`` - up axis: ``-Y``
        - ``"world"``  - forward axis: ``+X`` - up axis: ``+Z``
        """

  offset: OffsetCfg = field(default_factory=OffsetCfg)
  """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""

  data_types: list[str] = field(default_factory=lambda: ["distance_to_image_plane"])
  """List of sensor names/types to enable for the camera. Defaults to ["distance_to_image_plane"]."""

  update_period: float = 0.0
  """Camera refresh period in seconds.

    - ``<= 0``: refresh on every ``sim.sense()`` call.
    - ``> 0``: refresh at most once every ``update_period`` seconds.
    """

  depth_clipping_behavior: Literal["max", "zero", "none"] = "none"
  """Clipping behavior for values that exceed the maximum distance.

    - ``"max"``: Values are clipped to max_distance.
    - ``"zero"``: Values are set to zero.
    - ``"none"``: No clipping (``inf`` / ``nan`` for misses).
    """

  pattern: PinholeCameraPatternCfg = field(default_factory=PinholeCameraPatternCfg)
  """Pinhole camera pattern (height, width, fovy)."""

  focal_length: float | None = None
  """Optional focal length for aperture-based intrinsic construction."""

  horizontal_aperture: float | None = None
  """Optional horizontal aperture for aperture-based intrinsic construction."""

  vertical_aperture: float | None = None
  """Optional vertical aperture for aperture-based intrinsic construction."""

  horizontal_aperture_offset: float = 0.0
  """Horizontal aperture offset in aperture-based intrinsic construction."""

  vertical_aperture_offset: float = 0.0
  """Vertical aperture offset in aperture-based intrinsic construction."""

  def build(self) -> GroupedRayCasterCamera:
    return GroupedRayCasterCamera(self)
