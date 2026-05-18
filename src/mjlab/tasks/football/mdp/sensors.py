"""Custom raycast patterns for football tasks.

The only public class here is :class:`ForwardPinholeCameraPatternCfg`, a
duck-typed alternative to mjlab's built-in :class:`PinholeCameraPatternCfg`.
The built-in one fires rays along the MuJoCo camera convention
(-Z forward), so attaching it to ``ObjRef(type="body", name="pelvis")``
would point the camera at the floor unless you also add a rotated site
to the robot's MJCF. ``ForwardPinholeCameraPatternCfg`` fires rays
along the **frame +X** axis (= pelvis forward when attached to a
standing humanoid) — no robot-spec changes needed.

Image → frame coordinate mapping (pelvis local: +X forward, +Y left,
+Z up; image: +u right, +v down):

  pixel +u (right)  →  frame -Y (right side of the robot)
  pixel +v (down)   →  frame -Z (down)
  ray  forward      →  frame +X

The ``generate_rays`` signature exactly matches
``mjlab.sensor.raycast_sensor.GridPatternCfg.generate_rays`` /
``PinholeCameraPatternCfg.generate_rays`` so passing an instance to
``RayCastSensorCfg.pattern`` works (the field is not runtime
type-checked).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  import mujoco


@dataclass
class ForwardPinholeCameraPatternCfg:
  """Pinhole-camera ray pattern with optical axis along frame +X."""

  width: int = 80
  """Image width in pixels (number of columns)."""

  height: int = 45
  """Image height in pixels (number of rows). VisualMimic uses 80×45."""

  fovy: float = 58.0
  """Vertical FoV in degrees. RealSense D435i default is ~58°."""

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(local_offsets, local_directions)``, both ``[H*W, 3]``.

    All rays share the same origin (offsets are zero — a true pinhole
    camera diverges from one optical centre); directions are unit
    vectors in frame-local coordinates.
    """
    del mj_model  # Not needed — we don't read MJCF camera intrinsics.

    v_fov_rad = math.radians(self.fovy)
    aspect = self.width / self.height
    h_fov_rad = 2.0 * math.atan(math.tan(v_fov_rad / 2.0) * aspect)

    # Normalised pixel coords in [-1, 1]. ``indexing="xy"`` gives
    # ``grid_u`` shape ``[height, width]`` and ``flatten()`` is
    # row-major: row 0 (v=-1, image top) first.
    u = torch.linspace(-1.0, 1.0, self.width, device=device, dtype=torch.float32)
    v = torch.linspace(-1.0, 1.0, self.height, device=device, dtype=torch.float32)
    grid_u, grid_v = torch.meshgrid(u, v, indexing="xy")

    ray_x = torch.ones_like(grid_u.flatten())                # forward (+X)
    ray_y = -grid_u.flatten() * math.tan(h_fov_rad / 2.0)    # right → -Y
    ray_z = -grid_v.flatten() * math.tan(v_fov_rad / 2.0)    # down  → -Z

    directions = torch.stack([ray_x, ray_y, ray_z], dim=1)
    directions = directions / directions.norm(dim=1, keepdim=True)

    num_rays = directions.shape[0]
    local_offsets = torch.zeros(
      (num_rays, 3), device=device, dtype=torch.float32
    )
    return local_offsets, directions
