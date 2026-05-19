from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from mjlab.sensor.grouped_ray_caster import GroupedRayCasterCamera
from mjlab.utils.lab_api import math as math_utils

from .noisy_camera import NoisyCameraMixin

if TYPE_CHECKING:
  from .noisy_grouped_raycaster_camera_cfg import NoisyGroupedRayCasterCameraCfg


class NoisyGroupedRayCasterCamera(NoisyCameraMixin, GroupedRayCasterCamera):
  """GroupedRayCasterCamera with configurable noise pipeline and optional history buffers."""

  cfg: NoisyGroupedRayCasterCameraCfg

  def initialize(self, mj_model, model, data, device: str) -> None:
    super().initialize(mj_model, model, data, device)
    # Store nominal offset so we can re-apply perturbations on top each reset.
    self._nominal_offset_pos = self._offset_pos.clone()
    self._nominal_offset_quat = self._offset_quat.clone()
    self.build_noise_pipeline()
    self.build_history_buffers()

  def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None):
    super().reset(env_ids)
    self._apply_extrinsic_perturbation(env_ids)
    self._apply_intrinsic_perturbation(env_ids)
    self.reset_noise_pipeline(env_ids)
    self.reset_history_buffers(env_ids)

  def postprocess_rays(self) -> None:
    super().postprocess_rays()
    if self._update_period_s > 0.0:
      env_ids = self.refresh_mask.nonzero(as_tuple=False).squeeze(-1)
    else:
      env_ids = self._ALL_INDICES
    if env_ids.numel() == 0:
      return
    self.apply_noise_pipeline_to_all_data_types(env_ids)
    self.update_history_buffers(env_ids)

  # ------------------------------------------------------------------
  # Extrinsic / intrinsic perturbation helpers
  # ------------------------------------------------------------------

  def _apply_extrinsic_perturbation(
    self, env_ids: Sequence[int] | torch.Tensor | None
  ) -> None:
    """Perturb camera position and orientation at episode reset."""
    ext_cfg = self.cfg.extrinsic_perturbation
    if ext_cfg is None:
      return

    device = self._runtime_device
    assert device is not None
    ids = (
      self._ALL_INDICES
      if env_ids is None
      else torch.as_tensor(env_ids, device=device, dtype=torch.long)
    )
    n = ids.shape[0]

    # --- position perturbation ---
    px, py, pz = ext_cfg.pos_range
    pos_noise = torch.empty(n, 3, device=device).uniform_(-1.0, 1.0)
    pos_noise *= torch.tensor([px, py, pz], device=device)
    self._offset_pos[ids] = self._nominal_offset_pos[ids] + pos_noise

    # --- orientation perturbation (roll, pitch, yaw) ---
    roll = torch.empty(n, device=device).uniform_(
      -ext_cfg.roll_range, ext_cfg.roll_range
    )
    pitch = torch.empty(n, device=device).uniform_(
      -ext_cfg.pitch_range, ext_cfg.pitch_range
    )
    yaw = torch.empty(n, device=device).uniform_(-ext_cfg.yaw_range, ext_cfg.yaw_range)

    # Build perturbation quaternion from rpy.
    perturb_quat = _rpy_to_quat(roll, pitch, yaw, device=device)  # (n, 4) wxyz
    self._offset_quat[ids] = math_utils.quat_mul(
      self._nominal_offset_quat[ids], perturb_quat
    )

  def _apply_intrinsic_perturbation(
    self, env_ids: Sequence[int] | torch.Tensor | None
  ) -> None:
    """Perturb camera FOV and principal point at episode reset."""
    int_cfg = self.cfg.intrinsic_perturbation
    if int_cfg is None:
      return

    device = self._runtime_device
    assert device is not None
    ids = (
      self._ALL_INDICES
      if env_ids is None
      else torch.as_tensor(env_ids, device=device, dtype=torch.long)
    )
    n = ids.shape[0]

    # Clone nominal intrinsics for these envs.
    K = self._camera_data.intrinsic_matrices[ids].clone()  # (n, 3, 3)

    # FOV perturbation: scale f_x and f_y by the same factor.
    # Δfov drawn uniformly; new_fovy = nominal_fovy + Δfov → scale = tan(new/2)/tan(nom/2)
    nom_fy = K[:, 1, 1]  # (n,)
    h = float(self.cfg.pattern.height)
    nom_half_fovy = torch.atan(h / (2.0 * nom_fy))  # (n,)
    delta_fov_rad = torch.empty(n, device=device).uniform_(
      -int_cfg.fov_range, int_cfg.fov_range
    ) * (math.pi / 180.0)
    new_half_fovy = (nom_half_fovy + delta_fov_rad / 2.0).clamp(
      0.05, math.pi / 2 - 0.05
    )
    scale = torch.tan(nom_half_fovy) / torch.tan(new_half_fovy)  # (n,)
    K[:, 0, 0] = K[:, 0, 0] * scale
    K[:, 1, 1] = K[:, 1, 1] * scale

    # Principal point perturbation.
    K[:, 0, 2] += torch.empty(n, device=device).uniform_(
      -int_cfg.cx_range, int_cfg.cx_range
    )
    K[:, 1, 2] += torch.empty(n, device=device).uniform_(
      -int_cfg.cy_range, int_cfg.cy_range
    )

    self.set_intrinsic_matrices(
      K, focal_length=self._focal_length, env_ids=ids.tolist()
    )


def _rpy_to_quat(
  roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor, device: str
) -> torch.Tensor:
  """Convert roll/pitch/yaw (radians) to quaternion (w, x, y, z). Shape: (N, 4)."""
  cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
  cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
  cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy
  return torch.stack([w, x, y, z], dim=-1)
