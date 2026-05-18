"""Ball- and goal-related observation terms.

Ports of the ball observations from
``motionprior/.../mdp/observations.py``. All terms read the ball entity
by name via ``env.scene[ball_name]`` and transform into the robot's body
frame using ``quat_apply_inverse(robot_quat, ...)`` when relative
information is exposed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def dribbling_goal_position(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """``(dx, dy, dist)`` of the goal point in world XY frame relative to
  the robot. Shape ``(num_envs, 3)``."""
  command = env.command_manager.get_term(command_name)
  robot: Entity = env.scene["robot"]
  robot_xy = robot.data.root_link_pos_w[:, :2]
  goal_xy = command.goal_pos[:, :2]
  rel = goal_xy - robot_xy
  dist = torch.norm(rel, dim=-1, keepdim=True)
  return torch.cat([rel, dist], dim=-1)


def ball_relative_position(
  env: ManagerBasedRlEnv,
  ball_name: str,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Ball position relative to the robot, in the robot's body frame
  (so XY rotates with yaw). Shape ``(num_envs, 3)``."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene[asset_name]
  rel_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
  return quat_apply_inverse(robot.data.root_link_quat_w, rel_w)


def ball_velocity(env: ManagerBasedRlEnv, ball_name: str) -> torch.Tensor:
  """Ball linear velocity in the world frame. Shape ``(num_envs, 3)``."""
  ball: Entity = env.scene[ball_name]
  return ball.data.root_link_lin_vel_w


def ball_to_goal_vector(
  env: ManagerBasedRlEnv,
  ball_name: str,
  command_name: str,
) -> torch.Tensor:
  """XY vector from ball to goal, expressed in the robot body frame.
  Shape ``(num_envs, 2)``."""
  ball: Entity = env.scene[ball_name]
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_term(command_name)

  ball_xy = ball.data.root_link_pos_w[:, :2]
  goal_xy = command.goal_pos[:, :2]
  diff = goal_xy - ball_xy  # (N, 2)

  diff_3d = torch.cat(
    [diff, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
  )
  diff_b = quat_apply_inverse(robot.data.root_link_quat_w, diff_3d)
  return diff_b[:, :2]


def ball_absolute_position(
  env: ManagerBasedRlEnv, ball_name: str
) -> torch.Tensor:
  """Ball position in the world frame. Shape ``(num_envs, 3)``.

  Privileged obs only — for multi-env training each env has its own
  origin, so this carries a large absolute offset.
  """
  ball: Entity = env.scene[ball_name]
  return ball.data.root_link_pos_w


def passing_source_position(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """``(dx, dy, dist)`` of the passing source position relative to the
  robot, in the robot body frame. Shape ``(num_envs, 3)``."""
  command = env.command_manager.get_term(command_name)
  robot: Entity = env.scene["robot"]
  robot_xy = robot.data.root_link_pos_w[:, :2]
  src_xy = command.source_pos[:, :2]
  rel = src_xy - robot_xy
  dist = torch.norm(rel, dim=-1, keepdim=True)

  rel_3d = torch.cat([rel, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
  rel_b = quat_apply_inverse(robot.data.root_link_quat_w, rel_3d)
  return torch.cat([rel_b[:, :2], dist], dim=-1)


def depth_image(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  miss_value: float | None = None,
  scale: float = 1.0,
  image_height: int | None = None,
  image_width: int | None = None,
  apply_mask: bool = False,
  mask_max_per_env: int = 6,
  mask_prob_per_slot: float = 0.10,
  mask_max_h_frac: float = 0.30,
  mask_max_w_frac: float = 0.30,
  bottom_left_prob: float = 0.20,
  bottom_left_h_frac: float = 0.40,
  bottom_left_w_frac: float = 0.30,
) -> torch.Tensor:
  """Flattened depth image from a :class:`RayCastSensor`, with optional
  VisualMimic-style sim2real augmentation (random rectangular masks).

  Returns the raw per-ray distances (metres) as a flat ``(num_envs, N)``
  vector, where ``N = num_frames * num_rays_per_frame``. Miss rays
  (``distance < 0`` from the raycaster) are clamped to ``miss_value``
  (defaults to the sensor's ``max_distance``). ``scale`` lets the caller
  normalise into ``[0, 1]`` for the policy network.

  When ``apply_mask=True`` (training only — disable on play), random
  rectangular masks are drawn on the 2D image to mimic the holes /
  artefacts seen on a real RealSense depth image after spatial-temporal
  filtering. Mirrors VisualMimic §III-D-a: up to ``mask_max_per_env``
  rectangles (each with probability ``mask_prob_per_slot``, max size
  ``mask_max_h_frac × mask_max_w_frac`` of the frame) filled with
  white (max_distance), black (0) or random gray; plus a fixed
  bottom-left white mask with probability ``bottom_left_prob``.

  ``image_height`` / ``image_width`` must be passed when ``apply_mask``
  is True (so we can reshape the flat tensor to ``(H, W)``). For a
  :class:`ForwardPinholeCameraPatternCfg` they're just ``pattern.height``
  and ``pattern.width``.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  max_d = float(sensor.cfg.max_distance)
  if miss_value is None:
    miss_value = max_d
  distances = sensor.data.distances  # [B, N]
  out = torch.where(
    distances < 0, torch.full_like(distances, miss_value), distances
  )
  if apply_mask:
    if image_height is None or image_width is None:
      raise ValueError(
        "depth_image: apply_mask=True requires image_height + image_width."
      )
    out = _apply_random_rect_masks(
      out,
      image_height=image_height,
      image_width=image_width,
      max_distance=max_d,
      mask_max_per_env=mask_max_per_env,
      mask_prob_per_slot=mask_prob_per_slot,
      mask_max_h_frac=mask_max_h_frac,
      mask_max_w_frac=mask_max_w_frac,
      bottom_left_prob=bottom_left_prob,
      bottom_left_h_frac=bottom_left_h_frac,
      bottom_left_w_frac=bottom_left_w_frac,
    )
  if scale != 1.0:
    out = out * scale
  return out


def _apply_random_rect_masks(
  flat_depth: torch.Tensor,
  *,
  image_height: int,
  image_width: int,
  max_distance: float,
  mask_max_per_env: int,
  mask_prob_per_slot: float,
  mask_max_h_frac: float,
  mask_max_w_frac: float,
  bottom_left_prob: float,
  bottom_left_h_frac: float,
  bottom_left_w_frac: float,
) -> torch.Tensor:
  """Stamp up to ``mask_max_per_env`` random rectangles per env onto a
  flattened ``(B, H*W)`` depth tensor.

  The masks are drawn on a 2D ``(H, W)`` view in-place, then flattened
  back. Fill values are drawn per-rectangle from {white=max_distance,
  black=0, gray ~ U(0, max_distance)} with equal probability.
  """
  B = flat_depth.shape[0]
  H, W = image_height, image_width
  device = flat_depth.device
  view = flat_depth.view(B, H, W)

  # ---- fixed bottom-left white mask (occluder under the chin) ----
  bl_h = int(H * bottom_left_h_frac)
  bl_w = int(W * bottom_left_w_frac)
  if bl_h > 0 and bl_w > 0:
    bl_apply = torch.rand(B, device=device) < bottom_left_prob  # [B]
    if bl_apply.any():
      # ``view`` is a non-contiguous reshape of ``flat_depth``; index_put_
      # works through the underlying storage via the contiguous flat copy.
      idx = bl_apply.nonzero(as_tuple=False).squeeze(-1)  # [K]
      view[idx, H - bl_h : H, 0:bl_w] = max_distance

  # ---- up to ``mask_max_per_env`` random rectangles per env ----
  if mask_max_per_env > 0:
    max_h = max(1, int(H * mask_max_h_frac))
    max_w = max(1, int(W * mask_max_w_frac))
    for _ in range(mask_max_per_env):
      apply = torch.rand(B, device=device) < mask_prob_per_slot
      if not apply.any():
        continue
      idx = apply.nonzero(as_tuple=False).squeeze(-1)  # [K]
      K = idx.numel()
      rect_h = torch.randint(1, max_h + 1, (K,), device=device)
      rect_w = torch.randint(1, max_w + 1, (K,), device=device)
      top = (torch.rand(K, device=device) * (H - rect_h.float())).long()
      left = (torch.rand(K, device=device) * (W - rect_w.float())).long()
      # 0=white, 1=black, 2=gray
      fill_kind = torch.randint(0, 3, (K,), device=device)
      gray = torch.rand(K, device=device) * max_distance
      fill_val = torch.where(
        fill_kind == 0,
        torch.full((K,), max_distance, device=device),
        torch.where(fill_kind == 1, torch.zeros(K, device=device), gray),
      )
      # Loop is short (≤6 per call typically). Vectorising further is not
      # worth the readability cost.
      for k in range(K):
        b = int(idx[k].item())
        t = int(top[k].item())
        l = int(left[k].item())
        h = int(rect_h[k].item())
        w = int(rect_w[k].item())
        view[b, t : t + h, l : l + w] = float(fill_val[k].item())
  return view.reshape(B, H * W)
