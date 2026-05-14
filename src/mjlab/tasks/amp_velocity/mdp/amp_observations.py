"""Observation functions for the AMP feature vector.

Produces, in the anchor body's frame, for every tracked body:
  - position (3)
  - orientation as the first two columns of its rotation matrix (6)
  - body-local linear velocity (3)
  - body-local angular velocity (3)

The resulting per-step vector is fed to the AMP discriminator alongside the
expert clip features built by ``AMPLoader``. Both sides share the same
feature definition so the discriminator sees comparable distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Sentinel defaults. Callers must always pass explicit body_names; these exist
# only so the signature matches mjlab's MDP-term call convention.
_EMPTY_BODY_CFG = SceneEntityCfg("robot", body_names=())


def _anchor_id(cfg: SceneEntityCfg) -> int:
  """Extract the single anchor body id from a SceneEntityCfg."""
  ids = cfg.body_ids
  assert isinstance(ids, list) and len(ids) == 1, (
    f"AMP anchor cfg must resolve to exactly one body, got {ids!r}"
  )
  return ids[0]


def robot_body_pos_b(
  env: ManagerBasedRlEnv,
  anchor_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
  body_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
) -> torch.Tensor:
  """Tracked body positions expressed in the anchor body's frame."""
  asset: Entity = env.scene[anchor_cfg.name]
  anchor_id = _anchor_id(anchor_cfg)
  anchor_pos_w = asset.data.body_link_pos_w[:, anchor_id]
  anchor_quat_w = asset.data.body_link_quat_w[:, anchor_id]

  body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]

  num_bodies = body_pos_w.shape[1]
  pos_b, _ = subtract_frame_transforms(
    anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
    anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
    body_pos_w,
    body_quat_w,
  )
  return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
  env: ManagerBasedRlEnv,
  anchor_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
  body_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
) -> torch.Tensor:
  """Tracked body orientations in anchor frame, flattened to 6D per body.

  Uses the first two columns of the rotation matrix as a continuous,
  ambiguity-free representation of 3D rotation (Zhou et al. 2019).
  """
  asset: Entity = env.scene[anchor_cfg.name]
  anchor_id = _anchor_id(anchor_cfg)
  anchor_pos_w = asset.data.body_link_pos_w[:, anchor_id]
  anchor_quat_w = asset.data.body_link_quat_w[:, anchor_id]

  body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]

  num_bodies = body_pos_w.shape[1]
  _, ori_b = subtract_frame_transforms(
    anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
    anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
    body_pos_w,
    body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_lin_vel_b(
  env: ManagerBasedRlEnv,
  anchor_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
  body_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
) -> torch.Tensor:
  """Per-body linear velocity expressed in each body's own local frame.

  Note: ``anchor_cfg`` is unused (kept for API symmetry with the loader call
  signature). Velocities are rotated by each body's *own* world quaternion,
  matching the AMP_mjlab reference implementation.
  """
  del anchor_cfg
  asset: Entity = env.scene[body_cfg.name]
  body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]

  num_bodies = body_lin_vel_w.shape[1]
  body_lin_vel_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4),
    body_lin_vel_w.reshape(-1, 3),
  ).reshape(env.num_envs, num_bodies, 3)
  return body_lin_vel_b.reshape(env.num_envs, -1)


def robot_body_ang_vel_b(
  env: ManagerBasedRlEnv,
  anchor_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
  body_cfg: SceneEntityCfg = _EMPTY_BODY_CFG,
) -> torch.Tensor:
  """Per-body angular velocity expressed in each body's own local frame."""
  del anchor_cfg
  asset: Entity = env.scene[body_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]
  body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]

  num_bodies = body_ang_vel_w.shape[1]
  body_ang_vel_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4),
    body_ang_vel_w.reshape(-1, 3),
  ).reshape(env.num_envs, num_bodies, 3)
  return body_ang_vel_b.reshape(env.num_envs, -1)
