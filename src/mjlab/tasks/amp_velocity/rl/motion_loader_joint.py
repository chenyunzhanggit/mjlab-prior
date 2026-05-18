"""Joint-space expert motion loader for the telebotM2-style AMP variant.

Companion to :mod:`motion_loader` (body-space). Reads the same npz schema but
constructs a per-frame joint-space feature

    [base_lin_vel_b(3), base_ang_vel_b(3), joint_pos(J), joint_vel(J)]

and yields K-frame stacks ``(B, K, d)`` for the ``DiscriminatorMulti`` path.

Schema (one npz file, all keys mandatory):
  fps:            (1,)   float
  joint_pos:      (T, J_all) float
  joint_vel:      (T, J_all) float
  body_pos_w:     (T, B, 3)  (unused here)
  body_quat_w:    (T, B, 4)  used for pelvis to rotate base vels into body frame
  body_lin_vel_w: (T, B, 3)  pelvis row gives base_lin_vel_w
  body_ang_vel_w: (T, B, 3)  pelvis row gives base_ang_vel_w
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from mjlab.utils.lab_api.math import quat_apply_inverse


class AMPJointLoader:
  """Loads expert clips and yields K-frame stacks for the multi discriminator."""

  def __init__(
    self,
    motion_file: str,
    joint_names: Sequence[str],
    all_joint_names: Sequence[str],
    pelvis_body_name: str,
    all_body_names: Sequence[str],
    num_frames: int,
    device: str | torch.device = "cuda:0",
  ) -> None:
    """Initialize the joint-space AMP expert dataset.

    Args:
      motion_file: Path to a single ``.npz`` file or a directory of them.
      joint_names: Subset of joints to include in the AMP feature, in the
        desired order (must match the env-side ``SceneEntityCfg`` order).
      all_joint_names: Full ordered list of joint names matching the npz
        ``joint_pos`` / ``joint_vel`` second axis.
      pelvis_body_name: Body whose world-frame velocity is the AMP base vel.
      all_body_names: Full ordered list of body names matching the npz
        body_*_w second axis.
      num_frames: K — number of stacked frames per sample (matches
        ``DiscriminatorMulti.num_frames``).
      device: Torch device to hold the precomputed features on.
    """
    assert os.path.exists(motion_file), f"Invalid path: {motion_file}"
    assert num_frames >= 1, f"num_frames must be >= 1, got {num_frames}"
    all_joints_list = list(all_joint_names)
    self._joint_indexes = [all_joints_list.index(n) for n in joint_names]
    self._num_joints = len(self._joint_indexes)

    all_bodies_list = list(all_body_names)
    self._pelvis_index = all_bodies_list.index(pelvis_body_name)

    self._num_frames = num_frames
    self._feature_dim = 3 + 3 + 2 * self._num_joints  # lin + ang + qpos + qvel

    if os.path.isfile(motion_file):
      motion_files = [motion_file]
      motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
    elif os.path.isdir(motion_file):
      motion_names_unsorted: list[str] = []
      motion_files_unsorted: list[str] = []
      for root, _dirs, files in os.walk(motion_file):
        for filename in sorted(files):
          if filename.endswith(".npz"):
            motion_names_unsorted.append(os.path.splitext(filename)[0])
            motion_files_unsorted.append(os.path.join(root, filename))
      assert motion_files_unsorted, f"No npz files found in directory: {motion_file}"
      paired = sorted(zip(motion_files_unsorted, motion_names_unsorted, strict=True))
      motion_files = [p[0] for p in paired]
      motion_names = [p[1] for p in paired]
    else:
      raise ValueError(f"Path is neither a file nor a directory: {motion_file}")

    self.motion_names = motion_names
    self._features_list: list[torch.Tensor] = []  # each: (T, feature_dim)

    for motion_idx, (motion_name, motion_path) in enumerate(
      zip(motion_names, motion_files, strict=True)
    ):
      print(
        f"Processing joint-AMP motion {motion_idx + 1}/{len(motion_files)}: "
        f"{motion_name}"
      )
      data = np.load(motion_path)

      if motion_idx == 0:
        self.fps = float(np.asarray(data["fps"]).item())

      joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
      joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
      body_quat_w = torch.tensor(
        data["body_quat_w"], dtype=torch.float32, device=device
      )
      body_lin_vel_w = torch.tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=device
      )
      body_ang_vel_w = torch.tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=device
      )

      T = joint_pos.shape[0]
      pelvis_quat = body_quat_w[:, self._pelvis_index, :]  # (T, 4)
      pelvis_lin_w = body_lin_vel_w[:, self._pelvis_index, :]  # (T, 3)
      pelvis_ang_w = body_ang_vel_w[:, self._pelvis_index, :]  # (T, 3)

      base_lin_b = quat_apply_inverse(pelvis_quat, pelvis_lin_w)  # (T, 3)
      base_ang_b = quat_apply_inverse(pelvis_quat, pelvis_ang_w)  # (T, 3)

      idx_tensor = torch.tensor(self._joint_indexes, dtype=torch.long, device=device)
      qpos = joint_pos.index_select(dim=1, index=idx_tensor)  # (T, J)
      qvel = joint_vel.index_select(dim=1, index=idx_tensor)  # (T, J)

      # Sanity progress bar (mirrors AMPLoader's tqdm); no per-frame work.
      for _ in tqdm(range(T), desc=f"Preloading joint-AMP for {motion_name}"):
        pass

      feats = torch.cat([base_lin_b, base_ang_b, qpos, qvel], dim=-1)  # (T, d)
      assert feats.shape == (T, self._feature_dim), (
        f"AMPJointLoader: built feature shape {feats.shape}, expected "
        f"{(T, self._feature_dim)}"
      )
      self._features_list.append(feats)

  @property
  def observation_dim(self) -> int:
    """Per-step feature dim (not the discriminator input dim)."""
    return self._feature_dim

  @property
  def num_frames(self) -> int:
    return self._num_frames

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    """Yield ``num_mini_batch`` random K-frame stacks.

    Each yielded tensor has shape ``(mini_batch_size, K, d)`` and is consumed
    directly by ``DiscriminatorMulti``. The minibatch's source motion rotates
    round-robin across all loaded clips so every clip eventually contributes.
    """
    num_motions = len(self._features_list)
    for batch_idx in range(num_mini_batch):
      m = batch_idx % num_motions
      feats = self._features_list[m]  # (T, d)
      T = feats.shape[0]
      max_start = T - self._num_frames
      if max_start < 1:
        raise RuntimeError(
          f"Motion clip {self.motion_names[m]} has only {T} frames, which is "
          f"shorter than num_frames={self._num_frames}; cannot sample a stack."
        )
      starts = torch.randint(0, max_start, (mini_batch_size,), device=feats.device)

      # Build (B, K, d) by gathering K consecutive frames per sample.
      offsets = torch.arange(self._num_frames, device=feats.device)  # (K,)
      idxs = starts.unsqueeze(-1) + offsets.unsqueeze(0)  # (B, K)
      stack = feats[idxs]  # (B, K, d)
      yield stack
