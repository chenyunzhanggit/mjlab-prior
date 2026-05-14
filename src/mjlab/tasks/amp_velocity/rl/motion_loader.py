"""Expert motion data loader for AMP discriminator training.

Adapted from AMP_mjlab/rsl_rl/utils/motion_loader.py (BSD-3-Clause).

Reads ``.npz`` motion clips and precomputes per-frame body-relative features
in the anchor frame: position, rotation-matrix first 2 columns, body-local
linear/angular velocities. The discriminator consumes ``(s, s')`` pairs of
these features.

Schema (one npz file, all keys mandatory):
  fps:            (1,)   float
  joint_pos:      (T, J) float
  joint_vel:      (T, J) float
  body_pos_w:     (T, B, 3)  world position of each body
  body_quat_w:    (T, B, 4)  world orientation (wxyz)
  body_lin_vel_w: (T, B, 3)
  body_ang_vel_w: (T, B, 3)

B must match ``len(all_body_names)`` of the simulated robot; the loader
indexes ``body_indexes`` and ``anchor_index`` into the second dim.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

import mjlab.utils.lab_api.math as math_utils


class AMPLoader:
  """Loads expert motion clips and yields (s, s') minibatches for AMP."""

  def __init__(
    self,
    motion_file: str,
    body_names: Sequence[str],
    anchor_name: str,
    all_body_names: Sequence[str],
    device: str | torch.device = "cuda:0",
  ) -> None:
    """Initialize the AMP expert dataset.

    Args:
      motion_file: Path to a single ``.npz`` file or a directory of them.
      body_names: Subset of bodies to include in the AMP feature.
      anchor_name: Name of the anchor body (features are expressed in its frame).
      all_body_names: Full ordered list of body names matching the npz layout.
      device: Torch device to hold the precomputed features on.
    """
    assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

    all_names_list = list(all_body_names)
    self._body_indexes = [all_names_list.index(n) for n in body_names]
    self._anchor_indexes = all_names_list.index(anchor_name)
    self._num_bodies = len(self._body_indexes)

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
    self._body_pos_b_list: list[torch.Tensor] = []
    self._body_ori_b_list: list[torch.Tensor] = []
    self._body_lin_vel_b_list: list[torch.Tensor] = []
    self._body_ang_vel_b_list: list[torch.Tensor] = []

    for motion_idx, (motion_name, motion_path) in enumerate(
      zip(motion_names, motion_files, strict=True)
    ):
      print(f"Processing motion {motion_idx + 1}/{len(motion_files)}: {motion_name}")
      data = np.load(motion_path)

      if motion_idx == 0:
        self.fps = float(np.asarray(data["fps"]).item())

      _body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
      _body_quat_w = torch.tensor(
        data["body_quat_w"], dtype=torch.float32, device=device
      )
      _body_lin_vel_w = torch.tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=device
      )
      _body_ang_vel_w = torch.tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=device
      )
      T = _body_pos_w.shape[0]

      pos_b = torch.zeros((T, self._num_bodies, 3), dtype=torch.float32, device=device)
      ori_b = torch.zeros((T, self._num_bodies, 6), dtype=torch.float32, device=device)
      lin_b = torch.zeros((T, self._num_bodies, 3), dtype=torch.float32, device=device)
      ang_b = torch.zeros((T, self._num_bodies, 3), dtype=torch.float32, device=device)

      for f in tqdm(range(T), desc=f"Preloading AMP data for {motion_name}"):
        anchor_pos = (
          _body_pos_w[f, self._anchor_indexes, :]
          .squeeze()
          .unsqueeze(0)
          .repeat(self._num_bodies, 1)
        )
        anchor_quat = (
          _body_quat_w[f, self._anchor_indexes, :]
          .squeeze()
          .unsqueeze(0)
          .repeat(self._num_bodies, 1)
        )
        body_pos = _body_pos_w[f, self._body_indexes, :]
        body_quat = _body_quat_w[f, self._body_indexes, :]
        body_lin = _body_lin_vel_w[f, self._body_indexes, :]
        body_ang = _body_ang_vel_w[f, self._body_indexes, :]

        pos_b_f, quat_b_f = math_utils.subtract_frame_transforms(
          anchor_pos, anchor_quat, body_pos, body_quat
        )
        mat = math_utils.matrix_from_quat(quat_b_f)
        ori_b_f = mat[..., :, :2].reshape(self._num_bodies, 6)
        lin_b_f = math_utils.quat_apply_inverse(body_quat, body_lin)
        ang_b_f = math_utils.quat_apply_inverse(body_quat, body_ang)

        pos_b[f] = pos_b_f
        ori_b[f] = ori_b_f
        lin_b[f] = lin_b_f
        ang_b[f] = ang_b_f

      self._body_pos_b_list.append(pos_b)
      self._body_ori_b_list.append(ori_b)
      self._body_lin_vel_b_list.append(lin_b)
      self._body_ang_vel_b_list.append(ang_b)

  @property
  def observation_dim(self) -> int:
    # pos (3) + ori_b (6) + lin_vel (3) + ang_vel (3) per body
    return (3 + 6 + 3 + 3) * self._num_bodies

  def feed_forward_generator(self, num_mini_batch: int, mini_batch_size: int):
    """Yield ``num_mini_batch`` random (s, s') minibatches across all clips."""
    num_motions = len(self._body_pos_b_list)
    for batch_idx in range(num_mini_batch):
      m = batch_idx % num_motions
      pos = self._body_pos_b_list[m]
      ori = self._body_ori_b_list[m]
      lin = self._body_lin_vel_b_list[m]
      ang = self._body_ang_vel_b_list[m]
      T = pos.shape[0]

      idxs = torch.randint(0, T, (mini_batch_size,), device=pos.device)
      idxs = torch.clamp(idxs, max=T - 1)
      next_idxs = torch.clamp(idxs + 1, max=T - 1)

      s = torch.cat(
        [
          pos[idxs].reshape(mini_batch_size, -1),
          ori[idxs].reshape(mini_batch_size, -1),
          lin[idxs].reshape(mini_batch_size, -1),
          ang[idxs].reshape(mini_batch_size, -1),
        ],
        dim=-1,
      )
      s_next = torch.cat(
        [
          pos[next_idxs].reshape(mini_batch_size, -1),
          ori[next_idxs].reshape(mini_batch_size, -1),
          lin[next_idxs].reshape(mini_batch_size, -1),
          ang[next_idxs].reshape(mini_batch_size, -1),
        ],
        dim=-1,
      )
      yield s, s_next
