"""Reference State Initialization (RSI) events for AMP training.

On reset, instead of putting the robot in its default standing pose, sample a
random frame from the expert motion clips and write that frame's root pose,
root velocity, joint pos, and joint vel into the sim. This lets the policy
start each episode somewhere on the expert manifold, which is the standard
trick from the AMP paper for keeping the discriminator's signal non-vacuous
in the early stages of training.

Two event functions are exposed:

- :func:`init_motion_for_rsi` (``mode="startup"``): loads all ``.npz`` clips
  under ``motion_dir`` once, concatenates the per-frame fields into a single
  GPU-resident buffer, and stashes them on a process-global singleton keyed by
  ``motion_dir`` so multiple resets share the same buffer.
- :func:`reset_from_motion` (``mode="reset"``): samples ``len(env_ids)`` random
  frames and writes the corresponding root/joint state to the sim for the
  envs being reset.

The npz schema must match what :class:`AMPLoader` already expects:
``joint_pos / joint_vel / body_pos_w / body_quat_w / body_lin_vel_w /
body_ang_vel_w``, with body index 0 being the floating-base root.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class _RsiFrameBuffer:
  """Process-global cache of motion frames keyed by ``motion_dir``.

  Built lazily by :func:`init_motion_for_rsi` so the startup cost is paid once
  per training run regardless of how many envs reset. All tensors live on the
  env's device after init so sampling is a plain ``index_select`` with no
  host→device copy on the hot path.
  """

  _instance: "_RsiFrameBuffer | None" = None

  def __init__(self) -> None:
    # motion_dir -> dict of stacked tensors with shape (total_frames, ...).
    self._frames: dict[str, dict[str, torch.Tensor]] = {}

  @classmethod
  def get(cls) -> "_RsiFrameBuffer":
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def is_loaded(self, motion_dir: str) -> bool:
    return motion_dir in self._frames

  def load(self, motion_dir: str, device: torch.device | str) -> None:
    """Scan ``motion_dir`` for ``.npz`` clips and stack all frames on device."""
    assert os.path.isdir(motion_dir), f"RSI motion_dir is not a directory: {motion_dir}"

    npz_paths: list[str] = []
    for root, _dirs, files in os.walk(motion_dir):
      for filename in sorted(files):
        if filename.endswith(".npz"):
          npz_paths.append(os.path.join(root, filename))
    assert npz_paths, f"No .npz files found under RSI motion_dir: {motion_dir}"

    root_pos_chunks: list[torch.Tensor] = []
    root_quat_chunks: list[torch.Tensor] = []
    root_lin_vel_chunks: list[torch.Tensor] = []
    root_ang_vel_chunks: list[torch.Tensor] = []
    joint_pos_chunks: list[torch.Tensor] = []
    joint_vel_chunks: list[torch.Tensor] = []

    for path in sorted(npz_paths):
      data = np.load(path)
      # Body index 0 is the floating-base root (pelvis for G1).
      root_pos_chunks.append(
        torch.tensor(data["body_pos_w"][:, 0, :], dtype=torch.float32, device=device)
      )
      root_quat_chunks.append(
        torch.tensor(data["body_quat_w"][:, 0, :], dtype=torch.float32, device=device)
      )
      root_lin_vel_chunks.append(
        torch.tensor(
          data["body_lin_vel_w"][:, 0, :], dtype=torch.float32, device=device
        )
      )
      root_ang_vel_chunks.append(
        torch.tensor(
          data["body_ang_vel_w"][:, 0, :], dtype=torch.float32, device=device
        )
      )
      joint_pos_chunks.append(
        torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
      )
      joint_vel_chunks.append(
        torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
      )

    frames = {
      "root_pos": torch.cat(root_pos_chunks, dim=0),
      "root_quat": torch.cat(root_quat_chunks, dim=0),
      "root_lin_vel": torch.cat(root_lin_vel_chunks, dim=0),
      "root_ang_vel": torch.cat(root_ang_vel_chunks, dim=0),
      "joint_pos": torch.cat(joint_pos_chunks, dim=0),
      "joint_vel": torch.cat(joint_vel_chunks, dim=0),
    }
    total = frames["root_pos"].shape[0]
    print(f"[RSI] Loaded {len(npz_paths)} clips, {total} frames from {motion_dir}")
    self._frames[motion_dir] = frames

  def sample(
    self, motion_dir: str, num: int, device: torch.device | str
  ) -> dict[str, torch.Tensor]:
    """Return ``num`` random frames as a dict of tensors on ``device``."""
    assert motion_dir in self._frames, (
      f"RSI buffer for {motion_dir} not initialized — did you forget the "
      f"init_motion_for_rsi startup event?"
    )
    frames = self._frames[motion_dir]
    total = frames["root_pos"].shape[0]
    idx = torch.randint(0, total, (num,), device=device)
    return {k: v[idx] for k, v in frames.items()}


# --------------------------------------------------------------------------- #
# Event callbacks                                                             #
# --------------------------------------------------------------------------- #


def init_motion_for_rsi(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  motion_dir: str,
) -> None:
  """Startup event: preload all motion frames into the RSI buffer."""
  del env_ids  # startup event runs for all envs, env_ids is unused
  buf = _RsiFrameBuffer.get()
  if not buf.is_loaded(motion_dir):
    buf.load(motion_dir, device=env.device)


def reset_from_motion(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  motion_dir: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset event: place each reset env at a random expert motion frame.

  Root xy is placed at the env's grid origin; root z comes from the motion
  frame (preserves the standing height of the clip). Root orientation,
  linear/angular velocity, and the full joint state all come from the sampled
  frame. Joint positions are clamped to soft limits before writing.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  if len(env_ids) == 0:
    return

  asset: Entity = env.scene[asset_cfg.name]
  buf = _RsiFrameBuffer.get()
  sampled = buf.sample(motion_dir, num=len(env_ids), device=env.device)

  # --- Root pose: xy from env grid, z + quat from motion frame ---
  positions = env.scene.env_origins[env_ids].clone()
  positions[:, 2] = sampled["root_pos"][:, 2]
  root_pose = torch.cat([positions, sampled["root_quat"]], dim=-1)
  asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)

  # --- Root velocity ---
  root_vel = torch.cat([sampled["root_lin_vel"], sampled["root_ang_vel"]], dim=-1)
  asset.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

  # --- Joint state (clamp positions to soft limits) ---
  joint_ids = asset_cfg.joint_ids
  joint_pos = sampled["joint_pos"][:, joint_ids]
  joint_vel = sampled["joint_vel"][:, joint_ids]

  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None
  limits = soft_joint_pos_limits[env_ids][:, joint_ids]
  joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])

  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )
