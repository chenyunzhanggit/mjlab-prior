"""Multi-motion command for the tracking-style env.

Ports ``~/zcy/motionprior/source/whole_body_tracking/whole_body_tracking/mdp/
commands_multi_ada.py`` (`MultiMotionLoader` + `MotionCommand`) onto mjlab,
adding the isaaclab→mujoco joint/body reindex constants from
``~/zcy/tracking_bfm/src/mjlab/tasks/tracking/mdp/multi_commands.py``.

Only the pieces motion_prior distillation actually consumes are kept:

* ``MultiMotionLoader`` — list-of-tensors per clip, with isaaclab→mujoco
  reindex applied at load time (``motion_type="isaaclab"`` default).
  No ``HybridMultiMotionLoader``.
* ``MultiMotionCommand`` — per-env buffer of the currently-playing clip,
  uniform sampling by default, RSI on ``_resample_command``. Drops future
  N-step lookahead properties, adaptive sampling complexity, random
  static, and ghost / debug visualizer.
* ``MultiMotionCommandCfg`` — mjlab dataclass style. Either
  ``motion_files`` (explicit list) or ``motion_path`` (directory glob)
  may be set; ``motion_path`` is expanded inside ``__init__`` so CLI
  overrides land before glob runs.

End-of-file aliases ``MotionCommand`` / ``MotionCommandCfg`` make this
module a drop-in for the single-motion symbols when imported as
``from mjlab.tasks.tracking.mdp.multi_commands import (
    MotionCommand, MotionCommandCfg)``.
The single-motion implementation in ``commands.py`` is **unchanged**.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


# ---------------------------------------------------------------------------
# isaaclab → mujoco reindex (ported verbatim from tracking_bfm:32-167)
# ---------------------------------------------------------------------------

_ISAACLAB_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
  "waist_yaw_joint",
  "left_hip_roll_joint",
  "right_hip_roll_joint",
  "waist_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_yaw_joint",
  "waist_pitch_joint",
  "left_knee_joint",
  "right_knee_joint",
  "left_shoulder_pitch_joint",
  "right_shoulder_pitch_joint",
  "left_ankle_pitch_joint",
  "right_ankle_pitch_joint",
  "left_shoulder_roll_joint",
  "right_shoulder_roll_joint",
  "left_ankle_roll_joint",
  "right_ankle_roll_joint",
  "left_shoulder_yaw_joint",
  "right_shoulder_yaw_joint",
  "left_elbow_joint",
  "right_elbow_joint",
  "left_wrist_roll_joint",
  "right_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_wrist_yaw_joint",
]

_MUJOCO_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
]

_ISAACLAB_BODY_NAMES = [
  "pelvis",
  "left_hip_pitch_link",
  "right_hip_pitch_link",
  "waist_yaw_link",
  "left_hip_roll_link",
  "right_hip_roll_link",
  "waist_roll_link",
  "left_hip_yaw_link",
  "right_hip_yaw_link",
  "torso_link",
  "left_knee_link",
  "right_knee_link",
  "left_shoulder_pitch_link",
  "right_shoulder_pitch_link",
  "left_ankle_pitch_link",
  "right_ankle_pitch_link",
  "left_shoulder_roll_link",
  "right_shoulder_roll_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_shoulder_yaw_link",
  "right_shoulder_yaw_link",
  "left_elbow_link",
  "right_elbow_link",
  "left_wrist_roll_link",
  "right_wrist_roll_link",
  "left_wrist_pitch_link",
  "right_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
]

_MUJOCO_BODY_NAMES = [
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
]

_ISAACLAB_TO_MUJOCO_JOINT_REINDEX = [
  _ISAACLAB_JOINT_NAMES.index(name) for name in _MUJOCO_JOINT_NAMES
]
_ISAACLAB_TO_MUJOCO_BODY_REINDEX = [
  _ISAACLAB_BODY_NAMES.index(name) for name in _MUJOCO_BODY_NAMES
]


def _expand_motion_dir(motion_path: str) -> list[str]:
  """Mirror ``play_mp.py:get_data10K_motion_files`` — recursive .npz glob.

  Accepts a file (returned as a 1-element list) or a directory (recursive
  glob for ``**/*.npz``, sorted for determinism).
  """
  if os.path.isfile(motion_path):
    return [motion_path]
  if os.path.isdir(motion_path):
    pattern = os.path.join(motion_path, "**", "*.npz")
    files = sorted(f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f))
    if not files:
      raise ValueError(f"No .npz under: {motion_path}")
    return files
  raise ValueError(f"Invalid motion_path (not a file/dir): {motion_path}")


# ---------------------------------------------------------------------------
# MultiMotionLoader
# ---------------------------------------------------------------------------


class MultiMotionLoader:
  """Holds every motion clip in memory as a list of (T_i, ...) tensors.

  Two changes vs the motionprior reference:

  * ``motion_type`` parameter (``"isaaclab"`` default) drives whether
    the per-clip tensors get reindexed to mujoco joint/body order at load
    time. ``"mujoco"`` skips reindex (clip already in mujoco order).
  * ``body_indexes`` is a ``torch.Tensor`` instead of ``list[int]`` so it
    matches the single-motion ``MotionLoader`` and survives the same
    advanced indexing patterns downstream.
  """

  def __init__(
    self,
    motion_files: Sequence[str],
    body_indexes: torch.Tensor,
    motion_type: Literal["isaaclab", "mujoco"] = "isaaclab",
    device: str = "cpu",
  ) -> None:
    if len(motion_files) == 0:
      raise ValueError("MultiMotionLoader: motion_files is empty.")

    if motion_type == "isaaclab":
      joint_reindex: list[int] | None = _ISAACLAB_TO_MUJOCO_JOINT_REINDEX
      body_reindex: list[int] | None = _ISAACLAB_TO_MUJOCO_BODY_REINDEX
    elif motion_type == "mujoco":
      joint_reindex = None
      body_reindex = None
    else:
      raise ValueError(f"Unsupported motion_type: {motion_type}")

    self.num_files = len(motion_files)
    self.motion_files = list(motion_files)
    self._body_indexes = body_indexes
    self.device = device

    self.joint_pos_list: list[torch.Tensor] = []
    self.joint_vel_list: list[torch.Tensor] = []
    self._body_pos_w_list: list[torch.Tensor] = []
    self._body_quat_w_list: list[torch.Tensor] = []
    self._body_lin_vel_w_list: list[torch.Tensor] = []
    self._body_ang_vel_w_list: list[torch.Tensor] = []
    self.fps_list: list[np.ndarray] = []
    file_lengths: list[int] = []

    for motion_file in motion_files:
      if not os.path.isfile(motion_file):
        raise FileNotFoundError(f"Invalid file path: {motion_file}")
      data = np.load(motion_file)
      self.fps_list.append(data["fps"])

      jp = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
      jv = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
      bp = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
      bq = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
      blv = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
      bav = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)

      if joint_reindex is not None:
        jp = jp[:, joint_reindex]
        jv = jv[:, joint_reindex]
      if body_reindex is not None:
        bp = bp[:, body_reindex, :]
        bq = bq[:, body_reindex, :]
        blv = blv[:, body_reindex, :]
        bav = bav[:, body_reindex, :]

      self.joint_pos_list.append(jp)
      self.joint_vel_list.append(jv)
      self._body_pos_w_list.append(bp)
      self._body_quat_w_list.append(bq)
      self._body_lin_vel_w_list.append(blv)
      self._body_ang_vel_w_list.append(bav)
      file_lengths.append(jp.shape[0])

    self.file_lengths = torch.tensor(file_lengths, dtype=torch.long, device=self.device)
    self.fps = self.fps_list[0]

    # Uniform default; ``MultiMotionCommand._uniform_sampling`` calls
    # ``torch.multinomial`` against this so no extra branching is needed
    # if a user later wants weighted sampling.
    self.init_sampling_probabilities = torch.full(
      (self.num_files,), 1.0 / self.num_files, dtype=torch.float32, device=self.device
    )

  def get_motion_data_batch(
    self,
    motion_idx: int,
    time_steps_start: torch.Tensor,
    time_steps_end: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Slice clip ``motion_idx`` between ``[start, end)`` (clamped)."""
    time_steps_tensor = torch.arange(
      int(time_steps_start), int(time_steps_end), device=self.device
    )
    time_steps_tensor = torch.clamp(
      time_steps_tensor,
      torch.tensor(0, device=self.device),
      self.file_lengths[motion_idx] - 1,
    )
    return {
      "joint_pos": self.joint_pos_list[motion_idx][time_steps_tensor],
      "joint_vel": self.joint_vel_list[motion_idx][time_steps_tensor],
      "body_pos_w": self._body_pos_w_list[motion_idx][time_steps_tensor][
        :, self._body_indexes
      ],
      "body_quat_w": self._body_quat_w_list[motion_idx][time_steps_tensor][
        :, self._body_indexes
      ],
      "body_lin_vel_w": self._body_lin_vel_w_list[motion_idx][time_steps_tensor][
        :, self._body_indexes
      ],
      "body_ang_vel_w": self._body_ang_vel_w_list[motion_idx][time_steps_tensor][
        :, self._body_indexes
      ],
    }


# ---------------------------------------------------------------------------
# MultiMotionCommand
# ---------------------------------------------------------------------------


class MultiMotionCommand(CommandTerm):
  """Per-env motion-clip selector with a rolling-buffer property surface.

  Property contract (per ``single_motion_migration_audit.md``): all
  ``anchor_*_w`` are ``(N, 3)`` / ``(N, 4)``; ``body_*_w`` are
  ``(N, num_bodies, *)``. ``command`` is
  ``cat([joint_pos, joint_vel])`` to match the single-motion behavior
  the obs config expects.

  Sampling defaults to uniform. ``enable_adaptive_sampling=True`` would
  require a downstream success/fail bookkeeping path that motion_prior
  distillation does not currently use; left as a stub that falls back
  to uniform.
  """

  cfg: "MultiMotionCommandCfg"
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: "MultiMotionCommandCfg", env: ManagerBasedRlEnv) -> None:
    # Expand motion_path -> motion_files BEFORE any super-init that may
    # consume cfg fields (CLI override has landed by now per todo "风险点 #5").
    if cfg.motion_path:
      if cfg.motion_files:
        raise ValueError("Set either motion_files or motion_path, not both.")
      cfg.motion_files = _expand_motion_dir(cfg.motion_path)
    if not cfg.motion_files:
      raise ValueError(
        "MultiMotionCommandCfg requires either motion_files or motion_path."
      )

    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MultiMotionLoader(
      motion_files=cfg.motion_files,
      body_indexes=self.body_indexes,
      motion_type=cfg.motion_type,
      device=self.device,
    )

    self.num_motion: int = self.motion.num_files

    # Buffer covers a full episode for the currently-assigned clip per env.
    # ``+1`` to make ``time_steps == buffer_start_time + buffer_length - 1``
    # safe at episode end.
    longest = int(self.motion.file_lengths.max().item())
    self.buffer_length: int = int(min(env.max_episode_length, longest)) + 1

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_length = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self.buffer_start_time = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    self._init_buffers()

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    if cfg.if_log_metrics:
      for key in (
        "error_anchor_pos",
        "error_anchor_rot",
        "error_anchor_lin_vel",
        "error_anchor_ang_vel",
        "error_body_pos",
        "error_body_rot",
        "error_body_lin_vel",
        "error_body_ang_vel",
        "error_joint_pos",
        "error_joint_vel",
        "sampling_entropy",
        "sampling_top1_prob",
      ):
        self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  # --------------------------- buffer plumbing --------------------------- #

  def _init_buffers(self) -> None:
    joint_dim = self.motion.joint_pos_list[0].shape[1]
    body_dim = len(self.cfg.body_names)
    n = self.num_envs
    L = self.buffer_length
    dev = self.device

    self.joint_pos_buffer = torch.zeros(n, L, joint_dim, device=dev)
    self.joint_vel_buffer = torch.zeros(n, L, joint_dim, device=dev)
    self.body_pos_w_buffer = torch.zeros(n, L, body_dim, 3, device=dev)
    self.body_quat_w_buffer = torch.zeros(n, L, body_dim, 4, device=dev)
    self.body_quat_w_buffer[:, :, :, 0] = 1.0
    self.body_lin_vel_w_buffer = torch.zeros(n, L, body_dim, 3, device=dev)
    self.body_ang_vel_w_buffer = torch.zeros(n, L, body_dim, 3, device=dev)

  def _update_buffers(self, env_ids: torch.Tensor) -> None:
    """Refill per-env buffer slices from their assigned clip.

    Mirrors the motionprior reference's per-env Python loop. Vectorizing
    across ragged-length clips is not in scope per todo "不做的事".
    """
    if env_ids.numel() == 0:
      return
    for env_id in env_ids.tolist():
      data = self.motion.get_motion_data_batch(
        int(self.motion_idx[env_id].item()),
        self.buffer_start_time[env_id],
        self.buffer_start_time[env_id] + self.buffer_length,
      )
      self.joint_pos_buffer[env_id] = data["joint_pos"]
      self.joint_vel_buffer[env_id] = data["joint_vel"]
      self.body_pos_w_buffer[env_id] = data["body_pos_w"]
      self.body_quat_w_buffer[env_id] = data["body_quat_w"]
      self.body_lin_vel_w_buffer[env_id] = data["body_lin_vel_w"]
      self.body_ang_vel_w_buffer[env_id] = data["body_ang_vel_w"]

  # --------------------------- properties --------------------------- #

  def _buffer_idx(self) -> torch.Tensor:
    return torch.clamp(
      self.time_steps - self.buffer_start_time, 0, self.buffer_length - 1
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.joint_pos_buffer[torch.arange(self.num_envs, device=self.device), idx]

  @property
  def joint_vel(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.joint_vel_buffer[torch.arange(self.num_envs, device=self.device), idx]

  @property
  def body_pos_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return (
      self.body_pos_w_buffer[torch.arange(self.num_envs, device=self.device), idx]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.body_quat_w_buffer[torch.arange(self.num_envs, device=self.device), idx]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.body_lin_vel_w_buffer[
      torch.arange(self.num_envs, device=self.device), idx
    ]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.body_ang_vel_w_buffer[
      torch.arange(self.num_envs, device=self.device), idx
    ]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return (
      self.body_pos_w_buffer[
        torch.arange(self.num_envs, device=self.device),
        idx,
        self.motion_anchor_body_index,
      ]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.body_quat_w_buffer[
      torch.arange(self.num_envs, device=self.device),
      idx,
      self.motion_anchor_body_index,
    ]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.body_lin_vel_w_buffer[
      torch.arange(self.num_envs, device=self.device),
      idx,
      self.motion_anchor_body_index,
    ]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    idx = self._buffer_idx()
    return self.body_ang_vel_w_buffer[
      torch.arange(self.num_envs, device=self.device),
      idx,
      self.motion_anchor_body_index,
    ]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  # --------------------------- CommandTerm hooks --------------------------- #

  def _update_metrics(self) -> None:
    if not self.cfg.if_log_metrics:
      return
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _uniform_sampling(self, env_ids: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(
      self.motion.init_sampling_probabilities, len(env_ids), replacement=True
    )

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Same RSI write path as the single-motion implementation."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    # 1. Sample which clip and where in it to start.
    self.motion_idx[env_ids] = self._uniform_sampling(env_ids)
    self.motion_length[env_ids] = self.motion.file_lengths[self.motion_idx[env_ids]]
    if self.cfg.start_from_zero_step:
      self.time_steps[env_ids] = 0
    else:
      self.time_steps[env_ids] = (
        sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        * self.motion_length[env_ids].float()
      ).long()
    self.buffer_start_time[env_ids] = self.time_steps[env_ids].clone()

    # 2. Refill the per-env buffer slices.
    self._update_buffers(env_ids)

    # 3. Pull the start frame's root state for RSI initialization.
    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ("x", "y", "z", "roll", "pitch", "yaw")
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids].clone()
    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )

    self._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_*_relative_w`` after a manual reset (parity with single)."""
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def _update_command(self) -> None:
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion_length)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    self.update_relative_body_poses()

    if self.cfg.if_log_metrics:
      # Uniform sampling: maximum entropy by definition.
      self.metrics["sampling_entropy"][:] = 1.0
      self.metrics["sampling_top1_prob"][:] = 1.0 / float(self.num_motion)


# ---------------------------------------------------------------------------
# Cfg
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  """Config for :class:`MultiMotionCommand`.

  Either ``motion_files`` (an explicit list) or ``motion_path`` (a file
  or a directory expanded recursively for ``*.npz``) must be supplied.
  ``motion_path`` is glob-expanded inside the command's ``__init__`` so
  CLI overrides land before the filesystem walk.
  """

  motion_files: list[str] = field(default_factory=list)
  motion_path: str = ""
  motion_type: Literal["isaaclab", "mujoco"] = "isaaclab"

  entity_name: str
  anchor_body_name: str
  body_names: tuple[str, ...]

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)

  enable_adaptive_sampling: bool = False
  start_from_zero_step: bool = True #False
  if_log_metrics: bool = True

  def build(self, env: ManagerBasedRlEnv) -> MultiMotionCommand:
    return MultiMotionCommand(self, env)


# ---------------------------------------------------------------------------
# Drop-in aliases (per todo #2.5). Single-motion versions in ``commands.py``
# are NOT touched.
# ---------------------------------------------------------------------------

MotionCommand = MultiMotionCommand
MotionCommandCfg = MultiMotionCommandCfg
