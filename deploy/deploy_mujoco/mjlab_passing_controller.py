"""mjlab passing downstream-VQ controller for MuJoCo sim2sim.

Companion to ``mjlab_velocity_controller.py`` but for the football passing
task — the policy sees a ball flying toward the robot and learns to
redirect it back into a target ``source`` zone.

Differences vs the original ``passing_controller.py`` (IsaacLab port):

* **Two-file ckpt loading.** mjlab saves the frozen motion-prior backbone
  and the trainable downstream actor / critic as **separate** files:

    - motion_prior ckpt → top-level keys ``motion_prior``, ``decoder``,
      ``quantizer`` (``MotionPriorSingleVQOnPolicyRunner.save``).
    - downstream   ckpt → top-level keys ``actor``, ``critic``, ``std``
      (``DownStreamVQOnPolicyRunner.save``).

* **No IsaacLab joint-order reindex.** mjlab observes / acts in MJCF
  joint order, so both ``qpos[7:36]`` / ``qvel[6:35]`` and the policy
  output align with ``mujoco_joint_names`` directly.

* **Constants from mjlab.** ``action_scale`` / kp / kd /
  ``default_dof_pos`` are pulled out of the mjlab G1 cfgs at load time —
  guaranteeing zero kp / kd / scale drift between training and deploy.

* **Ball observations replicate** ``ball_relative_position`` /
  ``ball_velocity`` / ``passing_source_position`` from the mjlab MDP
  observation functions:

  - ``ball_relative_position``: ``quat_apply_inverse(robot_quat,
    ball_pos_w - robot_pos_w)`` — ball position in robot body frame.
  - ``ball_velocity``: raw world-frame linear velocity.
  - ``passing_source_position``: ``(dx_b, dy_b, dist_w)`` where the XY
    delta is rotated into body frame and ``dist_w`` is the unrotated
    world-frame XY distance.

Observation layout (matches mjlab passing env):

  prop_obs  (372)  = gravity(3×4) | ang_vel(3×4) | jpos(29×4) | jvel(29×4) | actions(29×4)
  policy_obs (381) = ball_rel(3)   | ball_vel(3) | passing_src(3) | prop_obs(372)
"""

from __future__ import annotations

import os
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_controller import BaseController  # noqa: E402
from common.bydmimic_utils import (  # noqa: E402
  get_gravity_orientation,
  mujoco_joint_names,
)


# ---------------------------------------------------------------------------
# Rotation helper (matches mjlab.utils.lab_api.math.quat_apply_inverse)
# ---------------------------------------------------------------------------


def _quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate vector v from world frame into the body frame defined by
  the quaternion ``q_wxyz``. Equivalent to mjlab's
  ``quat_apply_inverse(root_link_quat_w, v_w)``."""
  w = float(q_wxyz[0])
  qv = q_wxyz[1:].astype(np.float64)
  vv = v.astype(np.float64)
  a = vv * (2.0 * w * w - 1.0)
  b = np.cross(qv, vv) * (w * 2.0)
  c = qv * (np.dot(qv, vv) * 2.0)
  return (a - b + c).astype(np.float32)


# ---------------------------------------------------------------------------
# mjlab → MJCF-ordered constants
# ---------------------------------------------------------------------------


def _build_mjlab_constants() -> dict[str, np.ndarray]:
  """Pull training-time per-joint constants out of mjlab (action_scale, kp,
  kd, default_dof_pos) and arrange them in MJCF joint order."""
  try:
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
      G1_ACTION_SCALE,
      G1_ARTICULATION,
      KNEES_BENT_KEYFRAME,
    )
  except ImportError as e:
    raise ImportError(
      "Deploy needs ``mjlab.asset_zoo.robots.unitree_g1``. Install mjlab "
      "(``uv sync --group deploy``).\n  underlying error: " + str(e)
    ) from e

  num_joints = len(mujoco_joint_names)

  def _match(name: str, patterns) -> tuple[bool, str | None]:
    for pat in patterns:
      if pat.startswith(".*"):
        if name.endswith(pat[2:]):
          return True, pat
      else:
        if name == pat:
          return True, pat
    return False, None

  action_scale = np.zeros(num_joints, dtype=np.float32)
  for i, name in enumerate(mujoco_joint_names):
    ok, key = _match(name, G1_ACTION_SCALE.keys())
    if not ok:
      raise KeyError(f"No G1_ACTION_SCALE entry matches '{name}'")
    action_scale[i] = G1_ACTION_SCALE[key]

  kps = np.zeros(num_joints, dtype=np.float32)
  kds = np.zeros(num_joints, dtype=np.float32)
  for i, name in enumerate(mujoco_joint_names):
    matched = None
    for act in G1_ARTICULATION.actuators:
      ok, _ = _match(name, act.target_names_expr)
      if ok:
        matched = act
        break
    if matched is None:
      raise KeyError(f"No G1_ARTICULATION actuator matches '{name}'")
    kps[i] = float(matched.stiffness)
    kds[i] = float(matched.damping)

  default_dof_pos = np.zeros(num_joints, dtype=np.float32)
  kf_dict = KNEES_BENT_KEYFRAME.joint_pos or {}
  for i, name in enumerate(mujoco_joint_names):
    val = 0.0
    for pat, v in kf_dict.items():
      if pat.startswith(".*"):
        if name.endswith(pat[2:]):
          val = v
      elif re.fullmatch(pat, name):
        val = v
    default_dof_pos[i] = float(val)

  return dict(
    action_scale=action_scale,
    kps=kps,
    kds=kds,
    default_dof_pos=default_dof_pos,
  )


# ---------------------------------------------------------------------------
# Network building blocks (identical to mjlab_velocity_controller)
# ---------------------------------------------------------------------------


class _QuantizerInference(nn.Module):
  """Nearest-codebook lookup. Mirrors mjlab ``EMAQuantizer`` (only the
  ``codebook`` buffer is needed at inference; ``code_sum`` / ``code_count``
  are kept so ``load_state_dict(strict=True)`` works against the
  motion_prior ckpt's quantizer dict)."""

  def __init__(self, num_code: int, code_dim: int) -> None:
    super().__init__()
    self.register_buffer("codebook", torch.zeros(num_code, code_dim))
    self.register_buffer("code_sum", torch.zeros(num_code, code_dim))
    self.register_buffer("code_count", torch.zeros(num_code))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    k_w = self.codebook.t()
    dist = (
      torch.sum(x**2, dim=-1, keepdim=True)
      - 2.0 * torch.matmul(x, k_w)
      + torch.sum(k_w**2, dim=0, keepdim=True)
    )
    return F.embedding(dist.argmin(dim=-1), self.codebook)


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), nn.ELU()]
  for i in range(len(hidden_dims) - 1):
    layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ELU()]
  layers.append(nn.Linear(hidden_dims[-1], out_dim))
  return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Inference policy (mirrors mjlab DownStreamVQPolicy)
# ---------------------------------------------------------------------------


class _MjlabPassingVQPolicy(nn.Module):
  def __init__(
    self,
    num_obs: int,
    num_actions: int,
    prop_obs_dim: int,
    code_dim: int,
    num_code: int,
    actor_hidden_dims: list[int],
    motion_prior_hidden_dims: list[int],
    decoder_hidden_dims: list[int],
    lab_lambda: float = 3.0,
    use_lab: bool = True,
  ) -> None:
    super().__init__()
    self.use_lab = use_lab
    self.lab_lambda = lab_lambda

    self.motion_prior = _build_mlp(prop_obs_dim, motion_prior_hidden_dims, code_dim)
    self.quantizer = _QuantizerInference(num_code, code_dim)
    self.decoder = _build_mlp(
      prop_obs_dim + code_dim, decoder_hidden_dims, num_actions
    )
    self.actor = _build_mlp(num_obs, actor_hidden_dims, code_dim)

  @torch.no_grad()
  def inference(
    self, policy_obs: torch.Tensor, prop_obs: torch.Tensor
  ) -> torch.Tensor:
    raw = self.actor(policy_obs)
    prior = self.motion_prior(prop_obs)
    z = prior + self.lab_lambda * torch.tanh(raw) if self.use_lab else prior + raw
    q_z = self.quantizer(z)
    return self.decoder(torch.cat([prop_obs, q_z], dim=-1))


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class MjlabPassingVQController(BaseController):
  """Passing-task downstream-VQ controller for sim2sim.

  Args:
    motion_prior_ckpt_path: Path to the frozen mjlab motion-prior VQ ckpt
      (``motion_prior`` / ``decoder`` / ``quantizer`` top-level keys).
    downstream_ckpt_path: Path to the trainable mjlab downstream ckpt
      (``actor`` / ``critic`` / ``std`` top-level keys).
    device: ``"cpu"`` or ``"cuda:N"``.
    code_dim / num_code: VQ codebook dims; must match training (64 / 2048).
    actor_hidden_dims / motion_prior_hidden_dims / decoder_hidden_dims:
      MLP widths; defaults ``[512, 256, 128]`` mirror mjlab cfgs.
    lab_lambda / use_lab: LAB residual scale + on/off (default 3.0 / True).

  Use ``init_ball(model)`` after creating the simulator so the ball's
  qpos / qvel addresses are resolved from the compiled model, then call
  ``set_source_pos(np.ndarray)`` at every reset to inform the controller
  where the incoming ball was launched from (this becomes the
  ``passing_source_position`` task command).
  """

  HISTORY_LEN = 4
  NUM_JOINTS = 29
  PROP_OBS_DIM = 372
  # ball_rel_pos(3) + ball_vel(3) + passing_src(3) + prop_obs(372)
  OBS_DIM = 381

  def __init__(
    self,
    motion_prior_ckpt_path: str | Path,
    downstream_ckpt_path: str | Path,
    device: str = "cpu",
    code_dim: int = 64,
    num_code: int = 2048,
    actor_hidden_dims: list[int] | None = None,
    motion_prior_hidden_dims: list[int] | None = None,
    decoder_hidden_dims: list[int] | None = None,
    lab_lambda: float = 3.0,
    use_lab: bool = True,
  ) -> None:
    actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
    motion_prior_hidden_dims = motion_prior_hidden_dims or [512, 256, 128]
    decoder_hidden_dims = decoder_hidden_dims or [512, 256, 128]

    self.device = torch.device(device)
    self._policy = _MjlabPassingVQPolicy(
      num_obs=self.OBS_DIM,
      num_actions=self.NUM_JOINTS,
      prop_obs_dim=self.PROP_OBS_DIM,
      code_dim=code_dim,
      num_code=num_code,
      actor_hidden_dims=actor_hidden_dims,
      motion_prior_hidden_dims=motion_prior_hidden_dims,
      decoder_hidden_dims=decoder_hidden_dims,
      lab_lambda=lab_lambda,
      use_lab=use_lab,
    ).to(self.device)

    self._load_two_file_ckpt(motion_prior_ckpt_path, downstream_ckpt_path)
    self._policy.eval()

    # Task command: where the ball was launched from (world frame, set by
    # the main script at every reset).
    self.source_pos = np.zeros(3, dtype=np.float32)

    # Ball indices populated by ``init_ball``.
    self._ball_qpos_adr: int | None = None
    self._ball_dof_adr: int | None = None

    self._reset_buffers()

    # Per-joint constants pulled straight from mjlab.
    consts = _build_mjlab_constants()
    self.kps = consts["kps"]
    self.kds = consts["kds"]
    self.action_scale = consts["action_scale"]
    self.default_dof_pos = consts["default_dof_pos"]

    self.num_actions = self.NUM_JOINTS
    self.num_obs = self.OBS_DIM

  # ------------------------------------------------------------------
  # Checkpoint loading
  # ------------------------------------------------------------------

  def _load_two_file_ckpt(
    self,
    motion_prior_ckpt_path: str | Path,
    downstream_ckpt_path: str | Path,
  ) -> None:
    mp_path = Path(motion_prior_ckpt_path).expanduser()
    ds_path = Path(downstream_ckpt_path).expanduser()
    if not mp_path.is_file():
      raise FileNotFoundError(f"motion_prior ckpt not found: {mp_path}")
    if not ds_path.is_file():
      raise FileNotFoundError(f"downstream   ckpt not found: {ds_path}")

    mp_ckpt = torch.load(mp_path, map_location=self.device, weights_only=False)
    ds_ckpt = torch.load(ds_path, map_location=self.device, weights_only=False)

    for key in ("motion_prior", "decoder", "quantizer"):
      if key not in mp_ckpt:
        raise KeyError(
          f"motion_prior ckpt {mp_path} missing key {key!r}; "
          f"available top-level keys: {sorted(mp_ckpt.keys())}"
        )
    if "actor" not in ds_ckpt:
      raise KeyError(
        f"downstream ckpt {ds_path} missing key 'actor'; "
        f"available top-level keys: {sorted(ds_ckpt.keys())}"
      )

    self._policy.motion_prior.load_state_dict(mp_ckpt["motion_prior"], strict=True)
    self._policy.decoder.load_state_dict(mp_ckpt["decoder"], strict=True)
    self._policy.quantizer.load_state_dict(mp_ckpt["quantizer"], strict=True)
    self._policy.actor.load_state_dict(ds_ckpt["actor"], strict=True)

    print(f"[MjlabPassingVQController] motion_prior ckpt: {mp_path}")
    print(f"[MjlabPassingVQController] downstream   ckpt: {ds_path}")
    print(f"[MjlabPassingVQController] iter (mp / ds) = "
          f"{mp_ckpt.get('iter', '?')} / {ds_ckpt.get('iter', '?')}")

  # ------------------------------------------------------------------
  # Ball wiring
  # ------------------------------------------------------------------

  def init_ball(self, model) -> None:
    """Resolve the soccer ball's qpos / qvel addresses from the compiled
    MuJoCo model. Must be called before the first ``step()``."""
    ball_body_id = model.body("soccer_ball").id
    jnt_id = model.body_jntadr[ball_body_id]
    self._ball_qpos_adr = int(model.jnt_qposadr[jnt_id])
    self._ball_dof_adr = int(model.jnt_dofadr[jnt_id])
    print(
      f"[MjlabPassingVQController] Ball indices: "
      f"qpos_adr={self._ball_qpos_adr} dof_adr={self._ball_dof_adr}"
    )

  def set_source_pos(self, source_pos: np.ndarray) -> None:
    """Update the task command (where the ball was launched from)."""
    self.source_pos = np.asarray(source_pos, dtype=np.float32).copy()

  def _get_ball_state(self, mujoco_data) -> tuple[np.ndarray, np.ndarray]:
    assert self._ball_qpos_adr is not None and self._ball_dof_adr is not None, (
      "Call init_ball(model) before step()."
    )
    a = self._ball_qpos_adr
    d = self._ball_dof_adr
    ball_pos = mujoco_data.qpos[a : a + 3].astype(np.float32)
    ball_vel = mujoco_data.qvel[d : d + 3].astype(np.float32)
    return ball_pos, ball_vel

  # ------------------------------------------------------------------
  # Observation buffers
  # ------------------------------------------------------------------

  def _reset_buffers(self) -> None:
    h = self.HISTORY_LEN
    self._gravity_buf: deque[np.ndarray] = deque(
      [np.zeros(3, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._ang_vel_buf: deque[np.ndarray] = deque(
      [np.zeros(3, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._jpos_buf: deque[np.ndarray] = deque(
      [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._jvel_buf: deque[np.ndarray] = deque(
      [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._action_buf: deque[np.ndarray] = deque(
      [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(h)], maxlen=h
    )

  def _build_obs(self, mujoco_data) -> tuple[np.ndarray, np.ndarray]:
    # ---- Proprio (in MJCF joint order, no IsaacLab reindex) ----
    q_wxyz = mujoco_data.qpos[3:7]
    gravity = get_gravity_orientation(q_wxyz).astype(np.float32)
    ang_vel = mujoco_data.qvel[3:6].astype(np.float32)
    jpos_rel = (mujoco_data.qpos[7:36] - self.default_dof_pos).astype(np.float32)
    jvel = mujoco_data.qvel[6:35].astype(np.float32)

    self._gravity_buf.append(gravity)
    self._ang_vel_buf.append(ang_vel)
    self._jpos_buf.append(jpos_rel)
    self._jvel_buf.append(jvel)

    hist_gravity = np.concatenate(list(self._gravity_buf))     # 12
    hist_ang_vel = np.concatenate(list(self._ang_vel_buf))     # 12
    hist_jpos = np.concatenate(list(self._jpos_buf))           # 116
    hist_jvel = np.concatenate(list(self._jvel_buf))           # 116
    hist_actions = np.concatenate(list(self._action_buf))      # 116

    prop_obs = np.concatenate(
      [hist_gravity, hist_ang_vel, hist_jpos, hist_jvel, hist_actions]
    )

    # ---- Ball-related obs (current frame, no history) ----
    ball_pos_w, ball_vel_w = self._get_ball_state(mujoco_data)
    robot_pos_w = mujoco_data.qpos[0:3].astype(np.float32)

    # ball_relative_position: ball in robot body frame (3)
    rel_w = ball_pos_w - robot_pos_w
    ball_rel_pos = _quat_apply_inverse(q_wxyz, rel_w)

    # ball_velocity: world frame linear velocity (3)
    ball_vel = ball_vel_w

    # passing_source_position: (dx_b, dy_b, dist_w) — 2D XY delta rotated
    # into the body frame, plus the unrotated world-frame XY distance.
    src_rel_2d = self.source_pos[:2] - robot_pos_w[:2]                   # (2,)
    dist_w = float(np.linalg.norm(src_rel_2d))
    rel_3d = np.array([src_rel_2d[0], src_rel_2d[1], 0.0], dtype=np.float32)
    rel_b = _quat_apply_inverse(q_wxyz, rel_3d)
    passing_src = np.array(
      [rel_b[0], rel_b[1], dist_w], dtype=np.float32
    )

    policy_obs = np.concatenate(
      [ball_rel_pos, ball_vel, passing_src, prop_obs]
    )

    assert prop_obs.shape == (self.PROP_OBS_DIM,), prop_obs.shape
    assert policy_obs.shape == (self.OBS_DIM,), policy_obs.shape
    return policy_obs, prop_obs

  # ------------------------------------------------------------------
  # BaseController interface
  # ------------------------------------------------------------------

  def reset(self) -> None:
    self._reset_buffers()

  def step(self, mujoco_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy_obs_np, prop_obs_np = self._build_obs(mujoco_data)

    policy_obs_t = torch.from_numpy(policy_obs_np).unsqueeze(0).to(self.device)
    prop_obs_t = torch.from_numpy(prop_obs_np).unsqueeze(0).to(self.device)

    actions = (
      self._policy.inference(policy_obs_t, prop_obs_t).squeeze(0).cpu().numpy()
    )
    self._action_buf.append(actions.copy())

    target_q = actions * self.action_scale + self.default_dof_pos
    return target_q, self.kps, self.kds
