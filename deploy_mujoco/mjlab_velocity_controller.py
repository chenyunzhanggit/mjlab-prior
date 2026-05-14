"""mjlab velocity downstream-VQ controller for MuJoCo sim2sim.

Differences vs. the original ``downstream_controller.py`` (which was ported
from the IsaacLab motionprior framework):

* **Two-file ckpt loading.** mjlab saves the frozen motion-prior backbone
  and the trainable downstream actor / critic as **separate** files:

    - motion_prior ckpt → top-level keys ``motion_prior``, ``decoder``,
      ``quantizer`` (from ``MotionPriorSingleVQOnPolicyRunner.save``).
    - downstream ckpt   → top-level keys ``actor``, ``critic``, ``std``,
      ``optimizer``, ``iter`` (from ``DownStreamOnPolicyRunner.save``).

  IsaacLab reference packed everything into one model_state_dict. We need
  to load both files and stitch the modules together.

* **No IsaacLab joint-order reindex.** mjlab observes / acts in MJCF
  joint order (the policy's action vector is in MJCF order because
  ``find_joints_by_actuator_names`` preserves ``entity.joint_names``
  natural ordering, and observations use ``data.qpos[joint_q_adr]``
  which is the same). So both ``qpos[7:36]`` / ``qvel[6:35]`` and the
  policy output align with ``mujoco_joint_names``.

* **Constants imported directly from mjlab** instead of using the
  hand-tuned ``bydmimic_utils`` values, so ``action_scale`` / kp / kd
  / ``default_dof_pos`` exactly match the training-time
  ``BuiltinPositionActuatorCfg`` / ``KNEES_BENT_KEYFRAME``. This is
  critical for sim2sim fidelity — any kp/kd mismatch changes how the
  same target_q tracks, and any action_scale mismatch shifts the
  policy's output mapping.

* **No ``height_scan`` in motion_prior obs.** After the recent retrain
  (student obs is proprio-only), prop_obs is 372-dim.

Observation layout (matches our retrained motion_prior + downstream):

  prop_obs   (372): gravity(3×4) | ang_vel(3×4) | jpos(29×4) | jvel(29×4) | actions(29×4)
  policy_obs (375): vel_cmd(3)   | prop_obs(372)

History stacking is oldest-to-newest (mjlab's ``CircularBuffer.buffer``
property: index 0 oldest, index max_len-1 newest).

Inference path (VQ with optional LAB):

  raw   = actor(policy_obs)                       # (code_dim,)
  prior = motion_prior(prop_obs)                  # (code_dim,)
  z     = prior + lab_lambda * tanh(raw)          # use_lab=True
  q_z   = quantizer.nearest_lookup(z)             # (code_dim,)
  act   = decoder(cat([prop_obs, q_z]))           # (29,)
  target = act * action_scale + default_dof_pos   # MJCF joint order
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
# mjlab → MJCF-ordered constants (action_scale, kp, kd, default_pos)
# ---------------------------------------------------------------------------


def _build_mjlab_constants() -> dict[str, np.ndarray]:
  """Pull the exact training-time per-joint constants out of mjlab and
  arrange them in MJCF joint order.

  Returns a dict with keys ``action_scale``, ``kps``, ``kds``,
  ``default_dof_pos``, each a 29-d float32 array indexed by
  ``mujoco_joint_names``.
  """
  # Import lazily so the file can still be inspected without a full mjlab
  # install (e.g. for type-checking / docs); error message is informative.
  try:
    from mjlab.actuator.actuator_cfg import BuiltinPositionActuatorCfg  # noqa: F401
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
      G1_ACTION_SCALE,
      G1_ARTICULATION,
      KNEES_BENT_KEYFRAME,
    )
  except ImportError as e:
    raise ImportError(
      "Deploy needs to import ``mjlab.asset_zoo.robots.unitree_g1`` to read "
      "the exact action_scale / kp / kd / default_pos that match training. "
      "Make sure mjlab is installed in this env.\n  underlying error: " + str(e)
    ) from e

  num_joints = len(mujoco_joint_names)

  def _resolve_match(joint_name: str, patterns_iter) -> tuple[bool, str | None]:
    """Try each pattern (literal or `.*` regex) against ``joint_name``."""
    for pat in patterns_iter:
      if pat.startswith(".*"):
        if joint_name.endswith(pat[2:]):
          return True, pat
      else:
        if joint_name == pat:
          return True, pat
    return False, None

  # action_scale: G1_ACTION_SCALE is keyed by per-pattern entries already
  # (one entry per regex literal in each actuator cfg's target_names_expr).
  action_scale = np.zeros(num_joints, dtype=np.float32)
  for i, name in enumerate(mujoco_joint_names):
    found, key = _resolve_match(name, G1_ACTION_SCALE.keys())
    if not found:
      raise KeyError(
        f"No G1_ACTION_SCALE entry matches joint '{name}'. "
        f"Available patterns: {list(G1_ACTION_SCALE.keys())}"
      )
    action_scale[i] = G1_ACTION_SCALE[key]

  # kp / kd: walk G1_ARTICULATION.actuators (which are
  # BuiltinPositionActuatorCfg with .stiffness / .damping / .target_names_expr).
  kps = np.zeros(num_joints, dtype=np.float32)
  kds = np.zeros(num_joints, dtype=np.float32)
  for i, name in enumerate(mujoco_joint_names):
    matched_act = None
    for act in G1_ARTICULATION.actuators:
      ok, _ = _resolve_match(name, act.target_names_expr)
      if ok:
        matched_act = act
        break
    if matched_act is None:
      raise KeyError(
        f"No G1_ARTICULATION actuator matches joint '{name}'. "
        "Check actuator cfg target_names_expr coverage."
      )
    kps[i] = float(matched_act.stiffness)
    kds[i] = float(matched_act.damping)

  # default_dof_pos: KNEES_BENT_KEYFRAME.joint_pos is a dict of regex -> value;
  # unmatched joints default to 0.0 (mjlab's ``resolve_expr`` semantics).
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
# Network building blocks
# ---------------------------------------------------------------------------


class _QuantizerInference(nn.Module):
  """Nearest-codebook lookup. mjlab's ``EMAQuantizer`` exposes
  ``codebook`` / ``code_sum`` / ``code_count`` as registered buffers; we
  only need ``codebook`` at deploy time."""

  def __init__(self, num_code: int, code_dim: int) -> None:
    super().__init__()
    self.register_buffer("codebook", torch.zeros(num_code, code_dim))
    self.register_buffer("code_sum", torch.zeros(num_code, code_dim))
    self.register_buffer("code_count", torch.zeros(num_code))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    k_w = self.codebook.t()  # (D, K)
    dist = (
      torch.sum(x**2, dim=-1, keepdim=True)
      - 2.0 * torch.matmul(x, k_w)
      + torch.sum(k_w**2, dim=0, keepdim=True)
    )
    code_idx = dist.argmin(dim=-1)
    return F.embedding(code_idx, self.codebook)


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
  """Match ``rsl_rl.modules.MLP``'s topology: Linear -> activation pairs,
  last layer is plain Linear."""
  layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), nn.ELU()]
  for i in range(len(hidden_dims) - 1):
    layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ELU()]
  layers.append(nn.Linear(hidden_dims[-1], out_dim))
  return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Inference policy (mirrors mjlab.tasks.motion_prior.rl.policies.DownStreamVQPolicy)
# ---------------------------------------------------------------------------


class _MjlabDownstreamVQPolicy(nn.Module):
  """Deployment-only reconstruction of mjlab ``DownStreamVQPolicy``.

  Submodules and shapes are identical to training time so ``load_state_dict
  (strict=True)`` works against the two-file mjlab ckpt format.
  """

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

    self.motion_prior = _build_mlp(
      prop_obs_dim, motion_prior_hidden_dims, code_dim
    )
    self.quantizer = _QuantizerInference(num_code, code_dim)
    self.decoder = _build_mlp(
      prop_obs_dim + code_dim, decoder_hidden_dims, num_actions
    )
    self.actor = _build_mlp(num_obs, actor_hidden_dims, code_dim)

  @torch.no_grad()
  def inference(
    self, policy_obs: torch.Tensor, prop_obs: torch.Tensor
  ) -> torch.Tensor:
    """Deterministic forward — matches mjlab's ``policy_inference``."""
    raw = self.actor(policy_obs)
    prior = self.motion_prior(prop_obs)
    z = prior + self.lab_lambda * torch.tanh(raw) if self.use_lab else prior + raw
    q_z = self.quantizer(z)
    return self.decoder(torch.cat([prop_obs, q_z], dim=-1))


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class MjlabVelocityVQController(BaseController):
  """Velocity-tracking downstream-VQ controller for sim2sim.

  Args:
    motion_prior_ckpt_path: Path to the frozen mjlab motion-prior VQ ckpt
      (saved by ``MotionPriorSingleVQOnPolicyRunner.save``, top-level keys
      include ``motion_prior``, ``decoder``, ``quantizer``).
    downstream_ckpt_path: Path to the trainable mjlab downstream ckpt
      (saved by ``DownStreamOnPolicyRunner.save``, top-level keys include
      ``actor``, ``critic``, ``std``).
    device: ``"cpu"`` or ``"cuda:N"``.
    code_dim / num_code: VQ codebook dims. Must match the motion_prior
      training config (default: 64 / 2048 — mjlab defaults).
    actor_hidden_dims / motion_prior_hidden_dims / decoder_hidden_dims:
      MLP hidden widths; default ``[512, 256, 128]`` mirrors mjlab's
      ``RslRlDownstreamVQPolicyCfg`` / ``RslRlMotionPriorSingleVQPolicyCfg``.
    lab_lambda: Latent Action Barrier scale (default 3.0, matches mjlab).
    use_lab: Whether to use the LAB tanh-clip on the actor residual.
    velocity_commands: Initial (vx, vy, wz) command; defaults to zeros and
      is replaced each step from the keyboard helper.
  """

  HISTORY_LEN = 4
  NUM_JOINTS = 29
  PROP_OBS_DIM = 372
  OBS_DIM = 375  # prop_obs_dim + 3-D velocity command

  def __init__(
    self,
    motion_prior_ckpt_path: str | Path,
    downstream_ckpt_path: str | Path,
    device: str = "cpu",
    velocity_commands: np.ndarray | None = None,
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
    self._policy = _MjlabDownstreamVQPolicy(
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

    self.velocity_commands = (
      np.asarray(velocity_commands, dtype=np.float32)
      if velocity_commands is not None
      else np.zeros(3, dtype=np.float32)
    )

    self._reset_buffers()

    # Per-joint constants pulled straight from mjlab. These exactly match
    # what the training-time env saw, so sim2sim has zero kp/kd/scale
    # drift. Replaces the hand-tuned ``bydmimic_utils`` arrays.
    consts = _build_mjlab_constants()
    self.kps = consts["kps"]
    self.kds = consts["kds"]
    self.action_scale = consts["action_scale"]
    self.default_dof_pos = consts["default_dof_pos"]

    # BaseController required attrs (assigned directly; calling super()
    # would run the validator before these are set).
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
    """Stitch the frozen backbone (motion_prior / decoder / quantizer) and
    the trainable actor into the single inference policy module.

    Strict loading on each sub-state-dict — any mismatch (e.g. wrong
    ``code_dim`` / ``hidden_dims``) surfaces immediately instead of silently
    producing garbage.
    """
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

    print(f"[MjlabVelocityVQController] motion_prior ckpt: {mp_path}")
    print(f"[MjlabVelocityVQController] downstream   ckpt: {ds_path}")
    print(f"[MjlabVelocityVQController] iter (mp / ds) = "
          f"{mp_ckpt.get('iter', '?')} / {ds_ckpt.get('iter', '?')}")

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
    """Compute policy_obs (375) and prop_obs (372) from MuJoCo state.

    Order in qpos / qvel (MJCF, 29-DOF G1):
      qpos[0:3]   root position
      qpos[3:7]   root quaternion (w, x, y, z)
      qpos[7:36]  joint positions (MJCF order, matches mjlab obs ordering)
      qvel[0:3]   root linear velocity (world)
      qvel[3:6]   root angular velocity (body)
      qvel[6:35]  joint velocities (MJCF order)

    History is appended at this step but ``_action_buf`` still holds the
    PREVIOUS step's action (= ``last_action`` from mjlab's perspective).
    """
    # gravity: matches mjlab.envs.mdp.projected_gravity = R^T * (0,0,-1).
    # The bydmimic ``get_gravity_orientation`` formula is the same one in
    # algebraic form. Verified term-by-term against
    # ``data.projected_gravity_b = quat_apply_inverse(root_quat_w, gravity_vec_w)``.
    gravity = get_gravity_orientation(mujoco_data.qpos[3:7]).astype(np.float32)
    # ang_vel: matches mjlab's ``builtin_sensor("robot/imu_ang_vel")``.
    # That sensor is a ``<gyro>`` at the pelvis IMU site, which returns
    # angular velocity in BODY frame — same as ``qvel[3:6]`` for the
    # free joint of the pelvis (MuJoCo convention).
    ang_vel = mujoco_data.qvel[3:6].astype(np.float32)
    # joint_pos_rel: matches mjlab's ``joint_pos_rel = joint_pos - default_joint_pos``.
    # Uses the mjlab-derived ``default_dof_pos`` so the offset exactly
    # matches the training keyframe.
    jpos_rel = (mujoco_data.qpos[7:36] - self.default_dof_pos).astype(np.float32)
    # joint_vel_rel: mjlab subtracts ``default_joint_vel`` which is
    # all-zero (``KNEES_BENT_KEYFRAME.joint_vel={".*": 0.0}``), so raw
    # ``qvel[6:35]`` matches mjlab's value bit-for-bit.
    jvel = mujoco_data.qvel[6:35].astype(np.float32)

    self._gravity_buf.append(gravity)
    self._ang_vel_buf.append(ang_vel)
    self._jpos_buf.append(jpos_rel)
    self._jvel_buf.append(jvel)

    # Flatten oldest -> newest, matching mjlab's CircularBuffer.buffer
    # (chronological order, ``flatten_history_dim=True``).
    hist_gravity = np.concatenate(list(self._gravity_buf))    # 12
    hist_ang_vel = np.concatenate(list(self._ang_vel_buf))    # 12
    hist_jpos = np.concatenate(list(self._jpos_buf))          # 116
    hist_jvel = np.concatenate(list(self._jvel_buf))          # 116
    hist_actions = np.concatenate(list(self._action_buf))     # 116

    prop_obs = np.concatenate(
      [hist_gravity, hist_ang_vel, hist_jpos, hist_jvel, hist_actions]
    )
    policy_obs = np.concatenate([self.velocity_commands, prop_obs])

    assert prop_obs.shape == (self.PROP_OBS_DIM,), prop_obs.shape
    assert policy_obs.shape == (self.OBS_DIM,), policy_obs.shape
    return policy_obs, prop_obs

  # ------------------------------------------------------------------
  # BaseController interface
  # ------------------------------------------------------------------

  def reset(self) -> None:
    self._reset_buffers()

  def step(self, mujoco_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One policy step. Returns ``(target_q, kps, kds)`` in MJCF order."""
    policy_obs_np, prop_obs_np = self._build_obs(mujoco_data)

    policy_obs_t = torch.from_numpy(policy_obs_np).unsqueeze(0).to(self.device)
    prop_obs_t = torch.from_numpy(prop_obs_np).unsqueeze(0).to(self.device)

    actions = self._policy.inference(policy_obs_t, prop_obs_t).squeeze(0).cpu().numpy()
    self._action_buf.append(actions.copy())

    # ``JointPositionActionCfg(use_default_offset=True)`` semantics:
    #   target = action * action_scale + default_joint_pos
    target_q = actions * self.action_scale + self.default_dof_pos
    return target_q, self.kps, self.kds
