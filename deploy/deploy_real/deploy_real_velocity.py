"""Real-robot deploy of the mjlab G1 velocity-tracking VQ policy.

Mirrors the protocol of the original Unitree reference ``deploy_real.py``
(zero-torque → move-to-default → policy loop → damping) but swaps the
single-file TorchScript policy for the mjlab two-file checkpoint stack:

  prop_obs   (372): gravity(3×4) | ang_vel(3×4) | jpos_rel(29×4) | jvel(29×4) | actions(29×4)
  policy_obs (375): vel_cmd(3) | prop_obs(372)

  raw   = actor(policy_obs)
  prior = motion_prior(prop_obs)
  z     = prior + lab_lambda * tanh(raw)        # use_lab=True
  q_z   = quantizer.nearest(z)
  act   = decoder(cat([prop_obs, q_z]))
  target_q = act * action_scale + default_dof_pos   # MJCF joint order

All 29 joints are policy-controlled (no separate leg / arm-waist split
like the reference's locomotion ckpt that only drove the 12 legs).
``action_scale`` / kp / kd / default_dof_pos are pulled from mjlab's
``g1_constants`` so they exactly match training.

Joystick mapping (Unitree wireless remote, mirrors reference):
  ly  →  vx   (forward / back)
  lx  → -vy   (left / right — sign flipped so left stick left = +y)
  rx  → -wz   (turn)

Buttons:
  start  → after zero-torque, advance to move-to-default
  A      → after move-to-default, start policy loop
  select → exit policy loop (then damping)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Unitree SDK.
from unitree_sdk2py.core.channel import (  # type: ignore
  ChannelFactoryInitialize,
  ChannelPublisher,
  ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (  # type: ignore
  unitree_hg_msg_dds__LowCmd_,
  unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG  # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG  # type: ignore
from unitree_sdk2py.utils.crc import CRC  # type: ignore

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common.command_helper import (  # noqa: E402
  MotorMode,
  create_damping_cmd,
  create_zero_cmd,
  init_cmd_hg,
)
from common.remote_controller import KeyMap, RemoteController  # noqa: E402
from common.rotation_helper import get_gravity_orientation  # noqa: E402


# ---------------------------------------------------------------------------
# Policy reconstruction (same as deploy_mujoco's controller)
# ---------------------------------------------------------------------------


class _QuantizerInference(nn.Module):
  """Nearest-codebook lookup — mirrors mjlab's ``EMAQuantizer`` at deploy time."""

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
    code_idx = dist.argmin(dim=-1)
    return F.embedding(code_idx, self.codebook)


def _build_mlp(in_dim: int, hidden_dims, out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), nn.ELU()]
  for i in range(len(hidden_dims) - 1):
    layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ELU()]
  layers.append(nn.Linear(hidden_dims[-1], out_dim))
  return nn.Sequential(*layers)


class _MjlabDownstreamVQPolicy(nn.Module):
  """Same topology as ``deploy_mujoco/mjlab_velocity_controller.py``."""

  def __init__(
    self,
    num_obs: int,
    num_actions: int,
    prop_obs_dim: int,
    code_dim: int,
    num_code: int,
    actor_hidden_dims,
    motion_prior_hidden_dims,
    decoder_hidden_dims,
    lab_lambda: float,
    use_lab: bool,
  ) -> None:
    super().__init__()
    self.use_lab = use_lab
    self.lab_lambda = lab_lambda
    self.motion_prior = _build_mlp(prop_obs_dim, motion_prior_hidden_dims, code_dim)
    self.quantizer = _QuantizerInference(num_code, code_dim)
    self.decoder = _build_mlp(prop_obs_dim + code_dim, decoder_hidden_dims, num_actions)
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
# Per-joint constants from mjlab (MJCF order, 29-d)
# ---------------------------------------------------------------------------

# Same MJCF joint ordering used in deploy_mujoco. Match what mjlab observes
# (``data.qpos[joint_q_adr]``) and what mjlab acts on
# (``find_joints_by_actuator_names`` preserves entity.joint_names = MJCF
# declaration order).
MJCF_JOINT_NAMES: tuple[str, ...] = (
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
)
NUM_JOINTS = 29


def _build_mjlab_constants() -> dict[str, np.ndarray]:
  """Pull training-time per-joint constants out of mjlab. See
  ``deploy/deploy_mujoco/mjlab_velocity_controller.py`` for the same
  routine — kept duplicated here so ``deploy_real`` doesn't import
  anything from the sim2sim package."""
  try:
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
      G1_ACTION_SCALE,
      G1_ARTICULATION,
      KNEES_BENT_KEYFRAME,
    )
  except ImportError as e:
    raise ImportError(
      "Real deploy needs mjlab installed (``uv sync --group deploy``)."
    ) from e

  def _match(name: str, patterns) -> str | None:
    for pat in patterns:
      if pat.startswith(".*"):
        if name.endswith(pat[2:]):
          return pat
      elif name == pat:
        return pat
    return None

  action_scale = np.zeros(NUM_JOINTS, dtype=np.float32)
  kps = np.zeros(NUM_JOINTS, dtype=np.float32)
  kds = np.zeros(NUM_JOINTS, dtype=np.float32)
  default_pos = np.zeros(NUM_JOINTS, dtype=np.float32)

  for i, name in enumerate(MJCF_JOINT_NAMES):
    pat = _match(name, G1_ACTION_SCALE.keys())
    if pat is None:
      raise KeyError(f"G1_ACTION_SCALE missing pattern for {name}")
    action_scale[i] = G1_ACTION_SCALE[pat]

    matched = None
    for act in G1_ARTICULATION.actuators:
      if _match(name, act.target_names_expr) is not None:
        matched = act
        break
    if matched is None:
      raise KeyError(f"G1_ARTICULATION missing actuator for {name}")
    kps[i] = float(matched.stiffness)
    kds[i] = float(matched.damping)

    kf_dict = KNEES_BENT_KEYFRAME.joint_pos or {}
    val = 0.0
    for kpat, kv in kf_dict.items():
      if kpat.startswith(".*"):
        if name.endswith(kpat[2:]):
          val = kv
      elif name == kpat:
        val = kv
    default_pos[i] = float(val)

  return dict(action_scale=action_scale, kps=kps, kds=kds, default_pos=default_pos)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Config:
  """Loads YAML config + applies CLI overrides."""

  def __init__(self, cfg: dict) -> None:
    self.control_dt: float = float(cfg.get("control_dt", 0.02))
    self.msg_type: str = cfg.get("msg_type", "hg")
    self.imu_type: str = cfg.get("imu_type", "pelvis")
    self.lowcmd_topic: str = cfg.get("lowcmd_topic", "rt/lowcmd")
    self.lowstate_topic: str = cfg.get("lowstate_topic", "rt/lowstate")

    self.motion_prior_ckpt_path: str = cfg["motion_prior_ckpt_path"]
    self.downstream_ckpt_path: str = cfg["downstream_ckpt_path"]

    # Maps mjlab MJCF joint index → Unitree motor index. For G1 hg the
    # default order in low_state.motor_state matches the MJCF joint order
    # (0..28), so a plain identity is correct unless your firmware uses
    # a different motor map.
    self.joint2motor_idx: list[int] = cfg.get(
      "joint2motor_idx", list(range(NUM_JOINTS))
    )

    self.code_dim: int = int(cfg.get("code_dim", 64))
    self.num_code: int = int(cfg.get("num_code", 2048))
    self.lab_lambda: float = float(cfg.get("lab_lambda", 3.0))
    self.use_lab: bool = bool(cfg.get("use_lab", True))

    self.actor_hidden_dims = tuple(cfg.get("actor_hidden_dims", (512, 256, 128)))
    self.motion_prior_hidden_dims = tuple(
      cfg.get("motion_prior_hidden_dims", (512, 256, 128))
    )
    self.decoder_hidden_dims = tuple(
      cfg.get("decoder_hidden_dims", (512, 256, 128))
    )

    # Joystick scaling: max command magnitude reached at full stick.
    # max_cmd matches the training velocity range (mjlab vel env training
    # range is lin_vel_x ∈ (-1, 2), lin_vel_y ∈ (-1, 1), ang_vel_z
    # ∈ (-π, π)); we conservatively clip below those values for safety
    # on the real robot.
    self.max_cmd = np.array(
      cfg.get("max_cmd", [0.5, 0.3, 1.0]), dtype=np.float32
    )


def _load_config(yaml_path: str, overrides: dict) -> Config:
  with open(yaml_path, "r") as f:
    raw = yaml.safe_load(f) or {}
  raw.update({k: v for k, v in overrides.items() if v is not None})
  return Config(raw)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class VelocityRealController:
  """Real-robot controller for the mjlab velocity-VQ policy.

  Lifecycle:
    1. Construct → load policy, init DDS, wait for first lowstate.
    2. ``zero_torque_state()``  — wait for START button, keep zero torque.
    3. ``move_to_default_pos()`` — interpolate to KNEES_BENT keyframe.
    4. ``default_pos_state()``   — hold at default, wait for A button.
    5. ``run()`` in a loop until SELECT pressed, then damping & exit.
  """

  HISTORY_LEN = 4
  PROP_OBS_DIM = 372
  OBS_DIM = 375

  def __init__(self, config: Config) -> None:
    self.config = config
    self.remote_controller = RemoteController()

    # Constants (mjlab-derived, MJCF order).
    consts = _build_mjlab_constants()
    self.action_scale = consts["action_scale"]
    self.kps = consts["kps"]
    self.kds = consts["kds"]
    self.default_dof_pos = consts["default_pos"]

    # Policy.
    self.policy = _MjlabDownstreamVQPolicy(
      num_obs=self.OBS_DIM,
      num_actions=NUM_JOINTS,
      prop_obs_dim=self.PROP_OBS_DIM,
      code_dim=config.code_dim,
      num_code=config.num_code,
      actor_hidden_dims=config.actor_hidden_dims,
      motion_prior_hidden_dims=config.motion_prior_hidden_dims,
      decoder_hidden_dims=config.decoder_hidden_dims,
      lab_lambda=config.lab_lambda,
      use_lab=config.use_lab,
    )
    self._load_ckpts(config.motion_prior_ckpt_path, config.downstream_ckpt_path)
    self.policy.eval()

    # Observation buffers.
    h = self.HISTORY_LEN
    self._gravity_buf: deque[np.ndarray] = deque(
      [np.zeros(3, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._ang_vel_buf: deque[np.ndarray] = deque(
      [np.zeros(3, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._jpos_buf: deque[np.ndarray] = deque(
      [np.zeros(NUM_JOINTS, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._jvel_buf: deque[np.ndarray] = deque(
      [np.zeros(NUM_JOINTS, dtype=np.float32) for _ in range(h)], maxlen=h
    )
    self._action_buf: deque[np.ndarray] = deque(
      [np.zeros(NUM_JOINTS, dtype=np.float32) for _ in range(h)], maxlen=h
    )

    # Working buffers.
    self.qj = np.zeros(NUM_JOINTS, dtype=np.float32)
    self.dqj = np.zeros(NUM_JOINTS, dtype=np.float32)
    self.cmd = np.zeros(3, dtype=np.float32)

    # DDS.
    self.low_cmd = unitree_hg_msg_dds__LowCmd_()
    self.low_state = unitree_hg_msg_dds__LowState_()
    self.mode_pr_ = MotorMode.PR
    self.mode_machine_ = 0

    self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
    self.lowcmd_publisher_.Init()

    self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
    self.lowstate_subscriber.Init(self._on_lowstate, 10)

    self._wait_for_low_state()
    init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

  # ------------------------------------------------------------------
  # Setup helpers
  # ------------------------------------------------------------------

  def _load_ckpts(self, mp_path: str, ds_path: str) -> None:
    mp = Path(mp_path).expanduser()
    ds = Path(ds_path).expanduser()
    if not mp.is_file():
      raise FileNotFoundError(f"motion_prior ckpt not found: {mp}")
    if not ds.is_file():
      raise FileNotFoundError(f"downstream   ckpt not found: {ds}")
    mp_ckpt = torch.load(mp, map_location="cpu", weights_only=False)
    ds_ckpt = torch.load(ds, map_location="cpu", weights_only=False)
    for k in ("motion_prior", "decoder", "quantizer"):
      if k not in mp_ckpt:
        raise KeyError(
          f"motion_prior ckpt {mp} missing key {k!r}; "
          f"available: {sorted(mp_ckpt.keys())}"
        )
    if "actor" not in ds_ckpt:
      raise KeyError(
        f"downstream ckpt {ds} missing key 'actor'; "
        f"available: {sorted(ds_ckpt.keys())}"
      )
    self.policy.motion_prior.load_state_dict(mp_ckpt["motion_prior"], strict=True)
    self.policy.decoder.load_state_dict(mp_ckpt["decoder"], strict=True)
    self.policy.quantizer.load_state_dict(mp_ckpt["quantizer"], strict=True)
    self.policy.actor.load_state_dict(ds_ckpt["actor"], strict=True)
    print(f"[deploy_real] motion_prior ckpt: {mp}")
    print(f"[deploy_real] downstream   ckpt: {ds}")
    print(f"[deploy_real] iter (mp / ds) = "
          f"{mp_ckpt.get('iter', '?')} / {ds_ckpt.get('iter', '?')}")

  def _on_lowstate(self, msg: LowStateHG) -> None:
    self.low_state = msg
    self.mode_machine_ = self.low_state.mode_machine
    self.remote_controller.set(self.low_state.wireless_remote)

  def _wait_for_low_state(self) -> None:
    while self.low_state.tick == 0:
      time.sleep(self.config.control_dt)
    print("[deploy_real] Connected to robot.")

  def _send_cmd(self, cmd: LowCmdHG) -> None:
    cmd.crc = CRC().Crc(cmd)
    self.lowcmd_publisher_.Write(cmd)

  # ------------------------------------------------------------------
  # State machine (mirrors reference deploy_real.py)
  # ------------------------------------------------------------------

  def zero_torque_state(self) -> None:
    print("[deploy_real] Zero torque. Press START to continue.")
    while self.remote_controller.button[KeyMap.start] != 1:
      create_zero_cmd(self.low_cmd)
      self._send_cmd(self.low_cmd)
      time.sleep(self.config.control_dt)

  def move_to_default_pos(self) -> None:
    print("[deploy_real] Moving to default pos (2 s).")
    total_time = 2.0
    num_step = int(total_time / self.config.control_dt)
    motor_idx = self.config.joint2motor_idx

    init_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
    for i in range(NUM_JOINTS):
      init_pos[i] = self.low_state.motor_state[motor_idx[i]].q

    for step in range(num_step):
      alpha = step / num_step
      for j in range(NUM_JOINTS):
        mi = motor_idx[j]
        self.low_cmd.motor_cmd[mi].q = (
          init_pos[j] * (1 - alpha) + self.default_dof_pos[j] * alpha
        )
        self.low_cmd.motor_cmd[mi].qd = 0.0
        self.low_cmd.motor_cmd[mi].kp = float(self.kps[j])
        self.low_cmd.motor_cmd[mi].kd = float(self.kds[j])
        self.low_cmd.motor_cmd[mi].tau = 0.0
      self._send_cmd(self.low_cmd)
      time.sleep(self.config.control_dt)

  def default_pos_state(self) -> None:
    print("[deploy_real] Holding default pos. Press A to start policy.")
    motor_idx = self.config.joint2motor_idx
    while self.remote_controller.button[KeyMap.A] != 1:
      for i in range(NUM_JOINTS):
        mi = motor_idx[i]
        self.low_cmd.motor_cmd[mi].q = float(self.default_dof_pos[i])
        self.low_cmd.motor_cmd[mi].qd = 0.0
        self.low_cmd.motor_cmd[mi].kp = float(self.kps[i])
        self.low_cmd.motor_cmd[mi].kd = float(self.kds[i])
        self.low_cmd.motor_cmd[mi].tau = 0.0
      self._send_cmd(self.low_cmd)
      time.sleep(self.config.control_dt)

  # ------------------------------------------------------------------
  # Policy loop
  # ------------------------------------------------------------------

  def _read_state(self) -> tuple[np.ndarray, np.ndarray]:
    """Read joint pos/vel for all 29 joints (MJCF order)."""
    motor_idx = self.config.joint2motor_idx
    for i in range(NUM_JOINTS):
      mi = motor_idx[i]
      self.qj[i] = self.low_state.motor_state[mi].q
      self.dqj[i] = self.low_state.motor_state[mi].dq
    return self.qj.copy(), self.dqj.copy()

  def _build_obs(self) -> tuple[np.ndarray, np.ndarray]:
    # IMU.
    quat = self.low_state.imu_state.quaternion  # w,x,y,z
    ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
    if self.config.imu_type != "pelvis":
      raise NotImplementedError(
        "Only imu_type='pelvis' is supported (G1 default). For 'torso', "
        "port the waist-yaw rotation from reference's transform_imu_data."
      )
    gravity = get_gravity_orientation(quat).astype(np.float32)

    # Joint state.
    qj, dqj = self._read_state()
    jpos_rel = (qj - self.default_dof_pos).astype(np.float32)
    jvel = dqj.astype(np.float32)

    # Buffers (oldest→newest like mjlab's CircularBuffer.buffer).
    self._gravity_buf.append(gravity)
    self._ang_vel_buf.append(ang_vel)
    self._jpos_buf.append(jpos_rel)
    self._jvel_buf.append(jvel)

    hist_gravity = np.concatenate(list(self._gravity_buf))    # 12
    hist_ang_vel = np.concatenate(list(self._ang_vel_buf))    # 12
    hist_jpos = np.concatenate(list(self._jpos_buf))          # 116
    hist_jvel = np.concatenate(list(self._jvel_buf))          # 116
    hist_actions = np.concatenate(list(self._action_buf))     # 116

    prop_obs = np.concatenate(
      [hist_gravity, hist_ang_vel, hist_jpos, hist_jvel, hist_actions]
    )

    # Joystick → velocity command, scaled by max_cmd. Reference mapping.
    self.cmd[0] = self.remote_controller.ly                          # vx
    self.cmd[1] = -self.remote_controller.lx                         # vy
    self.cmd[2] = -self.remote_controller.rx                         # wz
    vel_cmd = (self.cmd * self.config.max_cmd).astype(np.float32)

    policy_obs = np.concatenate([vel_cmd, prop_obs])
    assert prop_obs.shape == (self.PROP_OBS_DIM,), prop_obs.shape
    assert policy_obs.shape == (self.OBS_DIM,), policy_obs.shape
    return policy_obs, prop_obs

  def run(self) -> None:
    policy_obs_np, prop_obs_np = self._build_obs()
    policy_obs_t = torch.from_numpy(policy_obs_np).unsqueeze(0)
    prop_obs_t = torch.from_numpy(prop_obs_np).unsqueeze(0)
    actions = self.policy.inference(policy_obs_t, prop_obs_t).squeeze(0).numpy()
    self._action_buf.append(actions.copy())

    target_q = actions * self.action_scale + self.default_dof_pos

    motor_idx = self.config.joint2motor_idx
    for i in range(NUM_JOINTS):
      mi = motor_idx[i]
      self.low_cmd.motor_cmd[mi].q = float(target_q[i])
      self.low_cmd.motor_cmd[mi].qd = 0.0
      self.low_cmd.motor_cmd[mi].kp = float(self.kps[i])
      self.low_cmd.motor_cmd[mi].kd = float(self.kds[i])
      self.low_cmd.motor_cmd[mi].tau = 0.0
    self._send_cmd(self.low_cmd)
    time.sleep(self.config.control_dt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Real-robot deploy of mjlab G1 velocity-tracking VQ policy.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument("net", type=str, help="network interface (e.g. eth0)")
  parser.add_argument("config", type=str, help="YAML config name in configs/")
  parser.add_argument("--motion-prior-ckpt-path", dest="motion_prior_ckpt_path",
                      type=str, default=None)
  parser.add_argument("--downstream-ckpt-path", dest="downstream_ckpt_path",
                      type=str, default=None)
  args = parser.parse_args()

  # YAML can be passed as a bare name (looked up in configs/) or full path.
  cfg_path = args.config
  if not Path(cfg_path).is_file():
    cfg_path = str(_HERE / "configs" / args.config)
  if not Path(cfg_path).is_file():
    raise FileNotFoundError(f"config not found: {cfg_path}")

  overrides = dict(
    motion_prior_ckpt_path=args.motion_prior_ckpt_path,
    downstream_ckpt_path=args.downstream_ckpt_path,
  )
  config = _load_config(cfg_path, overrides)

  ChannelFactoryInitialize(0, args.net)
  controller = VelocityRealController(config)

  controller.zero_torque_state()
  controller.move_to_default_pos()
  controller.default_pos_state()

  print("[deploy_real] Policy running. SELECT to exit.")
  while True:
    try:
      controller.run()
      if controller.remote_controller.button[KeyMap.select] == 1:
        break
    except KeyboardInterrupt:
      break

  create_damping_cmd(controller.low_cmd)
  controller._send_cmd(controller.low_cmd)
  print("[deploy_real] Damping engaged. Exit.")


if __name__ == "__main__":
  main()
