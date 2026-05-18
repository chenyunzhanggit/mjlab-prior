"""MuJoCo sim2sim for the mjlab G1 passing downstream-VQ policy.

Companion to ``deploy_mujoco_mjlab_velocity.py`` but for the passing task:
a ball is launched toward the robot from a random source position, and the
robot has to redirect it back into that source zone.

Differences vs. ``deploy_mujoco_passing.py`` (the IsaacLab port):

* Two-file ckpt — ``motion_prior_ckpt_path`` (frozen backbone) +
  ``downstream_ckpt_path`` (trainable actor / critic).
* MJCF joint order throughout (no IsaacLab reindex).
* ``action_scale`` / kp / kd / ``default_dof_pos`` pulled out of mjlab so
  sim2sim parameters match training exactly.

Usage:

  # YAML config (recommended)
  python deploy_mujoco/deploy_mujoco_mjlab_passing.py \\
      --config deploy_mujoco/configs/mjlab_g1_passing_vq.yaml

  # CLI flags
  python deploy_mujoco/deploy_mujoco_mjlab_passing.py \\
      --motion-prior-ckpt-path /path/to/motion_prior/model_XXXX.pt \\
      --downstream-ckpt-path   /path/to/downstream/model_XXXX.pt

Keyboard:
  R      : reset episode (re-randomises robot yaw + ball source / speed)
  Ctrl-C : exit
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import glfw  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from common.bydmimic_utils import pd_control  # noqa: E402
from common.keyboard_helper import KeyboardVelocityCommand  # noqa: E402
from mjlab_passing_controller import MjlabPassingVQController  # noqa: E402
from mujoco_sim import MujocoSimulator  # noqa: E402


SOCCER_BALL_RADIUS = 0.11


# ---------------------------------------------------------------------------
# Simulator subclass — handles the soccer ball freejoint.
# ---------------------------------------------------------------------------


class _MjlabMujocoSimulatorWithBall(MujocoSimulator):
  """``MujocoSimulator`` extended for the passing scene.

  Notes:
  * The robot uses the first freejoint (``qpos[0:7]`` / ``qvel[0:6]``) and
    joints ``qpos[7:36]`` / ``qvel[6:35]``. The soccer ball is appended as
    a second freejoint at the end of qpos / qvel — its addresses are
    resolved from the compiled model at construction time.
  * ``step()`` overrides the base PD to use the robot-only slices (the
    base uses ``qpos[7:]`` which would include the ball, wrong shape).
  * Ball damping is applied **outside** MuJoCo (we multiply ``qvel`` by
    ``(1 - damping * dt)`` each substep) so that the deploy matches the
    PhysX-style ``linear_damping`` / ``angular_damping`` the training env
    used. For passing the values come from the mjlab
    ``SoccerBallParams(linear_damping=0.3, angular_damping=0.2)``.
  * ``default_dof_pos`` is overwritten from mjlab constants by the caller
    via ``set_default_dof_pos`` (so the deploy resets to the same pose
    the training env's ``KNEES_BENT_KEYFRAME`` started from).
  """

  _BALL_LINEAR_DAMPING = 0.3
  _BALL_ANGULAR_DAMPING = 0.2

  def __init__(self, xml_path: str, dt: float, control_decimation: int) -> None:
    super().__init__(xml_path=xml_path, dt=dt, control_decimation=control_decimation)

    ball_id = self.model.body("soccer_ball").id
    jnt_id = self.model.body_jntadr[ball_id]
    self._ball_qpos_adr = int(self.model.jnt_qposadr[jnt_id])
    self._ball_dof_adr = int(self.model.jnt_dofadr[jnt_id])

    # Don't auto-log every sim step (the parent does this); flip on if
    # you want to plot torques / qpos curves after a run.
    self.use_log = False

  # ------------------------------------------------------------------
  # Setters
  # ------------------------------------------------------------------

  def set_default_dof_pos(self, default_dof_pos: np.ndarray) -> None:
    self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float32).copy()

  # ------------------------------------------------------------------
  # Reset
  # ------------------------------------------------------------------

  def reset(self) -> None:
    """Reset robot to the mjlab default keyframe. Ball is reset separately
    by ``reset_ball`` because the source / velocity sampling lives in
    the deploy main script (matches training semantics)."""
    mujoco.mj_resetData(self.model, self.data)
    # qpos[7:36] is the 29 robot joints — only those, not the ball
    # freejoint which sits at the end of qpos.
    self.data.qpos[7:36] = self.default_dof_pos
    # Also reset pelvis height to the keyframe value so the robot doesn't
    # spawn into the ground when KNEES_BENT_KEYFRAME's ``pos.z`` (0.76)
    # is below the default XML free-joint z (0.793).
    self.data.qpos[2] = 0.76
    mujoco.mj_step(self.model, self.data)
    self.sim_step_counter = 0
    self.control_step_counter = 0

  def reset_ball(self, source_pos: np.ndarray, ball_vel: np.ndarray) -> None:
    """Place the ball at ``source_pos`` with linear velocity ``ball_vel``
    (world frame). Quaternion is reset to identity; angular velocity to
    zero."""
    a = self._ball_qpos_adr
    d = self._ball_dof_adr
    self.data.qpos[a : a + 3] = source_pos
    self.data.qpos[a + 3] = 1.0
    self.data.qpos[a + 4 : a + 7] = 0.0
    self.data.qvel[d : d + 3] = ball_vel
    self.data.qvel[d + 3 : d + 6] = 0.0
    mujoco.mj_forward(self.model, self.data)

  # ------------------------------------------------------------------
  # Step
  # ------------------------------------------------------------------

  def step(self, target_q, kps, kds) -> None:
    """PD on the 29 robot joints; ball is purely passive."""
    tau = pd_control(
      target_q,
      self.data.qpos[7:36],
      kps,
      np.zeros_like(kds),
      self.data.qvel[6:35],
      kds,
    )
    self.data.ctrl[:] = tau
    mujoco.mj_step(self.model, self.data)

    # Apply ball damping: v *= (1 - damping * dt). Matches PhysX
    # RigidBodyPropertiesCfg semantics used in the training env.
    dt = self.model.opt.timestep
    d = self._ball_dof_adr
    lin = max(1.0 - self._BALL_LINEAR_DAMPING * dt, 0.0)
    ang = max(1.0 - self._BALL_ANGULAR_DAMPING * dt, 0.0)
    self.data.qvel[d : d + 3] *= lin
    self.data.qvel[d + 3 : d + 6] *= ang

    self.sim_step_counter += 1


# ---------------------------------------------------------------------------
# Episode init helper — matches training-time
# ``reset_ball_along_line_passing``.
# ---------------------------------------------------------------------------


def _sample_ball_init(
  rng: np.random.Generator,
  robot_xy: np.ndarray,
  theta: float,
  source_distance_range: tuple[float, float],
  source_lateral_range: tuple[float, float],
  ball_speed_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
  """Sample source position + ball velocity for one episode.

  Mirrors ``reset_ball_along_line_passing`` in mjlab:
    * source = robot_xy + forward(θ) * d + lateral(θ) * lat
    * ball velocity points from source back toward the robot at
      ``rand_uniform(*ball_speed_range)`` m/s.
  """
  cos_t = float(np.cos(theta))
  sin_t = float(np.sin(theta))

  d_src = float(rng.uniform(*source_distance_range))
  lat = float(rng.uniform(*source_lateral_range))

  src_x = robot_xy[0] + cos_t * d_src - sin_t * lat
  src_y = robot_xy[1] + sin_t * d_src + cos_t * lat
  source_pos = np.array([src_x, src_y, SOCCER_BALL_RADIUS], dtype=np.float32)

  # Velocity: from source toward robot, in the XY plane only.
  to_robot = robot_xy - source_pos[:2]
  norm = float(np.linalg.norm(to_robot))
  to_robot_unit = to_robot / max(norm, 1e-6)
  speed = float(rng.uniform(*ball_speed_range))
  ball_vel = np.array(
    [to_robot_unit[0] * speed, to_robot_unit[1] * speed, 0.0], dtype=np.float32
  )
  return source_pos, ball_vel


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict:
  with open(path, "r") as f:
    return yaml.safe_load(f) or {}


def _resolve_path(path: str, base_dir: str) -> str:
  if os.path.isabs(path):
    return path
  candidate = os.path.join(base_dir, path)
  if os.path.exists(candidate):
    return os.path.abspath(candidate)
  return os.path.abspath(path)


def build_config(args: argparse.Namespace) -> dict:
  cfg: dict = {}
  if args.config:
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = _load_yaml(args.config)
    for key in ("motion_prior_ckpt_path", "downstream_ckpt_path", "xml_path"):
      if cfg.get(key):
        cfg[key] = _resolve_path(cfg[key], cfg_dir)
  else:
    cfg_dir = os.getcwd()

  # CLI overrides.
  for key in (
    "motion_prior_ckpt_path",
    "downstream_ckpt_path",
    "xml_path",
  ):
    val = getattr(args, key, None)
    if val:
      cfg[key] = _resolve_path(val, cfg_dir)

  for key in (
    "device",
    "simulation_dt",
    "control_decimation",
    "render_decimation",
    "code_dim",
    "num_code",
    "lab_lambda",
    "use_lab",
  ):
    val = getattr(args, key, None)
    if val is not None:
      cfg[key] = val

  cfg.setdefault(
    "xml_path", os.path.join(_HERE, "xml", "g1_29dof_mjlab_passing.xml")
  )
  cfg.setdefault("device", "cpu")
  cfg.setdefault("simulation_dt", 0.002)
  cfg.setdefault("control_decimation", 10)
  cfg.setdefault("render_decimation", 1)
  cfg.setdefault("code_dim", 64)
  cfg.setdefault("num_code", 2048)
  cfg.setdefault("lab_lambda", 3.0)
  cfg.setdefault("use_lab", True)
  # Ball randomisation ranges — must match mjlab passing command cfg.
  cfg.setdefault("source_distance_range", [3.0, 6.0])
  cfg.setdefault("source_lateral_range", [-0.3, 0.3])
  cfg.setdefault("ball_speed_range", [5.0, 9.0])
  # Robot initial yaw range. Defaults to full circle to match training.
  cfg.setdefault("robot_yaw_range", [0.0, 6.283185307179586])

  for required in ("motion_prior_ckpt_path", "downstream_ckpt_path"):
    if not cfg.get(required):
      raise ValueError(
        f"{required} is required (set in YAML config or via "
        f"--{required.replace('_', '-')})"
      )

  return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="MuJoCo sim2sim for mjlab G1 passing VQ policy",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument("--config", type=str, default=None)
  parser.add_argument(
    "--motion-prior-ckpt-path",
    dest="motion_prior_ckpt_path",
    type=str,
    default=None,
    help="Frozen motion-prior VQ backbone (mjlab single-VQ ckpt).",
  )
  parser.add_argument(
    "--downstream-ckpt-path",
    dest="downstream_ckpt_path",
    type=str,
    default=None,
    help="Trainable downstream actor / critic (mjlab DS ckpt).",
  )
  parser.add_argument("--xml-path", dest="xml_path", type=str, default=None)
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument(
    "--simulation-dt", dest="simulation_dt", type=float, default=None
  )
  parser.add_argument(
    "--control-decimation", dest="control_decimation", type=int, default=None
  )
  parser.add_argument(
    "--render-decimation", dest="render_decimation", type=int, default=None
  )
  parser.add_argument("--code-dim", dest="code_dim", type=int, default=None)
  parser.add_argument("--num-code", dest="num_code", type=int, default=None)
  parser.add_argument("--lab-lambda", dest="lab_lambda", type=float, default=None)
  parser.add_argument("--no-lab", dest="use_lab", action="store_false", default=None)
  return parser.parse_args()


# ---------------------------------------------------------------------------
# Keyboard / reset hook
# ---------------------------------------------------------------------------


def _hook_glfw_keyboard(viewer, on_reset_fn) -> None:
  prev_cb = viewer._key_callback

  def _cb(window, key, scancode, action, mods):
    prev_cb(window, key, scancode, action, mods)
    if action == glfw.PRESS and key == glfw.KEY_R:
      on_reset_fn()

  glfw.set_key_callback(viewer.window, _cb)
  print(
    "[Keyboard] GLFW hook registered — press R in the MuJoCo window to reset."
  )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
  args = parse_args()
  cfg = build_config(args)

  dt = cfg["simulation_dt"]
  decimation = cfg["control_decimation"]
  render_decimation = cfg["render_decimation"]
  ctrl_freq = 1.0 / (dt * decimation)

  print("=" * 64)
  print("  mjlab G1 passing VQ — MuJoCo sim2sim")
  print("=" * 64)
  print(f"  motion_prior ckpt : {cfg['motion_prior_ckpt_path']}")
  print(f"  downstream   ckpt : {cfg['downstream_ckpt_path']}")
  print(f"  xml               : {cfg['xml_path']}")
  print(f"  device            : {cfg['device']}")
  print(f"  dt={dt}s  decim={decimation}  ctrl_freq={ctrl_freq:.1f} Hz")
  print(
    f"  code_dim={cfg['code_dim']}  num_code={cfg['num_code']}"
    f"  lab={cfg['use_lab']} (λ={cfg['lab_lambda']})"
  )
  print(f"  source_dist  : {cfg['source_distance_range']} m")
  print(f"  source_lat   : {cfg['source_lateral_range']} m")
  print(f"  ball_speed   : {cfg['ball_speed_range']} m/s")
  print("=" * 64)

  rng = np.random.default_rng()

  controller = MjlabPassingVQController(
    motion_prior_ckpt_path=cfg["motion_prior_ckpt_path"],
    downstream_ckpt_path=cfg["downstream_ckpt_path"],
    device=cfg["device"],
    code_dim=cfg["code_dim"],
    num_code=cfg["num_code"],
    lab_lambda=cfg["lab_lambda"],
    use_lab=cfg["use_lab"],
  )

  simulator = _MjlabMujocoSimulatorWithBall(
    xml_path=cfg["xml_path"],
    dt=dt,
    control_decimation=decimation,
  )
  # Override the simulator's default with mjlab's KNEES_BENT_KEYFRAME so
  # the reset pose matches training (the controller already pulled it).
  simulator.set_default_dof_pos(controller.default_dof_pos)

  # Wire ball indices into the controller.
  controller.init_ball(simulator.model)

  dist_range = tuple(cfg["source_distance_range"])
  lat_range = tuple(cfg["source_lateral_range"])
  speed_range = tuple(cfg["ball_speed_range"])
  yaw_range = tuple(cfg["robot_yaw_range"])

  def do_reset() -> None:
    simulator.reset()
    controller.reset()

    # Random robot yaw — matches training (G1 passing reset samples random θ).
    theta = float(rng.uniform(*yaw_range))
    half = theta / 2.0
    robot_quat = np.array(
      [np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64
    )
    simulator.data.qpos[3:7] = robot_quat
    mujoco.mj_forward(simulator.model, simulator.data)

    # Sample ball source + velocity, push them into both sim and controller.
    robot_xy = simulator.data.qpos[0:2].astype(np.float32)
    source_pos, ball_vel = _sample_ball_init(
      rng, robot_xy, theta, dist_range, lat_range, speed_range
    )
    controller.set_source_pos(source_pos)
    simulator.reset_ball(source_pos, ball_vel)

    print(
      f"[Reset] yaw={np.degrees(theta):.1f}°  "
      f"source=({source_pos[0]:.2f}, {source_pos[1]:.2f})  "
      f"dist={float(np.linalg.norm(source_pos[:2] - robot_xy)):.2f} m  "
      f"speed={float(np.linalg.norm(ball_vel)):.2f} m/s"
    )

  # KeyboardVelocityCommand only used for the Ctrl-C / lifecycle hook; the
  # passing policy has no velocity commands.
  kb = KeyboardVelocityCommand(on_reset=do_reset)
  _hook_glfw_keyboard(simulator.viewer, do_reset)

  do_reset()
  target_q = controller.default_dof_pos.copy()
  kp_arr = controller.kps
  kd_arr = controller.kds

  print("\n[SIM] Running. Press R to reset, Ctrl-C to exit.\n")
  try:
    while simulator.is_alive():
      if simulator.should_run_control():
        target_q, kp_arr, kd_arr = controller.step(simulator.data)
      simulator.step(target_q, kp_arr, kd_arr)
      if simulator.sim_step_counter % render_decimation == 0:
        simulator.render()
  except KeyboardInterrupt:
    print("\n[SIM] Interrupted.")
  finally:
    kb.stop()
    print("[SIM] Done.")


if __name__ == "__main__":
  main()
