"""MuJoCo sim2sim for the mjlab G1 velocity-tracking downstream-VQ policy.

Differences vs. ``deploy_mujoco_downstream.py`` (which targets the original
IsaacLab motionprior framework's single-file checkpoint):

* Takes **two** ckpt paths — ``motion_prior_ckpt_path`` (frozen backbone)
  and ``downstream_ckpt_path`` (trainable actor / critic) — to match
  mjlab's two-file save layout.
* No IsaacLab-style joint-order reindexing: mjlab observes / acts in MJCF
  order directly.
* prop_obs dim = 372 (no height_scan), matching the user's retrained
  motion-prior backbone.

Usage:

  # YAML config
  python deploy_mujoco/deploy_mujoco_mjlab_velocity.py \\
      --config deploy_mujoco/configs/mjlab_g1_velocity_vq.yaml

  # CLI flags
  python deploy_mujoco/deploy_mujoco_mjlab_velocity.py \\
      --motion-prior-ckpt-path /path/to/motion_prior/model_XXXX.pt \\
      --downstream-ckpt-path   /path/to/downstream/model_XXXX.pt

Keyboard control (same as the existing downstream deploy):
  ↑ / ↓ : vx += / -= step_vx
  Z / X : wz += / -= step_wz
  L     : zero command
  R     : reset sim
  Ctrl-C: exit
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import glfw  # noqa: E402
import yaml  # noqa: E402

from common.keyboard_helper import KeyboardVelocityCommand  # noqa: E402
from mjlab_velocity_controller import MjlabVelocityVQController  # noqa: E402
from mujoco_sim import MujocoSimulator  # noqa: E402


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

  # CLI overrides win when set.
  for src_key, dst_key in (
    ("motion_prior_ckpt_path", "motion_prior_ckpt_path"),
    ("downstream_ckpt_path", "downstream_ckpt_path"),
    ("xml_path", "xml_path"),
  ):
    val = getattr(args, src_key, None)
    if val:
      cfg[dst_key] = _resolve_path(val, cfg_dir)

  for key in (
    "device",
    "simulation_dt",
    "control_decimation",
    "render_decimation",
    "code_dim",
    "num_code",
    "lab_lambda",
    "use_lab",
    "step_vx",
    "step_wz",
    "max_vx",
    "max_wz",
  ):
    val = getattr(args, key, None)
    if val is not None:
      cfg[key] = val

  # Defaults. xml_path points at mjlab's own G1 MJCF — the deploy uses raw
  # MuJoCo + manual PD via ``mujoco_sim.MujocoSimulator``; kp/kd come from
  # ``common.bydmimic_utils``. (Switching to mjlab's compiled <position>
  # actuators is a future refactor.)
  cfg.setdefault(
    "xml_path",
    os.path.join(_HERE, "xml", "g1_29dof.xml"),
  )
  cfg.setdefault("device", "cpu")
  cfg.setdefault("simulation_dt", 0.002)
  cfg.setdefault("control_decimation", 10)
  cfg.setdefault("render_decimation", 1)
  cfg.setdefault("code_dim", 64)
  cfg.setdefault("num_code", 2048)
  cfg.setdefault("lab_lambda", 3.0)
  cfg.setdefault("use_lab", True)
  cfg.setdefault("step_vx", 0.1)
  cfg.setdefault("step_wz", 0.1)
  cfg.setdefault("max_vx", 2.0)
  cfg.setdefault("max_wz", 3.0)

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
    description="MuJoCo sim2sim for mjlab G1 velocity-tracking VQ policy",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
  )
  parser.add_argument("--config", type=str, default=None)
  parser.add_argument("--motion-prior-ckpt-path", dest="motion_prior_ckpt_path",
                      type=str, default=None,
                      help="Frozen motion-prior VQ backbone (mjlab single-VQ ckpt).")
  parser.add_argument("--downstream-ckpt-path", dest="downstream_ckpt_path",
                      type=str, default=None,
                      help="Trainable downstream actor / critic (mjlab DS ckpt).")
  parser.add_argument("--xml-path", dest="xml_path", type=str, default=None)
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument("--simulation-dt", dest="simulation_dt", type=float, default=None)
  parser.add_argument("--control-decimation", dest="control_decimation",
                      type=int, default=None)
  parser.add_argument("--render-decimation", dest="render_decimation",
                      type=int, default=None)
  parser.add_argument("--code-dim", dest="code_dim", type=int, default=None)
  parser.add_argument("--num-code", dest="num_code", type=int, default=None)
  parser.add_argument("--lab-lambda", dest="lab_lambda", type=float, default=None)
  parser.add_argument("--no-lab", dest="use_lab", action="store_false", default=None)
  parser.add_argument("--step-vx", dest="step_vx", type=float, default=None)
  parser.add_argument("--step-wz", dest="step_wz", type=float, default=None)
  parser.add_argument("--max-vx", dest="max_vx", type=float, default=None)
  parser.add_argument("--max-wz", dest="max_wz", type=float, default=None)
  return parser.parse_args()


# ---------------------------------------------------------------------------
# GLFW keyboard hook (so commands fire when the MuJoCo window has focus)
# ---------------------------------------------------------------------------

_GLFW_KEYMAP = {
  glfw.KEY_UP: "up",
  glfw.KEY_DOWN: "down",
  glfw.KEY_Z: "z",
  glfw.KEY_X: "x",
  glfw.KEY_L: "l",
  glfw.KEY_R: "r",
}


def _hook_glfw_keyboard(viewer, kb: KeyboardVelocityCommand) -> None:
  prev_cb = viewer._key_callback

  def _cb(window, key, scancode, action, mods):
    prev_cb(window, key, scancode, action, mods)
    if action == glfw.PRESS:
      name = _GLFW_KEYMAP.get(key)
      if name:
        kb._apply(name)

  glfw.set_key_callback(viewer.window, _cb)
  print("[Keyboard] GLFW hook registered — commands also work from the MuJoCo window.")


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
  print("  mjlab G1 velocity-tracking VQ — MuJoCo sim2sim")
  print("=" * 64)
  print(f"  motion_prior ckpt : {cfg['motion_prior_ckpt_path']}")
  print(f"  downstream   ckpt : {cfg['downstream_ckpt_path']}")
  print(f"  xml               : {cfg['xml_path']}")
  print(f"  device            : {cfg['device']}")
  print(f"  dt={dt}s  decim={decimation}  ctrl_freq={ctrl_freq:.1f} Hz")
  print(f"  code_dim={cfg['code_dim']}  num_code={cfg['num_code']}"
        f"  lab={cfg['use_lab']} (λ={cfg['lab_lambda']})")
  print("=" * 64)

  controller = MjlabVelocityVQController(
    motion_prior_ckpt_path=cfg["motion_prior_ckpt_path"],
    downstream_ckpt_path=cfg["downstream_ckpt_path"],
    device=cfg["device"],
    code_dim=cfg["code_dim"],
    num_code=cfg["num_code"],
    lab_lambda=cfg["lab_lambda"],
    use_lab=cfg["use_lab"],
  )

  simulator = MujocoSimulator(
    xml_path=cfg["xml_path"],
    dt=dt,
    control_decimation=decimation,
  )

  def on_reset() -> None:
    print("\n[R] Resetting simulation.")
    simulator.reset()
    controller.reset()

  kb = KeyboardVelocityCommand(
    step_vx=cfg["step_vx"],
    step_wz=cfg["step_wz"],
    max_vx=cfg["max_vx"],
    max_wz=cfg["max_wz"],
    on_reset=on_reset,
  )
  _hook_glfw_keyboard(simulator.viewer, kb)

  target_q = controller.default_dof_pos.copy()
  kp_arr = controller.kps
  kd_arr = controller.kds

  simulator.reset()
  controller.reset()

  print("\n[SIM] Running. Use keyboard to drive twist commands. Ctrl-C to exit.\n")
  try:
    while simulator.is_alive():
      if simulator.should_run_control():
        controller.velocity_commands = kb.command
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
