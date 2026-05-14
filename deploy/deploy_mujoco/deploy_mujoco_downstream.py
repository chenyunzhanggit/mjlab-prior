"""
MuJoCo sim2sim deployment for G1 downstream VQ task.

Usage (从项目根目录或 deploy_mujoco/ 目录运行均可):

  # 使用配置文件（推荐）
  cd /home/dx/motionprior
  python deploy_mujoco/deploy_mujoco_downstream.py \\
      --config deploy_mujoco/configs/g1_downstream_vq.yaml

  # 直接指定参数
  python deploy_mujoco/deploy_mujoco_downstream.py \\
      --policy_path logs/dt_velocity_tracking_vq/.../model_8000.pt

  # 配置文件 + 参数覆盖
  python deploy_mujoco/deploy_mujoco_downstream.py \\
      --config deploy_mujoco/configs/g1_downstream_vq.yaml \\
      --policy_path logs/dt_velocity_tracking_vq/.../model_10000.pt

键盘控制 (增量模式, 同 IsaacLab play_vel DebugKeyboardWithOneHot):
  Up Arrow   : vx += step_vx
  Down Arrow : vx -= step_vx
  Z          : wz += step_wz
  X          : wz -= step_wz
  L          : 速度归零
  R          : 重置仿真
  Ctrl-C     : 退出
"""

import argparse
import os
import sys

# Allow running from project root or deploy_mujoco/
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import glfw
import numpy as np
import yaml

from mujoco_sim import MujocoSimulator
from downstream_controller import DownstreamVQController
from common.keyboard_helper import KeyboardVelocityCommand


# -----------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path: str, base_dir: str) -> str:
    """Make relative paths absolute, anchored to base_dir."""
    if os.path.isabs(path):
        return path
    # Try relative to base_dir first, then cwd
    candidate = os.path.join(base_dir, path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(path)


def build_config(args) -> dict:
    """Merge YAML config with CLI overrides (CLI wins)."""
    cfg: dict = {}

    # 1. Load YAML base config
    if args.config:
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        cfg = _load_yaml(args.config)
        # Resolve relative paths inside YAML relative to the YAML file's directory
        for key in ("policy_path", "xml_path"):
            if key in cfg and cfg[key]:
                cfg[key] = _resolve_path(cfg[key], cfg_dir)
    else:
        cfg_dir = os.getcwd()

    # 2. CLI overrides (non-None values win)
    if args.policy_path:
        cfg["policy_path"] = _resolve_path(args.policy_path, cfg_dir)
    if args.xml_path:
        cfg["xml_path"] = _resolve_path(args.xml_path, cfg_dir)
    if args.device:
        cfg["device"] = args.device

    # Numeric overrides
    for key in ("simulation_dt", "control_decimation",
                "code_dim", "num_code", "lab_lambda",
                "step_vx", "step_wz", "max_vx", "max_wz",
                "render_decimation"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # Defaults
    cfg.setdefault("xml_path", os.path.join(
        _HERE, "..",
        "source/whole_body_tracking/whole_body_tracking/assets/"
        "unitree_description/mjcf/g1.xml",
    ))
    cfg.setdefault("device",             "cpu")
    cfg.setdefault("simulation_dt",      0.002)
    cfg.setdefault("control_decimation", 10)
    cfg.setdefault("code_dim",           64)
    cfg.setdefault("num_code",           2048)
    cfg.setdefault("lab_lambda",         3.0)
    cfg.setdefault("step_vx",            0.1)
    cfg.setdefault("step_wz",            0.1)
    cfg.setdefault("max_vx",             1.0)
    cfg.setdefault("max_wz",             1.0)
    cfg.setdefault("render_decimation",  1)   # render every N sim steps; >1 = faster display

    if not cfg.get("policy_path"):
        raise ValueError("policy_path is required (set in YAML config or via --policy_path)")

    return cfg


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MuJoCo sim2sim for G1 downstream VQ policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (e.g. configs/g1_downstream_vq.yaml)",
    )
    # CLI overrides – all optional when using --config
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--xml_path",    type=str, default=None)
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--simulation_dt",      type=float, default=None)
    parser.add_argument("--control_decimation", type=int,   default=None)
    parser.add_argument("--code_dim",    type=int,   default=None)
    parser.add_argument("--num_code",    type=int,   default=None)
    parser.add_argument("--lab_lambda",  type=float, default=None)
    parser.add_argument("--step_vx",  type=float, default=None, help="vx increment per keypress (m/s)")
    parser.add_argument("--step_wz",  type=float, default=None, help="wz increment per keypress (rad/s)")
    parser.add_argument("--max_vx",   type=float, default=None, help="max |vx| clamp")
    parser.add_argument("--max_wz",   type=float, default=None, help="max |wz| clamp")
    parser.add_argument("--render_decimation", type=int, default=None,
                        help="render every N sim steps (1=every step, 5=5x faster display)")
    return parser.parse_args()


def _hook_glfw_keyboard(viewer, kb) -> None:
    """
    Chain our velocity-increment callback onto the viewer's existing GLFW key handler.
    This makes keyboard commands work when the MuJoCo window has focus.
    The termios thread in KeyboardVelocityCommand handles the terminal-focus case.
    """
    _KEY_MAP = {
        glfw.KEY_UP:   "up",
        glfw.KEY_DOWN: "down",
        glfw.KEY_Z:    "z",
        glfw.KEY_X:    "x",
        glfw.KEY_L:    "l",
        glfw.KEY_R:    "r",
    }

    prev_cb = viewer._key_callback  # mujoco_viewer's existing handler (space=pause, etc.)

    def _cb(window, key, scancode, action, mods):
        prev_cb(window, key, scancode, action, mods)
        if action == glfw.PRESS:          # fire once per physical keypress, not on repeat
            name = _KEY_MAP.get(key)
            if name:
                kb._apply(name)

    glfw.set_key_callback(viewer.window, _cb)
    print("[Keyboard] GLFW hook registered — commands also work from the MuJoCo window.")


def main():
    args   = parse_args()
    cfg    = build_config(args)

    dt               = cfg["simulation_dt"]
    decimation       = cfg["control_decimation"]
    render_decimation = cfg["render_decimation"]
    ctrl_freq        = 1.0 / (dt * decimation)

    print("=" * 60)
    print("  G1 Downstream VQ – MuJoCo sim2sim")
    print("=" * 60)
    print(f"  policy : {cfg['policy_path']}")
    print(f"  xml    : {cfg['xml_path']}")
    print(f"  device : {cfg['device']}")
    print(f"  dt={dt}s  decimation={decimation}  ctrl_freq={ctrl_freq:.1f}Hz")
    print("=" * 60)

    # ---- Controller ----
    controller = DownstreamVQController(
        policy_path=cfg["policy_path"],
        device=cfg["device"],
        code_dim=cfg["code_dim"],
        num_code=cfg["num_code"],
        lab_lambda=cfg["lab_lambda"],
    )

    # ---- Simulator ----
    simulator = MujocoSimulator(
        xml_path=cfg["xml_path"],
        dt=dt,
        control_decimation=decimation,
    )

    # ---- Keyboard ----
    def on_reset():
        print("[R] Resetting simulation...")
        simulator.reset()
        controller.reset()

    kb = KeyboardVelocityCommand(
        step_vx=cfg["step_vx"],
        step_wz=cfg["step_wz"],
        max_vx=cfg["max_vx"],
        max_wz=cfg["max_wz"],
        on_reset=on_reset,
    )

    # Hook GLFW so commands also work when the MuJoCo viewer window has focus.
    # termios (above) handles the case when the terminal has focus.
    _hook_glfw_keyboard(simulator.viewer, kb)

    # ---- Run ----
    target_q = controller.default_dof_pos.copy()
    kps      = controller.kps
    kds      = controller.kds

    simulator.reset()
    controller.reset()

    print("\n[SIM] Running. Use keyboard to control (see above). Ctrl-C to exit.\n")

    try:
        while simulator.is_alive():
            if simulator.should_run_control():
                # Feed current keyboard command into controller
                controller.velocity_commands = kb.command
                target_q, kps, kds = controller.step(simulator.data)

            simulator.step(target_q, kps, kds)
            if simulator.sim_step_counter % render_decimation == 0:
                simulator.render()

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted.")
    finally:
        kb.stop()
        print("[SIM] Done.")


if __name__ == "__main__":
    main()
