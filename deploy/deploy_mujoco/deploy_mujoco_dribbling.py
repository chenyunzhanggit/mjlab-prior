"""
MuJoCo sim2sim deployment for G1 dribbling VQ task.

Usage:
  cd /home/dx/motionprior
  python deploy_mujoco/deploy_mujoco_dribbling.py \\
      --config deploy_mujoco/configs/g1_dribbling_vq.yaml

  python deploy_mujoco/deploy_mujoco_dribbling.py \\
      --policy_path logs/g1_dribbling_vq/.../model_XXXX.pt

键盘控制:
  R          : 重置仿真（重新随机化机器人朝向、球和目标位置）
  Ctrl-C     : 退出

Episode 流程:
  1. 机器人随机朝向 θ
  2. 球置于机器人正前方 0.4~0.7m，静止
  3. 目标置于机器人正前方 8~15m
  4. 机器人运球向目标前进
  5. 按 R 手动重置
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import glfw
import mujoco
import numpy as np
import yaml

from mujoco_sim import MujocoSimulator
from dribbling_controller import DribblingVQController
from common.keyboard_helper import KeyboardVelocityCommand
from common.bydmimic_utils import pd_control


SOCCER_BALL_RADIUS = 0.11


# ---------------------------------------------------------------------------
# MujocoSimulator subclass with ball support (dribbling physics)
# ---------------------------------------------------------------------------

class MujocoSimulatorWithBall(MujocoSimulator):
    """Extends MujocoSimulator with soccer ball freejoint support.

    Dribbling physics: linear_damping=0.8, angular_damping=0.0
    """

    _BALL_LINEAR_DAMPING  = 0.8
    _BALL_ANGULAR_DAMPING = 0.0

    def __init__(self, xml_path: str, dt: float, control_decimation: int):
        super().__init__(xml_path=xml_path, dt=dt, control_decimation=control_decimation)

        ball_id  = self.model.body("soccer_ball").id
        jnt_id   = self.model.body_jntadr[ball_id]
        self._ball_qpos_adr = self.model.jnt_qposadr[jnt_id]
        self._ball_dof_adr  = self.model.jnt_dofadr[jnt_id]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:36] = self.default_dof_pos
        mujoco.mj_step(self.model, self.data)
        self.sim_step_counter   = 0
        self.control_step_counter = 0

    def reset_ball(self, ball_pos: np.ndarray) -> None:
        """Place ball at ball_pos with zero velocity."""
        a = self._ball_qpos_adr
        d = self._ball_dof_adr
        self.data.qpos[a:a + 3] = ball_pos
        self.data.qpos[a + 3]       = 1.0
        self.data.qpos[a + 4:a + 7] = 0.0
        self.data.qvel[d:d + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, target_q, kps, kds):
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

        # PhysX-matched damping: linear_damping=0.8, angular_damping=0.0
        dt  = self.model.opt.timestep
        d   = self._ball_dof_adr
        lin = max(1.0 - self._BALL_LINEAR_DAMPING  * dt, 0.0)
        ang = max(1.0 - self._BALL_ANGULAR_DAMPING * dt, 0.0)
        self.data.qvel[d:d + 3]     *= lin
        self.data.qvel[d + 3:d + 6] *= ang

        if self.use_log:
            self.log["time_step"].append(self.sim_step_counter)
            self.log["target_q"].append(target_q.copy())
            self.log["q"].append(self.data.qpos[7:36].copy())
            self.log["dq"].append(self.data.qvel[6:35].copy())
            self.log["tau"].append(tau.copy())

        self.sim_step_counter += 1

    def get_ball_pos(self) -> np.ndarray:
        a = self._ball_qpos_adr
        return self.data.qpos[a:a + 3].copy()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    candidate = os.path.join(base_dir, path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(path)


def build_config(args) -> dict:
    cfg: dict = {}

    if args.config:
        cfg_dir = os.path.dirname(os.path.abspath(args.config))
        cfg = _load_yaml(args.config)
        for key in ("policy_path", "xml_path"):
            if key in cfg and cfg[key]:
                cfg[key] = _resolve_path(cfg[key], cfg_dir)
    else:
        cfg_dir = os.getcwd()

    if args.policy_path:
        cfg["policy_path"] = _resolve_path(args.policy_path, cfg_dir)
    if args.xml_path:
        cfg["xml_path"] = _resolve_path(args.xml_path, cfg_dir)
    if args.device:
        cfg["device"] = args.device

    for key in ("simulation_dt", "control_decimation",
                "code_dim", "num_code", "lab_lambda", "render_decimation"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    cfg.setdefault("xml_path", os.path.join(_HERE, "xml", "g1_29dof_ball_dribbling.xml"))
    cfg.setdefault("device",             "cpu")
    cfg.setdefault("simulation_dt",      0.002)
    cfg.setdefault("control_decimation", 10)
    cfg.setdefault("code_dim",           64)
    cfg.setdefault("num_code",           2048)
    cfg.setdefault("lab_lambda",         3.0)
    cfg.setdefault("render_decimation",  1)
    # Ball/goal init ranges — must match G1DribblingEnv._reset_scene_along_line
    cfg.setdefault("ball_distance_range", [0.4, 0.7])   # ball forward distance from robot [m]
    cfg.setdefault("goal_distance_range", [8.0, 15.0])  # goal forward distance from robot [m]

    if not cfg.get("policy_path"):
        raise ValueError("policy_path is required (set in YAML config or via --policy_path)")

    return cfg


# ---------------------------------------------------------------------------
# GLFW keyboard hook
# ---------------------------------------------------------------------------

def _hook_glfw_keyboard(viewer, on_reset_fn) -> None:
    prev_cb = viewer._key_callback

    def _cb(window, key, scancode, action, mods):
        prev_cb(window, key, scancode, action, mods)
        if action == glfw.PRESS and key == glfw.KEY_R:
            on_reset_fn()

    glfw.set_key_callback(viewer.window, _cb)
    print("[Keyboard] GLFW hook registered — press R from the MuJoCo window to reset.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="MuJoCo sim2sim for G1 dribbling VQ policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",      type=str,   default=None)
    parser.add_argument("--policy_path", type=str,   default=None)
    parser.add_argument("--xml_path",    type=str,   default=None)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--simulation_dt",      type=float, default=None)
    parser.add_argument("--control_decimation", type=int,   default=None)
    parser.add_argument("--code_dim",    type=int,   default=None)
    parser.add_argument("--num_code",    type=int,   default=None)
    parser.add_argument("--lab_lambda",  type=float, default=None)
    parser.add_argument("--render_decimation", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = build_config(args)

    dt                = cfg["simulation_dt"]
    decimation        = cfg["control_decimation"]
    render_decimation = cfg["render_decimation"]
    ctrl_freq         = 1.0 / (dt * decimation)

    print("=" * 60)
    print("  G1 Dribbling VQ – MuJoCo sim2sim")
    print("=" * 60)
    print(f"  policy : {cfg['policy_path']}")
    print(f"  xml    : {cfg['xml_path']}")
    print(f"  device : {cfg['device']}")
    print(f"  dt={dt}s  decimation={decimation}  ctrl_freq={ctrl_freq:.1f}Hz")
    print(f"  ball_dist : {cfg['ball_distance_range']} m (robot forward)")
    print(f"  goal_dist : {cfg['goal_distance_range']} m (robot forward)")
    print("=" * 60)

    rng = np.random.default_rng()

    # ---- Controller ----
    controller = DribblingVQController(
        policy_path=cfg["policy_path"],
        device=cfg["device"],
        code_dim=cfg["code_dim"],
        num_code=cfg["num_code"],
        lab_lambda=cfg["lab_lambda"],
    )

    # ---- Simulator ----
    simulator = MujocoSimulatorWithBall(
        xml_path=cfg["xml_path"],
        dt=dt,
        control_decimation=decimation,
    )

    controller.init_ball(simulator.model)

    _ball_range = tuple(cfg["ball_distance_range"])
    _goal_range = tuple(cfg["goal_distance_range"])

    # ---- Reset helper ----
    def do_reset():
        simulator.reset()
        controller.reset()

        # Random robot facing direction — matches training (G1DribblingEnv samples random theta)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        half  = theta / 2.0
        robot_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
        simulator.data.qpos[3:7] = robot_quat
        mujoco.mj_forward(simulator.model, simulator.data)

        robot_pos = simulator.data.qpos[0:3].copy()
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

        # Ball: 0.4~0.7m forward along theta, stationary
        d_ball = rng.uniform(*_ball_range)
        ball_pos = np.array([
            robot_pos[0] + cos_t * d_ball,
            robot_pos[1] + sin_t * d_ball,
            SOCCER_BALL_RADIUS,
        ], dtype=np.float32)
        simulator.reset_ball(ball_pos)

        # Goal: 8~15m forward along theta
        d_goal = rng.uniform(*_goal_range)
        goal_pos = np.array([
            robot_pos[0] + cos_t * d_goal,
            robot_pos[1] + sin_t * d_goal,
            0.0,
        ], dtype=np.float32)
        controller.goal_pos = goal_pos

        print(f"[Reset] theta={np.degrees(theta):.1f}°  "
              f"ball=[{ball_pos[0]:.2f},{ball_pos[1]:.2f}]  d_ball={d_ball:.2f}m  "
              f"goal=[{goal_pos[0]:.2f},{goal_pos[1]:.2f}]  d_goal={d_goal:.1f}m")

    # ---- Keyboard (terminal focus) ----
    kb = KeyboardVelocityCommand(on_reset=do_reset)

    # ---- GLFW hook (window focus) ----
    _hook_glfw_keyboard(simulator.viewer, do_reset)

    # ---- Initial reset ----
    do_reset()

    target_q = controller.default_dof_pos.copy()
    kps_arr  = controller.kps
    kds_arr  = controller.kds

    print("\n[SIM] Running. Press R to reset, Ctrl-C to exit.\n")

    try:
        while simulator.is_alive():
            if simulator.should_run_control():
                target_q, kps_arr, kds_arr = controller.step(simulator.data)

            simulator.step(target_q, kps_arr, kds_arr)
            if simulator.sim_step_counter % render_decimation == 0:
                simulator.render()

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted.")
    finally:
        kb.stop()
        print("[SIM] Done.")


if __name__ == "__main__":
    main()
