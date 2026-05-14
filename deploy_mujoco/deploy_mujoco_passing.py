"""
MuJoCo sim2sim deployment for G1 passing VQ task.

Usage:
  cd /home/dx/motionprior
  python deploy_mujoco/deploy_mujoco_passing.py \\
      --config deploy_mujoco/configs/g1_passing_vq.yaml

  python deploy_mujoco/deploy_mujoco_passing.py \\
      --policy_path logs/g1_passing_vq/.../model_XXXX.pt

键盘控制:
  R          : 重置仿真（重新随机化来球位置和速度）
  Ctrl-C     : 退出

Episode 流程:
  1. 来球从 source_pos 以随机速度射向机器人
  2. 机器人用脚停球或直接将球踢回 source 区域
  3. 按 R 手动重置
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
from passing_controller import PassingVQController
from common.keyboard_helper import KeyboardVelocityCommand
from common.bydmimic_utils import pd_control


SOCCER_BALL_RADIUS = 0.11


# ---------------------------------------------------------------------------
# MujocoSimulator subclass with ball support
# ---------------------------------------------------------------------------

class MujocoSimulatorWithBall(MujocoSimulator):
    """Extends MujocoSimulator to handle the extra soccer ball freejoint in qpos."""

    # Matching RigidBodyPropertiesCfg in G1PassingSceneCfg
    _BALL_LINEAR_DAMPING  = 0.3
    _BALL_ANGULAR_DAMPING = 0.2

    def __init__(self, xml_path: str, dt: float, control_decimation: int):
        super().__init__(xml_path=xml_path, dt=dt, control_decimation=control_decimation)

        # Resolve ball joint addresses from compiled model
        ball_id  = self.model.body("soccer_ball").id
        jnt_id   = self.model.body_jntadr[ball_id]
        self._ball_qpos_adr = self.model.jnt_qposadr[jnt_id]
        self._ball_dof_adr  = self.model.jnt_dofadr[jnt_id]

    def reset(self) -> None:
        """Reset robot to default pose; ball stays at XML default until reset_ball()."""
        mujoco.mj_resetData(self.model, self.data)
        # Only set robot joint angles (qpos[7:36]), not the ball freejoint
        self.data.qpos[7:36] = self.default_dof_pos
        mujoco.mj_step(self.model, self.data)
        self.sim_step_counter   = 0
        self.control_step_counter = 0

    def reset_ball(self, source_pos: np.ndarray, ball_vel: np.ndarray) -> None:
        """Place ball at source_pos with initial velocity ball_vel (world frame)."""
        a = self._ball_qpos_adr
        d = self._ball_dof_adr
        # position
        self.data.qpos[a:a + 3] = source_pos
        # quaternion (identity)
        self.data.qpos[a + 3]       = 1.0
        self.data.qpos[a + 4:a + 7] = 0.0
        # linear velocity
        self.data.qvel[d:d + 3] = ball_vel
        # angular velocity (zero)
        self.data.qvel[d + 3:d + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, target_q, kps, kds):
        """Override to use robot-only qpos/qvel slices (not ball DOFs)."""
        tau = pd_control(
            target_q,
            self.data.qpos[7:36],       # robot joints only
            kps,
            np.zeros_like(kds),
            self.data.qvel[6:35],       # robot joint velocities only
            kds,
        )
        self.data.ctrl[:] = tau
        mujoco.mj_step(self.model, self.data)

        # Apply ball damping: v *= (1 - damping * dt), matching PhysX linear/angular
        # damping semantics (RigidBodyPropertiesCfg: linear_damping=0.3, angular_damping=0.2)
        dt  = self.model.opt.timestep
        d   = self._ball_dof_adr
        lin = max(1.0 - self._BALL_LINEAR_DAMPING  * dt, 0.0)
        ang = max(1.0 - self._BALL_ANGULAR_DAMPING * dt, 0.0)
        self.data.qvel[d:d + 3]   *= lin
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
# Episode initialisation helpers
# ---------------------------------------------------------------------------

def _compute_ball_init(
    rng: np.random.Generator,
    theta: float,
    source_distance_range: tuple,
    source_lateral_range: tuple,
    ball_speed_range: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ball source position and initial velocity for a given robot heading theta.

    Mirrors G1PassingEnv._reset_scene_along_line exactly:
      - source is placed in the robot's FORWARD direction (theta) at distance d_src
        with lateral offset lat
      - ball velocity points from source toward robot (i.e., backward along theta)

    Args:
        theta: robot yaw angle [rad] — robot faces (cos_t, sin_t) direction.

    Returns:
        source_pos [3]: ball start position in world frame
        ball_vel   [3]: initial ball linear velocity in world frame
    """
    robot_pos = np.array([0.0, 0.0, 0.793], dtype=np.float32)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    d_src = rng.uniform(*source_distance_range)
    lat   = rng.uniform(*source_lateral_range)

    # Source is at robot's front: robot_pos + forward * d_src + left * lat
    src_x = robot_pos[0] + cos_t * d_src - sin_t * lat
    src_y = robot_pos[1] + sin_t * d_src + cos_t * lat
    source_pos = np.array([src_x, src_y, SOCCER_BALL_RADIUS], dtype=np.float32)

    # Velocity: from source toward robot (opposite of forward direction)
    direction = robot_pos[:2] - source_pos[:2]
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    speed     = rng.uniform(*ball_speed_range)
    ball_vel  = np.array([direction[0] * speed, direction[1] * speed, 0.0], dtype=np.float32)

    return source_pos, ball_vel


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

    cfg.setdefault("xml_path", os.path.join(_HERE, "xml", "g1_29dof_ball.xml"))
    cfg.setdefault("device",             "cpu")
    cfg.setdefault("simulation_dt",      0.002)
    cfg.setdefault("control_decimation", 10)
    cfg.setdefault("code_dim",           64)
    cfg.setdefault("num_code",           2048)
    cfg.setdefault("lab_lambda",         3.0)
    cfg.setdefault("render_decimation",  1)
    # Ball randomisation ranges — must match G1PassingCommandsCfg from training
    cfg.setdefault("source_distance_range", [3.0, 6.0])
    cfg.setdefault("source_lateral_range",  [-0.3, 0.3])
    cfg.setdefault("ball_speed_range",      [5.0, 9.0])

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
        description="MuJoCo sim2sim for G1 passing VQ policy",
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
    print("  G1 Passing VQ – MuJoCo sim2sim")
    print("=" * 60)
    print(f"  policy : {cfg['policy_path']}")
    print(f"  xml    : {cfg['xml_path']}")
    print(f"  device : {cfg['device']}")
    print(f"  dt={dt}s  decimation={decimation}  ctrl_freq={ctrl_freq:.1f}Hz")
    print(f"  source_dist  : {cfg['source_distance_range']} m")
    print(f"  source_lat   : {cfg['source_lateral_range']} m")
    print(f"  ball_speed   : {cfg['ball_speed_range']} m/s")
    print("=" * 60)

    rng = np.random.default_rng()

    # ---- Controller ----
    controller = PassingVQController(
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

    # Wire ball indices into controller
    controller.init_ball(simulator.model)

    _dist_range  = tuple(cfg["source_distance_range"])
    _lat_range   = tuple(cfg["source_lateral_range"])
    _speed_range = tuple(cfg["ball_speed_range"])

    # ---- Reset helper ----
    def do_reset():
        simulator.reset()
        controller.reset()

        # Random robot facing direction — matches training (G1PassingEnv samples random theta)
        theta     = rng.uniform(0.0, 2.0 * np.pi)
        half      = theta / 2.0
        # Quaternion [w, x, y, z] for yaw=theta around world z-axis
        robot_quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
        simulator.data.qpos[3:7] = robot_quat
        mujoco.mj_forward(simulator.model, simulator.data)

        # Ball source is in the robot's FORWARD direction (theta) — same as training
        source_pos, ball_vel = _compute_ball_init(
            rng, theta, _dist_range, _lat_range, _speed_range,
        )
        controller.source_pos = source_pos
        simulator.reset_ball(source_pos, ball_vel)

        dist  = np.linalg.norm(source_pos[:2])
        speed = np.linalg.norm(ball_vel)
        print(f"[Reset] theta={np.degrees(theta):.1f}°  "
              f"source=[{source_pos[0]:.2f},{source_pos[1]:.2f}]  "
              f"dist={dist:.1f}m  speed={speed:.1f}m/s")

    # ---- Keyboard (terminal focus) ----
    # Reuse KeyboardVelocityCommand only for the R-key / Ctrl-C handling;
    # velocity commands are unused by the passing policy.
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
