"""
Passing VQ Policy controller for MuJoCo sim2sim deployment.

Observation layout (from G1PassingObservationsCfg):
  policy (actor input, 381):
    ball_rel_pos(3) | ball_vel(3) | passing_src(3)
    | gravity(3×4) | ang_vel(3×4) | jpos(29×4) | jvel(29×4) | actions(29×4)
  prob_obs (motion_prior, 372):
    gravity(3×4) | ang_vel(3×4) | jpos(29×4) | jvel(29×4) | actions(29×4)

Ball observations have no history — always current values.
History order is oldest-to-newest (IsaacLab convention).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common.bydmimic_utils import (
    default_angles,
    kps,
    kds,
    action_scale,
    mujoco_to_isaaclab_reindex,
    isaaclab_to_mujoco_reindex,
    get_gravity_orientation,
)
from base_controller import BaseController

SOCCER_BALL_RADIUS = 0.11


# ---------------------------------------------------------------------------
# Rotation helper
# ---------------------------------------------------------------------------

def _quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world frame to body frame given q = [w, x, y, z].

    Equivalent to IsaacLab's quat_apply_inverse.
    """
    w = q_wxyz[0]
    qv = q_wxyz[1:]
    a = v * (2.0 * w ** 2 - 1.0)
    b = np.cross(qv, v) * (w * 2.0)
    c = qv * (np.dot(qv, v) * 2.0)
    return a - b + c


# ---------------------------------------------------------------------------
# Lightweight network building blocks (identical to downstream_controller.py)
# ---------------------------------------------------------------------------

class _QuantizerInference(nn.Module):
    def __init__(self, nb_code: int, code_dim: int):
        super().__init__()
        self.register_buffer("codebook", torch.zeros(nb_code, code_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k_w = self.codebook.t()
        dist = (
            torch.sum(x ** 2, dim=-1, keepdim=True)
            - 2.0 * torch.matmul(x, k_w)
            + torch.sum(k_w ** 2, dim=0, keepdim=True)
        )
        return F.embedding(dist.argmin(dim=-1), self.codebook)


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), nn.ELU()]
    for i in range(len(hidden_dims) - 1):
        layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ELU()]
    layers.append(nn.Linear(hidden_dims[-1], out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Inference policy
# ---------------------------------------------------------------------------

class _PassingVQPolicyInference(nn.Module):
    """Deployment-only reconstruction of DownStreamVQPolicy for passing task."""

    PROB_OBS_DIM = 372  # same as vel tracking (no ball info)

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        code_dim: int,
        num_code: int,
        hidden_dims: list[int],
        lab_lambda: float = 3.0,
        use_mp: bool = True,
    ):
        super().__init__()
        self.use_mp = use_mp
        self.lab_lambda = lab_lambda
        prob_dim = self.PROB_OBS_DIM

        self.actor        = _build_mlp(num_obs,              hidden_dims, code_dim)
        self.motion_prior = _build_mlp(prob_dim,             hidden_dims, code_dim)
        self.quantizer    = _QuantizerInference(num_code,    code_dim)
        self.decoder      = _build_mlp(prob_dim + code_dim,  hidden_dims, num_actions)

    @torch.no_grad()
    def inference(self, obs: torch.Tensor, prob_obs: torch.Tensor) -> torch.Tensor:
        raw = self.actor(obs)
        if self.use_mp:
            mp  = self.motion_prior(prob_obs)
            z   = mp + self.lab_lambda * torch.tanh(raw)
        else:
            z = raw
        q_z     = self.quantizer(z)
        actions = self.decoder(torch.cat([prob_obs, q_z], dim=-1))
        return actions


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class PassingVQController(BaseController):
    """
    Passing VQ policy controller for MuJoCo sim2sim.

    Observation layout (381-dim policy obs):
      ball_rel_pos (3) | ball_vel (3) | passing_src (3)
      | gravity×4 (12) | ang_vel×4 (12) | jpos×4 (116) | jvel×4 (116) | actions×4 (116)

    Call init_ball(model) after creating the MujocoSimulator so that body/joint
    indices for the soccer ball are resolved from the compiled model.

    Set source_pos (world-frame [x,y,z]) before each episode to tell the
    controller where the ball was launched from.
    """

    HISTORY_LEN  = 4
    NUM_JOINTS   = 29
    # policy obs: 3+3+3 (ball) + 12+12+116+116+116 (proprioception with history)
    OBS_DIM      = 381
    PROB_OBS_DIM = 372

    def __init__(
        self,
        policy_path: str,
        device: str = "cpu",
        code_dim: int = 64,
        num_code: int = 2048,
        hidden_dims: list[int] | None = None,
        lab_lambda: float = 3.0,
        use_mp: bool = True,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.device = torch.device(device)

        self._policy = _PassingVQPolicyInference(
            num_obs=self.OBS_DIM,
            num_actions=self.NUM_JOINTS,
            code_dim=code_dim,
            num_code=num_code,
            hidden_dims=hidden_dims,
            lab_lambda=lab_lambda,
            use_mp=use_mp,
        ).to(self.device)

        ckpt       = torch.load(policy_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        remapped: dict[str, torch.Tensor] = {}
        prefix_allow = {"actor.", "motion_prior.", "decoder."}
        for k, v in state_dict.items():
            if k == "quantizer.codebook":
                remapped["quantizer.codebook"] = v
            elif any(k.startswith(p) for p in prefix_allow):
                remapped[k] = v

        missing, unexpected = self._policy.load_state_dict(remapped, strict=False)
        if missing:
            print(f"[PassingVQController] WARNING – missing weights: {missing}")
        if unexpected:
            print(f"[PassingVQController] WARNING – unexpected keys (ignored): {unexpected}")
        self._policy.eval()

        print(f"[PassingVQController] Loaded checkpoint: {policy_path}")
        print(f"  code_dim={code_dim}, num_code={num_code}, lab_lambda={lab_lambda}, use_mp={use_mp}")

        # Source position (world frame): set by main script at each episode reset
        self.source_pos = np.zeros(3, dtype=np.float32)

        # Ball model indices — populated by init_ball()
        self._ball_body_id: int | None = None
        self._ball_qpos_adr: int | None = None
        self._ball_dof_adr: int | None = None

        self._reset_buffers()

        # BaseController required fields
        self.num_actions     = self.NUM_JOINTS
        self.num_obs         = self.OBS_DIM
        self.kps             = kps.copy()
        self.kds             = kds.copy()
        self.action_scale    = action_scale.copy()
        self.default_dof_pos = default_angles.copy()

    # ------------------------------------------------------------------
    # Ball initialisation
    # ------------------------------------------------------------------

    def init_ball(self, model) -> None:
        """Resolve soccer ball body/joint indices from the compiled MuJoCo model."""
        ball_body_id       = model.body("soccer_ball").id
        jnt_id             = model.body_jntadr[ball_body_id]
        self._ball_body_id  = ball_body_id
        self._ball_qpos_adr = model.jnt_qposadr[jnt_id]
        self._ball_dof_adr  = model.jnt_dofadr[jnt_id]
        print(f"[PassingVQController] Ball: body_id={ball_body_id}, "
              f"qpos_adr={self._ball_qpos_adr}, dof_adr={self._ball_dof_adr}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_buffers(self) -> None:
        H = self.HISTORY_LEN
        self._gravity_buf: deque[np.ndarray] = deque(
            [np.zeros(3,               dtype=np.float32) for _ in range(H)], maxlen=H)
        self._ang_vel_buf: deque[np.ndarray] = deque(
            [np.zeros(3,               dtype=np.float32) for _ in range(H)], maxlen=H)
        self._jpos_buf:    deque[np.ndarray] = deque(
            [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(H)], maxlen=H)
        self._jvel_buf:    deque[np.ndarray] = deque(
            [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(H)], maxlen=H)
        self._action_buf:  deque[np.ndarray] = deque(
            [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(H)], maxlen=H)

    def _get_ball_state(self, mujoco_data) -> tuple[np.ndarray, np.ndarray]:
        """Return ball (position, linear_velocity) in world frame."""
        adr = self._ball_qpos_adr
        dof = self._ball_dof_adr
        ball_pos = mujoco_data.qpos[adr:adr + 3].copy().astype(np.float32)
        ball_vel = mujoco_data.qvel[dof:dof + 3].copy().astype(np.float32)
        return ball_pos, ball_vel

    def _build_obs(self, mujoco_data) -> tuple[np.ndarray, np.ndarray]:
        # ---- Current proprioception in IsaacLab joint order ----
        q_wxyz   = mujoco_data.qpos[3:7]            # robot base quaternion [w,x,y,z]
        gravity  = get_gravity_orientation(q_wxyz).astype(np.float32)
        ang_vel  = mujoco_data.qvel[3:6].astype(np.float32)
        jpos_rel = (mujoco_data.qpos[7:36] - default_angles).astype(np.float32)
        jpos_il  = jpos_rel[mujoco_to_isaaclab_reindex]
        jvel_il  = mujoco_data.qvel[6:35].astype(np.float32)[mujoco_to_isaaclab_reindex]

        self._gravity_buf.append(gravity)
        self._ang_vel_buf.append(ang_vel)
        self._jpos_buf.append(jpos_il)
        self._jvel_buf.append(jvel_il)

        hist_gravity  = np.concatenate(list(self._gravity_buf))   # 12
        hist_ang_vel  = np.concatenate(list(self._ang_vel_buf))   # 12
        hist_jpos     = np.concatenate(list(self._jpos_buf))      # 116
        hist_jvel     = np.concatenate(list(self._jvel_buf))      # 116
        hist_actions  = np.concatenate(list(self._action_buf))    # 116

        # ---- Ball observations (current frame only, no history) ----
        ball_pos_w, ball_vel_w = self._get_ball_state(mujoco_data)
        robot_pos = mujoco_data.qpos[0:3].astype(np.float32)

        # ball_relative_position: ball in robot body frame
        rel_w = ball_pos_w - robot_pos
        ball_rel_pos = _quat_apply_inverse(q_wxyz, rel_w).astype(np.float32)  # [3]

        # ball_velocity: world frame linear velocity
        ball_vel = ball_vel_w  # [3]

        # passing_source_position: (x_b, y_b, distance) in robot frame
        src_rel_2d = self.source_pos[:2] - robot_pos[:2]             # [2] world
        distance   = float(np.linalg.norm(src_rel_2d))
        src_rel_3d = np.array([src_rel_2d[0], src_rel_2d[1], 0.0], dtype=np.float32)
        src_rel_b  = _quat_apply_inverse(q_wxyz, src_rel_3d).astype(np.float32)
        passing_src = np.array([src_rel_b[0], src_rel_b[1], distance], dtype=np.float32)  # [3]

        # ---- Assemble ----
        prob_obs = np.concatenate([
            hist_gravity, hist_ang_vel, hist_jpos, hist_jvel, hist_actions,
        ])  # 372

        obs = np.concatenate([
            ball_rel_pos, ball_vel, passing_src,
            hist_gravity, hist_ang_vel, hist_jpos, hist_jvel, hist_actions,
        ])  # 381

        assert obs.shape      == (self.OBS_DIM,),      f"obs shape mismatch: {obs.shape}"
        assert prob_obs.shape == (self.PROB_OBS_DIM,), f"prob_obs shape mismatch: {prob_obs.shape}"
        return obs, prob_obs

    # ------------------------------------------------------------------
    # Public API (BaseController interface)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._reset_buffers()

    def step(self, mujoco_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_np, prob_obs_np = self._build_obs(mujoco_data)

        obs_t      = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        prob_obs_t = torch.from_numpy(prob_obs_np).unsqueeze(0).to(self.device)

        actions_il = self._policy.inference(obs_t, prob_obs_t).squeeze(0).cpu().numpy()

        self._action_buf.append(actions_il.copy())

        target_q = (
            actions_il[isaaclab_to_mujoco_reindex] * self.action_scale
            + self.default_dof_pos
        )
        return target_q, self.kps, self.kds
