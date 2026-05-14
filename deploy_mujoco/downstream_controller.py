"""
Downstream VQ Policy controller for MuJoCo sim2sim deployment.

Observation layout (from G1DownstreamVQEnvCfg):
  obs      (actor input,  375): vel_cmd(3) | gravity(3×4) | ang_vel(3×4) | jpos(29×4) | jvel(29×4) | actions(29×4)
  prob_obs (motion_prior, 372):             | gravity(3×4) | ang_vel(3×4) | jpos(29×4) | jvel(29×4) | actions(29×4)

History stacking is oldest-to-newest (IsaacLab convention, verified from ObservationManager docs).

Policy inference (VQ version with LAB):
  raw  = actor(obs)                              # [code_dim]
  mp   = motion_prior(prob_obs)                  # [code_dim]
  z    = mp + lab_lambda * tanh(raw)             # LAB constraint
  q_z  = quantizer.nearest_lookup(z)            # [code_dim]
  act  = decoder(cat([prob_obs, q_z]))           # [29]
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


# ---------------------------------------------------------------------------
# Lightweight network building blocks
# ---------------------------------------------------------------------------

class _QuantizerInference(nn.Module):
    """Nearest-codebook lookup; no EMA updates needed at deploy time."""

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
        code_idx = dist.argmin(dim=-1)
        return F.embedding(code_idx, self.codebook)


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), nn.ELU()]
    for i in range(len(hidden_dims) - 1):
        layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ELU()]
    layers.append(nn.Linear(hidden_dims[-1], out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Standalone inference policy
# ---------------------------------------------------------------------------

class _DownstreamVQPolicyInference(nn.Module):
    """
    Deployment-only reconstruction of DownStreamVQPolicy.
    Loads: actor, motion_prior, quantizer.codebook, decoder.
    """

    PROB_OBS_DIM = 372

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

        self.actor        = _build_mlp(num_obs,           hidden_dims, code_dim)
        self.motion_prior = _build_mlp(prob_dim,          hidden_dims, code_dim)
        self.quantizer    = _QuantizerInference(num_code, code_dim)
        self.decoder      = _build_mlp(prob_dim + code_dim, hidden_dims, num_actions)

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

class DownstreamVQController(BaseController):
    """
    Downstream VQ policy controller for MuJoCo sim2sim.

    Args:
        policy_path:       Path to the downstream .pt checkpoint.
        device:            "cpu" or "cuda:N".
        velocity_commands: [vx, vy, omega_z] command (default all-zeros; model was
                           typically trained with zero velocity commands).
        code_dim:          VQ codebook dimension (default 64, read from checkpoint).
        num_code:          VQ codebook size (default 2048, read from checkpoint).
        hidden_dims:       MLP hidden dims [512, 256, 128] to match training config.
        lab_lambda:        LAB scale factor (default 3.0).
        use_mp:            Whether the motion prior branch is active (default True).
    """

    HISTORY_LEN  = 4
    NUM_JOINTS   = 29
    OBS_DIM      = 375
    PROB_OBS_DIM = 372

    def __init__(
        self,
        policy_path: str,
        device: str = "cpu",
        velocity_commands: np.ndarray | None = None,
        code_dim: int = 64,
        num_code: int = 2048,
        hidden_dims: list[int] | None = None,
        lab_lambda: float = 3.0,
        use_mp: bool = True,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.device = torch.device(device)

        # Build policy network
        self._policy = _DownstreamVQPolicyInference(
            num_obs=self.OBS_DIM,
            num_actions=self.NUM_JOINTS,
            code_dim=code_dim,
            num_code=num_code,
            hidden_dims=hidden_dims,
            lab_lambda=lab_lambda,
            use_mp=use_mp,
        ).to(self.device)

        # Load weights
        ckpt       = torch.load(policy_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Map original QuantizeEMAReset key → our _QuantizerInference key
        remapped: dict[str, torch.Tensor] = {}
        prefix_allow = {"actor.", "motion_prior.", "decoder."}
        for k, v in state_dict.items():
            if k == "quantizer.codebook":
                remapped["quantizer.codebook"] = v
            elif any(k.startswith(p) for p in prefix_allow):
                remapped[k] = v

        missing, unexpected = self._policy.load_state_dict(remapped, strict=False)
        if missing:
            print(f"[DownstreamVQController] WARNING – missing weights: {missing}")
        if unexpected:
            print(f"[DownstreamVQController] WARNING – unexpected keys (ignored): {unexpected}")
        self._policy.eval()

        print(f"[DownstreamVQController] Loaded checkpoint: {policy_path}")
        print(f"  code_dim={code_dim}, num_code={num_code}, lab_lambda={lab_lambda}, use_mp={use_mp}")

        # Velocity command [vx, vy, omega_z]
        self.velocity_commands = (
            np.asarray(velocity_commands, dtype=np.float32)
            if velocity_commands is not None
            else np.zeros(3, dtype=np.float32)
        )

        # Observation history buffers (deque: left=oldest, right=newest)
        self._reset_buffers()

        # BaseController required fields (set without calling super().__init__()
        # because the parent validator runs before subclass attrs are set)
        self.num_actions     = self.NUM_JOINTS
        self.num_obs         = self.OBS_DIM
        self.kps             = kps.copy()
        self.kds             = kds.copy()
        self.action_scale    = action_scale.copy()
        self.default_dof_pos = default_angles.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_buffers(self) -> None:
        H = self.HISTORY_LEN
        self._gravity_buf: deque[np.ndarray] = deque(
            [np.zeros(3,              dtype=np.float32) for _ in range(H)], maxlen=H)
        self._ang_vel_buf: deque[np.ndarray] = deque(
            [np.zeros(3,              dtype=np.float32) for _ in range(H)], maxlen=H)
        self._jpos_buf:    deque[np.ndarray] = deque(
            [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(H)], maxlen=H)
        self._jvel_buf:    deque[np.ndarray] = deque(
            [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(H)], maxlen=H)
        self._action_buf:  deque[np.ndarray] = deque(
            [np.zeros(self.NUM_JOINTS, dtype=np.float32) for _ in range(H)], maxlen=H)

    def _build_obs(self, mujoco_data) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute obs and prob_obs from current MuJoCo state.

        Timing:
          - state buffers (gravity/ang_vel/jpos/jvel) are updated with current values
          - action_buf still holds the PREVIOUS step's action, i.e. the "last_action"
            seen by the environment when computing this step's observation
        """
        # Current-step proprioceptive features in IsaacLab joint order
        gravity  = get_gravity_orientation(mujoco_data.qpos[3:7]).astype(np.float32)
        ang_vel  = mujoco_data.qvel[3:6].astype(np.float32)
        jpos_rel = (mujoco_data.qpos[7:36] - default_angles).astype(np.float32)
        jpos_il  = jpos_rel[mujoco_to_isaaclab_reindex]
        jvel_il  = mujoco_data.qvel[6:35].astype(np.float32)[mujoco_to_isaaclab_reindex]

        # Push current values into rolling buffers (action_buf pushed AFTER inference)
        self._gravity_buf.append(gravity)
        self._ang_vel_buf.append(ang_vel)
        self._jpos_buf.append(jpos_il)
        self._jvel_buf.append(jvel_il)

        # Flatten histories oldest→newest (IsaacLab convention)
        hist_gravity  = np.concatenate(list(self._gravity_buf))   # 12
        hist_ang_vel  = np.concatenate(list(self._ang_vel_buf))   # 12
        hist_jpos     = np.concatenate(list(self._jpos_buf))      # 116
        hist_jvel     = np.concatenate(list(self._jvel_buf))      # 116
        hist_actions  = np.concatenate(list(self._action_buf))    # 116  (last action)

        # prob_obs: 372 (motion_prior input)
        prob_obs = np.concatenate([
            hist_gravity, hist_ang_vel, hist_jpos, hist_jvel, hist_actions
        ])

        # obs: 375 (actor input = vel_cmd + prob_obs)
        obs = np.concatenate([self.velocity_commands, prob_obs])

        assert obs.shape == (self.OBS_DIM,), f"obs shape mismatch: {obs.shape}"
        assert prob_obs.shape == (self.PROB_OBS_DIM,), f"prob_obs shape mismatch: {prob_obs.shape}"

        return obs, prob_obs

    # ------------------------------------------------------------------
    # Public API (BaseController interface)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._reset_buffers()

    def step(self, mujoco_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one policy step.

        Args:
            mujoco_data: mujoco.MjData from the current simulation state.

        Returns:
            target_q:  target joint positions in MuJoCo order  [29]
            kps:       joint stiffness gains                   [29]
            kds:       joint damping gains                     [29]
        """
        obs_np, prob_obs_np = self._build_obs(mujoco_data)

        obs_t      = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        prob_obs_t = torch.from_numpy(prob_obs_np).unsqueeze(0).to(self.device)

        # Inference: returns [1, 29] in IsaacLab joint order
        actions_il = self._policy.inference(obs_t, prob_obs_t).squeeze(0).cpu().numpy()

        # Store raw action (IsaacLab order) for next obs's "last_action"
        self._action_buf.append(actions_il.copy())

        # Convert to MuJoCo target joint positions
        target_q = (
            actions_il[isaaclab_to_mujoco_reindex] * self.action_scale
            + self.default_dof_pos
        )

        return target_q, self.kps, self.kds
