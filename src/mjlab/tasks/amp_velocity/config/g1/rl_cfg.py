"""Typed runner / algorithm config for AMP-augmented G1 velocity training.

Layers AMP-specific fields on top of the standard G1 velocity PPO config:
  - everything in ``RslRlOnPolicyRunnerCfg`` (actor / critic / algorithm) is
    inherited from ``unitree_g1_ppo_runner_cfg`` so the locomotion side
    matches the baseline.
  - ``amp`` (``RslRlAmpCfg``) controls the discriminator, motion-data path,
    reward lerp, and replay buffer.

The ``AmpVelocityOnPolicyRunner`` reads these via ``dataclasses.asdict`` (the
mjlab train script always converts agent cfg to a plain dict before
constructing the runner), so field names here must match the keys the runner
expects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from mjlab.rl.config import RslRlOnPolicyRunnerCfg
from mjlab.tasks.amp_velocity.config.g1.env_cfgs import (
  AMP_ANCHOR_BODY,
  AMP_G1_ALL_JOINTS,
  AMP_PELVIS_BODY,
  AMP_TRACKED_BODIES,
)
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg


def _default_motion_dir() -> str:
  """Resolve the default expert motion directory.

  Priority:
    1. ``MJLAB_AMP_MOTION_DIR`` env var (escape hatch for CI / custom layout).
    2. ``$repo_root/src/mjlab/motions/g1/amp/WalkandRun`` (user-supplied data).
  """
  override = os.environ.get("MJLAB_AMP_MOTION_DIR")
  if override:
    return override
  # rl_cfg.py is at .../src/mjlab/tasks/amp_velocity/config/g1/rl_cfg.py
  here = os.path.dirname(os.path.abspath(__file__))
  # [NOTE] change AMP path
  return os.path.abspath(
    os.path.join(here, "..", "..", "..", "..", "motions", "g1", "amp", "WalkandRun")
  )


@dataclass
class RslRlAmpCfg:
  """AMP discriminator / reward / replay knobs.

  Two variants are supported, selected by ``variant``:

  - ``"body"`` (default): AMP_mjlab-style (s, s') discriminator over body-space
    features. ``tracked_bodies`` / ``anchor_body`` apply.
  - ``"joint"``: telebotM2-style K-frame stack discriminator over joint-space
    features. ``amp_joints`` / ``pelvis_body`` / ``num_frames`` apply.

  When changing ``variant`` here, also pass the same value to
  ``g1_amp_velocity_rough_env_cfg(amp_variant=...)``; otherwise the env's
  ``amp`` obs group will disagree with the loader / discriminator built in the
  runner and the dim assert will fire.
  """

  variant: str = "body"
  """Selects which AMP feature / discriminator variant to use ('body' or 'joint')."""

  motion_dir: str = field(default_factory=_default_motion_dir)
  """Path to a directory of expert motion ``.npz`` files (or a single file)."""

  # ---- body variant ---------------------------------------------------- #
  tracked_bodies: tuple[str, ...] = AMP_TRACKED_BODIES
  """Bodies included in the AMP feature vector (body variant only). Must match
  the env's ``amp`` obs group exactly — order is part of the feature definition."""

  anchor_body: str = AMP_ANCHOR_BODY
  """Body whose frame the AMP features are expressed in (body variant only)."""

  # ---- joint variant --------------------------------------------------- #
  amp_joints: tuple[str, ...] = AMP_G1_ALL_JOINTS
  """Joints included in the AMP feature vector (joint variant only). Order
  must match the env's ``amp`` obs group exactly."""

  pelvis_body: str = AMP_PELVIS_BODY
  """Body whose world-frame velocity gives base lin/ang vel (joint variant only)."""

  num_frames: int = 5
  """K — number of consecutive frames stacked per discriminator sample (joint
  variant only). Matches telebotM2's ``amp_num_frames=5``."""

  # ---- shared ---------------------------------------------------------- #
  discriminator_hidden: tuple[int, ...] = (1024, 512, 256)
  """Hidden layer sizes of the discriminator trunk."""

  reward_coef: float = 0.5
  """Scalar multiplier on the AMP-derived reward before lerp."""

  task_reward_lerp: float = 0.6
  """Blend factor between AMP reward (0.0) and env task reward (1.0).

  0.0 → reward = amp_reward (pure style imitation).
  Recommend 0.3–0.5 to balance task and style.
  """

  grad_pen_lambda: float = 10.0
  """Coefficient on the discriminator gradient penalty (LSGAN-style)."""

  replay_buffer_size: int = 1_000_000
  """Capacity of the policy-side AMP (s, s') ring buffer."""

  discriminator_lr: float = 1.0e-4
  """Learning rate for the discriminator's dedicated optimizer.

  Kept separate from the PPO optimizer (which trains actor + critic);
  this lets us tune them independently and avoids weight-decay leakage.
  """

  discriminator_weight_decay: float = 1.0e-3
  """Weight decay on the discriminator *trunk* parameters.

  Matches AMP_mjlab's ``amp_trunk`` group (1e-3). The previous unified value
  (1e-2) was 10× too large for the trunk and 10× too small for the head; the
  resulting disc saturated within tens of iterations (logits pinned at ±1).
  """

  discriminator_head_weight_decay: float = 1.0e-1
  """Weight decay on the discriminator *output head* (``amp_linear``).

  AMP_mjlab uses 1e-1 here — 100× larger than the trunk. This is the key
  trick that keeps the LSGAN logits from running away to ±∞: it bounds the
  norm of the final linear layer so ``d_expert`` / ``d_policy`` stay near
  ±0.3-0.5 rather than collapsing to ±1 (which kills the AMP reward).
  """


@dataclass
class RslRlAmpRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner cfg for the AMP-augmented velocity task.

  Inherits actor / critic / algorithm / save / log fields from
  ``RslRlOnPolicyRunnerCfg`` so the locomotion PPO setup is identical to
  the baseline. The ``amp`` block adds AMP-specific knobs.
  """

  class_name: str = "AmpVelocityOnPolicyRunner"
  amp: RslRlAmpCfg = field(default_factory=RslRlAmpCfg)


def unitree_g1_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Build the G1 AMP-velocity runner cfg by extending the velocity baseline.

  Mirrors ``unitree_g1_ppo_runner_cfg`` but adds:
    - a ``std_range=(0.05, 1.0)`` lower bound on the actor's Gaussian std,
      matching AMP_mjlab's ``min_normalized_std=[0.05]*29``. Without this,
      the policy noise can collapse under the GAN-style AMP loss and the
      discriminator pins the policy in a degenerate local optimum.
  """
  base = unitree_g1_ppo_runner_cfg()
  # The baseline cfg leaves distribution_cfg as a dict; augment it with an
  # explicit std lower bound. Copy first so we don't mutate the singleton.
  actor_cfg = base.actor
  if actor_cfg.distribution_cfg is not None:
    new_dist_cfg = dict(actor_cfg.distribution_cfg)
    new_dist_cfg.setdefault("std_range", (0.05, 1.0))
    actor_cfg.distribution_cfg = new_dist_cfg
  # Lift every field of the baseline into an RslRlAmpRunnerCfg so the AMP
  # task tracks any future tuning of the locomotion side automatically.
  return RslRlAmpRunnerCfg(
    seed=base.seed,
    num_steps_per_env=base.num_steps_per_env,
    max_iterations=base.max_iterations,
    obs_groups=base.obs_groups,
    save_interval=base.save_interval,
    experiment_name="g1_amp_velocity",
    run_name=base.run_name,
    logger=base.logger,
    wandb_project=base.wandb_project,
    wandb_tags=base.wandb_tags,
    resume=base.resume,
    load_run=base.load_run,
    load_checkpoint=base.load_checkpoint,
    clip_actions=base.clip_actions,
    upload_model=base.upload_model,
    actor=actor_cfg,
    critic=base.critic,
    algorithm=base.algorithm,
  )
