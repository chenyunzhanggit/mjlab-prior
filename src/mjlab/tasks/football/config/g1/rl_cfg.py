"""Runner configs for the three G1 football tasks.

All three share the ``DownStreamVQOnPolicyRunner`` (VQ flavor of the
downstream PPO loop). ``motion_prior_ckpt_path`` is left empty here and
must be set via ``--agent.motion-prior-ckpt-path`` on the train CLI so a
specific VQ motion-prior checkpoint can be selected per run.
"""

from __future__ import annotations

from mjlab.tasks.motion_prior.downstream_rl_cfg import (
  RslRlDownstreamVQPolicyCfg,
  RslRlDownstreamVQRunnerCfg,
)


def _default_vq_runner_cfg(experiment_name: str) -> RslRlDownstreamVQRunnerCfg:
  """Shared PPO hyperparams across the three football tasks."""
  return RslRlDownstreamVQRunnerCfg(
    experiment_name=experiment_name,
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
    policy=RslRlDownstreamVQPolicyCfg(
      init_noise_std=1.0,
      actor_hidden_dims=(512, 256, 128),
      decoder_hidden_dims=(512, 256, 128),
      motion_prior_hidden_dims=(512, 256, 128),
      num_code=2048,
      code_dim=64,
      activation="elu",
    ),
  )


def unitree_g1_dribbling_vq_runner_cfg() -> RslRlDownstreamVQRunnerCfg:
  return _default_vq_runner_cfg("g1_football_dribbling_vq")


def unitree_g1_kicking_vq_runner_cfg() -> RslRlDownstreamVQRunnerCfg:
  return _default_vq_runner_cfg("g1_football_kicking_vq")


def unitree_g1_passing_vq_runner_cfg() -> RslRlDownstreamVQRunnerCfg:
  return _default_vq_runner_cfg("g1_football_passing_vq")


def unitree_g1_passing_perception_vq_runner_cfg() -> RslRlDownstreamVQRunnerCfg:
  """Same PPO hyperparams as plain passing, but the policy is the
  vision-aware :class:`DownStreamVQVisionPolicy`: the depth-image
  suffix of policy_obs is routed through a small CNN encoder before
  being concatenated with proprio+cmd and fed to an MLP.

  CNN config follows VisualMimic-style recipes for low-resolution
  depth inputs: 3 strided convs (16→32→32), then Linear projection to
  a 64-dim embedding. The MLP head reuses the standard
  ``[512, 256, 128]`` widths used by the all-MLP downstream baseline.

  ``image_height`` / ``image_width`` toggle the runner into vision mode
  (see ``DownStreamVQOnPolicyRunner._build_policy``). Keeping them None
  would fall back to the all-MLP actor (broken on this task: it'd try
  to feed 3600 raw depth pixels through a dense Linear layer).
  """
  from mjlab.tasks.football.config.g1.env_cfgs import (
    PERCEPTION_IMAGE_HEIGHT,
    PERCEPTION_IMAGE_WIDTH,
  )

  cfg = _default_vq_runner_cfg("g1_football_passing_perception_vq")
  cfg.policy.image_height = PERCEPTION_IMAGE_HEIGHT
  cfg.policy.image_width = PERCEPTION_IMAGE_WIDTH
  cfg.policy.depth_embedding_dim = 64
  cfg.policy.depth_channels = (16, 32, 32)
  return cfg
