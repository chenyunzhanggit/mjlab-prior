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


def _perception_vq_runner_cfg(experiment_name: str) -> RslRlDownstreamVQRunnerCfg:
  """Shared vision-aware :class:`DownStreamVQVisionPolicy` runner cfg.

  Image shape is **auto-detected** from the env's ``depth`` obs group at
  runtime (see ``DownStreamVQOnPolicyRunner._build_policy``); the
  ``image_height`` / ``image_width`` fields on the policy cfg are only
  used as fallback when the env doesn't expose a depth group. We leave
  them at defaults (None) since perception env_cfgs always provide one.

  CNN: 3 strided convs (16→32→32) → Linear → 64-d embedding, mirroring
  the topology mjlab-prior-main / mjlab-loco use for the same input size.
  """
  cfg = _default_vq_runner_cfg(experiment_name)
  cfg.policy.depth_embedding_dim = 64
  cfg.policy.depth_channels = (16, 32, 32)
  return cfg


def unitree_g1_passing_perception_vq_runner_cfg() -> RslRlDownstreamVQRunnerCfg:
  return _perception_vq_runner_cfg("g1_football_passing_perception_vq")


def unitree_g1_kicking_perception_vq_runner_cfg() -> RslRlDownstreamVQRunnerCfg:
  return _perception_vq_runner_cfg("g1_football_kicking_perception_vq")


def unitree_g1_dribbling_perception_vq_runner_cfg() -> RslRlDownstreamVQRunnerCfg:
  return _perception_vq_runner_cfg("g1_football_dribbling_perception_vq")
