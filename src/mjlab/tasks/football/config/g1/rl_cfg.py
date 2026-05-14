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
  """Same PPO hyperparams as plain passing — the only diff is the env's
  policy obs (depth image replaces direct ball state). The actor MLP
  width (512/256/128) is plenty for a 176-dim depth input + proprio."""
  return _default_vq_runner_cfg("g1_football_passing_perception_vq")
