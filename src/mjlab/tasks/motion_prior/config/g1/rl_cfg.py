"""RL configuration for Unitree G1 motion-prior distillation tasks."""

from mjlab.tasks.motion_prior.rl_cfg import (
  RslRlMotionPriorRunnerCfg,
  RslRlMotionPriorVQRunnerCfg,
)


def unitree_g1_motion_prior_runner_cfg() -> RslRlMotionPriorRunnerCfg:
  """Create RL runner configuration for G1 VAE motion-prior distillation."""
  return RslRlMotionPriorRunnerCfg(
    experiment_name="g1_motion_prior",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def unitree_g1_motion_prior_vq_runner_cfg() -> RslRlMotionPriorVQRunnerCfg:
  """Create RL runner configuration for G1 VQ-VAE motion-prior distillation."""
  return RslRlMotionPriorVQRunnerCfg(
    experiment_name="g1_motion_prior_vq",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )
