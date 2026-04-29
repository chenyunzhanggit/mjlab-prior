"""RL configuration for Unitree G1 motion-prior distillation tasks."""

from mjlab.tasks.motion_prior.rl_cfg import (
  RslRlMotionPriorAlgoCfg,
  RslRlMotionPriorRunnerCfg,
)


def unitree_g1_motion_prior_runner_cfg() -> RslRlMotionPriorRunnerCfg:
  """Create RL runner configuration for G1 VAE motion-prior distillation."""
  return RslRlMotionPriorRunnerCfg(
    experiment_name="g1_motion_prior",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def unitree_g1_motion_prior_vq_runner_cfg() -> RslRlMotionPriorRunnerCfg:
  """Create RL runner configuration for G1 VQ-VAE motion-prior distillation.

  Currently shares the VAE schedule; VQ-specific knobs (codebook size,
  commitment loss coeff) land in prior.md task #12 along with the VQ
  policy / algorithm.
  """
  return RslRlMotionPriorRunnerCfg(
    experiment_name="g1_motion_prior_vq",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
    algorithm=RslRlMotionPriorAlgoCfg(
      # Placeholder until VQ algorithm lands; keeps shape-compat for now.
    ),
  )
