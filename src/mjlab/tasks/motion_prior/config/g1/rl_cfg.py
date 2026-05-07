"""RL configuration for Unitree G1 motion-prior distillation tasks."""

from mjlab.tasks.motion_prior.rl_cfg import (
  RslRlMotionPriorRunnerCfg,
  RslRlMotionPriorSingleRunnerCfg,
  RslRlMotionPriorSingleVQRunnerCfg,
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


def unitree_g1_motion_prior_single_runner_cfg() -> RslRlMotionPriorSingleRunnerCfg:
  """Single-encoder VAE motion-prior runner cfg.

  ``teacher_policy_path`` defaults to empty — the user must supply the
  multi-motion tracking actor checkpoint via
  ``--agent.teacher-policy-path`` on the CLI.
  """
  return RslRlMotionPriorSingleRunnerCfg(
    experiment_name="g1_motion_prior_single",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def unitree_g1_motion_prior_single_vq_runner_cfg() -> RslRlMotionPriorSingleVQRunnerCfg:
  """Single-encoder VQ motion-prior runner cfg."""
  return RslRlMotionPriorSingleVQRunnerCfg(
    experiment_name="g1_motion_prior_single_vq",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )
