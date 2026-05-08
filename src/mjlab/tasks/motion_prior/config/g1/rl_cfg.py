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
  """Create RL runner configuration for G1 single-encoder motion-prior distillation.

  Single trackingbfm teacher → single encoder. The default
  ``teacher_policy_path`` is a placeholder; in practice the user supplies it
  via CLI ``--agent.teacher-policy-path /path/to/trackingbfm_model.pt``.
  """
  return RslRlMotionPriorSingleRunnerCfg(
    experiment_name="g1_motion_prior_single",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )


def unitree_g1_motion_prior_single_vq_runner_cfg() -> RslRlMotionPriorSingleVQRunnerCfg:
  """Create RL runner configuration for G1 single-encoder VQ-VAE motion-prior distillation."""
  return RslRlMotionPriorSingleVQRunnerCfg(
    experiment_name="g1_motion_prior_single_vq",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=100_000,
  )
