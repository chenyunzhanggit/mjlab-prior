"""Unitree G1 AMP velocity task registration.

Registers two task IDs with the mjlab task registry:

  - ``Mjlab-AMP-Velocity-Rough-Unitree-G1``: default reset (uniform root + zero
    joint offset). Robot spawns in the standing pose every episode.
  - ``Mjlab-AMP-Velocity-Rough-Unitree-G1-RSI``: Reference State Initialization
    reset. Robot spawns at a random expert motion frame every episode.

The two share everything else (rewards, terminations, AMP feature definition,
PPO + discriminator config). Switching task ID is a clean A/B experiment.

The custom runner is wired here so ``scripts/train.py`` picks it up via
``load_runner_cls``.
"""

from mjlab.tasks.amp_velocity.config.g1.env_cfgs import g1_amp_velocity_rough_env_cfg
from mjlab.tasks.amp_velocity.config.g1.rl_cfg import unitree_g1_amp_ppo_runner_cfg

# Runner class is imported lazily inside register_mjlab_task's consumer
# (scripts/train.py) so we only pull torch / heavy deps at training time.
# But registry needs the class object now — keep this import here.
from mjlab.tasks.amp_velocity.rl.runner import AmpVelocityOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

# Default reset: standing pose + small uniform perturbation.
register_mjlab_task(
  task_id="Mjlab-AMP-Velocity-Rough-Unitree-G1",
  env_cfg=g1_amp_velocity_rough_env_cfg(use_rsi=False),
  play_env_cfg=g1_amp_velocity_rough_env_cfg(play=True, use_rsi=False),
  rl_cfg=unitree_g1_amp_ppo_runner_cfg(),
  runner_cls=AmpVelocityOnPolicyRunner,
)

# RSI reset: every reset, sample a random frame from the expert motion clips.
register_mjlab_task(
  task_id="Mjlab-AMP-Velocity-Rough-Unitree-G1-RSI",
  env_cfg=g1_amp_velocity_rough_env_cfg(use_rsi=True),
  play_env_cfg=g1_amp_velocity_rough_env_cfg(play=True, use_rsi=True),
  rl_cfg=unitree_g1_amp_ppo_runner_cfg(),
  runner_cls=AmpVelocityOnPolicyRunner,
)
