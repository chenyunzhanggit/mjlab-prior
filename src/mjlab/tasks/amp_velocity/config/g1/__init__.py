"""Unitree G1 AMP velocity task registration.

Registers ``Mjlab-AMP-Velocity-Rough-Unitree-G1`` with the mjlab task registry.
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

register_mjlab_task(
  task_id="Mjlab-AMP-Velocity-Rough-Unitree-G1",
  env_cfg=g1_amp_velocity_rough_env_cfg(),
  play_env_cfg=g1_amp_velocity_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_amp_ppo_runner_cfg(),
  runner_cls=AmpVelocityOnPolicyRunner,
)
