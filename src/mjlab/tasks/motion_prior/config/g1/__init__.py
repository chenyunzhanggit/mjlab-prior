from mjlab.tasks.motion_prior.rl import (
  MotionPriorOnPolicyRunner,
  MotionPriorVQOnPolicyRunner,
)
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_flat_motion_prior_env_cfg,
  unitree_g1_flat_motion_prior_vq_env_cfg,
  unitree_g1_rough_motion_prior_env_cfg,
)
from .rl_cfg import (
  unitree_g1_motion_prior_runner_cfg,
  unitree_g1_motion_prior_vq_runner_cfg,
)

# teacher_a (Teleopit TemporalCNN) on flat terrain.
register_mjlab_task(
  task_id="Mjlab-MotionPrior-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_motion_prior_env_cfg(),
  play_env_cfg=unitree_g1_flat_motion_prior_env_cfg(play=True),
  rl_cfg=unitree_g1_motion_prior_runner_cfg(),
  runner_cls=MotionPriorOnPolicyRunner,
)

# teacher_b (mjlab Velocity-Rough MLP) on rough terrain.
register_mjlab_task(
  task_id="Mjlab-MotionPrior-Rough-Unitree-G1",
  env_cfg=unitree_g1_rough_motion_prior_env_cfg(),
  play_env_cfg=unitree_g1_rough_motion_prior_env_cfg(play=True),
  rl_cfg=unitree_g1_motion_prior_runner_cfg(),
  runner_cls=MotionPriorOnPolicyRunner,
)

# VQ variant (same flat env, different algorithm).
register_mjlab_task(
  task_id="Mjlab-MotionPrior-VQ-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_motion_prior_vq_env_cfg(),
  play_env_cfg=unitree_g1_flat_motion_prior_vq_env_cfg(play=True),
  rl_cfg=unitree_g1_motion_prior_vq_runner_cfg(),
  runner_cls=MotionPriorVQOnPolicyRunner,
)
