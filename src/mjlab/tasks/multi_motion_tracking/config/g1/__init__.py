"""Register Unitree G1 multi-motion tracking tasks."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_flat_multi_motion_tracking_env_cfg
from .rl_cfg import unitree_g1_multi_motion_tracking_ppo_runner_cfg

# Multi-motion tracking on flat terrain. ``MjlabOnPolicyRunner`` is used
# instead of ``MotionTrackingOnPolicyRunner`` because the latter's ONNX
# export bundles the entire motion clip as ONNX buffers, which doesn't
# generalize to a clip-pool: the caller picks the clip per env at runtime.
register_mjlab_task(
  task_id="Mjlab-MultiMotionTracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_multi_motion_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_multi_motion_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_multi_motion_tracking_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
