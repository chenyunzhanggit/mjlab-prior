"""Register Unitree G1 multi-motion tracking tasks."""

from mjlab.tasks.multi_motion_tracking.rl import MultiMotionTrackingOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_flat_multi_motion_tracking_env_cfg
from .rl_cfg import unitree_g1_multi_motion_tracking_ppo_runner_cfg

# Multi-motion tracking on flat terrain. ``MultiMotionTrackingOnPolicyRunner``
# is a thin subclass of :class:`MjlabOnPolicyRunner` that accepts the
# ``registry_name`` kwarg ``train.py`` forwards to tracking tasks; we use
# it instead of the single-motion :class:`MotionTrackingOnPolicyRunner`
# because the latter's ONNX export bundles the entire motion clip into
# the graph, which doesn't generalize to a clip-pool (the caller picks
# the clip per env at runtime).
register_mjlab_task(
  task_id="Mjlab-MultiMotionTracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_multi_motion_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_multi_motion_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_multi_motion_tracking_ppo_runner_cfg(),
  runner_cls=MultiMotionTrackingOnPolicyRunner,
)
