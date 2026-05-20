"""Register Unitree G1 football tasks."""

from mjlab.tasks.football.config.g1.env_cfgs import (
  unitree_g1_dribbling_env_cfg,
  unitree_g1_dribbling_perception_env_cfg,
  unitree_g1_kicking_env_cfg,
  unitree_g1_kicking_perception_env_cfg,
  unitree_g1_passing_env_cfg,
  unitree_g1_passing_perception_env_cfg,
)
from mjlab.tasks.football.config.g1.rl_cfg import (
  unitree_g1_dribbling_perception_vq_runner_cfg,
  unitree_g1_dribbling_vq_runner_cfg,
  unitree_g1_kicking_perception_vq_runner_cfg,
  unitree_g1_kicking_vq_runner_cfg,
  unitree_g1_passing_perception_vq_runner_cfg,
  unitree_g1_passing_vq_runner_cfg,
)
from mjlab.tasks.motion_prior.rl import DownStreamVQOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

# Dribbling — VQ-flavor downstream task. The frozen VQ motion-prior
# backbone is loaded from ``--agent.motion-prior-ckpt-path``; the soccer
# ball uses linear=0.8 / angular=0.0 damping per the reference cfg.
register_mjlab_task(
  task_id="Mjlab-Football-Dribbling-VQ-Unitree-G1",
  env_cfg=unitree_g1_dribbling_env_cfg(),
  play_env_cfg=unitree_g1_dribbling_env_cfg(play=True),
  rl_cfg=unitree_g1_dribbling_vq_runner_cfg(),
  runner_cls=DownStreamVQOnPolicyRunner,
)

# Kicking — same backbone, different env (kicking layout + reward + termination).
# Ball damping linear=0.4 / angular=0.2.
register_mjlab_task(
  task_id="Mjlab-Football-Kicking-VQ-Unitree-G1",
  env_cfg=unitree_g1_kicking_env_cfg(),
  play_env_cfg=unitree_g1_kicking_env_cfg(play=True),
  rl_cfg=unitree_g1_kicking_vq_runner_cfg(),
  runner_cls=DownStreamVQOnPolicyRunner,
)

# Passing — incoming ball launched toward the robot; redirect back into
# the source zone. Ball damping linear=0.3 / angular=0.2.
register_mjlab_task(
  task_id="Mjlab-Football-Passing-VQ-Unitree-G1",
  env_cfg=unitree_g1_passing_env_cfg(),
  play_env_cfg=unitree_g1_passing_env_cfg(play=True),
  rl_cfg=unitree_g1_passing_vq_runner_cfg(),
  runner_cls=DownStreamVQOnPolicyRunner,
)

# ===========================================================================
# Perception-only variants — policy reads a pelvis depth camera (D405 16:9)
# instead of direct ball state. Critic keeps privileged ball state
# (asymmetric actor-critic). All three share the vision-aware
# DownStreamVQVisionPolicy (CNN depth encoder + MLP).
# ===========================================================================

# Passing-Perception — incoming ball, redirect into source zone, vision-only.
register_mjlab_task(
  task_id="Mjlab-Football-Passing-Perception-Unitree-G1",
  env_cfg=unitree_g1_passing_perception_env_cfg(),
  play_env_cfg=unitree_g1_passing_perception_env_cfg(play=True),
  rl_cfg=unitree_g1_passing_perception_vq_runner_cfg(),
  runner_cls=DownStreamVQOnPolicyRunner,
)

# Kicking-Perception — strike the ball toward the goal, vision-only.
register_mjlab_task(
  task_id="Mjlab-Football-Kicking-Perception-Unitree-G1",
  env_cfg=unitree_g1_kicking_perception_env_cfg(),
  play_env_cfg=unitree_g1_kicking_perception_env_cfg(play=True),
  rl_cfg=unitree_g1_kicking_perception_vq_runner_cfg(),
  runner_cls=DownStreamVQOnPolicyRunner,
)

# Dribbling-Perception — push the ball toward the goal, vision-only.
register_mjlab_task(
  task_id="Mjlab-Football-Dribbling-Perception-Unitree-G1",
  env_cfg=unitree_g1_dribbling_perception_env_cfg(),
  play_env_cfg=unitree_g1_dribbling_perception_env_cfg(play=True),
  rl_cfg=unitree_g1_dribbling_perception_vq_runner_cfg(),
  runner_cls=DownStreamVQOnPolicyRunner,
)
