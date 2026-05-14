"""Register Unitree G1 football tasks."""

from mjlab.tasks.football.config.g1.env_cfgs import (
  unitree_g1_dribbling_env_cfg,
  unitree_g1_kicking_env_cfg,
  unitree_g1_passing_env_cfg,
  unitree_g1_passing_perception_env_cfg,
)
from mjlab.tasks.football.config.g1.rl_cfg import (
  unitree_g1_dribbling_vq_runner_cfg,
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

# Passing-Perception — same task as Passing but the policy cannot read
# ball state directly. A 176-dim forward LiDAR depth scan replaces
# ``ball_relative_position`` + ``ball_velocity`` in the policy obs.
# Critic keeps the privileged ball state for asymmetric A2C training.
register_mjlab_task(
  task_id="Mjlab-Football-Passing-Perception-Unitree-G1",
  env_cfg=unitree_g1_passing_perception_env_cfg(),
  play_env_cfg=unitree_g1_passing_perception_env_cfg(play=True),
  rl_cfg=unitree_g1_passing_perception_vq_runner_cfg(),
  runner_cls=DownStreamVQOnPolicyRunner,
)
