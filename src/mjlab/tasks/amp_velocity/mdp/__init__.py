"""MDP terms specific to the AMP velocity task."""

from mjlab.tasks.amp_velocity.mdp.amp_observations import (
  robot_body_ang_vel_b,
  robot_body_lin_vel_b,
  robot_body_ori_b,
  robot_body_pos_b,
)

__all__ = (
  "robot_body_ang_vel_b",
  "robot_body_lin_vel_b",
  "robot_body_ori_b",
  "robot_body_pos_b",
)
