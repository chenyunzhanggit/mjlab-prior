"""Frozen teacher utilities for motion prior distillation.

Two teacher families are supported:

* ``load_teleopit_teacher`` — Teleopit ``track.pt`` (TemporalCNN, 1-D + 3-D).
* ``load_velocity_teacher`` — mjlab Velocity-Rough actor (plain MLP, 1-D).
"""

from mjlab.tasks.motion_prior.teacher.conv1d_encoder import Conv1dEncoder
from mjlab.tasks.motion_prior.teacher.downstream_ckpt_loader import (
  load_motion_prior_components,
  load_motion_prior_vq_components,
)
from mjlab.tasks.motion_prior.teacher.loader import (
  TELEOPIT_TEACHER_CFG,
  TeleopitTeacherCfg,
  build_teleopit_teacher,
  load_teleopit_teacher,
  make_dummy_obs,
)
from mjlab.tasks.motion_prior.teacher.temporal_cnn_model import TemporalCNNModel
from mjlab.tasks.motion_prior.teacher.velocity_loader import (
  VELOCITY_TEACHER_CFG,
  VelocityTeacherCfg,
  build_velocity_teacher,
  load_velocity_teacher,
)

__all__ = [
  "Conv1dEncoder",
  "TemporalCNNModel",
  "TELEOPIT_TEACHER_CFG",
  "TeleopitTeacherCfg",
  "build_teleopit_teacher",
  "load_teleopit_teacher",
  "load_motion_prior_components",
  "load_motion_prior_vq_components",
  "make_dummy_obs",
  "VELOCITY_TEACHER_CFG",
  "VelocityTeacherCfg",
  "build_velocity_teacher",
  "load_velocity_teacher",
]
