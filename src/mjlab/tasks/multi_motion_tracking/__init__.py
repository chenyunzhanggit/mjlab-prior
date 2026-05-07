"""Multi-motion tracking task.

Mirrors :mod:`mjlab.tasks.tracking` but the per-env motion command is a
:class:`~mjlab.tasks.tracking.mdp.multi_commands.MultiMotionCommand` so a
single training run can iterate over a directory of motion clips.

The observation set follows the upstream ``motionprior`` reference's
``TrackingMultiMotionTeacherEnvCfg`` (``tracking_env_cfg_multi_motion_teacher.py``),
adapted to mjlab-prior's available MDP terms:

* Anchor reference signals (``motion_anchor_pos_b`` / ``motion_anchor_ori_b``)
  use the non-``_future`` variants — :class:`MultiMotionCommand` does not
  expose ``motion_anchor_pos`` / ``motion_anchor_quat`` future-step
  buffers (per ``single_motion_2_multi_motion_todo.md``); for the
  reference's default ``future_steps=1`` the shapes coincide.
* Self-collision penalty replaces upstream's ``undesired_contacts`` —
  mjlab-prior's contact sensor framework exposes ``self_collision_cost``
  rather than per-body filter rules.

Reward / termination layouts are otherwise faithful to the reference.
"""
