"""Football downstream tasks (dribbling / kicking / passing).

These build on a frozen motion_prior backbone (VQ flavor) using
``DownStreamVQOnPolicyRunner``, but the env itself is the soccer scene
(plane + soccer ball + goal command). One shared ``SoccerBallEntityCfg``
backs all three tasks; per-task parameters (ball damping, init layout,
goal command, rewards, terminations) live in ``config/g1/env_cfgs.py``.

Port notes vs ``motionprior/source/.../envs/unitree_g1/g1_{dribbling,kicking,passing}_vq_cfg.py``:

* Reference is IsaacLab + USD. We use mjlab + MjSpec. The soccer ball
  is built as a single body with 3 prismatic + 3 revolute joints (instead
  of one free joint) so per-axis linear/angular damping can be set
  independently — matches PhysX ``RigidBodyPropertiesCfg.linear_damping /
  angular_damping`` semantics 1:1.
* Terrain restitution defaults to 0 (MuJoCo) — reference's 0.5~0.8 bounce
  is dropped per migration plan. Effect: dribbling rolls less freely;
  this is a known intentional simplification.
* ``_reset_scene_along_line`` is implemented as a ``mode="reset"``
  EventTerm rather than overriding the env's reset method, so we don't
  fork ``ManagerBasedRlEnv``.
* mid-episode motion resample re-placement (reference: triggers when
  motion command rolls over within an episode) is intentionally NOT
  carried — downstream tasks here run with ``init_from_motion=False``
  and motion never resamples mid-episode.
"""
