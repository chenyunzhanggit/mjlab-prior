"""Soccer ball entity (free-floating sphere with per-axis damping).

mjlab/MuJoCo equivalent of the reference's
``RigidObjectCfg(spawn=UsdFileCfg(rigid_props=RigidBodyPropertiesCfg(
linear_damping=..., angular_damping=...)))``.

A MuJoCo free joint has a single ``damping`` scalar that applies to all
6 DOFs equally. To match PhysX's separate linear / angular damping we
build the ball as a single body with **3 slide + 3 hinge joints** in
world axes; each joint owns its own damping. The body's pose is then
written via ``write_joint_state_to_sim`` instead of
``write_root_state_to_sim``.

Joint layout (DOF order in qpos / qvel):

  0: slide_x (axis = world x, damping = linear_damping)
  1: slide_y (axis = world y, damping = linear_damping)
  2: slide_z (axis = world z, damping = linear_damping)
  3: hinge_x (axis = body  x, damping = angular_damping)
  4: hinge_y (axis = body  y, damping = angular_damping)
  5: hinge_z (axis = body  z, damping = angular_damping)

Slides are evaluated in the (still-aligned) parent frame, so qpos[0:3]
== world position. Hinges chain in body frame (XYZ intrinsic), which
gives Euler-XYZ orientation; for a sphere the orientation rarely
matters for rewards/terminations (we use root_link_pos_w + linear
velocity exclusively in the reward path), so gimbal-lock at the poles
is acceptable.

Ground contact behaviour is unaffected — collisions are geom-vs-geom in
MuJoCo regardless of how the kinematic tree is built. ``mass`` is set on
the geom (FIFA spec: ~0.45 kg, radius 0.11 m).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from mjlab.entity import EntityCfg

SOCCER_BALL_RADIUS: float = 0.11
"""Regulation soccer ball radius in metres (FIFA: circumference 68-70 cm)."""

SOCCER_BALL_MASS: float = 0.45
"""Regulation soccer ball mass in kg (FIFA: 410-450 g)."""

_DEFAULT_RGBA: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SoccerBallParams:
  """Per-task ball physics overrides.

  Reference values (from motionprior reference cfgs):
    dribbling: linear=0.8 / angular=0.0
    kicking:   linear=0.4 / angular=0.2
    passing:   linear=0.3 / angular=0.2
  """

  linear_damping: float
  angular_damping: float
  radius: float = SOCCER_BALL_RADIUS
  mass: float = SOCCER_BALL_MASS
  rgba: tuple[float, float, float, float] = _DEFAULT_RGBA


def make_soccer_ball_spec(params: SoccerBallParams):
  """Build an ``MjSpec`` for one soccer ball.

  Called by ``EntityCfg.build()`` during ``Scene._add_entities``; returns
  a fresh spec each call so the ``Scene.attach`` machinery can prefix
  body / joint / geom names per env without aliasing.
  """

  def _spec_fn() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball")

    # 3 prismatic joints (world-aligned at rest) for linear DOFs.
    body.add_joint(
      name="slide_x",
      type=mujoco.mjtJoint.mjJNT_SLIDE,
      axis=(1.0, 0.0, 0.0),
      damping=params.linear_damping,
    )
    body.add_joint(
      name="slide_y",
      type=mujoco.mjtJoint.mjJNT_SLIDE,
      axis=(0.0, 1.0, 0.0),
      damping=params.linear_damping,
    )
    body.add_joint(
      name="slide_z",
      type=mujoco.mjtJoint.mjJNT_SLIDE,
      axis=(0.0, 0.0, 1.0),
      damping=params.linear_damping,
    )
    # 3 revolute joints (intrinsic XYZ Euler) for angular DOFs.
    body.add_joint(
      name="hinge_x",
      type=mujoco.mjtJoint.mjJNT_HINGE,
      axis=(1.0, 0.0, 0.0),
      damping=params.angular_damping,
    )
    body.add_joint(
      name="hinge_y",
      type=mujoco.mjtJoint.mjJNT_HINGE,
      axis=(0.0, 1.0, 0.0),
      damping=params.angular_damping,
    )
    body.add_joint(
      name="hinge_z",
      type=mujoco.mjtJoint.mjJNT_HINGE,
      axis=(0.0, 0.0, 1.0),
      damping=params.angular_damping,
    )

    body.add_geom(
      name="ball_geom",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(params.radius, 0.0, 0.0),
      mass=params.mass,
      rgba=params.rgba,
    )
    return spec

  return _spec_fn


# Joint names in the order they appear in qpos / qvel for the ball.
# Other modules can resolve indices via ``entity.find_joints(BALL_JOINT_NAMES)``.
BALL_JOINT_NAMES: tuple[str, ...] = (
  "slide_x",
  "slide_y",
  "slide_z",
  "hinge_x",
  "hinge_y",
  "hinge_z",
)


def soccer_ball_entity_cfg(params: SoccerBallParams) -> EntityCfg:
  """Wrap ``make_soccer_ball_spec(params)`` into an ``EntityCfg``."""
  return EntityCfg(spec_fn=make_soccer_ball_spec(params))
