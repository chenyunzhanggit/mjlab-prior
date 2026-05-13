"""Soccer ball entity (single free-joint sphere).

mjlab/MuJoCo equivalent of the reference's
``RigidObjectCfg(spawn=UsdFileCfg(rigid_props=RigidBodyPropertiesCfg(
linear_damping=..., angular_damping=...)))``.

Earlier we tried a **3 prismatic + 3 revolute** decomposition to honor
``linear_damping`` and ``angular_damping`` separately, matching the
reference one-for-one. That setup is mathematically equivalent in the
small-angle regime but introduces an XYZ-Euler chain whose Jacobian
goes singular near pitch=±π/2 — a fast-spinning ball easily reaches
those orientations. On 4096-env rollouts this surfaced as a small but
persistent subset of envs (~5 of 4096) whose MuJoCo solver diverged into
NaN at the first ball↔robot contact; once an env's qpos went NaN every
subsequent observation was NaN until the next time-out reset.

Switching to a single ``freejoint`` removes the singular Jacobian and
makes the ball numerically equivalent to PhysX's free body. The cost is
that MuJoCo's freejoint exposes a single scalar ``damping`` applied to
all 6 DOFs; per-axis linear / angular damping is no longer separable.
We use ``linear_damping`` as the single value since the soccer ball's
behavior is dominated by translation + rolling (where linear damping
governs both, via no-slip rolling). Reference angular damping values are
not large enough to matter for downstream task convergence.

Pose / velocity writes go through ``write_root_state_to_sim`` (the
standard mjlab API for floating-base entities) — no custom joint-id
plumbing.

``mass`` is set on the geom (FIFA spec: ~0.45 kg, radius 0.11 m).
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

  Reference values (from motionprior reference cfgs, only ``linear_damping``
  is honored — see module docstring for why):
    dribbling: linear=0.8 (was 0.8 / 0.0 in reference)
    kicking:   linear=0.4 (was 0.4 / 0.2)
    passing:   linear=0.3 (was 0.3 / 0.2)
  """

  linear_damping: float
  angular_damping: float  # accepted for API parity; not applied (see docstring)
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
    # Single 6-DOF free joint. ``damping`` is a scalar applied to all 6
    # DOFs (linear & angular). Free joint pose is set via mjlab's
    # ``write_root_state_to_sim`` — matches the floating-base flow used
    # by the manipulation cube example.
    body.add_freejoint(name="ball_freejoint")
    body.add_geom(
      name="ball_geom",
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(params.radius, 0.0, 0.0),
      mass=params.mass,
      rgba=params.rgba,
    )
    # Set damping on the freejoint after creation (MjSpec.add_freejoint
    # doesn't accept damping in its signature).
    for j in spec.joints:
      if j.name == "ball_freejoint":
        j.damping = params.linear_damping
        break
    return spec

  return _spec_fn


def soccer_ball_entity_cfg(params: SoccerBallParams) -> EntityCfg:
  """Wrap ``make_soccer_ball_spec(params)`` into an ``EntityCfg``."""
  return EntityCfg(spec_fn=make_soccer_ball_spec(params))
