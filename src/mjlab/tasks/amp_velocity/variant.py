"""Helpers for switching AMP feature/discriminator variants at runtime.

Centralizes the (env_cfg ↔ agent_cfg) consistency check used by train.py and
play.py so both scripts present a single-knob workflow:

    --agent.amp.variant {body,joint}

At training time the user sets this directly via CLI. At play time the
variant must match what the checkpoint was trained with; we auto-detect it
from the checkpoint's ``discriminator_state_dict`` keys (spectral_norm leaves
``weight_orig / weight_u / weight_v`` instead of ``weight``), so the user
doesn't have to remember.
"""

from __future__ import annotations

from typing import Any


def sync_amp_variant(env_cfg, agent_cfg) -> None:
  """Rebuild env-side AMP obs group to match ``agent_cfg.amp.variant``.

  No-op when:
    * ``agent_cfg`` has no ``amp`` field (non-AMP task), or
    * ``env_cfg.observations`` has no ``amp`` group (non-AMP env).

  Otherwise reads the rl-side ``variant`` flag and overwrites
  ``env_cfg.observations["amp"]`` with the matching ObservationGroupCfg.
  """
  amp_cfg = getattr(agent_cfg, "amp", None)
  if amp_cfg is None:
    return
  if "amp" not in env_cfg.observations:
    return
  variant = getattr(amp_cfg, "variant", "body")

  # Lazy import: only needed when this branch fires.
  from mjlab.tasks.amp_velocity.config.g1.env_cfgs import (
    _build_amp_obs_group_body,
    _build_amp_obs_group_joint,
  )

  if variant == "body":
    env_cfg.observations["amp"] = _build_amp_obs_group_body()
  elif variant == "joint":
    env_cfg.observations["amp"] = _build_amp_obs_group_joint()
  else:
    raise ValueError(f"Unknown amp variant {variant!r}; expected 'body' or 'joint'.")
  print(f"[INFO] AMP variant: {variant}")


def detect_variant_from_checkpoint(state_dict: dict[str, Any]) -> str | None:
  """Infer the AMP variant from a saved checkpoint's discriminator weights.

  Joint variant uses ``DiscriminatorMulti``, which applies spectral_norm to
  every Linear → the saved keys are ``trunk.<n>.weight_orig``,
  ``...weight_u``, ``...weight_v`` instead of plain ``trunk.<n>.weight``.

  Returns ``"joint"`` / ``"body"``, or ``None`` if the checkpoint has no
  AMP discriminator state at all.
  """
  disc_sd = state_dict.get("discriminator_state_dict")
  if not isinstance(disc_sd, dict) or not disc_sd:
    return None
  for k in disc_sd:
    if k.endswith("weight_orig"):
      return "joint"
  return "body"
