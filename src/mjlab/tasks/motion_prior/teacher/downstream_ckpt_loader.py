"""Loader for the frozen ``motion_prior`` / ``mp_mu`` / ``decoder`` triplet
that ``DownStreamPolicy`` consumes as its frozen backbone.

The downstream task does not need the encoders, the per-encoder reparameterize
heads, or the optimizer state stored alongside a full motion-prior training
checkpoint — it only reads the three submodules below. Centralising the
extraction here keeps the policy module agnostic of how
``MotionPriorOnPolicyRunner.save()`` lays out the file.

Checkpoint layout (see ``MotionPriorOnPolicyRunner.save``):

    ckpt = {
        "encoder_a":    state_dict,
        "encoder_b":    state_dict,
        "es_a_mu":      state_dict,
        "es_a_var":     state_dict,
        "es_b_mu":      state_dict,
        "es_b_var":     state_dict,
        "decoder":      state_dict,   ← needed
        "motion_prior": state_dict,   ← needed
        "mp_mu":        state_dict,   ← needed (VAE)
        "mp_var":       state_dict,
        "optimizer":    state_dict,
        "iter":         int,
        "infos":        dict,
    }

Shape contract (also documented in downstream_migration_audit.md):

    motion_prior: MLP(prop_obs_dim → motion_prior_hidden_dims → latent_z_dims)
    mp_mu:        Linear(latent_z_dims, latent_z_dims)
    decoder:      MLP(prop_obs_dim + latent_z_dims → decoder_hidden_dims → num_actions)

This deliberately matches the ``MotionPriorPolicy`` architecture rather than
the reference ``DownStreamPolicy`` (which routes a fixed 64-d intermediate
through ``mp_mu``). See audit doc §3.
"""

from __future__ import annotations

from pathlib import Path

import torch

REQUIRED_KEYS = ("decoder", "motion_prior", "mp_mu")


def load_motion_prior_components(
  ckpt_path: str | Path,
  *,
  device: str | torch.device = "cpu",
) -> dict[str, dict]:
  """Load the three frozen sub-state-dicts from a motion-prior checkpoint.

  Returns:
    Dict with exactly the keys ``"decoder"``, ``"motion_prior"``,
    ``"mp_mu"`` — each value is a state_dict ready to feed into
    ``module.load_state_dict(..., strict=True)``.

  Raises:
    FileNotFoundError: ``ckpt_path`` does not exist.
    KeyError: ckpt is missing one of the required submodule entries.
  """
  ckpt_path = Path(ckpt_path).expanduser()
  if not ckpt_path.is_file():
    raise FileNotFoundError(f"motion_prior checkpoint not found: {ckpt_path}")

  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  missing = [k for k in REQUIRED_KEYS if k not in ckpt]
  if missing:
    raise KeyError(
      f"motion_prior checkpoint at {ckpt_path} is missing keys {missing}; "
      f"top-level keys: {sorted(ckpt.keys())}. "
      "DownStreamPolicy needs decoder / motion_prior / mp_mu."
    )

  return {k: ckpt[k] for k in REQUIRED_KEYS}
