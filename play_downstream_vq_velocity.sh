#!/usr/bin/env bash
# Play the trained VQ downstream velocity-tracking policy.
#
# Inference chain (DownStreamVQPolicy.policy_inference, deterministic):
#   prior_latent = motion_prior(prop_obs)                # frozen, code_dim
#   raw_action   = actor(policy_obs)                     # task latent, code_dim
#                                                         # policy_obs holds the
#                                                         # twist command + prop hist + height_scan
#   z            = prior_latent + lab_lambda * tanh(raw_action)   # use_lab=True
#                  (or prior_latent + raw_action when use_lab=False)
#   q            = quantizer(z)                          # frozen codebook
#   action       = decoder(cat([prop_obs, q]))           # frozen
#
# So the actor IS the "task latent" — it's how the velocity command enters
# the chain. Compared to play_motion_prior_vq_flat.sh (no task latent), the
# robot here actually walks toward the commanded twist.
#
# play.py doesn't expose --agent.* fields, so the VQ motion-prior ckpt
# (carrying motion_prior + decoder + quantizer) is injected via env var.
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_downstream_vq_velocity/CHANGE_ME}"
CKPT="${CKPT:-model_15000.pt}"
MOTION_PRIOR_CKPT="${MOTION_PRIOR_CKPT:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior_vq/CHANGE_ME/model_19999.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

MJLAB_MOTION_PRIOR_CKPT="$MOTION_PRIOR_CKPT" \
uv run python -m mjlab.scripts.play \
  Mjlab-Downstream-VQ-Velocity-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --num-envs "$NUM_ENVS" \
  "$@"
