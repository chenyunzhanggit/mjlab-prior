#!/usr/bin/env bash
# Play the trained downstream velocity-tracking policy (frozen motion-prior
# backbone + trained actor + decoder).
#
# play.py reads the agent cfg from the task registry where
# motion_prior_ckpt_path defaults to empty; we inject it via the
# MJLAB_MOTION_PRIOR_CKPT env var because play.py does not expose
# --agent.* fields.
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-//home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_downstream_velocity/2026-04-30_23-43-15}"
CKPT="${CKPT:-model_15000.pt}"
MOTION_PRIOR_CKPT="${MOTION_PRIOR_CKPT:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior/2026-04-29_22-52-01/model_19999.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

MJLAB_MOTION_PRIOR_CKPT="$MOTION_PRIOR_CKPT" \
uv run python -m mjlab.scripts.play \
  Mjlab-Downstream-Velocity-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --num-envs 1 \
  "$@"
