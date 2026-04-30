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
RUN="${RUN:-/home/bcj/zcy/mjlab-prior/logs/rsl_rl/g1_downstream_velocity/<timestamp>}"
CKPT="${CKPT:-model_99000.pt}"
MOTION_PRIOR_CKPT="${MOTION_PRIOR_CKPT:-/home/bcj/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/<timestamp>/model_xxx.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

MJLAB_MOTION_PRIOR_CKPT="$MOTION_PRIOR_CKPT" \
uv run python -m mjlab.scripts.play \
  Mjlab-Downstream-Velocity-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --num-envs "$NUM_ENVS" \
  "$@"
