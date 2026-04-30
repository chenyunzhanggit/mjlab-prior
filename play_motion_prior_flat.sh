#!/usr/bin/env bash
# Play the trained motion-prior student on the flat (motion-tracking) env.
# Path 3 deploy: prop_obs -> motion_prior -> decoder -> action. Teacher obs
# are computed by the env but ignored by get_inference_policy().
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior/2026-04-29_22-52-01}"
CKPT="${CKPT:-model_19999.pt}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/g1_retargeted_data/npz/Data10k/}"
MOTION_TYPE="${MOTION_TYPE:-isaaclab}"   
TEACHER_A="${TEACHER_A:-/home/lenovo/project/Teleopit/track.pt}"
TEACHER_B="${TEACHER_B:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

uv run python -m mjlab.scripts.play \
  Mjlab-MotionPrior-Flat-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --motion-path "$MOTION_PATH" \
  --motion-type "$MOTION_TYPE" \
  --num-envs 1 \
  --teacher-a-policy-path "$TEACHER_A" \
  --teacher-b-policy-path "$TEACHER_B" \
  "$@"
