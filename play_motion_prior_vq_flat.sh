#!/usr/bin/env bash
# Play the trained VQ-VAE motion-prior student on the FLAT (motion-tracking)
# env. Defaults to the encoder_a path so the student follows the motion
# command (Scenario 1: flat tracking with motion latent).
#
# To run the prop-only Path 3 deploy chain instead (no task latent at all,
# the robot drifts on whatever motion_prior(prop) decodes to):
#   MJLAB_MP_INFERENCE_PATH=deploy ./play_motion_prior_vq_flat.sh
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior_vq/2026-05-06_13-39-15}"
CKPT="${CKPT:-model_1500.pt}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/g1_retargeted_data/npz/Data10k/}"
MOTION_TYPE="${MOTION_TYPE:-isaaclab}"
TEACHER_A="${TEACHER_A:-/home/lenovo/project/Teleopit/track.pt}"
TEACHER_B="${TEACHER_B:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
MJLAB_MP_INFERENCE_PATH="${MJLAB_MP_INFERENCE_PATH:-encoder_a}"
# -------------------------------------------------------

cd "$(dirname "$0")"

MJLAB_MP_INFERENCE_PATH="$MJLAB_MP_INFERENCE_PATH" \
uv run python -m mjlab.scripts.play \
  Mjlab-MotionPrior-VQ-Flat-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --motion-path "$MOTION_PATH" \
  --motion-type "$MOTION_TYPE" \
  --num-envs "$NUM_ENVS" \
  --teacher-a-policy-path "$TEACHER_A" \
  --teacher-b-policy-path "$TEACHER_B" \
  "$@"
