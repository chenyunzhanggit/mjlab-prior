#!/usr/bin/env bash
# Play the trained VQ-VAE motion-prior student on the ROUGH (velocity-rough)
# env. Defaults to the encoder_b path so the student follows the twist
# command + height_scan that teacher_b sees (Scenario 2: rough locomotion
# with velocity latent).
#
# To run the prop-only Path 3 deploy chain instead:
#   MJLAB_MP_INFERENCE_PATH=deploy ./play_motion_prior_vq_rough.sh
#
# Loads the SAME ckpt that was trained under Mjlab-MotionPrior-VQ-Flat;
# only the trainable submodules (encoder_a/b, decoder, motion_prior,
# quantizer) are restored — frozen teachers reload from their ckpt paths.
#
# Note: the runner builds a secondary env on top of the primary one; when
# the primary is rough, you end up with two rough envs each at NUM_ENVS.
# Keep NUM_ENVS small (1) for visualization to avoid GPU OOM.
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior_vq/CHANGE_ME}"
CKPT="${CKPT:-model_19999.pt}"
TEACHER_A="${TEACHER_A:-/home/lenovo/project/Teleopit/track.pt}"
TEACHER_B="${TEACHER_B:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
MJLAB_MP_INFERENCE_PATH="${MJLAB_MP_INFERENCE_PATH:-encoder_b}"
# -------------------------------------------------------

cd "$(dirname "$0")"

MJLAB_MP_INFERENCE_PATH="$MJLAB_MP_INFERENCE_PATH" \
uv run python -m mjlab.scripts.play \
  Mjlab-MotionPrior-VQ-Rough-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --num-envs "$NUM_ENVS" \
  --teacher-a-policy-path "$TEACHER_A" \
  --teacher-b-policy-path "$TEACHER_B" \
  "$@"
