#!/usr/bin/env bash
# Play the trained motion-prior student on the rough (velocity / terrain) env.
# Path 3 deploy: prop_obs -> motion_prior -> decoder -> action. teacher_b's
# twist command and height_scan are visible to the env but ignored by
# get_inference_policy() — student only sees proprioception + last_action.
#
# The same checkpoint trained under Mjlab-MotionPrior-Flat-Unitree-G1 is
# loaded here; only the trainable submodules (motion_prior + decoder + the
# two encoders) are restored, frozen teachers reload from their ckpt paths.
#
# Rough env uses a `twist` velocity command, NOT a motion command, so no
# --motion-path / --motion-type flags are passed.
#
# Note: motion_prior runner builds a secondary env on top of the primary one;
# when the primary is already rough, you end up with two rough envs sharing
# NUM_ENVS each. Keep NUM_ENVS small (1 for visualization) to avoid GPU OOM.
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior/2026-04-29_22-52-01}"
CKPT="${CKPT:-model_19999.pt}"
TEACHER_A="${TEACHER_A:-/home/lenovo/project/Teleopit/track.pt}"
TEACHER_B="${TEACHER_B:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

uv run python -m mjlab.scripts.play \
  Mjlab-MotionPrior-Rough-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --num-envs "$NUM_ENVS" \
  --teacher-a-policy-path "$TEACHER_A" \
  --teacher-b-policy-path "$TEACHER_B" \
  "$@"
