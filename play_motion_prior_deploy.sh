#!/usr/bin/env bash
# Play the trained VAE motion-prior student under Path 3 deploy:
#   prop_obs -> motion_prior -> mp_mu -> decoder([prop, mp_mu]) -> action
#
# NO task latent at all. The student only sees proprioception, so behavior
# collapses to the trained "average motion" — robot drifts on whatever
# the motion_prior MLP regresses to from prop alone (typically a
# stand/sway). Same chain the on-robot ONNX runs.
#
# Uses the FLAT env scaffolding (any env with the matching `student` obs
# group works since the deploy chain ignores teacher obs); switch
# `--task` to Mjlab-MotionPrior-Rough-Unitree-G1 to see Path 3 over
# rough terrain instead.
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

MJLAB_MP_INFERENCE_PATH=deploy \
uv run python -m mjlab.scripts.play \
  Mjlab-MotionPrior-Flat-Unitree-G1 \
  --checkpoint-file "$RUN/$CKPT" \
  --motion-path "$MOTION_PATH" \
  --motion-type "$MOTION_TYPE" \
  --num-envs "$NUM_ENVS" \
  --teacher-a-policy-path "$TEACHER_A" \
  --teacher-b-policy-path "$TEACHER_B" \
  "$@"
