#!/usr/bin/env bash
# Play the trained VAE motion-prior student under Path 3 deploy:
#   prop_obs -> motion_prior -> mp_mu -> decoder([prop, mp_mu]) -> action
#
# NO task latent at all. The student only sees proprioception, so behavior
# collapses to the trained "average motion" — robot drifts on whatever
# the motion_prior MLP regresses to from prop alone (typically a
# stand/sway). Same chain the on-robot ONNX runs.
#
# TASK selects the env scaffolding:
#   Mjlab-MotionPrior-Flat-Unitree-G1   (default) — flat terrain, motion env
#   Mjlab-MotionPrior-Rough-Unitree-G1            — rough terrain, twist env
#
# Either way the deploy chain ignores teacher / command obs. Only difference
# is the terrain the robot is dropped onto, so this is exactly how you watch
# "motion_prior + prop_obs on rough terrain":
#   TASK=Mjlab-MotionPrior-Rough-Unitree-G1 ./play_motion_prior_deploy.sh
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior/2026-04-29_22-52-01}"
CKPT="${CKPT:-model_19999.pt}"
TASK="${TASK:-Mjlab-MotionPrior-Rough-Unitree-G1}"
MOTION_PATH="${MOTION_PATH:-/home/lenovo/g1_retargeted_data/npz/Data10k/}"
MOTION_TYPE="${MOTION_TYPE:-isaaclab}"
TEACHER_A="${TEACHER_A:-/home/lenovo/project/Teleopit/track.pt}"
TEACHER_B="${TEACHER_B:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

# Rough/twist env doesn't accept --motion-path / --motion-type; only the
# flat motion-tracking env needs them.
motion_args=()
if [[ "$TASK" == *"Flat"* ]]; then
  motion_args=(--motion-path "$MOTION_PATH" --motion-type "$MOTION_TYPE")
fi

MJLAB_MP_INFERENCE_PATH=deploy \
uv run python -m mjlab.scripts.play \
  "$TASK" \
  --checkpoint-file "$RUN/$CKPT" \
  "${motion_args[@]}" \
  --num-envs "$NUM_ENVS" \
  --teacher-a-policy-path "$TEACHER_A" \
  --teacher-b-policy-path "$TEACHER_B" \
  "$@"
