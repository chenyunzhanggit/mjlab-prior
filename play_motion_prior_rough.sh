#!/usr/bin/env bash
# Play the trained motion-prior student on the rough (velocity / terrain) env.
# Path 3 deploy: prop_obs -> motion_prior -> decoder -> action. teacher_b's
# twist command and height_scan are visible to the env but ignored by
# get_inference_policy() — student only sees proprioception + last_action.
#
# The same checkpoint trained under Mjlab-MotionPrior-Flat-Unitree-G1 is
# loaded here; only the trainable submodules (motion_prior + decoder + the
# two encoders) are restored, frozen teachers reload from their ckpt paths.
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/bcj/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/<timestamp>}"
CKPT="${CKPT:-model_99000.pt}"
TEACHER_A="${TEACHER_A:-/home/bcj/zcy/Teleopit/track.pt}"
TEACHER_B="${TEACHER_B:-/home/bcj/zcy/mjlab-prior/logs/model_21000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

uv run python -m mjlab.scripts.play \
  --task Mjlab-MotionPrior-Rough-Unitree-G1 \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.secondary-num-envs "$NUM_ENVS" \
  --agent.load-run "$RUN" \
  --agent.load-checkpoint "$CKPT" \
  --agent.teacher-a-policy-path "$TEACHER_A" \
  --agent.teacher-b-policy-path "$TEACHER_B" \
  "$@"
