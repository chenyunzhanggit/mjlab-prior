#!/usr/bin/env bash
# Play the trained motion-prior student on the flat (motion-tracking) env.
# Path 3 deploy: prop_obs -> motion_prior -> decoder -> action. Teacher obs
# are computed by the env but ignored by get_inference_policy().
set -euo pipefail

# ---- override via env vars or edit defaults below ----
RUN="${RUN:-/home/bcj/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/<timestamp>}"
CKPT="${CKPT:-model_99000.pt}"
MOTION_PATH="${MOTION_PATH:-/tmp/motions_tiny}"
MOTION_TYPE="${MOTION_TYPE:-mujoco}"   # set to "isaaclab" for Data10k
TEACHER_A="${TEACHER_A:-/home/bcj/zcy/Teleopit/track.pt}"
TEACHER_B="${TEACHER_B:-/home/bcj/zcy/mjlab-prior/logs/model_21000.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
# -------------------------------------------------------

cd "$(dirname "$0")"

uv run python -m mjlab.scripts.play \
  --task Mjlab-MotionPrior-Flat-Unitree-G1 \
  --env.commands.motion.motion-path "$MOTION_PATH" \
  --env.commands.motion.motion-type "$MOTION_TYPE" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.secondary-num-envs "$NUM_ENVS" \
  --agent.load-run "$RUN" \
  --agent.load-checkpoint "$CKPT" \
  --agent.teacher-a-policy-path "$TEACHER_A" \
  --agent.teacher-b-policy-path "$TEACHER_B" \
  "$@"
