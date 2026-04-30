#!/usr/bin/env bash
# Train the VQ downstream velocity-tracking PPO on top of a frozen
# **VQ** motion-prior backbone (motion_prior MLP + frozen codebook +
# decoder loaded from MOTION_PRIOR_VQ_CKPT).
#
# Differs from run_downstream_velocity.sh only in the registered task
# (DownStreamVQOnPolicyRunner builds DownStreamVQPolicy with a quantizer
# instead of mp_mu) and the expected ckpt format (saved by
# MotionPriorVQOnPolicyRunner).
set -euo pipefail

# ---- override via env vars or edit defaults below ----
MOTION_PRIOR_VQ_CKPT="${MOTION_PRIOR_VQ_CKPT:-/home/bcj/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior_vq/<timestamp>/model_xxx.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-100000}"
# -------------------------------------------------------

cd "$(dirname "$0")"

uv run python -m mjlab.scripts.train Mjlab-Downstream-VQ-Velocity-Unitree-G1 \
  --agent.motion-prior-ckpt-path "$MOTION_PRIOR_VQ_CKPT" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITER" \
  "$@"
