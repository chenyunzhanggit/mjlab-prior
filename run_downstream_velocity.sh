#!/usr/bin/env bash
# Train the downstream velocity-tracking PPO on top of a frozen motion-prior
# backbone (decoder + motion_prior + mp_mu loaded from MOTION_PRIOR_CKPT).
#
# Single env (no secondary). The trained policy outputs latent residuals
# in 32-d space; the frozen decoder maps them to G1 joint commands.
set -euo pipefail

# ---- override via env vars or edit defaults below ----
MOTION_PRIOR_CKPT="${MOTION_PRIOR_CKPT:-/home/bcj/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/<timestamp>/model_xxx.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-100000}"
# -------------------------------------------------------

cd "$(dirname "$0")"

uv run python -m mjlab.scripts.train Mjlab-Downstream-Velocity-Unitree-G1 \
  --agent.motion-prior-ckpt-path "$MOTION_PRIOR_CKPT" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITER" \
  "$@"
