#!/usr/bin/env bash
# Launch downstream velocity-tracking PPO on top of a frozen motion-prior backbone.
#
# Three modes via ``MODE`` env var:
#   smoke  — 64 envs × 20 iter, sanity check (default if no ckpt edited)
#   full   — 4096 envs × 100k iter, real training
#   vq     — same as ``full`` but uses the VQ task + needs a VQ ckpt
#
# Examples:
#   MODE=smoke ./run_downstream_velocity.sh
#   MODE=full MOTION_PRIOR_CKPT=/path/to/model.pt ./run_downstream_velocity.sh
#   MODE=vq   MOTION_PRIOR_CKPT=/path/to/vq_model.pt ./run_downstream_velocity.sh
#
# Pass extra flags through:
#   ./run_downstream_velocity.sh --agent.algorithm.entropy-coef 0.01
set -euo pipefail

# ============================================================================
# Edit these paths once, override the rest via env vars or CLI flags.
# ============================================================================

# VAE motion-prior ckpt (used by MODE=smoke / MODE=full).
MOTION_PRIOR_CKPT="${MOTION_PRIOR_CKPT:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior/2026-04-29_22-52-01/model_19999.pt}"

# VQ motion-prior ckpt (used by MODE=vq). Edit when you have a VQ run.
MOTION_PRIOR_VQ_CKPT="${MOTION_PRIOR_VQ_CKPT:-/home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_motion_prior_vq/2026-05-06_15-39-54/model_9000.pt}"

# ============================================================================
# Tunable knobs (env-var overridable).
# ============================================================================

MODE="${MODE:-full}"               # smoke / full / vq
NUM_ENVS="${NUM_ENVS:-}"             # leave empty to let MODE pick a default
MAX_ITER="${MAX_ITER:-}"             # same
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LR="${LR:-1e-3}"
ENTROPY_COEF="${ENTROPY_COEF:-0.005}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-24}"
LOGGER="${LOGGER:-wandb}"            # wandb / tensorboard
RUN_NAME="${RUN_NAME:-}"             # optional descriptive tag for the log dir

# ============================================================================
# Resolve mode → task ID + ckpt + scale defaults.
# ============================================================================

case "$MODE" in
  smoke)
    TASK="Mjlab-Downstream-Velocity-Unitree-G1"
    CKPT="$MOTION_PRIOR_CKPT"
    : "${NUM_ENVS:=64}"
    : "${MAX_ITER:=20}"
    ;;
  full)
    TASK="Mjlab-Downstream-Velocity-Unitree-G1"
    CKPT="$MOTION_PRIOR_CKPT"
    : "${NUM_ENVS:=3500}"
    : "${MAX_ITER:=100000}"
    ;;
  vq)
    TASK="Mjlab-Downstream-VQ-Velocity-Unitree-G1"
    CKPT="$MOTION_PRIOR_VQ_CKPT"
    : "${NUM_ENVS:=3500}"
    : "${MAX_ITER:=100000}"
    ;;
  *)
    echo "ERROR: unknown MODE=$MODE; use smoke / full / vq" >&2
    exit 1
    ;;
esac

# ============================================================================
# Pre-flight: confirm ckpt exists, print resolved config.
# ============================================================================

if [[ "$CKPT" == *"<timestamp>"* ]] || [[ ! -f "$CKPT" ]]; then
  echo "ERROR: motion-prior ckpt not found: $CKPT" >&2
  echo "       Edit the script's default or set MOTION_PRIOR_CKPT / MOTION_PRIOR_VQ_CKPT." >&2
  exit 1
fi

cd "$(dirname "$0")"

echo "============================================================"
echo "  task           : $TASK"
echo "  mode           : $MODE"
echo "  motion_prior   : $CKPT"
echo "  num_envs       : $NUM_ENVS"
echo "  max_iter       : $MAX_ITER"
echo "  num_steps/env  : $NUM_STEPS_PER_ENV"
echo "  save_interval  : $SAVE_INTERVAL"
echo "  lr             : $LR"
echo "  entropy_coef   : $ENTROPY_COEF"
echo "  logger         : $LOGGER"
[[ -n "$RUN_NAME" ]] && echo "  run_name       : $RUN_NAME"
echo "  extra args     : $*"
echo "============================================================"

# ============================================================================
# Launch.
# ============================================================================

CMD=(
  uv run python -m mjlab.scripts.train "$TASK"
  --agent.motion-prior-ckpt-path "$CKPT"
  --env.scene.num-envs "$NUM_ENVS"
  --agent.max-iterations "$MAX_ITER"
  --agent.num-steps-per-env "$NUM_STEPS_PER_ENV"
  --agent.save-interval "$SAVE_INTERVAL"
  --agent.algorithm.learning-rate "$LR"
  --agent.algorithm.entropy-coef "$ENTROPY_COEF"
  --agent.logger "$LOGGER"
)

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--agent.run-name "$RUN_NAME")
fi

exec "${CMD[@]}" "$@"
