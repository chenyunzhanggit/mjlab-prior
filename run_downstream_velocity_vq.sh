#!/usr/bin/env bash
# Thin wrapper: delegates to ``run_downstream_velocity.sh`` with MODE=vq.
# Kept for backwards-compatible invocation; new usage prefers
# ``MODE=vq ./run_downstream_velocity.sh`` directly.
set -euo pipefail
MODE=vq exec "$(dirname "$0")/run_downstream_velocity.sh" "$@"
