#!/usr/bin/env bash
set -euo pipefail

cd ~/project/mjlab_prior
source .venv/bin/activate

uv run python -m mjlab.scripts.train Mjlab-MotionPrior-Flat-Unitree-G1 \
  --env.commands.motion.motion-path /home/lenovo/g1_retargeted_data/npz/Data10k \
  --env.scene.num-envs 64 \
  --agent.secondary-num-envs 64 \
  --agent.max-iterations 50 \
  --agent.teacher-a-policy-path /home/lenovo/project/Teleopit/track.pt \
  --agent.teacher-b-policy-path /home/lenovo/project/mjlab_prior/logs/rsl_rl/g1_velocity/2026-04-28_16-16-06/model_21000.pt
