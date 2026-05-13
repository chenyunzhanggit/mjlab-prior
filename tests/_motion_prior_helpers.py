"""Shared helpers for motion_prior policy/runner tests.

Centralises ckpt discovery + the depth-image dummy shape so individual test
files don't all hard-code paths that drift as we retrain teachers.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Default depth shape — must match what motion_prior env exposes (D10: 1x60x60).
DEFAULT_DEPTH_SHAPE: tuple[int, int, int] = (1, 60, 60)


def _candidate_teacher_a_paths() -> list[Path]:
  env = os.environ.get("MJLAB_TEACHER_A_CKPT")
  candidates: list[Path] = []
  if env:
    candidates.append(Path(env).expanduser())
  candidates.extend(
    Path(p).expanduser()
    for p in (
      "~/project/Teleopit/track.pt",
      "/home/lenovo/project/Teleopit/track.pt",
    )
  )
  return candidates


def _candidate_teacher_b_dirs() -> list[Path]:
  env = os.environ.get("MJLAB_TEACHER_B_CKPT")
  if env:
    p = Path(env).expanduser()
    return [p.parent] if p.is_file() else [p]
  return [
    Path("~/project/mjlab_prior/logs/rsl_rl/g1_velocity").expanduser(),
  ]


def _latest_velocity_ckpt() -> Path | None:
  """Pick the newest ``model_*.pt`` under any g1_velocity run dir."""
  env = os.environ.get("MJLAB_TEACHER_B_CKPT")
  if env:
    p = Path(env).expanduser()
    if p.is_file():
      return p
  best: tuple[int, Path] | None = None
  for root in _candidate_teacher_b_dirs():
    if not root.exists():
      continue
    for ckpt in root.glob("*/model_*.pt"):
      try:
        step = int(ckpt.stem.split("_")[-1])
      except ValueError:
        continue
      if best is None or step > best[0]:
        best = (step, ckpt)
  return best[1] if best else None


def teacher_ckpts_or_skip() -> tuple[Path, Path]:
  """Return ``(teacher_a, teacher_b)`` paths, or ``pytest.skip``."""
  a: Path | None = None
  for cand in _candidate_teacher_a_paths():
    if cand.is_file():
      a = cand
      break
  if a is None:
    pytest.skip("teacher_a checkpoint not found; set MJLAB_TEACHER_A_CKPT")
  b = _latest_velocity_ckpt()
  if b is None or not b.is_file():
    pytest.skip("teacher_b checkpoint not found; set MJLAB_TEACHER_B_CKPT")
  return a, b
