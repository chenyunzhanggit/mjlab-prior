"""End-to-end smoke test for the dual-env motion-prior distillation.

Spins up the **real** flat (motion-tracking) and rough (velocity) envs at
small ``num_envs`` and runs a handful of training iterations on GPU.

Skipped automatically when:
  * no CUDA device,
  * Teleopit teacher_a checkpoint missing,
  * mjlab velocity teacher_b checkpoint missing,
  * the small debug motion file is not present.

Marked ``slow`` so ``make test-fast`` ignores it.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest
import torch
from tensordict import TensorDict


def _t(td: TensorDict, key: str) -> torch.Tensor:
  return cast(torch.Tensor, td[key])


TEACHER_A_CKPT = Path("~/zcy/Teleopit/track.pt").expanduser()
TEACHER_B_CKPT = Path("~/zcy/mjlab-prior/logs/model_21000.pt").expanduser()
MOTION_FILE = Path("~/zcy/Teleopit/data/one_motion_for_debug.npz").expanduser()


def _requires_e2e() -> None:
  if not torch.cuda.is_available():
    pytest.skip("e2e smoke needs CUDA")
  if not TEACHER_A_CKPT.is_file():
    pytest.skip(f"teacher_a ckpt missing: {TEACHER_A_CKPT}")
  if not TEACHER_B_CKPT.is_file():
    pytest.skip(f"teacher_b ckpt missing: {TEACHER_B_CKPT}")
  if not MOTION_FILE.is_file():
    pytest.skip(f"motion file missing: {MOTION_FILE}")


@pytest.mark.slow
def test_dual_env_distillation_smoke(tmp_path: Path) -> None:
  _requires_e2e()
  # Defer heavy imports until after the skips fire.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
  from mjlab.tasks.motion_prior.rl.runner import MotionPriorOnPolicyRunner
  from mjlab.tasks.motion_prior.rl_cfg import (
    RslRlMotionPriorAlgoCfg,
    RslRlMotionPriorPolicyCfg,
    RslRlMotionPriorRunnerCfg,
  )
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.tasks.tracking.mdp import MotionCommandCfg

  device = "cuda:0"
  num_envs = 32

  # ----- primary (flat / teacher_a) env --------------------------------
  flat_cfg = load_env_cfg("Mjlab-MotionPrior-Flat-Unitree-G1")
  flat_cfg.scene.num_envs = num_envs
  motion_cmd = flat_cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = str(MOTION_FILE)
  flat_env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=flat_cfg, device=device))

  # Verify obs schema matches teacher_a expectations BEFORE going further.
  obs = flat_env.get_observations()
  assert _t(obs, "teacher_a").shape == (num_envs, 166), (
    f"teacher_a obs dim mismatch: got {_t(obs, 'teacher_a').shape}, expected (B, 166)"
  )
  assert _t(obs, "teacher_a_history").shape == (num_envs, 10, 166), (
    f"teacher_a_history dim mismatch: got {_t(obs, 'teacher_a_history').shape}"
  )
  expected_student = (3 + 3 + 29 + 29 + 29) * 4
  assert _t(obs, "student").shape == (num_envs, expected_student), (
    f"student dim mismatch: got {_t(obs, 'student').shape}, "
    f"expected (B, {expected_student})"
  )

  # ----- runner cfg with knobs sized for a quick run -------------------
  cfg = RslRlMotionPriorRunnerCfg(
    experiment_name="g1_motion_prior_e2e",
    num_steps_per_env=8,
    save_interval=10,
    secondary_task_id="Mjlab-MotionPrior-Rough-Unitree-G1",
    secondary_num_envs=num_envs,
    teacher_a_policy_path=str(TEACHER_A_CKPT),
    teacher_b_policy_path=str(TEACHER_B_CKPT),
    policy=RslRlMotionPriorPolicyCfg(latent_z_dims=32),
    algorithm=RslRlMotionPriorAlgoCfg(
      num_learning_epochs=2,
      learning_rate=5e-4,
      max_grad_norm=1.0,
    ),
    upload_model=False,
  )
  cfg_dict = asdict(cfg)
  cfg_dict["logger"] = "none"  # skip wandb / tensorboard for the test

  runner = MotionPriorOnPolicyRunner(
    env=flat_env,
    train_cfg=cfg_dict,
    log_dir=str(tmp_path),
    device=device,
  )
  assert runner.env_b.num_envs == num_envs

  # ----- run a handful of iterations and capture last loss --------------
  num_iters = 5
  runner.learn(num_learning_iterations=num_iters)
  assert runner.current_learning_iteration == num_iters - 1

  # save / load roundtrip on a real checkpoint
  ckpt_path = tmp_path / "last.pt"
  runner.save(str(ckpt_path))
  assert ckpt_path.is_file()
  prev = {
    k: v.detach().cpu().clone() for k, v in runner.policy.encoder_a.state_dict().items()
  }
  with torch.no_grad():
    for p in runner.policy.encoder_a.parameters():
      p.add_(1.0)
  runner.load(str(ckpt_path))
  for k, v in runner.policy.encoder_a.state_dict().items():
    assert torch.allclose(prev[k], v.detach().cpu()), f"encoder_a[{k}] not restored"

  # Final sanity: a fresh forward must produce finite outputs.
  obs_a = flat_env.get_observations()
  obs_b = runner.env_b.get_observations()
  with torch.no_grad():
    sa_a = runner.policy.policy_inference_a(
      _t(obs_a, "student"), _t(obs_a, "teacher_a")
    )
    sa_b = runner.policy.policy_inference_b(
      _t(obs_b, "student"), _t(obs_b, "teacher_b")
    )
  assert torch.isfinite(sa_a).all(), "non-finite student action on flat env"
  assert torch.isfinite(sa_b).all(), "non-finite student action on rough env"
  assert math.isfinite(sa_a.abs().mean().item())
  assert math.isfinite(sa_b.abs().mean().item())

  flat_env.close()
  runner.env_b.close()
