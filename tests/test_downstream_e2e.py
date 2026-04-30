"""End-to-end smoke for the downstream-task PPO loop on a real GPU env.

Spins up the real velocity-rough env at ``num_envs=32``, builds the
``DownStreamOnPolicyRunner`` with a discovered motion-prior ckpt, and
runs a handful of PPO iterations. Skipped automatically when:

  * no CUDA device,
  * no motion_prior ckpt under ``~/zcy/mjlab-prior/logs/...``.

Marked ``slow`` so ``make test-fast`` ignores it.
"""

from __future__ import annotations

import glob
from dataclasses import asdict
from pathlib import Path

import pytest
import torch


def _find_motion_prior_ckpt() -> Path | None:
  pattern = str(
    Path("~/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/*/model_*.pt").expanduser()
  )
  matches = sorted(glob.glob(pattern))
  return Path(matches[-1]) if matches else None


def _requires_e2e() -> Path:
  if not torch.cuda.is_available():
    pytest.skip("downstream e2e needs CUDA")
  ckpt = _find_motion_prior_ckpt()
  if ckpt is None:
    pytest.skip("no motion_prior ckpt under ~/zcy/mjlab-prior/logs/...")
  return ckpt


@pytest.mark.slow
def test_downstream_velocity_smoke(tmp_path: Path) -> None:
  ckpt = _requires_e2e()
  # Defer heavy imports until after skips fire.
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
  from mjlab.tasks.motion_prior.downstream_rl_cfg import (
    RslRlDownstreamPolicyCfg,
    RslRlDownstreamPpoCfg,
    RslRlDownstreamRunnerCfg,
  )
  from mjlab.tasks.motion_prior.rl.downstream_runner import DownStreamOnPolicyRunner
  from mjlab.tasks.registry import load_env_cfg

  device = "cuda:0"
  num_envs = 32

  env_cfg = load_env_cfg("Mjlab-Downstream-Velocity-Unitree-G1")
  env_cfg.scene.num_envs = num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))

  # Confirm obs schema matches what DownStreamPolicy expects.
  obs = env.get_observations()
  assert "policy" in obs.keys()
  assert "motion_prior_obs" in obs.keys()
  assert "critic" in obs.keys()
  prop_dim = obs["motion_prior_obs"].shape[-1]
  print(f"[smoke] motion_prior_obs dim = {prop_dim}")

  cfg = RslRlDownstreamRunnerCfg(
    experiment_name="g1_downstream_e2e",
    num_steps_per_env=8,
    save_interval=10,
    motion_prior_ckpt_path=str(ckpt),
    policy=RslRlDownstreamPolicyCfg(latent_z_dims=32),
    algorithm=RslRlDownstreamPpoCfg(
      num_learning_epochs=2,
      num_mini_batches=2,
      learning_rate=1e-3,
    ),
    upload_model=False,
  )
  cfg_dict = asdict(cfg)
  cfg_dict["logger"] = "none"

  runner = DownStreamOnPolicyRunner(
    env=env,
    train_cfg=cfg_dict,
    log_dir=str(tmp_path),
    device=device,
  )
  num_iters = 5
  runner.learn(num_learning_iterations=num_iters)
  assert runner.current_learning_iteration == num_iters - 1

  ckpt_out = tmp_path / "last.pt"
  runner.save(str(ckpt_out))
  assert ckpt_out.is_file()

  # Inference sanity.
  obs2 = env.get_observations()
  policy_fn = runner.get_inference_policy(device=device)
  with torch.no_grad():
    act = policy_fn(obs2)
  assert act.shape == (num_envs, runner.env.num_actions)
  assert torch.isfinite(act).all()

  env.close()
