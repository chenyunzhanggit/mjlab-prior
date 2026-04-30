"""ONNX export tests for the downstream policy.

Covers both export modes (``combined`` and ``actor``) plus the runner's
``export_policy_to_onnx`` hook. Skipped when no motion_prior ckpt is
available locally.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from mjlab.tasks.motion_prior.onnx import (
  DownStreamActorDeployModel,
  DownStreamCombinedDeployModel,
  export_downstream_to_onnx,
)
from mjlab.tasks.motion_prior.rl.policies import DownStreamPolicy

PROP = 559
LATENT = 32
NUM_ACTIONS = 29
NUM_OBS = PROP + 3
NUM_PRIV = NUM_OBS + 3
B = 4


def _find_motion_prior_ckpt() -> Path | None:
  pattern = str(
    Path("~/zcy/mjlab-prior/logs/rsl_rl/g1_motion_prior/*/model_*.pt").expanduser()
  )
  matches = sorted(glob.glob(pattern))
  return Path(matches[-1]) if matches else None


@pytest.fixture(scope="module")
def policy() -> DownStreamPolicy:
  ckpt = _find_motion_prior_ckpt()
  if ckpt is None:
    pytest.skip("no motion_prior ckpt available")
  return DownStreamPolicy(
    num_obs=NUM_OBS,
    num_actions=NUM_ACTIONS,
    num_privileged_obs=NUM_PRIV,
    prop_obs_dim=PROP,
    motion_prior_ckpt_path=ckpt,
    latent_z_dims=LATENT,
    device="cpu",
  )


# ---------------------------------------------------------------------------
# Deploy module surface
# ---------------------------------------------------------------------------


def test_combined_deploy_shapes(policy: DownStreamPolicy) -> None:
  m = DownStreamCombinedDeployModel(policy)
  prop = torch.randn(B, PROP)
  pol = torch.randn(B, NUM_OBS)
  out = m(prop, pol)
  assert out.shape == (B, NUM_ACTIONS)
  # deepcopied params: not aliasing the original.
  for p_src, p_dst in zip(policy.actor.parameters(), m.actor.parameters(), strict=True):
    assert p_src.data_ptr() != p_dst.data_ptr()


def test_actor_deploy_shapes(policy: DownStreamPolicy) -> None:
  m = DownStreamActorDeployModel(policy)
  pol = torch.randn(B, NUM_OBS)
  out = m(pol)
  assert out.shape == (B, LATENT)


# ---------------------------------------------------------------------------
# PyTorch ↔ onnxruntime parity
# ---------------------------------------------------------------------------


def test_combined_onnx_matches_pytorch(
  policy: DownStreamPolicy, tmp_path: Path
) -> None:
  out = tmp_path / "downstream_combined.onnx"
  export_downstream_to_onnx(policy, out, mode="combined")
  assert out.is_file()

  m = DownStreamCombinedDeployModel(policy).to("cpu").eval()
  gen = torch.Generator().manual_seed(0)
  prop = torch.randn(B, PROP, generator=gen)
  pol = torch.randn(B, NUM_OBS, generator=gen)
  with torch.no_grad():
    torch_out = m(prop, pol).cpu().numpy()

  sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
  ort_out = sess.run(None, {"prop_obs": prop.numpy(), "policy_obs": pol.numpy()})[0]
  np.testing.assert_allclose(torch_out, ort_out, atol=1e-5, rtol=1e-5)


def test_actor_onnx_matches_pytorch(policy: DownStreamPolicy, tmp_path: Path) -> None:
  out = tmp_path / "downstream_actor.onnx"
  export_downstream_to_onnx(policy, out, mode="actor")
  assert out.is_file()

  m = DownStreamActorDeployModel(policy).to("cpu").eval()
  pol = torch.randn(B, NUM_OBS, generator=torch.Generator().manual_seed(1))
  with torch.no_grad():
    torch_out = m(pol).cpu().numpy()

  sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
  ort_out = sess.run(None, {"policy_obs": pol.numpy()})[0]
  np.testing.assert_allclose(torch_out, ort_out, atol=1e-5, rtol=1e-5)


def test_export_rejects_unknown_mode(policy: DownStreamPolicy, tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="Unknown export mode"):
    export_downstream_to_onnx(policy, tmp_path / "x.onnx", mode="nope")


# ---------------------------------------------------------------------------
# Runner.export_policy_to_onnx hook
# ---------------------------------------------------------------------------


def test_runner_exports_via_save(tmp_path: Path) -> None:
  ckpt = _find_motion_prior_ckpt()
  if ckpt is None:
    pytest.skip("no motion_prior ckpt available")

  from tensordict import TensorDict

  from mjlab.tasks.motion_prior.rl.downstream_runner import DownStreamOnPolicyRunner

  class _FakeEnv:
    def __init__(self) -> None:
      self.num_envs = 4
      self.num_actions = NUM_ACTIONS
      self.device = torch.device("cpu")
      self.episode_length_buf = torch.zeros(4, dtype=torch.long)

    def get_observations(self) -> TensorDict:
      return TensorDict(
        {
          "policy": torch.randn(4, NUM_OBS),
          "motion_prior_obs": torch.randn(4, PROP),
          "critic": torch.randn(4, NUM_PRIV),
        },
        batch_size=[4],
      )

    def step(self, _):
      raise NotImplementedError

  runner = DownStreamOnPolicyRunner(
    env=_FakeEnv(),  # type: ignore[arg-type]
    train_cfg={
      "num_steps_per_env": 4,
      "save_interval": 100,
      "motion_prior_ckpt_path": str(ckpt),
      "policy": {"latent_z_dims": LATENT, "actor_hidden_dims": (64, 32)},
      "algorithm": {"num_learning_epochs": 1, "num_mini_batches": 1},
      "logger": "none",
    },
    log_dir=str(tmp_path),
    device="cpu",
  )

  ckpt_path = tmp_path / "model_0.pt"
  runner.save(str(ckpt_path))
  # save() should auto-dump model_0.onnx alongside.
  onnx_path = ckpt_path.with_suffix(".onnx")
  assert onnx_path.is_file()
  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  prop = np.random.randn(2, PROP).astype(np.float32)
  pol = np.random.randn(2, NUM_OBS).astype(np.float32)
  out = sess.run(None, {"prop_obs": prop, "policy_obs": pol})[0]
  assert out.shape == (2, NUM_ACTIONS)
