"""ONNX export + ``get_inference_policy`` smoke tests for both VAE and VQ.

Verifies:
  * the deploy wrapper for either policy type only consumes ``prop_obs``
    (no teacher obs, no history),
  * PyTorch and onnxruntime agree on the same inputs within ``atol=1e-5``,
  * ``runner.get_inference_policy`` runs end-to-end on a TensorDict whose
    teacher groups are absent or junk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.motion_prior.onnx import (
  MotionPriorVAEDeployModel,
  MotionPriorVQDeployModel,
  build_deploy_model,
  export_motion_prior_to_onnx,
)
from mjlab.tasks.motion_prior.rl.policies import (
  MotionPriorPolicy,
  MotionPriorVQPolicy,
)

TEACHER_A_CKPT = Path("~/zcy/Teleopit/track.pt").expanduser()
TEACHER_B_CKPT = Path("~/zcy/mjlab-prior/logs/model_21000.pt").expanduser()

PROP_OBS_DIM = (3 + 3 + 29 + 29 + 29) * 4
NUM_ACTIONS = 29
B = 4


def _ckpts_or_skip() -> tuple[Path, Path]:
  if not TEACHER_A_CKPT.is_file():
    pytest.skip(f"teacher_a checkpoint missing: {TEACHER_A_CKPT}")
  if not TEACHER_B_CKPT.is_file():
    pytest.skip(f"teacher_b checkpoint missing: {TEACHER_B_CKPT}")
  return TEACHER_A_CKPT, TEACHER_B_CKPT


@pytest.fixture(scope="module")
def vae_policy() -> MotionPriorPolicy:
  a, b = _ckpts_or_skip()
  return MotionPriorPolicy(
    prop_obs_dim=PROP_OBS_DIM,
    num_actions=NUM_ACTIONS,
    teacher_a_policy_path=a,
    teacher_b_policy_path=b,
    encoder_hidden_dims=(64, 32),
    decoder_hidden_dims=(64, 32),
    motion_prior_hidden_dims=(64, 32),
    latent_z_dims=16,
    device="cpu",
  )


@pytest.fixture(scope="module")
def vq_policy() -> MotionPriorVQPolicy:
  a, b = _ckpts_or_skip()
  return MotionPriorVQPolicy(
    prop_obs_dim=PROP_OBS_DIM,
    num_actions=NUM_ACTIONS,
    teacher_a_policy_path=a,
    teacher_b_policy_path=b,
    num_code=64,
    code_dim=16,
    encoder_hidden_dims=(64, 32),
    decoder_hidden_dims=(64, 32),
    motion_prior_hidden_dims=(64, 32),
    device="cpu",
  )


# ---------------------------------------------------------------------------
# build_deploy_model dispatch
# ---------------------------------------------------------------------------


def test_build_deploy_model_dispatches(vae_policy, vq_policy) -> None:
  assert isinstance(build_deploy_model(vae_policy), MotionPriorVAEDeployModel)
  assert isinstance(build_deploy_model(vq_policy), MotionPriorVQDeployModel)


def test_build_deploy_rejects_other_types() -> None:
  with pytest.raises(TypeError):
    build_deploy_model(torch.nn.Linear(4, 4))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Deploy module surface
# ---------------------------------------------------------------------------


def test_vae_deploy_input_is_prop_obs_only(vae_policy) -> None:
  m = MotionPriorVAEDeployModel(vae_policy)
  prop = torch.randn(B, PROP_OBS_DIM)
  out = m(prop)
  assert out.shape == (B, NUM_ACTIONS)
  # Module owns deepcopies — should not be sharing storage with the
  # original parameters (so saving the ONNX after further training of
  # the policy wouldn't drift this snapshot).
  for p_src, p_dst in zip(
    vae_policy.motion_prior.parameters(),
    m.motion_prior.parameters(),
    strict=True,
  ):
    assert p_src.data_ptr() != p_dst.data_ptr()


def test_vq_deploy_input_is_prop_obs_only(vq_policy) -> None:
  m = MotionPriorVQDeployModel(vq_policy)
  prop = torch.randn(B, PROP_OBS_DIM)
  out = m(prop)
  assert out.shape == (B, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# PyTorch <-> onnxruntime parity
# ---------------------------------------------------------------------------


def _check_onnx_parity(
  policy: MotionPriorPolicy | MotionPriorVQPolicy, tmp_path: Path
) -> None:
  out = tmp_path / "deploy.onnx"
  export_motion_prior_to_onnx(policy, out)
  assert out.is_file()

  deploy = build_deploy_model(policy).to("cpu").eval()
  prop = torch.randn(B, PROP_OBS_DIM, generator=torch.Generator().manual_seed(0))
  with torch.no_grad():
    torch_out = deploy(prop).cpu().numpy()

  sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
  ort_out = sess.run(None, {"prop_obs": prop.numpy()})[0]
  np.testing.assert_allclose(torch_out, ort_out, atol=1e-5, rtol=1e-5)


def test_vae_onnx_matches_pytorch(vae_policy, tmp_path: Path) -> None:
  _check_onnx_parity(vae_policy, tmp_path)


def test_vq_onnx_matches_pytorch(vq_policy, tmp_path: Path) -> None:
  _check_onnx_parity(vq_policy, tmp_path)


# ---------------------------------------------------------------------------
# get_inference_policy via the runner (uses Path 3, ignores teacher obs)
# ---------------------------------------------------------------------------


class _FakeFlatEnv:
  """Minimal stand-in that supplies the obs groups the runner inspects."""

  def __init__(self) -> None:
    self.num_envs = 4
    self.num_actions = NUM_ACTIONS
    self.device = torch.device("cpu")
    self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)

  def get_observations(self) -> TensorDict:
    return TensorDict(
      {
        "student": torch.randn(self.num_envs, PROP_OBS_DIM),
        "teacher_a": torch.randn(self.num_envs, 166),
        "teacher_a_history": torch.randn(self.num_envs, 10, 166),
      },
      batch_size=[self.num_envs],
    )

  def step(self, actions):  # unused in this test
    raise NotImplementedError


def test_runner_get_inference_policy_path3(monkeypatch, tmp_path: Path) -> None:
  ta, tb = _ckpts_or_skip()
  from mjlab.tasks.motion_prior.rl import runner as runner_mod
  from mjlab.tasks.motion_prior.rl.runner import MotionPriorOnPolicyRunner

  monkeypatch.setattr(
    runner_mod,
    "_build_secondary_env",
    lambda *a, **k: _FakeFlatEnv(),
  )

  runner = MotionPriorOnPolicyRunner(
    env=_FakeFlatEnv(),  # type: ignore[arg-type]
    train_cfg={
      "num_steps_per_env": 4,
      "save_interval": 100,
      "secondary_task_id": "_",
      "secondary_num_envs": 4,
      "teacher_a_policy_path": str(ta),
      "teacher_b_policy_path": str(tb),
      "policy": {
        "encoder_hidden_dims": (64, 32),
        "decoder_hidden_dims": (64, 32),
        "motion_prior_hidden_dims": (64, 32),
        "latent_z_dims": 16,
      },
      "algorithm": {"num_learning_epochs": 1, "learning_rate": 1e-3},
      "logger": "none",
    },
    log_dir=str(tmp_path),
    device="cpu",
  )

  policy = runner.get_inference_policy()
  obs = runner.env.get_observations()
  action = policy(obs)
  assert action.shape == (runner.env.num_envs, NUM_ACTIONS)

  # Drop teacher groups entirely — Path 3 must still work because deploy
  # only consumes ``student``.
  obs_no_teacher = TensorDict(
    {"student": obs["student"]}, batch_size=[runner.env.num_envs]
  )
  action_again = policy(obs_no_teacher)
  assert action_again.shape == (runner.env.num_envs, NUM_ACTIONS)


def test_runner_export_policy_to_onnx(monkeypatch, tmp_path: Path) -> None:
  ta, tb = _ckpts_or_skip()
  from mjlab.tasks.motion_prior.rl import runner as runner_mod
  from mjlab.tasks.motion_prior.rl.runner import MotionPriorOnPolicyRunner

  monkeypatch.setattr(
    runner_mod,
    "_build_secondary_env",
    lambda *a, **k: _FakeFlatEnv(),
  )

  runner = MotionPriorOnPolicyRunner(
    env=_FakeFlatEnv(),  # type: ignore[arg-type]
    train_cfg={
      "num_steps_per_env": 4,
      "save_interval": 100,
      "secondary_task_id": "_",
      "secondary_num_envs": 4,
      "teacher_a_policy_path": str(ta),
      "teacher_b_policy_path": str(tb),
      "policy": {
        "encoder_hidden_dims": (64, 32),
        "decoder_hidden_dims": (64, 32),
        "motion_prior_hidden_dims": (64, 32),
        "latent_z_dims": 16,
      },
      "algorithm": {"num_learning_epochs": 1, "learning_rate": 1e-3},
      "logger": "none",
    },
    log_dir=str(tmp_path),
    device="cpu",
  )

  onnx_path = tmp_path / "policy.onnx"
  runner.export_policy_to_onnx(str(onnx_path))
  assert onnx_path.is_file()
  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  prop = np.random.randn(2, PROP_OBS_DIM).astype(np.float32)
  out = sess.run(None, {"prop_obs": prop})[0]
  assert out.shape == (2, NUM_ACTIONS)
