"""Smoke tests for the Teleopit teacher loader.

These tests verify that ``load_teleopit_teacher`` correctly reconstructs the
frozen TemporalCNN teacher from a Teleopit ``track.pt`` checkpoint:

  1. structural sanity (shapes, freeze flag, ckpt key coverage)
  2. PyTorch ↔ ONNX self-consistency: export the loaded model with
     ``as_onnx()``, run both PyTorch and onnxruntime on identical random
     inputs, and require ``atol=1e-5`` agreement. This proves that the
     loader's normalize/Conv1D/MLP wiring is internally self-consistent —
     i.e. nothing was silently dropped at load time.

The full ONNX file ``~/project/Teleopit/track.onnx`` (and its dated
siblings) was exported from earlier checkpoints (~10k–22k iters) than
``track.pt`` (iter=30000), so we deliberately do not use it as ground
truth. We export a fresh ONNX from the loaded model itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from tensordict import TensorDict

from mjlab.tasks.motion_prior.teacher import (
  TELEOPIT_TEACHER_CFG,
  build_teleopit_teacher,
  load_teleopit_teacher,
)

TELEOPIT_CKPT = Path("~/project/Teleopit/track.pt").expanduser()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ckpt_or_skip() -> Path:
  if not TELEOPIT_CKPT.is_file():
    pytest.skip(f"Teleopit checkpoint not present at {TELEOPIT_CKPT}")
  return TELEOPIT_CKPT


@pytest.fixture(scope="module")
def teacher_ckpt() -> Path:
  return _ckpt_or_skip()


@pytest.fixture(scope="module")
def loaded_teacher(teacher_ckpt: Path) -> torch.nn.Module:
  return load_teleopit_teacher(teacher_ckpt, device="cpu")


@pytest.fixture(scope="module")
def dummy_batch() -> tuple[torch.Tensor, torch.Tensor]:
  """A fixed-seed (B=2) batch of obs + obs_history."""
  cfg = TELEOPIT_TEACHER_CFG
  gen = torch.Generator().manual_seed(42)
  obs = torch.randn(2, cfg.actor_obs_dim, generator=gen)
  obs_hist = torch.randn(
    2, cfg.actor_history_length, cfg.actor_history_obs_dim, generator=gen
  )
  return obs, obs_hist


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


def test_build_random_weights_has_expected_shapes() -> None:
  """The model built from TELEOPIT_TEACHER_CFG must match the ckpt's shapes."""
  m = build_teleopit_teacher()
  cfg = TELEOPIT_TEACHER_CFG

  assert m.obs_groups_1d == ["actor"]
  assert m.obs_groups_3d == ["actor_history"]
  assert m.obs_dim == cfg.actor_obs_dim
  assert m.cnn_latent_dim == cfg.cnn_output_channels[-1]
  assert m.history_lengths == [cfg.actor_history_length]

  # Conv1D first layer in_channels must equal actor obs dim.
  encoder: torch.nn.Sequential = m.cnn_encoders["actor_history"].net  # type: ignore[assignment]
  conv0: torch.nn.Conv1d = encoder[0]  # type: ignore[assignment]
  assert conv0.in_channels == cfg.actor_history_obs_dim
  assert conv0.out_channels == cfg.cnn_output_channels[0]

  # MLP first layer in_features = obs_dim_1d + cnn_latent_dim.
  mlp: torch.nn.Sequential = m.mlp
  mlp0: torch.nn.Linear = mlp[0]  # type: ignore[assignment]
  assert mlp0.in_features == cfg.actor_obs_dim + cfg.cnn_output_channels[-1]
  assert mlp0.out_features == cfg.hidden_dims[0]

  # MLP last layer outputs num_actions.
  mlp_last: torch.nn.Linear = mlp[-1]  # type: ignore[assignment]
  assert mlp_last.out_features == cfg.num_actions


def test_load_ckpt_strict_no_missing_or_unexpected(loaded_teacher) -> None:
  """``strict=True`` already guarantees this; sanity-check the count."""
  total = sum(p.numel() for p in loaded_teacher.parameters())
  total_buf = sum(b.numel() for b in loaded_teacher.buffers())
  # ~1.06M params for the architecture.
  assert total > 1_000_000
  # Buffers include EmpiricalNormalization running stats (mean/var/std/count)
  # for both the 1-D and 3-D paths.
  assert total_buf > 0


def test_loaded_teacher_is_frozen(loaded_teacher) -> None:
  assert loaded_teacher.training is False
  for name, p in loaded_teacher.named_parameters():
    assert not p.requires_grad, f"parameter {name} is not frozen"


def test_obs_normalizer_running_stats_are_loaded(loaded_teacher) -> None:
  """Running stats must come from the ckpt, not be Identity / zeros."""
  norm_1d = loaded_teacher.obs_normalizer
  norm_3d = loaded_teacher.obs_normalizers_3d["actor_history"]

  # _mean / _var / _std should exist as buffers and not be all zeros.
  for normalizer, label in [(norm_1d, "1-D"), (norm_3d, "3-D")]:
    mean = normalizer._mean
    var = normalizer._var
    assert mean.shape == (1, TELEOPIT_TEACHER_CFG.actor_obs_dim), label
    assert var.shape == (1, TELEOPIT_TEACHER_CFG.actor_obs_dim), label
    assert mean.abs().sum() > 0, f"{label} normalizer mean is zero — not loaded?"
    # variance should be strictly positive (running_var initialized to 1).
    assert var.min() > 0, f"{label} normalizer var has non-positive entries"


# ---------------------------------------------------------------------------
# PyTorch ↔ ONNX self-consistency
# ---------------------------------------------------------------------------


def test_pytorch_forward_paths_match(loaded_teacher, dummy_batch) -> None:
  """``forward(td)``, ``mlp(get_latent(td))``, and ``as_onnx()`` should agree."""
  obs, obs_hist = dummy_batch
  td = TensorDict({"actor": obs, "actor_history": obs_hist}, batch_size=[obs.shape[0]])

  with torch.no_grad():
    out_forward = loaded_teacher(td)
    latent = loaded_teacher.get_latent(td)
    out_mlp = loaded_teacher.mlp(latent)
    onnx_wrap = loaded_teacher.as_onnx()
    out_onnx_wrap = onnx_wrap(obs, obs_hist)

  assert out_forward.shape == (obs.shape[0], TELEOPIT_TEACHER_CFG.num_actions)
  assert torch.allclose(out_forward, out_mlp, atol=1e-6)
  assert torch.allclose(out_forward, out_onnx_wrap, atol=1e-6)


def test_pytorch_matches_freshly_exported_onnx(
  loaded_teacher, dummy_batch, tmp_path: Path
) -> None:
  """Export → load → run via onnxruntime; must match PyTorch within 1e-5."""
  obs, obs_hist = dummy_batch
  cfg = TELEOPIT_TEACHER_CFG

  # 1) export
  onnx_wrap = loaded_teacher.as_onnx()
  onnx_path = tmp_path / "teacher.onnx"
  dummy_inputs = (
    torch.zeros(1, cfg.actor_obs_dim),
    torch.zeros(1, cfg.actor_history_length, cfg.actor_history_obs_dim),
  )
  torch.onnx.export(
    onnx_wrap,
    dummy_inputs,
    str(onnx_path),
    input_names=onnx_wrap.input_names,  # ["obs", "obs_history"]
    output_names=onnx_wrap.output_names,  # ["actions"]
    dynamic_axes={
      "obs": {0: "B"},
      "obs_history": {0: "B"},
      "actions": {0: "B"},
    },
    opset_version=18,
    dynamo=False,
  )

  # 2) PyTorch reference
  with torch.no_grad():
    torch_out = loaded_teacher(
      TensorDict({"actor": obs, "actor_history": obs_hist}, batch_size=[obs.shape[0]])
    )

  # 3) onnxruntime
  sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  ort_out = sess.run(
    None,
    {"obs": obs.numpy(), "obs_history": obs_hist.numpy()},
  )[0]

  np.testing.assert_allclose(
    np.asarray(torch_out.numpy()),
    np.asarray(ort_out),
    atol=1e-5,
    rtol=1e-5,
  )


# ---------------------------------------------------------------------------
# Autograd: teacher does not produce gradients, downstream modules do
# ---------------------------------------------------------------------------


def test_teacher_inference_does_not_create_grad(loaded_teacher, dummy_batch) -> None:
  obs, obs_hist = dummy_batch
  td = TensorDict({"actor": obs, "actor_history": obs_hist}, batch_size=[obs.shape[0]])
  out = loaded_teacher(td)  # no torch.no_grad() on purpose
  assert out.requires_grad is False  # all teacher params frozen


def test_downstream_module_grad_flows(loaded_teacher, dummy_batch) -> None:
  """A trainable layer fed teacher outputs must still receive grad."""
  obs, obs_hist = dummy_batch
  td = TensorDict({"actor": obs, "actor_history": obs_hist}, batch_size=[obs.shape[0]])

  head = torch.nn.Linear(TELEOPIT_TEACHER_CFG.num_actions, 4)
  out = head(loaded_teacher(td))
  loss = out.pow(2).sum()
  loss.backward()
  assert head.weight.grad is not None
  assert head.weight.grad.abs().sum() > 0
  for name, p in loaded_teacher.named_parameters():
    assert p.grad is None, f"teacher param {name} got a grad — not frozen?"
