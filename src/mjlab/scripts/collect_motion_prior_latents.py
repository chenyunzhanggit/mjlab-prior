"""Collect VQ-VAE motion-prior encoder latents from a trained checkpoint.

Rolls out the flat and rough envs in parallel, encodes each step's
``teacher_a`` / ``teacher_b`` obs through the trained encoders + shared
quantizer, and dumps per encoder:

  * ``enc``    — raw encoder output, shape (N, code_dim)
  * ``q``      — quantized code (post-codebook lookup), shape (N, code_dim)
  * ``idx``    — codebook index, shape (N,)

into a single ``.npz`` for offline t-SNE / UMAP / histogram visualization.

This script bypasses ``MotionPriorVQOnPolicyRunner`` and ``MotionPriorVQPolicy``
on purpose — neither frozen teacher is needed for latent collection, and
skipping them lets the script work even when the teacher ckpts on disk
no longer match the architecture baked into the loaders (e.g. after a
velocity-teacher architecture change). It loads only the submodules it
needs straight from the saved ``state_dict``:

  * ``encoder_a``                — MLP ([teacher_a_obs] → code_dim)
  * ``encoder_b``                — MLP ([prop, depth_latent] → code_dim)
  * ``depth_cnn``                — CNNWithProjection ([B,1,H,W] → depth_latent)
  * ``quantizer``                — EMAQuantizer (codebook + EMA buffers)
  * ``decoder``                  — MLP (used to step the rollout on-policy)

Note: after the depth migration ``encoder_b`` no longer reads teacher_b's
286-dim scandot obs; it reads ``[prop, depth_cnn(depth)]``. So the rough
env's ``teacher_b`` group is unused here, but the ``depth`` group is
required on both envs.

Usage::

  uv run python -m mjlab.scripts.collect_motion_prior_latents \\
      --checkpoint logs/.../model_XXXX.pt \\
      --motion-path /path/to/motions \\
      --num-envs 64 --num-steps 200 \\
      --out /tmp/mp_latents.npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
import tyro
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.cnn_proj import CNNWithProjection
from mjlab.tasks.motion_prior.rl.policies.quantizer import EMAQuantizer
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp.multi_commands import (
  MotionCommandCfg as MultiMotionCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends

# Mirror motion_prior_policy.py's depth defaults so a ckpt trained with the
# defaults loads with strict=True out of the box. Override on the CLI only if
# your trained policy used non-default values.
_DEFAULT_DEPTH_SHAPE: tuple[int, int, int] = (1, 60, 60)
_DEFAULT_DEPTH_LATENT_DIM = 128
_DEFAULT_DEPTH_CNN_CFG: dict = {
  "output_channels": [32, 64, 128],
  "kernel_size": [5, 3, 3],
  "stride": [2, 2, 2],
  "padding": "none",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "avg",
}


def _t(td: TensorDict, key: str) -> torch.Tensor:
  """Narrow TensorDict indexing (union return) to Tensor."""
  return cast(torch.Tensor, td[key])


@dataclass(frozen=True)
class CollectConfig:
  checkpoint: str
  """Path to ``model_XXXX.pt`` produced by the VQ motion-prior runner."""

  task_a: str = "Mjlab-MotionPrior-Flat-Unitree-G1"
  """Primary (flat / tracking) task ID — drives encoder_a."""
  task_b: str = "Mjlab-MotionPrior-Rough-Unitree-G1"
  """Secondary (rough / velocity) task ID — drives encoder_b."""

  num_envs: int = 64
  """Number of parallel envs PER task (so total batch per step = 2*num_envs)."""
  num_steps: int = 200
  """Rollout horizon. Total samples per encoder ≈ num_envs * num_steps."""
  warmup_steps: int = 10
  """Steps to roll before recording — lets RSI / initial-state transients decay."""

  motion_path: str | None = None
  """Directory of .npz motion files (recursive glob). Required when task_a
  is a multi-motion tracking task (default flat env is). Mirrors play.py."""

  # Architecture overrides — defaults match the VQ rl_cfg.py defaults. Only
  # set these if your trained policy used non-default hidden dims / code dims.
  encoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  decoder_hidden_dims: tuple[int, ...] = (512, 256, 128)
  num_code: int = 2048
  code_dim: int = 64
  activation: str = "elu"

  # Depth-encoder architecture (must match training). encoder_b now reads
  # ``[prop_obs, depth_cnn(depth)]`` instead of teacher_b_obs, so we need to
  # reproduce the CNN here to load its frozen weights from the ckpt.
  depth_shape: tuple[int, int, int] = _DEFAULT_DEPTH_SHAPE
  depth_latent_dim: int = _DEFAULT_DEPTH_LATENT_DIM

  device: str | None = None
  out: str = "/tmp/mp_latents.npz"
  seed: int | None = None


def _build_env(
  task_id: str,
  num_envs: int,
  device: str,
  motion_path: str | None = None,
) -> RslRlVecEnvWrapper:
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = num_envs

  motion_cmd = env_cfg.commands.get("motion")
  if isinstance(motion_cmd, MultiMotionCommandCfg):
    if motion_path is not None:
      motion_cmd.motion_path = motion_path
      motion_cmd.motion_files = []
    if not motion_cmd.motion_path and not motion_cmd.motion_files:
      raise ValueError(
        f"Task '{task_id}' uses MultiMotionCommandCfg but no motions are "
        "configured. Pass --motion-path /path/to/dir."
      )

  raw = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  return RslRlVecEnvWrapper(raw)


@dataclass
class _LatentModules:
  encoder_a: MLP
  encoder_b: MLP
  decoder: MLP
  quantizer: EMAQuantizer
  depth_cnn: CNNWithProjection


def _load_latent_modules(
  cfg: CollectConfig,
  device: str,
  teacher_a_obs_dim: int,
  prop_obs_dim: int,
  num_actions: int,
) -> _LatentModules:
  """Construct + load only the submodules needed for latent collection.

  Bypasses ``MotionPriorVQPolicy.__init__`` (which would also try to load
  the two frozen teacher ckpts). State-dict keys come straight from
  ``MotionPriorVQOnPolicyRunner.save`` (see runner.py:776-787).

  After the depth migration ``encoder_b`` reads ``[prop, depth_latent]``
  (not teacher_b_obs), so we also rebuild the shared ``depth_cnn`` here.
  """
  state = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
  required = {"encoder_a", "encoder_b", "decoder", "quantizer", "depth_cnn"}
  missing = required - set(state.keys())
  if missing:
    raise KeyError(
      f"Checkpoint {cfg.checkpoint} is missing required keys: {sorted(missing)}. "
      f"Top-level keys present: {sorted(state.keys())}"
    )

  cnn_channels, cnn_h, cnn_w = cfg.depth_shape
  depth_cnn = CNNWithProjection(
    input_dim=(cnn_h, cnn_w),
    input_channels=cnn_channels,
    proj_dim=cfg.depth_latent_dim,
    proj_activation=cfg.activation,
    **_DEFAULT_DEPTH_CNN_CFG,
  ).to(device)
  encoder_a = MLP(
    teacher_a_obs_dim,
    cfg.code_dim,
    hidden_dims=cfg.encoder_hidden_dims,
    activation=cfg.activation,
  ).to(device)
  encoder_b = MLP(
    prop_obs_dim + cfg.depth_latent_dim,
    cfg.code_dim,
    hidden_dims=cfg.encoder_hidden_dims,
    activation=cfg.activation,
  ).to(device)
  decoder = MLP(
    prop_obs_dim + cfg.code_dim,
    num_actions,
    hidden_dims=cfg.decoder_hidden_dims,
    activation=cfg.activation,
  ).to(device)
  quantizer = EMAQuantizer(num_code=cfg.num_code, code_dim=cfg.code_dim).to(device)

  depth_cnn.load_state_dict(state["depth_cnn"], strict=True)
  encoder_a.load_state_dict(state["encoder_a"], strict=True)
  encoder_b.load_state_dict(state["encoder_b"], strict=True)
  decoder.load_state_dict(state["decoder"], strict=True)
  quantizer.load_state_dict(state["quantizer"], strict=True)

  for m in (depth_cnn, encoder_a, encoder_b, decoder, quantizer):
    m.eval()
    for p in m.parameters():
      p.requires_grad = False

  return _LatentModules(encoder_a, encoder_b, decoder, quantizer, depth_cnn)


def main(cfg: CollectConfig) -> None:
  configure_torch_backends()
  if cfg.seed is not None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[collect] building env_a={cfg.task_a} num_envs={cfg.num_envs}")
  env_a = _build_env(cfg.task_a, cfg.num_envs, device, motion_path=cfg.motion_path)
  print(f"[collect] building env_b={cfg.task_b} num_envs={cfg.num_envs}")
  env_b = _build_env(cfg.task_b, cfg.num_envs, device)

  obs_a = env_a.get_observations()
  obs_b = env_b.get_observations()
  teacher_a_obs_dim = _t(obs_a, "teacher_a").shape[-1]
  student_dim_a = _t(obs_a, "student").shape[-1]
  student_dim_b = _t(obs_b, "student").shape[-1]
  num_actions = int(env_a.num_actions)
  if int(env_b.num_actions) != num_actions:
    raise RuntimeError(
      f"env_a / env_b num_actions disagree: {num_actions} vs {int(env_b.num_actions)}"
    )
  # encoder_b reads [prop, depth_latent] now (depth migration). Both envs
  # must expose a "depth" group; the camera is registered by env_cfgs.py
  # when use_depth=True. Bail loudly here so we don't silently misuse a
  # zero tensor at rollout time.
  for label, obs in (("env_a", obs_a), ("env_b", obs_b)):
    if "depth" not in obs.keys():
      raise RuntimeError(
        f"{label} obs is missing the 'depth' group. The depth-migrated "
        "encoder_b cannot run without it. Make sure the task was built with "
        "use_depth=True (default in env_cfgs.py)."
      )
  depth_shape_a = tuple(_t(obs_a, "depth").shape[1:])
  depth_shape_b = tuple(_t(obs_b, "depth").shape[1:])
  if depth_shape_a != depth_shape_b:
    raise RuntimeError(
      f"env_a / env_b depth shapes disagree: {depth_shape_a} vs {depth_shape_b}"
    )
  if depth_shape_a != cfg.depth_shape:
    raise RuntimeError(
      f"Env depth shape {depth_shape_a} doesn't match --depth-shape "
      f"{cfg.depth_shape}. Pass --depth-shape C H W to override."
    )

  # Recover the *training-time* prop_obs_dim from the checkpoint's
  # decoder.0.weight. The decoder is built as MLP(prop+code, action), so
  # in_features = prop_obs_dim + code_dim. This lets us detect obs-schema
  # drift between training and now without trusting either env blindly.
  raw_state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
  expected_keys = {"encoder_a", "encoder_b", "decoder", "quantizer", "depth_cnn"}
  missing = expected_keys - set(raw_state.keys())
  if missing:
    # Most common cause: user pointed at a *downstream PPO* checkpoint (which
    # saves {actor, critic, std, optimizer, ...}) instead of the upstream VQ
    # motion-prior distillation checkpoint (which saves encoder_a/b, decoder,
    # quantizer, etc.). Help them debug.
    raise RuntimeError(
      f"Checkpoint {cfg.checkpoint} does not look like a VQ motion-prior "
      f"distillation checkpoint.\n"
      f"  expected top-level keys: {sorted(expected_keys)}\n"
      f"  missing: {sorted(missing)}\n"
      f"  actually present: {sorted(raw_state.keys())}\n"
      f"\n"
      f"If you see keys like 'actor' / 'critic' / 'std', you are loading a "
      f"downstream PPO checkpoint — encoders / quantizer / decoder are not "
      f"in there. Use a checkpoint produced by "
      f"MotionPriorVQOnPolicyRunner.save (the distillation stage), typically "
      f"under logs/rsl_rl/.../<vq run>/model_XXXX.pt."
    )
  dec_w = raw_state["decoder"]["0.weight"]
  ckpt_prop_obs_dim = int(dec_w.shape[1]) - cfg.code_dim
  if ckpt_prop_obs_dim <= 0:
    raise RuntimeError(
      f"Recovered prop_obs_dim={ckpt_prop_obs_dim} <= 0 from "
      f"decoder.0.weight.shape={tuple(dec_w.shape)} with code_dim={cfg.code_dim}. "
      f"Is --code-dim correct?"
    )
  del raw_state

  agent_cfg = load_rl_cfg(cfg.task_a)
  print(
    f"[collect] obs dims: teacher_a={teacher_a_obs_dim} "
    f"student=({student_dim_a} flat, {student_dim_b} rough) "
    f"depth={depth_shape_a} "
    f"num_actions={num_actions}\n"
    f"[collect] arch: encoder_hidden={cfg.encoder_hidden_dims} "
    f"decoder_hidden={cfg.decoder_hidden_dims} "
    f"num_code={cfg.num_code} code_dim={cfg.code_dim} "
    f"depth_latent_dim={cfg.depth_latent_dim}\n"
    f"[collect] training-time prop_obs_dim (recovered from ckpt) = "
    f"{ckpt_prop_obs_dim}\n"
    f"[collect] (rl_cfg defaults for {cfg.task_a} loaded for reference: "
    f"{type(agent_cfg).__name__})"
  )

  # Decide stepping strategy per env: use the trained decoder only when the
  # env's student-obs schema still matches what the decoder was trained on.
  use_decoder_a = student_dim_a == ckpt_prop_obs_dim
  use_decoder_b = student_dim_b == ckpt_prop_obs_dim
  for label, use_dec, dim in (
    ("env_a (flat)", use_decoder_a, student_dim_a),
    ("env_b (rough)", use_decoder_b, student_dim_b),
  ):
    if use_dec:
      print(f"[collect] {label}: stepping with trained decoder (student dim ok)")
    else:
      print(
        f"[collect] {label}: student dim {dim} != ckpt {ckpt_prop_obs_dim} "
        "— stepping with zero actions. Encoder latents are still valid; "
        "the rollout distribution is just whatever the env produces under "
        "zero action."
      )

  print(f"[collect] loading latent modules from {cfg.checkpoint}")
  mods = _load_latent_modules(
    cfg,
    device,
    teacher_a_obs_dim=teacher_a_obs_dim,
    prop_obs_dim=ckpt_prop_obs_dim,
    num_actions=num_actions,
  )

  zero_action_a = torch.zeros(env_a.num_envs, num_actions, device=device)
  zero_action_b = torch.zeros(env_b.num_envs, num_actions, device=device)

  enc_a_list, q_a_list, idx_a_list = [], [], []
  enc_b_list, q_b_list, idx_b_list = [], [], []

  total = cfg.warmup_steps + cfg.num_steps
  # encoder_b takes [prop, depth_latent]: prop comes from the env's student
  # group if its dim matches the ckpt, else zeros. Either way the latent is
  # still meaningful — depth is what gives encoder_b terrain awareness.
  prop_b_zero = torch.zeros(env_b.num_envs, ckpt_prop_obs_dim, device=device)
  with torch.no_grad():
    for step in range(total):
      ta_obs = _t(obs_a, "teacher_a")
      depth_b = _t(obs_b, "depth")
      depth_latent_b = mods.depth_cnn(depth_b)
      prop_b = _t(obs_b, "student") if use_decoder_b else prop_b_zero

      enc_a = mods.encoder_a(ta_obs)  # (E, code_dim)
      enc_b = mods.encoder_b(torch.cat([prop_b, depth_latent_b], dim=-1))
      idx_a = mods.quantizer.quantize(enc_a)  # (E,)
      idx_b = mods.quantizer.quantize(enc_b)
      q_a = mods.quantizer.dequantize(idx_a)
      q_b = mods.quantizer.dequantize(idx_b)

      if step >= cfg.warmup_steps:
        enc_a_list.append(enc_a.cpu().numpy())
        q_a_list.append(q_a.cpu().numpy())
        idx_a_list.append(idx_a.cpu().numpy())
        enc_b_list.append(enc_b.cpu().numpy())
        q_b_list.append(q_b.cpu().numpy())
        idx_b_list.append(idx_b.cpu().numpy())

      # Per-env stepping: trained decoder when student-obs dim matches the
      # checkpoint, otherwise zero actions (encoder still valid; rollout
      # distribution differs from training, but that's the best we can do
      # without retraining the decoder against the new student schema).
      if use_decoder_a:
        action_a = mods.decoder(torch.cat([_t(obs_a, "student"), q_a], dim=-1))
      else:
        action_a = zero_action_a
      if use_decoder_b:
        action_b = mods.decoder(torch.cat([_t(obs_b, "student"), q_b], dim=-1))
      else:
        action_b = zero_action_b
      obs_a, *_ = env_a.step(action_a)
      obs_b, *_ = env_b.step(action_b)

      if (step + 1) % 25 == 0:
        print(f"[collect] step {step + 1}/{total}")

  enc_a_arr = np.concatenate(enc_a_list, axis=0)
  q_a_arr = np.concatenate(q_a_list, axis=0)
  idx_a_arr = np.concatenate(idx_a_list, axis=0)
  enc_b_arr = np.concatenate(enc_b_list, axis=0)
  q_b_arr = np.concatenate(q_b_list, axis=0)
  idx_b_arr = np.concatenate(idx_b_list, axis=0)

  out_path = Path(cfg.out).expanduser()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  codebook = cast(torch.Tensor, mods.quantizer.codebook).detach().cpu().numpy()
  np.savez(
    out_path,
    enc_a=enc_a_arr,
    q_a=q_a_arr,
    idx_a=idx_a_arr,
    enc_b=enc_b_arr,
    q_b=q_b_arr,
    idx_b=idx_b_arr,
    codebook=codebook,
  )
  print(
    f"[collect] saved {out_path}\n"
    f"  enc_a {enc_a_arr.shape}  enc_b {enc_b_arr.shape}\n"
    f"  q_a   {q_a_arr.shape}    q_b   {q_b_arr.shape}\n"
    f"  idx_a {idx_a_arr.shape}  idx_b {idx_b_arr.shape}\n"
    f"  codebook {codebook.shape}"
  )


if __name__ == "__main__":
  main(tyro.cli(CollectConfig))
