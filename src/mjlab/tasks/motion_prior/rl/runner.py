"""Motion-prior distillation runners.

The runner orchestrates a **dual** rollout: one ``ManagerBasedRlEnv`` runs
the flat / motion-tracking env (teacher_a's home) and a second one runs the
rough / velocity env (teacher_b's home). A single ``MotionPriorPolicy`` is
evaluated on both, and the loss combines per-teacher behavior MSE plus
shared AR(1) and KL regularizers (see ``DistillationMotionPrior``).

Train-script contract: ``train.py`` registers the **primary (flat)** env
and constructs the runner with it. The runner reads the rough env's task
ID from ``train_cfg`` and instantiates the second env internally.

VQ variant is a stub that subclasses the VAE runner — its body lands in
prior.md task #12.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.motion_prior.rl.algorithms import DistillationMotionPrior
from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior import (
  DistillationLossCfg,
)
from mjlab.tasks.motion_prior.rl.policies import MotionPriorPolicy
from mjlab.tasks.registry import load_env_cfg


def _build_secondary_env(
  task_id: str,
  num_envs: int,
  device: str,
) -> RslRlVecEnvWrapper:
  """Create the second ``ManagerBasedRlEnv`` from a registered task ID.

  ``num_envs`` overrides the registered scene cfg so the two envs can be
  sized independently.
  """
  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = num_envs
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  return RslRlVecEnvWrapper(raw_env)


def _t(td: TensorDict, key: str) -> torch.Tensor:
  """Type-narrowing accessor: TensorDict indexing returns a union, but
  obs groups in this pipeline are always plain Tensors."""
  return cast(torch.Tensor, td[key])


class MotionPriorOnPolicyRunner:
  """Dual-env VAE motion-prior distillation runner."""

  env: RslRlVecEnvWrapper
  alg: DistillationMotionPrior

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(
        f"MotionPriorOnPolicyRunner got unexpected kwargs (ignored): {list(kwargs)}"
      )
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.current_learning_iteration: int = 0

    # ----- Primary env (flat / motion-tracking, hosts teacher_a obs) -----
    # Duck-typed: anything that quacks like ``RslRlVecEnvWrapper``
    # (``get_observations``, ``step``, ``num_envs``, ``episode_length_buf``)
    # works — that lets tests substitute a fake env without GPU.
    self.env = env  # type: ignore[assignment]

    # ----- Secondary env (rough / velocity, hosts teacher_b obs) ----------
    secondary_task_id = train_cfg.get(
      "secondary_task_id", "Mjlab-MotionPrior-Rough-Unitree-G1"
    )
    secondary_num_envs = int(train_cfg.get("secondary_num_envs", self.env.num_envs))
    print(
      f"[MotionPrior] building secondary env '{secondary_task_id}' "
      f"with num_envs={secondary_num_envs} on {device}"
    )
    self.env_b = _build_secondary_env(secondary_task_id, secondary_num_envs, device)

    # ----- Policy ---------------------------------------------------------
    teacher_a_path = train_cfg["teacher_a_policy_path"]
    teacher_b_path = train_cfg["teacher_b_policy_path"]
    student_obs_dim = _t(self.env.get_observations(), "student").shape[-1]
    num_actions = int(self.env.num_actions)
    policy_cfg = train_cfg.get("policy", {})

    self.policy = MotionPriorPolicy(
      prop_obs_dim=student_obs_dim,
      num_actions=num_actions,
      teacher_a_policy_path=teacher_a_path,
      teacher_b_policy_path=teacher_b_path,
      encoder_hidden_dims=tuple(policy_cfg.get("encoder_hidden_dims", (512, 256, 128))),
      decoder_hidden_dims=tuple(policy_cfg.get("decoder_hidden_dims", (512, 256, 128))),
      motion_prior_hidden_dims=tuple(
        policy_cfg.get("motion_prior_hidden_dims", (512, 256, 128))
      ),
      latent_z_dims=int(policy_cfg.get("latent_z_dims", 32)),
      activation=str(policy_cfg.get("activation", "elu")),
      device=device,
    )
    self._latent_z_dims = self.policy.latent_z_dims

    # ----- Algorithm ------------------------------------------------------
    algo_cfg = train_cfg.get("algorithm", {})
    loss_cfg = DistillationLossCfg(
      loss_type=str(algo_cfg.get("loss_type", "mse")),
      behavior_weight_a=float(algo_cfg.get("behavior_weight_a", 1.0)),
      behavior_weight_b=float(algo_cfg.get("behavior_weight_b", 1.0)),
      mu_regu_loss_coeff=float(algo_cfg.get("mu_regu_loss_coeff", 0.01)),
      ar1_phi=float(algo_cfg.get("ar1_phi", 0.99)),
      kl_loss_coeff_max=float(algo_cfg.get("kl_loss_coeff_max", 0.01)),
      kl_loss_coeff_min=float(algo_cfg.get("kl_loss_coeff_min", 0.001)),
      anneal_start_iter=int(algo_cfg.get("anneal_start_iter", 2500)),
      anneal_end_iter=int(algo_cfg.get("anneal_end_iter", 5000)),
      align_loss_coeff=float(algo_cfg.get("align_loss_coeff", 0.0)),
    )
    self.alg = DistillationMotionPrior(
      self.policy,
      learning_rate=float(algo_cfg.get("learning_rate", 5e-4)),
      max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
      loss_cfg=loss_cfg,
      device=device,
    )
    self.num_learning_epochs = int(algo_cfg.get("num_learning_epochs", 5))

    # ----- Rollout knobs --------------------------------------------------
    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.save_interval = int(train_cfg.get("save_interval", 500))
    self.upload_model = bool(train_cfg.get("upload_model", False))

    # ----- Optional SummaryWriter ----------------------------------------
    self._writer = None
    if log_dir is not None and train_cfg.get("logger", "tensorboard") == "tensorboard":
      try:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
      except Exception as e:
        print(f"[MotionPrior] tensorboard unavailable, skipping ({e})")

    # ----- Episode reward / length tracking (per-env) --------------------
    self._rew_buf_a: deque[float] = deque(maxlen=100)
    self._len_buf_a: deque[float] = deque(maxlen=100)
    self._rew_buf_b: deque[float] = deque(maxlen=100)
    self._len_buf_b: deque[float] = deque(maxlen=100)
    self._cur_rew_a = torch.zeros(self.env.num_envs, device=device)
    self._cur_len_a = torch.zeros(self.env.num_envs, device=device)
    self._cur_rew_b = torch.zeros(self.env_b.num_envs, device=device)
    self._cur_len_b = torch.zeros(self.env_b.num_envs, device=device)

  # --------------------------------------------------------------------- #
  # train.py interface                                                    #
  # --------------------------------------------------------------------- #

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    # Logger-feature shim: we don't track git status, but train.py calls this.
    pass

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    del init_at_random_ep_len  # both envs already RSI-randomize on reset.
    obs_a = self.env.get_observations()
    obs_b = self.env_b.get_observations()

    start_it = self.current_learning_iteration
    end_it = start_it + num_learning_iterations
    for it in range(start_it, end_it):
      t0 = time.time()
      rollout, obs_a, obs_b = self._collect_rollout(obs_a, obs_b)
      collect_t = time.time() - t0

      t1 = time.time()
      loss_dict: dict[str, float] = {}
      for _ in range(self.num_learning_epochs):
        loss_dict = self._epoch_step(rollout, cur_iter_num=it)
      learn_t = time.time() - t1

      self.current_learning_iteration = it
      self._log_iteration(it, loss_dict, collect_t, learn_t)

      if self.log_dir is not None and it % self.save_interval == 0:
        self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

    if self.log_dir is not None:
      self.save(
        os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
    if self._writer is not None:
      self._writer.close()

  # --------------------------------------------------------------------- #
  # Rollout                                                               #
  # --------------------------------------------------------------------- #

  def _collect_rollout(
    self, obs_a: TensorDict, obs_b: TensorDict
  ) -> tuple[dict[str, torch.Tensor], TensorDict, TensorDict]:
    """Roll out ``num_steps_per_env`` transitions on both envs.

    Stores **only detached inputs** (student / teacher obs) and frozen
    teacher targets, so the per-epoch loop in ``learn()`` can re-forward
    the trainable submodules and rebuild a fresh autograd graph every
    pass without ``retain_graph=True``.

    Stepping uses the policy under ``torch.no_grad`` to keep rollouts
    cheap and avoid retaining a graph that would be discarded anyway.
    """
    T = self.num_steps_per_env

    # Per-step buffers — every entry has shape (n_envs, ...).
    prop_a, prop_b = [], []
    ta_obs, ta_hist, tb_obs = [], [], []
    ta_act, tb_act = [], []
    prog_a, prog_b = [], []

    for _ in range(T):
      with torch.no_grad():
        # Sample student actions for stepping; gradients are reconstructed
        # in the per-epoch re-forward.
        _, _, _, sa_a_step, _, _ = self.policy.forward_a(
          _t(obs_a, "student"), _t(obs_a, "teacher_a")
        )
        _, _, _, sa_b_step, _, _ = self.policy.forward_b(
          _t(obs_b, "student"), _t(obs_b, "teacher_b")
        )
        # Frozen teacher targets — never need grad.
        teacher_a_action = self.policy.evaluate_a(
          _t(obs_a, "teacher_a"), _t(obs_a, "teacher_a_history")
        )
        teacher_b_action = self.policy.evaluate_b(_t(obs_b, "teacher_b"))

      # Detached input snapshots BEFORE stepping the env.
      prop_a.append(_t(obs_a, "student").detach().clone())
      ta_obs.append(_t(obs_a, "teacher_a").detach().clone())
      ta_hist.append(_t(obs_a, "teacher_a_history").detach().clone())
      ta_act.append(teacher_a_action.detach().clone())

      prop_b.append(_t(obs_b, "student").detach().clone())
      tb_obs.append(_t(obs_b, "teacher_b").detach().clone())
      tb_act.append(teacher_b_action.detach().clone())

      prog_a.append(self.env.episode_length_buf.detach().clone().to(self.device))
      prog_b.append(self.env_b.episode_length_buf.detach().clone().to(self.device))

      # Step both envs with detached student actions.
      obs_a, rew_a, done_a, _ = self.env.step(sa_a_step.detach())
      obs_b, rew_b, done_b, _ = self.env_b.step(sa_b_step.detach())
      self._track_episode_stats(rew_a, done_a, rew_b, done_b)

    # ---- stack to (n_envs, T, ...) and flatten to (n_envs * T, ...) ------
    def _flat(seq: list[torch.Tensor]) -> torch.Tensor:
      # stack(dim=1) → (n_envs, T, *), flatten(0,1) → (n_envs*T, *).
      return torch.stack(seq, dim=1).flatten(0, 1)

    rollout = dict(
      prop_obs_a=_flat(prop_a),
      teacher_a_obs=_flat(ta_obs),
      teacher_a_history_obs=_flat(ta_hist),
      actions_teacher_a=_flat(ta_act),
      prop_obs_b=_flat(prop_b),
      teacher_b_obs=_flat(tb_obs),
      actions_teacher_b=_flat(tb_act),
      progress_buf_a=torch.stack(prog_a, dim=1).unsqueeze(-1).float(),
      progress_buf_b=torch.stack(prog_b, dim=1).unsqueeze(-1).float(),
    )
    return rollout, obs_a, obs_b

  def _epoch_step(
    self,
    rollout: dict[str, torch.Tensor],
    cur_iter_num: int,
  ) -> dict[str, float]:
    """Re-forward trainable modules on stored detached inputs and apply
    one optimization step. Run once per ``num_learning_epochs``."""
    n_envs_a = self.env.num_envs
    n_envs_b = self.env_b.num_envs
    T = self.num_steps_per_env
    Z = self._latent_z_dims

    enc_mu_a, enc_lv_a, _, sa_a, mp_mu_a, mp_lv_a = self.policy.forward_a(
      rollout["prop_obs_a"], rollout["teacher_a_obs"]
    )
    enc_mu_b, enc_lv_b, _, sa_b, mp_mu_b, mp_lv_b = self.policy.forward_b(
      rollout["prop_obs_b"], rollout["teacher_b_obs"]
    )

    # Reshape (n_envs*T, Z) → (n_envs, T, Z) for AR(1).
    enc_mu_a_t = enc_mu_a.view(n_envs_a, T, Z)
    enc_mu_b_t = enc_mu_b.view(n_envs_b, T, Z)

    return self.alg.compute_loss_one_batch(
      actions_teacher_a=rollout["actions_teacher_a"],
      actions_student_a=sa_a,
      enc_mu_a=enc_mu_a,
      enc_log_var_a=enc_lv_a,
      mp_mu_a=mp_mu_a,
      mp_log_var_a=mp_lv_a,
      enc_mu_a_time_stack=enc_mu_a_t,
      progress_buf_a=rollout["progress_buf_a"],
      actions_teacher_b=rollout["actions_teacher_b"],
      actions_student_b=sa_b,
      enc_mu_b=enc_mu_b,
      enc_log_var_b=enc_lv_b,
      mp_mu_b=mp_mu_b,
      mp_log_var_b=mp_lv_b,
      enc_mu_b_time_stack=enc_mu_b_t,
      progress_buf_b=rollout["progress_buf_b"],
      cur_iter_num=cur_iter_num,
    )

  def _track_episode_stats(
    self,
    rew_a: torch.Tensor,
    done_a: torch.Tensor,
    rew_b: torch.Tensor,
    done_b: torch.Tensor,
  ) -> None:
    self._cur_rew_a += rew_a.float()
    self._cur_len_a += 1
    self._cur_rew_b += rew_b.float()
    self._cur_len_b += 1
    for cur_rew, cur_len, done, rbuf, lbuf in (
      (self._cur_rew_a, self._cur_len_a, done_a, self._rew_buf_a, self._len_buf_a),
      (self._cur_rew_b, self._cur_len_b, done_b, self._rew_buf_b, self._len_buf_b),
    ):
      done_idx = done.nonzero(as_tuple=False).squeeze(-1)
      if done_idx.numel() > 0:
        rbuf.extend(cur_rew[done_idx].tolist())
        lbuf.extend(cur_len[done_idx].tolist())
        cur_rew[done_idx] = 0.0
        cur_len[done_idx] = 0.0

  # --------------------------------------------------------------------- #
  # Logging                                                               #
  # --------------------------------------------------------------------- #

  def _log_iteration(
    self,
    it: int,
    loss_dict: dict[str, float],
    collect_t: float,
    learn_t: float,
  ) -> None:
    rew_a = statistics.fmean(self._rew_buf_a) if self._rew_buf_a else 0.0
    rew_b = statistics.fmean(self._rew_buf_b) if self._rew_buf_b else 0.0
    len_a = statistics.fmean(self._len_buf_a) if self._len_buf_a else 0.0
    len_b = statistics.fmean(self._len_buf_b) if self._len_buf_b else 0.0
    print(
      f"[it {it}] "
      f"behavior_a={loss_dict.get('loss/behavior_a', 0):.4f} "
      f"behavior_b={loss_dict.get('loss/behavior_b', 0):.4f} "
      f"kl_a={loss_dict.get('loss/kl_a', 0):.4f} "
      f"kl_b={loss_dict.get('loss/kl_b', 0):.4f} "
      f"ar1_a={loss_dict.get('loss/ar1_a', 0):.4f} "
      f"ar1_b={loss_dict.get('loss/ar1_b', 0):.4f} "
      f"rew_a={rew_a:.2f} rew_b={rew_b:.2f} "
      f"len_a={len_a:.0f} len_b={len_b:.0f} "
      f"t_collect={collect_t:.2f}s t_learn={learn_t:.2f}s"
    )
    if self._writer is not None:
      for k, v in loss_dict.items():
        self._writer.add_scalar(k, v, it)
      self._writer.add_scalar("episode/reward_a", rew_a, it)
      self._writer.add_scalar("episode/reward_b", rew_b, it)
      self._writer.add_scalar("episode/length_a", len_a, it)
      self._writer.add_scalar("episode/length_b", len_b, it)
      self._writer.add_scalar("time/collect", collect_t, it)
      self._writer.add_scalar("time/learn", learn_t, it)

  # --------------------------------------------------------------------- #
  # Save / load                                                           #
  # --------------------------------------------------------------------- #

  def save(self, path: str, infos: dict | None = None) -> None:
    """Persist policy + optimizer + iter. Frozen teachers are NOT saved
    (they are reloaded from their ckpt paths on construction).

    Also dumps a deploy-ready ``policy.onnx`` next to the checkpoint
    (Path 3: ``prop_obs → motion_prior → decoder → action``); ONNX
    failures are logged but do not break training.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
      # Slice trainable submodules; the two frozen teachers re-load from disk.
      "encoder_a": self.policy.encoder_a.state_dict(),
      "encoder_b": self.policy.encoder_b.state_dict(),
      "es_a_mu": self.policy.es_a_mu.state_dict(),
      "es_a_var": self.policy.es_a_var.state_dict(),
      "es_b_mu": self.policy.es_b_mu.state_dict(),
      "es_b_var": self.policy.es_b_var.state_dict(),
      "decoder": self.policy.decoder.state_dict(),
      "motion_prior": self.policy.motion_prior.state_dict(),
      "mp_mu": self.policy.mp_mu.state_dict(),
      "mp_var": self.policy.mp_var.state_dict(),
      "optimizer": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos or {},
    }
    torch.save(state, path)
    print(f"[MotionPrior] saved checkpoint to {path}")
    try:
      onnx_path = Path(path).with_suffix(".onnx")
      self.export_policy_to_onnx(str(onnx_path))
    except Exception as e:
      print(f"[MotionPrior] ONNX export failed (training continues): {e}")

  def export_policy_to_onnx(
    self, path: str, filename: str | None = None, verbose: bool = False
  ) -> None:
    """Export Path 3 (prop_obs → motion_prior → decoder → action) to ONNX."""
    from mjlab.tasks.motion_prior.onnx import export_motion_prior_to_onnx

    output = Path(path)
    if filename is not None:
      output = output / filename
    export_motion_prior_to_onnx(self.policy, output, verbose=verbose)

  def get_inference_policy(self, device: str | torch.device | None = None):
    """Return a callable ``(obs_td) -> action`` that runs Path 3 deploy.

    Used by ``mjlab.scripts.play``: takes the env's TensorDict, slices the
    ``student`` group (proprioception only), and runs the trained
    motion_prior + decoder under ``torch.no_grad``. teacher_obs groups
    are ignored — at deploy time they aren't available.
    """
    from mjlab.tasks.motion_prior.onnx import build_deploy_model

    deploy = build_deploy_model(self.policy)
    if device is not None:
      deploy = deploy.to(device)
    deploy.eval()

    def _policy(obs_td) -> torch.Tensor:
      prop = _t(obs_td, "student")
      with torch.no_grad():
        return deploy(prop)

    return _policy

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    self.policy.encoder_a.load_state_dict(state["encoder_a"], strict=strict)
    self.policy.encoder_b.load_state_dict(state["encoder_b"], strict=strict)
    self.policy.es_a_mu.load_state_dict(state["es_a_mu"], strict=strict)
    self.policy.es_a_var.load_state_dict(state["es_a_var"], strict=strict)
    self.policy.es_b_mu.load_state_dict(state["es_b_mu"], strict=strict)
    self.policy.es_b_var.load_state_dict(state["es_b_var"], strict=strict)
    self.policy.decoder.load_state_dict(state["decoder"], strict=strict)
    self.policy.motion_prior.load_state_dict(state["motion_prior"], strict=strict)
    self.policy.mp_mu.load_state_dict(state["mp_mu"], strict=strict)
    self.policy.mp_var.load_state_dict(state["mp_var"], strict=strict)
    if "optimizer" in state and state["optimizer"]:
      self.alg.optimizer.load_state_dict(state["optimizer"])
    self.current_learning_iteration = int(state.get("iter", 0))
    return state.get("infos", {})


class MotionPriorVQOnPolicyRunner(MotionPriorOnPolicyRunner):
  """VQ-VAE motion-prior runner — body lands in prior.md task #12."""
