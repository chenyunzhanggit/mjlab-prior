"""Single-teacher motion-prior runners (VAE + VQ).

Counterpart of :class:`MotionPriorOnPolicyRunner` (dual env) for the
single-encoder student. Only one ``ManagerBasedRlEnv`` is rolled out;
the rough secondary env / dual-stream rollout / per-stream stats are
not built. The training loop mirrors the dual implementation:

* Rollout under ``torch.no_grad`` storing only detached inputs +
  frozen-teacher targets.
* Per-epoch re-forward of trainable submodules so each
  ``num_learning_epochs`` pass starts with a fresh autograd graph.

ONNX export uses the **deploy path** (``prop → motion_prior → decoder``);
the encoder + teacher path is also available via
``runner.get_inference_policy(path="encoder")``.
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

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior_single import (
  DistillationMotionPriorSingle,
  DistillationSingleLossCfg,
)
from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior_single_vq import (
  DistillationMotionPriorSingleVQ,
  DistillationSingleVQLossCfg,
)
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_policy import (
  MotionPriorSinglePolicy,
)
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_vq_policy import (
  MotionPriorSingleVQPolicy,
)


def _t(td: TensorDict, key: str) -> torch.Tensor:
  return cast(torch.Tensor, td[key])


class MotionPriorSingleOnPolicyRunner:
  """Single-env, single-teacher VAE motion-prior distillation runner."""

  env: RslRlVecEnvWrapper
  policy: MotionPriorSinglePolicy
  alg: DistillationMotionPriorSingle

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
        f"MotionPriorSingleOnPolicyRunner got unexpected kwargs (ignored): "
        f"{list(kwargs)}"
      )
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.current_learning_iteration: int = 0

    self.env = env  # type: ignore[assignment]

    obs0 = self.env.get_observations()
    student_obs_dim = _t(obs0, "student").shape[-1]
    teacher_obs_dim = _t(obs0, "teacher_tracking").shape[-1]
    num_actions = int(self.env.num_actions)

    self._build_policy_and_alg(
      train_cfg=train_cfg,
      student_obs_dim=student_obs_dim,
      teacher_obs_dim=teacher_obs_dim,
      num_actions=num_actions,
      device=device,
    )

    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.save_interval = int(train_cfg.get("save_interval", 500))
    self.upload_model = bool(train_cfg.get("upload_model", False))

    self._writer = None
    if log_dir is not None and train_cfg.get("logger", "tensorboard") == "tensorboard":
      try:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
      except Exception as e:
        print(f"[MotionPriorSingle] tensorboard unavailable, skipping ({e})")

    self._rew_buf: deque[float] = deque(maxlen=100)
    self._len_buf: deque[float] = deque(maxlen=100)
    self._cur_rew = torch.zeros(self.env.num_envs, device=device)
    self._cur_len = torch.zeros(self.env.num_envs, device=device)

  # --------------------------------------------------------------------- #
  # Policy / algorithm construction (overridable by VQ subclass)          #
  # --------------------------------------------------------------------- #

  def _build_policy_and_alg(
    self,
    *,
    train_cfg: dict,
    student_obs_dim: int,
    teacher_obs_dim: int,
    num_actions: int,
    device: str,
  ) -> None:
    teacher_path = train_cfg["teacher_policy_path"]
    policy_cfg = train_cfg.get("policy", {})

    self.policy = MotionPriorSinglePolicy(
      prop_obs_dim=student_obs_dim,
      teacher_obs_dim=teacher_obs_dim,
      num_actions=num_actions,
      teacher_policy_path=teacher_path,
      encoder_hidden_dims=tuple(policy_cfg.get("encoder_hidden_dims", (512, 256, 128))),
      decoder_hidden_dims=tuple(policy_cfg.get("decoder_hidden_dims", (512, 256, 128))),
      motion_prior_hidden_dims=tuple(
        policy_cfg.get("motion_prior_hidden_dims", (512, 256, 128))
      ),
      teacher_hidden_dims=tuple(policy_cfg.get("teacher_hidden_dims", (512, 256, 128))),
      teacher_activation=str(policy_cfg.get("teacher_activation", "elu")),
      latent_z_dims=int(policy_cfg.get("latent_z_dims", 32)),
      activation=str(policy_cfg.get("activation", "elu")),
      device=device,
    )
    self._latent_z_dims = self.policy.latent_z_dims

    algo_cfg = train_cfg.get("algorithm", {})
    loss_cfg = DistillationSingleLossCfg(
      loss_type=str(algo_cfg.get("loss_type", "mse")),
      behavior_weight=float(algo_cfg.get("behavior_weight", 1.0)),
      mu_regu_loss_coeff=float(algo_cfg.get("mu_regu_loss_coeff", 0.01)),
      ar1_phi=float(algo_cfg.get("ar1_phi", 0.99)),
      kl_loss_coeff_max=float(algo_cfg.get("kl_loss_coeff_max", 0.01)),
      kl_loss_coeff_min=float(algo_cfg.get("kl_loss_coeff_min", 0.001)),
      anneal_start_iter=int(algo_cfg.get("anneal_start_iter", 2500)),
      anneal_end_iter=int(algo_cfg.get("anneal_end_iter", 5000)),
    )
    self.alg = DistillationMotionPriorSingle(
      self.policy,
      learning_rate=float(algo_cfg.get("learning_rate", 5e-4)),
      max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
      loss_cfg=loss_cfg,
      device=device,
    )
    self.num_learning_epochs = int(algo_cfg.get("num_learning_epochs", 5))

  # --------------------------------------------------------------------- #
  # train.py interface                                                    #
  # --------------------------------------------------------------------- #

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    pass

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    del init_at_random_ep_len
    obs = self.env.get_observations()

    start_it = self.current_learning_iteration
    end_it = start_it + num_learning_iterations
    for it in range(start_it, end_it):
      t0 = time.time()
      rollout, obs = self._collect_rollout(obs)
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

  def _policy_step_actions(self, obs: TensorDict) -> torch.Tensor:
    """Return ``student_action`` for env stepping (VAE: 4th of 6 outputs)."""
    _, _, _, sa_step, _, _ = self.policy(
      _t(obs, "student"), _t(obs, "teacher_tracking")
    )
    return sa_step

  def _collect_rollout(
    self, obs: TensorDict
  ) -> tuple[dict[str, torch.Tensor], TensorDict]:
    T = self.num_steps_per_env

    prop, ta_obs, ta_act, prog = [], [], [], []
    for _ in range(T):
      with torch.no_grad():
        sa_step = self._policy_step_actions(obs)
        teacher_action = self.policy.evaluate(_t(obs, "teacher_tracking"))

      prop.append(_t(obs, "student").detach().clone())
      ta_obs.append(_t(obs, "teacher_tracking").detach().clone())
      ta_act.append(teacher_action.detach().clone())
      prog.append(self.env.episode_length_buf.detach().clone().to(self.device))

      obs, rew, done, _ = self.env.step(sa_step.detach())
      self._track_episode_stats(rew, done)

    def _flat(seq: list[torch.Tensor]) -> torch.Tensor:
      return torch.stack(seq, dim=1).flatten(0, 1)

    rollout = dict(
      prop_obs=_flat(prop),
      teacher_obs=_flat(ta_obs),
      actions_teacher=_flat(ta_act),
      progress_buf=torch.stack(prog, dim=1).unsqueeze(-1).float(),
    )
    return rollout, obs

  def _epoch_step(
    self, rollout: dict[str, torch.Tensor], cur_iter_num: int
  ) -> dict[str, float]:
    n_envs = self.env.num_envs
    T = self.num_steps_per_env
    Z = self._latent_z_dims

    enc_mu, enc_lv, _, sa, mp_mu, mp_lv = self.policy(
      rollout["prop_obs"], rollout["teacher_obs"]
    )
    enc_mu_t = enc_mu.view(n_envs, T, Z)

    return self.alg.compute_loss_one_batch(
      actions_teacher=rollout["actions_teacher"],
      actions_student=sa,
      enc_mu=enc_mu,
      enc_log_var=enc_lv,
      mp_mu=mp_mu,
      mp_log_var=mp_lv,
      enc_mu_time_stack=enc_mu_t,
      progress_buf=rollout["progress_buf"],
      cur_iter_num=cur_iter_num,
    )

  def _track_episode_stats(self, rew: torch.Tensor, done: torch.Tensor) -> None:
    self._cur_rew += rew.float()
    self._cur_len += 1
    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
    if done_idx.numel() > 0:
      self._rew_buf.extend(self._cur_rew[done_idx].tolist())
      self._len_buf.extend(self._cur_len[done_idx].tolist())
      self._cur_rew[done_idx] = 0.0
      self._cur_len[done_idx] = 0.0

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
    rew = statistics.fmean(self._rew_buf) if self._rew_buf else 0.0
    ep_len = statistics.fmean(self._len_buf) if self._len_buf else 0.0
    print(
      f"[it {it}] "
      f"behavior={loss_dict.get('loss/behavior', 0):.4f} "
      f"kl={loss_dict.get('loss/kl', 0):.4f} "
      f"ar1={loss_dict.get('loss/ar1', 0):.4f} "
      f"rew={rew:.2f} len={ep_len:.0f} "
      f"t_collect={collect_t:.2f}s t_learn={learn_t:.2f}s"
    )
    if self._writer is not None:
      for k, v in loss_dict.items():
        self._writer.add_scalar(k, v, it)
      self._writer.add_scalar("episode/reward", rew, it)
      self._writer.add_scalar("episode/length", ep_len, it)
      self._writer.add_scalar("time/collect", collect_t, it)
      self._writer.add_scalar("time/learn", learn_t, it)

  # --------------------------------------------------------------------- #
  # Save / load / inference                                               #
  # --------------------------------------------------------------------- #

  def save(self, path: str, infos: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
      "encoder": self.policy.encoder.state_dict(),
      "es_mu": self.policy.es_mu.state_dict(),
      "es_var": self.policy.es_var.state_dict(),
      "decoder": self.policy.decoder.state_dict(),
      "motion_prior": self.policy.motion_prior.state_dict(),
      "mp_mu": self.policy.mp_mu.state_dict(),
      "mp_var": self.policy.mp_var.state_dict(),
      "optimizer": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos or {},
    }
    torch.save(state, path)
    print(f"[MotionPriorSingle] saved checkpoint to {path}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    self.policy.encoder.load_state_dict(state["encoder"], strict=strict)
    self.policy.es_mu.load_state_dict(state["es_mu"], strict=strict)
    self.policy.es_var.load_state_dict(state["es_var"], strict=strict)
    self.policy.decoder.load_state_dict(state["decoder"], strict=strict)
    self.policy.motion_prior.load_state_dict(state["motion_prior"], strict=strict)
    self.policy.mp_mu.load_state_dict(state["mp_mu"], strict=strict)
    self.policy.mp_var.load_state_dict(state["mp_var"], strict=strict)
    if "optimizer" in state and state["optimizer"]:
      self.alg.optimizer.load_state_dict(state["optimizer"])
    self.current_learning_iteration = int(state.get("iter", 0))
    return state.get("infos", {})

  def get_inference_policy(
    self,
    device: str | torch.device | None = None,
    path: str | None = None,
  ):
    """Return a callable ``(obs_td) -> action`` for play mode.

    Two paths are supported:
    * ``"encoder"`` — ``encoder(teacher_obs) → enc_mu → decoder([prop, enc_mu])``
    * ``"deploy"`` (default) — ``motion_prior(prop) → mp_mu → decoder([prop, mp_mu])``
    """
    if path is None:
      env_path = os.environ.get("MJLAB_MP_INFERENCE_PATH", "").strip().lower()
      if env_path and env_path != "auto":
        path = env_path
    chosen_path = path or "deploy"
    if chosen_path not in {"encoder", "deploy"}:
      raise ValueError(
        f"Unknown inference path: {chosen_path!r}. Expected 'encoder' or 'deploy'."
      )

    def _policy(obs_td) -> torch.Tensor:
      prop = _t(obs_td, "student")
      with torch.no_grad():
        if chosen_path == "encoder":
          return self.policy.policy_inference(prop, _t(obs_td, "teacher_tracking"))
        mp_mu = self.policy.motion_prior_inference(prop)
        return self.policy.decoder_inference(prop, mp_mu)

    return _policy


class MotionPriorSingleVQOnPolicyRunner(MotionPriorSingleOnPolicyRunner):
  """Single-teacher VQ motion-prior runner (subclass of the VAE runner)."""

  policy: MotionPriorSingleVQPolicy  # type: ignore[assignment]
  alg: DistillationMotionPriorSingleVQ  # type: ignore[assignment]

  def _build_policy_and_alg(
    self,
    *,
    train_cfg: dict,
    student_obs_dim: int,
    teacher_obs_dim: int,
    num_actions: int,
    device: str,
  ) -> None:
    teacher_path = train_cfg["teacher_policy_path"]
    policy_cfg = train_cfg.get("policy", {})

    self.policy = MotionPriorSingleVQPolicy(
      prop_obs_dim=student_obs_dim,
      teacher_obs_dim=teacher_obs_dim,
      num_actions=num_actions,
      teacher_policy_path=teacher_path,
      encoder_hidden_dims=tuple(policy_cfg.get("encoder_hidden_dims", (512, 256, 128))),
      decoder_hidden_dims=tuple(policy_cfg.get("decoder_hidden_dims", (512, 256, 128))),
      motion_prior_hidden_dims=tuple(
        policy_cfg.get("motion_prior_hidden_dims", (512, 256, 128))
      ),
      teacher_hidden_dims=tuple(policy_cfg.get("teacher_hidden_dims", (512, 256, 128))),
      teacher_activation=str(policy_cfg.get("teacher_activation", "elu")),
      num_code=int(policy_cfg.get("num_code", 2048)),
      code_dim=int(policy_cfg.get("code_dim", 64)),
      ema_decay=float(policy_cfg.get("ema_decay", 0.99)),
      activation=str(policy_cfg.get("activation", "elu")),
      device=device,
    )
    # AR(1) reshape uses code_dim for VQ.
    self._latent_z_dims = int(policy_cfg.get("code_dim", 64))

    algo_cfg = train_cfg.get("algorithm", {})
    loss_cfg = DistillationSingleVQLossCfg(
      loss_type=str(algo_cfg.get("loss_type", "mse")),
      behavior_weight=float(algo_cfg.get("behavior_weight", 1.0)),
      mu_regu_loss_coeff=float(algo_cfg.get("mu_regu_loss_coeff", 0.0)),
      ar1_phi=float(algo_cfg.get("ar1_phi", 0.99)),
      commit_loss_coeff=float(algo_cfg.get("commit_loss_coeff", 1.0)),
      mp_loss_coeff=float(algo_cfg.get("mp_loss_coeff", 1.0)),
    )
    self.alg = DistillationMotionPriorSingleVQ(
      self.policy,
      learning_rate=float(algo_cfg.get("learning_rate", 1e-3)),
      max_grad_norm=algo_cfg.get("max_grad_norm", None),
      loss_cfg=loss_cfg,
      device=device,
    )
    self.num_learning_epochs = int(algo_cfg.get("num_learning_epochs", 5))

  def _policy_step_actions(self, obs: TensorDict) -> torch.Tensor:
    """VQ ``forward`` returns ``(student_act, q, enc, mp_code, commit, perp)``."""
    sa_step, *_ = self.policy(
      _t(obs, "student"), _t(obs, "teacher_tracking"), training=False
    )
    return sa_step

  def _epoch_step(
    self, rollout: dict[str, torch.Tensor], cur_iter_num: int
  ) -> dict[str, float]:
    n_envs = self.env.num_envs
    T = self.num_steps_per_env
    Z = self._latent_z_dims

    sa, q, enc, mp_code, commit, perplexity = self.policy(
      rollout["prop_obs"], rollout["teacher_obs"], training=True
    )
    assert commit is not None
    enc_t = enc.view(n_envs, T, Z)

    return self.alg.compute_loss_one_batch(
      actions_teacher=rollout["actions_teacher"],
      actions_student=sa,
      enc=enc,
      q=q,
      mp_code=mp_code,
      commit=commit,
      perplexity=perplexity,
      enc_time_stack=enc_t,
      progress_buf=rollout["progress_buf"],
      cur_iter_num=cur_iter_num,
    )

  def _log_iteration(
    self,
    it: int,
    loss_dict: dict[str, float],
    collect_t: float,
    learn_t: float,
  ) -> None:
    rew = statistics.fmean(self._rew_buf) if self._rew_buf else 0.0
    ep_len = statistics.fmean(self._len_buf) if self._len_buf else 0.0
    print(
      f"[it {it}] "
      f"behavior={loss_dict.get('loss/behavior', 0):.4f} "
      f"commit={loss_dict.get('loss/commit', 0):.4f} "
      f"mp={loss_dict.get('loss/mp', 0):.4f} "
      f"ar1={loss_dict.get('loss/ar1', 0):.4f} "
      f"perp={loss_dict.get('perplexity', 0):.1f} "
      f"rew={rew:.2f} len={ep_len:.0f} "
      f"t_collect={collect_t:.2f}s t_learn={learn_t:.2f}s"
    )
    if self._writer is not None:
      for k, v in loss_dict.items():
        self._writer.add_scalar(k, v, it)
      self._writer.add_scalar("episode/reward", rew, it)
      self._writer.add_scalar("episode/length", ep_len, it)
      self._writer.add_scalar("time/collect", collect_t, it)
      self._writer.add_scalar("time/learn", learn_t, it)

  def save(self, path: str, infos: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
      "encoder": self.policy.encoder.state_dict(),
      "decoder": self.policy.decoder.state_dict(),
      "motion_prior": self.policy.motion_prior.state_dict(),
      "quantizer": self.policy.quantizer.state_dict(),
      "optimizer": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos or {},
    }
    torch.save(state, path)
    print(f"[MotionPriorSingleVQ] saved checkpoint to {path}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    self.policy.encoder.load_state_dict(state["encoder"], strict=strict)
    self.policy.decoder.load_state_dict(state["decoder"], strict=strict)
    self.policy.motion_prior.load_state_dict(state["motion_prior"], strict=strict)
    self.policy.quantizer.load_state_dict(state["quantizer"], strict=strict)
    if "optimizer" in state and state["optimizer"]:
      self.alg.optimizer.load_state_dict(state["optimizer"])
    self.current_learning_iteration = int(state.get("iter", 0))
    return state.get("infos", {})

  def get_inference_policy(
    self,
    device: str | torch.device | None = None,
    path: str | None = None,
  ):
    """VQ inference path: ``"encoder"`` or ``"deploy"`` (motion_prior)."""
    if path is None:
      env_path = os.environ.get("MJLAB_MP_INFERENCE_PATH", "").strip().lower()
      if env_path and env_path != "auto":
        path = env_path
    chosen_path = path or "deploy"
    if chosen_path not in {"encoder", "deploy"}:
      raise ValueError(
        f"Unknown inference path: {chosen_path!r}. Expected 'encoder' or 'deploy'."
      )

    def _policy(obs_td) -> torch.Tensor:
      prop = _t(obs_td, "student")
      with torch.no_grad():
        if chosen_path == "encoder":
          return self.policy.policy_inference(prop, _t(obs_td, "teacher_tracking"))
        # Deploy: prop → motion_prior(prop) → decoder([prop, mp_code])
        mp_code = self.policy.motion_prior_inference(prop)
        return self.policy.decoder_inference(prop, mp_code)

    return _policy
