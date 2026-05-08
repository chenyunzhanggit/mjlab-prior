"""Single-teacher motion-prior distillation runner.

One env, one frozen teacher, one encoder. Mirrors
:class:`MotionPriorOnPolicyRunner` with the secondary env / teacher_b
branch removed; the policy / algorithm classes are the single-encoder
variants.

Train-script contract: ``train.py`` registers the env (the
multi-motion-tracking flat env, which already exposes the trackingbfm
actor obs schema needed by the frozen teacher) and constructs the runner
with it. There is no secondary env to instantiate.
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
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_encoder_policy import (
  MotionPriorSingleEncoderPolicy,
)
from mjlab.tasks.motion_prior.rl.runner import _average_ep_infos


def _t(td: TensorDict, key: str) -> torch.Tensor:
  """Type-narrowing accessor for plain-tensor obs groups."""
  return cast(torch.Tensor, td[key])


class MotionPriorSingleOnPolicyRunner:
  """Single-env VAE motion-prior distillation runner."""

  env: RslRlVecEnvWrapper
  policy: MotionPriorSingleEncoderPolicy
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
        "MotionPriorSingleOnPolicyRunner got unexpected kwargs (ignored): "
        f"{list(kwargs)}"
      )
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.current_learning_iteration: int = 0

    # ----- Env --------------------------------------------------------
    self.env = env  # type: ignore[assignment]

    # ----- Policy + Algorithm (overridable hook for VQ subclass) -----
    initial_obs = self.env.get_observations()
    student_obs_dim = _t(initial_obs, "student").shape[-1]
    teacher_obs_dim = _t(initial_obs, "teacher_t").shape[-1]
    num_actions = int(self.env.num_actions)
    self._build_policy_and_alg(
      train_cfg=train_cfg,
      student_obs_dim=student_obs_dim,
      teacher_obs_dim=teacher_obs_dim,
      num_actions=num_actions,
      device=device,
    )

    # ----- Rollout knobs ---------------------------------------------
    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.save_interval = int(train_cfg.get("save_interval", 500))
    self.upload_model = bool(train_cfg.get("upload_model", False))

    # ----- Optional SummaryWriter ------------------------------------
    self._writer = None
    if log_dir is not None and train_cfg.get("logger", "tensorboard") == "tensorboard":
      try:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
      except Exception as e:
        print(f"[MotionPriorSingle] tensorboard unavailable, skipping ({e})")

    # ----- Episode reward / length tracking --------------------------
    self._rew_buf: deque[float] = deque(maxlen=100)
    self._len_buf: deque[float] = deque(maxlen=100)
    self._cur_rew = torch.zeros(self.env.num_envs, device=device)
    self._cur_len = torch.zeros(self.env.num_envs, device=device)

    # ----- ep_info accumulators (per-iteration, flushed in
    # ``_log_iteration``). Same format as the dual runner —
    # ``ManagerBasedRlEnv`` puts per-manager reset metrics under
    # ``extras["log"]`` whenever an env resets.
    self._ep_infos: list[dict[str, Any]] = []

    # ----- Cumulative timing for ETA estimation. -------------------------
    self._tot_timesteps: int = 0
    self._tot_time: float = 0.0
    self._start_iter: int = 0
    self._num_learning_iterations: int = 0

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
    """Build the VAE policy + algorithm. VQ subclass overrides."""
    teacher_policy_path = train_cfg["teacher_policy_path"]
    policy_cfg = train_cfg.get("policy", {})

    self.policy = MotionPriorSingleEncoderPolicy(
      prop_obs_dim=student_obs_dim,
      num_actions=num_actions,
      teacher_obs_dim=teacher_obs_dim,
      teacher_policy_path=teacher_policy_path,
      teacher_hidden_dims=tuple(
        policy_cfg.get(
          "teacher_hidden_dims",
          (2048, 2048, 1024, 1024, 512, 256, 128),
        )
      ),
      teacher_activation=str(policy_cfg.get("teacher_activation", "elu")),
      teacher_obs_normalization=bool(
        policy_cfg.get("teacher_obs_normalization", True)
      ),
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
    del init_at_random_ep_len  # env already RSI-randomizes on reset.
    obs = self.env.get_observations()

    start_it = self.current_learning_iteration
    self._start_iter = start_it
    self._num_learning_iterations = num_learning_iterations
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

  def _policy_step_action(self, obs: TensorDict) -> torch.Tensor:
    _, _, _, sa_step, _, _ = self.policy.forward(
      _t(obs, "student"), _t(obs, "teacher_t")
    )
    return sa_step

  def _collect_rollout(
    self, obs: TensorDict
  ) -> tuple[dict[str, torch.Tensor], TensorDict]:
    """Roll out ``num_steps_per_env`` transitions on the env.

    Stores **only detached inputs** (student / teacher obs) and the frozen
    teacher target, so the per-epoch loop in ``learn()`` can re-forward
    the trainable submodules without ``retain_graph=True``.
    """
    T = self.num_steps_per_env

    prop_buf, t_obs_buf, t_act_buf, prog_buf = [], [], [], []

    for _ in range(T):
      with torch.no_grad():
        sa_step = self._policy_step_action(obs)
        teacher_action = self.policy.evaluate(_t(obs, "teacher_t"))

      prop_buf.append(_t(obs, "student").detach().clone())
      t_obs_buf.append(_t(obs, "teacher_t").detach().clone())
      t_act_buf.append(teacher_action.detach().clone())
      prog_buf.append(self.env.episode_length_buf.detach().clone().to(self.device))

      obs, rew, done, infos = self.env.step(sa_step.detach())
      self._track_episode_stats(rew, done)
      log = infos.get("log") if isinstance(infos, dict) else None
      if log:
        self._ep_infos.append(log)

    def _flat(seq: list[torch.Tensor]) -> torch.Tensor:
      return torch.stack(seq, dim=1).flatten(0, 1)

    rollout = dict(
      prop_obs=_flat(prop_buf),
      teacher_obs=_flat(t_obs_buf),
      actions_teacher=_flat(t_act_buf),
      progress_buf=torch.stack(prog_buf, dim=1).unsqueeze(-1).float(),
    )
    return rollout, obs

  def _epoch_step(
    self,
    rollout: dict[str, torch.Tensor],
    cur_iter_num: int,
  ) -> dict[str, float]:
    n_envs = self.env.num_envs
    T = self.num_steps_per_env
    Z = self._latent_z_dims

    enc_mu, enc_lv, _, sa, mp_mu, mp_lv = self.policy.forward(
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
    """Console + writer log. Format mirrors ``rsl_rl.utils.logger.Logger.log``
    (the same code path the mjlab tracking task runs through), so output
    is bit-aligned with the reference: width=80, pad=40, banner with
    ``\\033[1m...\\033[0m``, ``Episode/``, ``Loss/``, ``Perf/``,
    ``Train/`` writer prefixes.
    """
    width = 80
    pad = 40

    iteration_time = collect_t + learn_t
    collection_size = self.num_steps_per_env * self.env.num_envs
    self._tot_timesteps += collection_size
    self._tot_time += iteration_time
    fps = int(collection_size / max(iteration_time, 1e-9))

    rew = statistics.fmean(self._rew_buf) if self._rew_buf else 0.0
    length = statistics.fmean(self._len_buf) if self._len_buf else 0.0

    # ---- Writer scalars ----------------------------------------------
    if self._writer is not None:
      for k, v in loss_dict.items():
        name = k.split("/", 1)[1] if k.startswith("loss/") else k
        self._writer.add_scalar(f"Loss/{name}", v, it)
      self._writer.add_scalar("Perf/total_fps", fps, it)
      self._writer.add_scalar("Perf/collection_time", collect_t, it)
      self._writer.add_scalar("Perf/learning_time", learn_t, it)
      self._writer.add_scalar("Train/mean_reward", rew, it)
      self._writer.add_scalar("Train/mean_episode_length", length, it)

    # ---- Episode extras flush ----------------------------------------
    averaged = _average_ep_infos(self._ep_infos)
    self._ep_infos.clear()
    extras_string = ""
    for k, v in averaged.items():
      if "/" in k:
        if self._writer is not None:
          self._writer.add_scalar(k, v, it)
        extras_string += f"{f'{k}:':>{pad}} {v:.4f}\n"
      else:
        if self._writer is not None:
          self._writer.add_scalar(f"Episode/{k}", v, it)
        extras_string += f"{f'Mean episode {k}:':>{pad}} {v:.4f}\n"

    # ---- Console (matches rsl_rl Logger.log layout exactly) ----------
    log_string = "#" * width + "\n"
    log_string += (
      f"\033[1m{f' Learning iteration {it}/{self._start_iter + self._num_learning_iterations} '.center(width)}\033[0m \n\n"
    )
    log_string += (
      f"{'Total steps:':>{pad}} {self._tot_timesteps} \n"
      f"{'Steps per second:':>{pad}} {fps:.0f} \n"
      f"{'Collection time:':>{pad}} {collect_t:.3f}s \n"
      f"{'Learning time:':>{pad}} {learn_t:.3f}s \n"
    )
    for k, v in loss_dict.items():
      name = k.split("/", 1)[1] if k.startswith("loss/") else k
      log_string += f"{f'Mean {name} loss:':>{pad}} {v:.4f}\n"
    if self._rew_buf:
      log_string += f"{'Mean reward:':>{pad}} {rew:.2f}\n"
      log_string += f"{'Mean episode length:':>{pad}} {length:.2f}\n"
    log_string += extras_string

    done_it = it + 1 - self._start_iter
    remaining_it = self._num_learning_iterations - done_it
    if done_it > 0 and remaining_it > 0:
      eta_secs = self._tot_time / done_it * remaining_it
      eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))
    else:
      eta_str = "--:--:--"
    log_string += "-" * width + "\n"
    log_string += f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
    log_string += (
      f"{'Time elapsed:':>{pad}} "
      f"{time.strftime('%H:%M:%S', time.gmtime(self._tot_time))}\n"
    )
    log_string += f"{'ETA:':>{pad}} {eta_str}\n"
    print(log_string, flush=True)

  # --------------------------------------------------------------------- #
  # Save / load                                                           #
  # --------------------------------------------------------------------- #

  def save(self, path: str, infos: dict | None = None) -> None:
    """Persist trainable submodules + optimizer + iter.

    Frozen teacher is NOT saved (re-loaded from ``teacher_policy_path`` on
    construction).
    """
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

  def get_inference_policy(
    self,
    device: str | torch.device | None = None,
    path: str | None = None,
  ):
    """Return a callable ``(obs_td) -> action`` running the encoder path.

    Two paths:

    * ``"encoder"`` (training path) — encoder(teacher_t_obs) → enc_mu →
      decoder([prop, enc_mu]). Used when ``teacher_t`` is in the obs.
    * ``"deploy"`` (proprio-only) — motion_prior(prop) → mp_mu →
      decoder([prop, mp_mu]). Used when ``teacher_t`` is absent.

    Resolution: explicit ``path=`` arg → ``MJLAB_MP_INFERENCE_PATH`` env
    var → auto-detect (encoder if ``teacher_t`` in obs, else deploy).
    """
    if path is None:
      env_path = os.environ.get("MJLAB_MP_INFERENCE_PATH", "").strip().lower()
      if env_path and env_path != "auto":
        path = env_path
    if path is not None and path not in {"encoder", "deploy"}:
      raise ValueError(
        f"Unknown inference path: {path!r}. Expected 'encoder', 'deploy', or None."
      )

    def _resolve(obs_td) -> str:
      if path is not None:
        return path
      return "encoder" if "teacher_t" in set(obs_td.keys()) else "deploy"

    def _policy(obs_td) -> torch.Tensor:
      prop = _t(obs_td, "student")
      chosen = _resolve(obs_td)
      with torch.no_grad():
        if chosen == "encoder":
          return self.policy.policy_inference(prop, _t(obs_td, "teacher_t"))
        # deploy path
        mp_mu = self.policy.motion_prior_inference(prop)
        return self.policy.decoder_inference(prop, mp_mu)

    return _policy

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
