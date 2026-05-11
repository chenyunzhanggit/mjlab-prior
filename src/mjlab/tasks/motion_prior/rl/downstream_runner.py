"""Runner for the downstream-task PPO loop.

Mirrors ``mjlab.scripts.train``'s contract (``runner_cls(env, train_cfg,
log_dir, device)``), but does not inherit ``rsl_rl.runners.OnPolicyRunner``
because rsl_rl 5.2's PPO expects an ``MLPModel`` actor — see
``DownStreamPPO`` module docstring.

env contract: a single ``RslRlVecEnvWrapper`` that exposes three obs
groups in its TensorDict:

  * ``policy``           — actor input (task command + proprio history)
  * ``motion_prior_obs`` — frozen ``motion_prior`` MLP input (proprio only,
                            schema MUST match motion_prior training-time
                            student obs)
  * ``critic``           — privileged obs (policy + base_lin_vel etc.)

Save / load only persists the **trainable** submodules (actor / critic /
``std`` parameter) plus the optimizer state. The frozen backbone reloads
from ``motion_prior_ckpt_path`` on every construction; we don't redundantly
serialize it.
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

from mjlab.tasks.motion_prior.rl.algorithms import DownStreamPPO, DownStreamPpoCfg
from mjlab.tasks.motion_prior.rl.policies import DownStreamPolicy
from mjlab.tasks.motion_prior.rl.runner import _average_ep_infos, _make_writer


def _t(td: TensorDict, key: str) -> torch.Tensor:
  """Type-narrowing accessor — same trick used elsewhere in this package."""
  return cast(torch.Tensor, td[key])


class DownStreamOnPolicyRunner:
  """PPO runner with a frozen motion-prior backbone."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ) -> None:
    if kwargs:
      print(f"DownStreamOnPolicyRunner got unexpected kwargs (ignored): {list(kwargs)}")
    self.cfg = train_cfg
    self.device = device
    self.log_dir = log_dir
    self.current_learning_iteration: int = 0

    self.env = env

    # ---- Resolve obs dims from the env's TensorDict -------------------- #
    obs0 = self.env.get_observations()
    policy_obs = _t(obs0, "policy")
    prop_obs = _t(obs0, "motion_prior_obs")
    critic_obs = _t(obs0, "critic")
    self._policy_obs_shape = tuple(policy_obs.shape[1:])
    self._prop_obs_shape = tuple(prop_obs.shape[1:])
    self._critic_obs_shape = tuple(critic_obs.shape[1:])

    # ---- Build policy ---------------------------------------------------- #
    # play.py doesn't expose --agent.* flags, so env var is the only way to
    # inject the motion-prior ckpt path into a play-time runner. Train.py
    # users still pass --agent.motion-prior-ckpt-path normally; the env var
    # is just a fallback when the cfg field is empty.
    motion_prior_ckpt = train_cfg.get("motion_prior_ckpt_path") or os.environ.get(
      "MJLAB_MOTION_PRIOR_CKPT", ""
    )
    if not motion_prior_ckpt:
      raise ValueError(
        "DownStreamOnPolicyRunner needs a motion_prior checkpoint. Set either "
        "--agent.motion-prior-ckpt-path on the train CLI, or the "
        "MJLAB_MOTION_PRIOR_CKPT environment variable for play."
      )
    policy_cfg = train_cfg.get("policy", {})

    self.policy = self._build_policy(
      num_obs=int(policy_obs.shape[-1]),
      num_actions=int(self.env.num_actions),
      num_privileged_obs=int(critic_obs.shape[-1]),
      prop_obs_dim=int(prop_obs.shape[-1]),
      motion_prior_ckpt_path=motion_prior_ckpt,
      policy_cfg=policy_cfg,
      device=device,
    )

    # ---- Build algorithm + storage --------------------------------------- #
    algo_cfg = train_cfg.get("algorithm", {})
    ppo_cfg = DownStreamPpoCfg(
      num_learning_epochs=int(algo_cfg.get("num_learning_epochs", 5)),
      num_mini_batches=int(algo_cfg.get("num_mini_batches", 4)),
      clip_param=float(algo_cfg.get("clip_param", 0.2)),
      gamma=float(algo_cfg.get("gamma", 0.99)),
      lam=float(algo_cfg.get("lam", 0.95)),
      value_loss_coef=float(algo_cfg.get("value_loss_coef", 1.0)),
      entropy_coef=float(algo_cfg.get("entropy_coef", 0.005)),
      learning_rate=float(algo_cfg.get("learning_rate", 1.0e-3)),
      max_grad_norm=float(algo_cfg.get("max_grad_norm", 1.0)),
      use_clipped_value_loss=bool(algo_cfg.get("use_clipped_value_loss", True)),
      schedule=str(algo_cfg.get("schedule", "adaptive")),
      desired_kl=float(algo_cfg.get("desired_kl", 0.01)),
    )
    self.alg = DownStreamPPO(self.policy, cfg=ppo_cfg, device=device)
    self.alg.init_storage(
      num_envs=self.env.num_envs,
      num_steps_per_env=int(train_cfg.get("num_steps_per_env", 24)),
      policy_obs_shape=self._policy_obs_shape,
      prop_obs_shape=self._prop_obs_shape,
      critic_obs_shape=self._critic_obs_shape,
    )

    self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
    self.save_interval = int(train_cfg.get("save_interval", 500))

    # ---- Writer (wandb / tensorboard / neptune) ------------------------ #
    # Same dispatcher as the motion_prior runners, so the downstream task
    # lands in the same wandb / tensorboard dashboards as the tracking
    # task. ``logger`` is read from ``train_cfg`` (default ``"wandb"`` per
    # ``RslRlBaseRunnerCfg``).
    self._writer = _make_writer(log_dir, train_cfg, label="DownStream")

    # ---- Episode reward / length tracking ------------------------------ #
    self._rew_buf: deque[float] = deque(maxlen=100)
    self._len_buf: deque[float] = deque(maxlen=100)
    self._cur_rew = torch.zeros(self.env.num_envs, device=device)
    self._cur_len = torch.zeros(self.env.num_envs, device=device)

    # ---- ep_info accumulator (same shape as motion_prior runners) ------ #
    # ``ManagerBasedRlEnv`` puts per-manager reset metrics
    # (``Episode_Reward/*``, ``Episode_Termination/*``, command metrics)
    # under ``extras["log"]`` whenever an env resets. Snapshots are
    # appended each step that has any reset_env_ids; ``_log_iteration``
    # averages and flushes them, then clears.
    self._ep_infos: list[dict[str, Any]] = []

    # ---- Cumulative timing for ETA estimation -------------------------- #
    self._tot_timesteps: int = 0
    self._tot_time: float = 0.0
    self._start_iter: int = 0
    self._num_learning_iterations: int = 0

  def _build_policy(
    self,
    *,
    num_obs: int,
    num_actions: int,
    num_privileged_obs: int,
    prop_obs_dim: int,
    motion_prior_ckpt_path: str,
    policy_cfg: dict,
    device: str | torch.device,
  ) -> DownStreamPolicy:
    """Construct the trainable policy. Subclasses (e.g. VQ runner) override
    to swap in a different policy class with different hyperparams."""
    return DownStreamPolicy(
      num_obs=num_obs,
      num_actions=num_actions,
      num_privileged_obs=num_privileged_obs,
      prop_obs_dim=prop_obs_dim,
      motion_prior_ckpt_path=motion_prior_ckpt_path,
      latent_z_dims=int(policy_cfg.get("latent_z_dims", 32)),
      motion_prior_hidden_dims=tuple(
        policy_cfg.get("motion_prior_hidden_dims", (512, 256, 128))
      ),
      decoder_hidden_dims=tuple(policy_cfg.get("decoder_hidden_dims", (512, 256, 128))),
      actor_hidden_dims=tuple(policy_cfg.get("actor_hidden_dims", (512, 256, 128))),
      critic_hidden_dims=tuple(policy_cfg.get("critic_hidden_dims", (512, 256, 128))),
      activation=str(policy_cfg.get("activation", "elu")),
      init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
      device=device,
    )

  # ------------------------------------------------------------------ #
  # train.py interface                                                  #
  # ------------------------------------------------------------------ #

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
    self._start_iter = start_it
    self._num_learning_iterations = num_learning_iterations
    end_it = start_it + num_learning_iterations
    for it in range(start_it, end_it):
      t0 = time.time()
      obs = self._collect_rollout(obs)
      collect_t = time.time() - t0

      t1 = time.time()
      self.alg.compute_returns(_t(obs, "critic"))
      loss_dict = self.alg.update()
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
      # WandbSummaryWriter / NeptuneSummaryWriter expose ``stop()``;
      # plain ``SummaryWriter`` only has ``close()``.
      stop_fn = getattr(self._writer, "stop", None)
      if callable(stop_fn):
        stop_fn()
      self._writer.close()

  # ------------------------------------------------------------------ #
  # rollout                                                             #
  # ------------------------------------------------------------------ #

  def _collect_rollout(self, obs: TensorDict) -> TensorDict:
    for _ in range(self.num_steps_per_env):
      policy_obs = _t(obs, "policy")
      prop_obs = _t(obs, "motion_prior_obs")
      critic_obs = _t(obs, "critic")

      action = self.alg.act(policy_obs, prop_obs, critic_obs)
      obs, rew, dones, extras = self.env.step(action)
      self.alg.process_env_step(rew, dones)
      self._track_episode_stats(rew, dones)
      self._track_episode_log(extras)
    return obs

  def _track_episode_stats(self, rew: torch.Tensor, done: torch.Tensor) -> None:
    self._cur_rew += rew.float()
    self._cur_len += 1
    done_idx = done.nonzero(as_tuple=False).squeeze(-1)
    if done_idx.numel() > 0:
      self._rew_buf.extend(self._cur_rew[done_idx].tolist())
      self._len_buf.extend(self._cur_len[done_idx].tolist())
      self._cur_rew[done_idx] = 0.0
      self._cur_len[done_idx] = 0.0

  def _track_episode_log(self, extras: dict) -> None:
    """Append a snapshot of ``extras["log"]`` for this step.

    ``ManagerBasedRlEnv`` overwrites ``extras["log"]`` each step (with
    that step's reset metrics from ``RewardManager`` / ``TerminationManager``
    / ``CommandManager``); we keep a list of all non-empty snapshots and
    let ``_average_ep_infos`` collapse them in ``_log_iteration``.
    """
    log = extras.get("log") if isinstance(extras, dict) else None
    if log:
      self._ep_infos.append(log)

  # ------------------------------------------------------------------ #
  # logging                                                             #
  # ------------------------------------------------------------------ #

  def _log_iteration(
    self,
    it: int,
    loss_dict: dict[str, float],
    collect_t: float,
    learn_t: float,
  ) -> None:
    """Console + writer log. Format mirrors ``rsl_rl.utils.logger.Logger.log``
    (the same code path the mjlab tracking task and motion_prior runners
    run through), so output is bit-aligned with the reference: width=80,
    pad=40, banner with ``\\033[1m...\\033[0m``, ``Episode/`` /
    ``Loss/`` / ``Perf/`` / ``Train/`` writer prefixes.
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

    # ---- Writer scalars (same prefixes as rsl_rl Logger) -------------
    if self._writer is not None:
      for k, v in loss_dict.items():
        # ``loss_dict`` keys come in as ``loss/<name>`` or
        # ``schedule/<name>`` from the algo; rsl_rl Logger puts losses
        # under ``Loss/<name>``. Strip the namespace prefix.
        name = k.split("/", 1)[1] if "/" in k else k
        self._writer.add_scalar(f"Loss/{name}", v, it)
      self._writer.add_scalar("Perf/total_fps", fps, it)
      self._writer.add_scalar("Perf/collection_time", collect_t, it)
      self._writer.add_scalar("Perf/learning_time", learn_t, it)
      self._writer.add_scalar("Train/mean_reward", rew, it)
      self._writer.add_scalar("Train/mean_episode_length", length, it)

    # ---- Episode extras flush (Episode_Reward / Episode_Termination / etc) ----
    averaged = _average_ep_infos(self._ep_infos)
    self._ep_infos.clear()
    extras_string = ""
    for k, v in averaged.items():
      if "/" in k:
        # Already namespaced (e.g. ``Episode_Reward/track_lin_vel_xy_exp``).
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
      name = k.split("/", 1)[1] if "/" in k else k
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

  # ------------------------------------------------------------------ #
  # save / load                                                         #
  # ------------------------------------------------------------------ #

  def save(self, path: str, infos: dict | None = None) -> None:
    """Persist trainable submodules + optimizer + iter.

    Frozen motion_prior / mp_mu / decoder are NOT saved — they reload from
    ``motion_prior_ckpt_path`` on construction. Also dumps a deploy-ready
    ``policy.onnx`` (combined prop_obs + policy_obs → action) next to the
    checkpoint; failures are logged but do not break training.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
      "actor": self.policy.actor.state_dict(),
      "critic": self.policy.critic.state_dict(),
      "std": self.policy.std.detach().cpu().clone(),
      "optimizer": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos or {},
    }
    torch.save(state, path)
    print(f"[DownStream] saved checkpoint to {path}")
    try:
      onnx_path = Path(path).with_suffix(".onnx")
      self.export_policy_to_onnx(str(onnx_path))
    except Exception as e:
      print(f"[DownStream] ONNX export failed (training continues): {e}")

  def export_policy_to_onnx(
    self,
    path: str,
    filename: str | None = None,
    *,
    mode: str = "combined",
    verbose: bool = False,
  ) -> None:
    """Export deploy ONNX. ``mode='combined'`` writes the full chain
    (prop_obs + policy_obs → action); ``'actor'`` writes only the
    trainable actor (policy_obs → raw_latent)."""
    from mjlab.tasks.motion_prior.onnx import export_downstream_to_onnx

    output = Path(path)
    if filename is not None:
      output = output / filename
    export_downstream_to_onnx(self.policy, output, mode=mode, verbose=verbose)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    state = torch.load(path, map_location=map_location, weights_only=False)
    self.policy.actor.load_state_dict(state["actor"], strict=strict)
    self.policy.critic.load_state_dict(state["critic"], strict=strict)
    with torch.no_grad():
      self.policy.std.copy_(state["std"].to(self.policy.std.device))
    if "optimizer" in state and state["optimizer"]:
      self.alg.optimizer.load_state_dict(state["optimizer"])
    self.current_learning_iteration = int(state.get("iter", 0))
    return state.get("infos", {})

  # ------------------------------------------------------------------ #
  # play.py hook                                                        #
  # ------------------------------------------------------------------ #

  def get_inference_policy(
    self,
    device: str | torch.device | None = None,
  ):
    """Return ``(obs_td) -> action`` for ``mjlab.scripts.play``."""
    if device is not None:
      self.policy.to(device)
    self.policy.eval()

    def _policy(obs_td: TensorDict) -> torch.Tensor:
      with torch.no_grad():
        return self.policy.policy_inference(
          _t(obs_td, "policy"), _t(obs_td, "motion_prior_obs")
        )

    return _policy


class DownStreamVQOnPolicyRunner(DownStreamOnPolicyRunner):
  """Downstream PPO runner that swaps in a VQ-VAE motion-prior backbone.

  Identical to :class:`DownStreamOnPolicyRunner` except ``_build_policy``
  constructs a :class:`DownStreamVQPolicy` (frozen motion_prior MLP +
  shared codebook quantizer + frozen decoder, instead of the VAE
  motion_prior MLP + ``mp_mu`` + decoder).

  Save/load and ONNX export inherit unchanged: the trainable parameters
  (actor / critic / std) and persistence layout are the same; the
  combined ONNX deploy module dispatches by policy type.
  """

  def _build_policy(
    self,
    *,
    num_obs: int,
    num_actions: int,
    num_privileged_obs: int,
    prop_obs_dim: int,
    motion_prior_ckpt_path: str,
    policy_cfg: dict,
    device: str | torch.device,
  ):
    from mjlab.tasks.motion_prior.rl.policies.downstream_vq_policy import (
      DownStreamVQPolicy,
    )

    return DownStreamVQPolicy(
      num_obs=num_obs,
      num_actions=num_actions,
      num_privileged_obs=num_privileged_obs,
      prop_obs_dim=prop_obs_dim,
      motion_prior_ckpt_path=motion_prior_ckpt_path,
      num_code=int(policy_cfg.get("num_code", 2048)),
      code_dim=int(policy_cfg.get("code_dim", 64)),
      motion_prior_hidden_dims=tuple(
        policy_cfg.get("motion_prior_hidden_dims", (512, 256, 128))
      ),
      decoder_hidden_dims=tuple(policy_cfg.get("decoder_hidden_dims", (512, 256, 128))),
      actor_hidden_dims=tuple(policy_cfg.get("actor_hidden_dims", (512, 256, 128))),
      critic_hidden_dims=tuple(policy_cfg.get("critic_hidden_dims", (512, 256, 128))),
      activation=str(policy_cfg.get("activation", "elu")),
      init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
      use_lab=bool(policy_cfg.get("use_lab", True)),
      lab_lambda=float(policy_cfg.get("lab_lambda", 3.0)),
      device=device,
    )
