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

    # ---- Optional SummaryWriter ---------------------------------------- #
    self._writer = None
    if log_dir is not None and train_cfg.get("logger", "tensorboard") == "tensorboard":
      try:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
      except Exception as e:
        print(f"[DownStream] tensorboard unavailable, skipping ({e})")

    # ---- Episode reward / length tracking ------------------------------ #
    self._rew_buf: deque[float] = deque(maxlen=100)
    self._len_buf: deque[float] = deque(maxlen=100)
    self._cur_rew = torch.zeros(self.env.num_envs, device=device)
    self._cur_len = torch.zeros(self.env.num_envs, device=device)

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
      obs, rew, dones, _ = self.env.step(action)
      self.alg.process_env_step(rew, dones)
      self._track_episode_stats(rew, dones)
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
    rew = statistics.fmean(self._rew_buf) if self._rew_buf else 0.0
    length = statistics.fmean(self._len_buf) if self._len_buf else 0.0
    print(
      f"[it {it}] "
      f"surrogate={loss_dict.get('loss/surrogate', 0):.4f} "
      f"value={loss_dict.get('loss/value', 0):.4f} "
      f"entropy={loss_dict.get('loss/entropy', 0):.4f} "
      f"lr={loss_dict.get('schedule/learning_rate', 0):.2e} "
      f"rew={rew:.2f} len={length:.0f} "
      f"t_collect={collect_t:.2f}s t_learn={learn_t:.2f}s"
    )
    if self._writer is not None:
      for k, v in loss_dict.items():
        self._writer.add_scalar(k, v, it)
      self._writer.add_scalar("episode/reward", rew, it)
      self._writer.add_scalar("episode/length", length, it)
      self._writer.add_scalar("time/collect", collect_t, it)
      self._writer.add_scalar("time/learn", learn_t, it)

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
