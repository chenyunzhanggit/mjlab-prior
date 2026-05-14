"""AMP-augmented PPO runner for the velocity task.

Replaces the stub from Phase B with a real runner that:
  - mirrors the standard ``OnPolicyRunner`` __init__ but constructs an
    :class:`AMPPPO` instead of a plain PPO, plus a Discriminator / AMPLoader /
    Normalizer / ReplayBuffer wired into it;
  - overrides ``learn()`` to (a) read the ``amp`` obs group at every step,
    (b) blend the discriminator-derived AMP reward into the env reward, and
    (c) push the policy-side (s, s') stream into the AMP ReplayBuffer;
  - extends ``save()`` to honour ``MjlabOnPolicyRunner``'s common_step_counter
    persistence and ONNX export side-effects;
  - delegates ``load()`` to ``MjlabOnPolicyRunner`` (legacy checkpoint
    migration + env_state restore) and lets AMPPPO.load attach the
    discriminator / normalizer dicts on top.

Reward injection mode is AMP_mjlab's R1: ``reward = discriminator.predict_amp_reward``
(internally lerp'd with task reward when ``task_reward_lerp > 0``). This
reward is what PPO sees, exactly as in the reference implementation.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from rsl_rl.env import VecEnv
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.amp_velocity.rl.amp_ppo import AMPPPO
from mjlab.tasks.amp_velocity.rl.discriminator import Discriminator
from mjlab.tasks.amp_velocity.rl.motion_loader import AMPLoader
from mjlab.tasks.amp_velocity.rl.normalizer import Normalizer
from mjlab.tasks.amp_velocity.rl.replay_buffer import ReplayBuffer


def _resolve_body_names_for_amp_loader(env: VecEnv) -> list[str]:
  """Resolve the ordered list of robot body names matching the npz layout.

  AMP_mjlab packs ``body_pos_w[t, idx, :]`` indexed by the robot's body order
  in the MuJoCo model. We need the same ordering so AMPLoader can pick
  ``body_indexes`` correctly. ``env.unwrapped.scene["robot"]`` exposes the
  bound entity whose ``.body_names`` is the canonical order.
  """
  robot = env.unwrapped.scene["robot"]  # type: ignore[attr-defined]
  return list(robot.body_names)


class AmpVelocityOnPolicyRunner(MjlabOnPolicyRunner):
  """Runner that co-trains PPO with an AMP discriminator on the velocity task.

  Inherits the ONNX-export and legacy-checkpoint behaviour of
  ``MjlabOnPolicyRunner`` but replaces ``__init__`` so the constructed
  algorithm is an ``AMPPPO``. ``learn()`` is rewritten to feed AMP obs into
  the discriminator and lerp the resulting reward into the env reward.
  """

  alg: AMPPPO
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **_: Any,
  ) -> None:
    # --- Replicate OnPolicyRunner.__init__ minus the algorithm build -----
    # We *cannot* call ``super().__init__`` here because both parents build a
    # plain PPO via ``construct_algorithm``; we want AMPPPO instead.

    # Strip None-valued optional configs (same logic as MjlabOnPolicyRunner).
    for key in ("actor", "critic"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
        if train_cfg[key].get("rnn_type") is None:
          for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(opt, None)

    self.env = env  # type: ignore[assignment]
    self.cfg = train_cfg
    self.device = device

    # Multi-GPU setup (mirrors the base class).
    self._configure_multi_gpu()

    # First obs sample tells construct_algorithm how to size storage / models.
    obs = self.env.get_observations()

    # --- Build the AMP-augmented algorithm -------------------------------
    self.alg = self._construct_amp_algorithm(obs, env, train_cfg, device)

    # --- Logger (same shape as OnPolicyRunner) ----------------------------
    self.logger = Logger(
      log_dir=log_dir,
      cfg=self.cfg,
      env_cfg=self.env.cfg,
      num_envs=self.env.num_envs,
      is_distributed=self.is_distributed,
      gpu_world_size=self.gpu_world_size,
      gpu_global_rank=self.gpu_global_rank,
      device=self.device,
    )

    self.current_learning_iteration = 0

  # ------------------------------------------------------------------ #
  # Algorithm construction                                             #
  # ------------------------------------------------------------------ #

  def _construct_amp_algorithm(
    self,
    obs,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> AMPPPO:
    """Build actor / critic / storage / AMPPPO and the AMP support objects.

    Heavily inspired by ``PPO.construct_algorithm`` but bypasses its
    ``cfg["algorithm"].pop`` flow so AMP-specific cfg lives in its own
    ``cfg["amp"]`` block.
    """
    from rsl_rl.extensions import (
      resolve_rnd_config,
      resolve_symmetry_config,
    )
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from rsl_rl.utils import compile_model, resolve_obs_groups

    alg_cfg = dict(cfg["algorithm"])  # shallow copy: we'll pop keys
    alg_cfg.pop("class_name", None)
    actor_cfg = dict(cfg["actor"])
    critic_cfg = dict(cfg["critic"])
    actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore[assignment]
    critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore[assignment]

    default_sets = ["actor", "critic"]
    if alg_cfg.get("rnd_cfg") is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
    alg_cfg = resolve_symmetry_config(alg_cfg, env)
    # Logger expects these keys to live on cfg["algorithm"] regardless of
    # whether rnd / symmetry are enabled; mirror the resolved values back.
    cfg["algorithm"]["rnd_cfg"] = alg_cfg.get("rnd_cfg")
    cfg["algorithm"]["symmetry_cfg"] = alg_cfg.get("symmetry_cfg")

    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg
    ).to(device)
    print(f"Actor Model: {actor}")
    if alg_cfg.pop("share_cnn_encoders", None):
      critic_cfg["cnns"] = actor.cnns
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **critic_cfg
    ).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
    )

    # --- AMP-side construction -------------------------------------------
    amp_cfg = cfg["amp"]
    body_names_in_order = _resolve_body_names_for_amp_loader(env)
    amp_data = AMPLoader(
      motion_file=amp_cfg["motion_dir"],
      body_names=amp_cfg["tracked_bodies"],
      anchor_name=amp_cfg["anchor_body"],
      all_body_names=body_names_in_order,
      device=device,
    )
    amp_obs_dim = amp_data.observation_dim
    # Sanity: the env's amp obs group must produce the same dim.
    env_amp_dim = int(obs["amp"].shape[-1])
    assert env_amp_dim == amp_obs_dim, (
      f"AMP obs dim mismatch: env produces {env_amp_dim}, loader expects "
      f"{amp_obs_dim}. Check tracked_bodies / anchor_body match between "
      f"env_cfgs and rl_cfg."
    )

    discriminator = Discriminator(
      input_dim=2 * amp_obs_dim,
      amp_reward_coef=amp_cfg["reward_coef"],
      hidden_layer_sizes=list(amp_cfg["discriminator_hidden"]),
      device=device,
      task_reward_lerp=amp_cfg["task_reward_lerp"],
    )
    amp_normalizer = Normalizer(input_dim=amp_obs_dim)
    amp_storage = ReplayBuffer(
      obs_dim=amp_obs_dim,
      buffer_size=int(amp_cfg["replay_buffer_size"]),
      device=device,
    )

    alg = AMPPPO(
      actor,
      critic,
      storage,
      device=device,
      **alg_cfg,
      multi_gpu_cfg=cfg["multi_gpu"],
      discriminator=discriminator,
      amp_data=amp_data,
      amp_normalizer=amp_normalizer,
      amp_storage=amp_storage,
      discriminator_lr=amp_cfg["discriminator_lr"],
      discriminator_weight_decay=amp_cfg["discriminator_weight_decay"],
      grad_pen_lambda=amp_cfg["grad_pen_lambda"],
    )

    # Honor torch_compile_mode if set.
    mode = cfg.get("torch_compile_mode")
    if mode is not None:
      alg.actor = compile_model(alg._raw_actor, mode)  # type: ignore[assignment]
      alg.critic = compile_model(alg._raw_critic, mode)  # type: ignore[assignment]

    return alg

  # ------------------------------------------------------------------ #
  # Learn loop                                                         #
  # ------------------------------------------------------------------ #

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """On-policy learn loop with AMP reward injection.

    Mirrors ``OnPolicyRunner.learn`` step-for-step; the only deltas are
    (a) pulling ``obs["amp"]`` before/after each env step, (b) replacing
    env rewards with the discriminator's ``predict_amp_reward`` blend, and
    (c) inserting the policy-side AMP transition into the ReplayBuffer.
    """
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    amp_obs = obs["amp"].to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      mean_amp_reward_buf: list[float] = []
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          # Sample actions via PPO.act.
          actions = self.alg.act(obs)
          # Step the env; obs is the *next* TensorDict.
          obs, env_rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, env_rewards, dones)
          obs = obs.to(self.device)
          env_rewards = env_rewards.to(self.device)
          dones = dones.to(self.device)
          next_amp_obs = obs["amp"].to(self.device)

          # Per AMP_mjlab: at reset boundaries, the post-reset amp obs does
          # NOT belong to the same episode as the pre-step amp obs, so the
          # transition is meaningless to the discriminator. Replace those
          # rows with the pre-step value so (s, s') stays self-consistent.
          reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
          next_amp_obs_with_term = next_amp_obs.clone()
          if reset_env_ids.numel() > 0:
            next_amp_obs_with_term[reset_env_ids] = amp_obs[reset_env_ids]

          # AMP reward (optionally lerp'd with env reward); shape (num_envs,).
          amp_reward, _ = self.alg.discriminator.predict_amp_reward(
            amp_obs,
            next_amp_obs_with_term,
            env_rewards,
            normalizer=self.alg.amp_normalizer,
          )
          mean_amp_reward_buf.append(amp_reward.mean().item())

          # Feed the discriminator-blended reward through PPO; insert policy
          # AMP sample for the next disc update.
          self.alg.process_env_step(obs, amp_reward, dones, extras)
          self.alg.amp_storage.insert(amp_obs, next_amp_obs_with_term)

          intrinsic_rewards = self.alg.intrinsic_rewards if self.alg.rnd else None
          # Log raw env reward; AMP reward is logged separately.
          self.logger.process_env_step(env_rewards, dones, extras, intrinsic_rewards)

          amp_obs = next_amp_obs

        stop = time.time()
        collect_time = stop - start
        start = stop

        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      if mean_amp_reward_buf:
        loss_dict["amp_reward_mean"] = sum(mean_amp_reward_buf) / len(
          mean_amp_reward_buf
        )

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.alg.rnd else None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")  # type: ignore[arg-type]
      )
      self.logger.stop_logging_writer()
