"""PPO for the downstream task with a frozen motion-prior backbone.

The hot loop is standard PPO (surrogate-clip + clipped value loss + entropy
bonus + adaptive-KL learning-rate schedule). Two adaptations vs vanilla
PPO:

* ``act()`` / ``act_inference()`` take **two** obs tensors:
  ``policy_obs`` (actor input, contains task command) and ``prop_obs``
  (motion-prior backbone input, proprioception only).
* The action stored in the rollout buffer is the **latent residual**
  (``DownStreamPolicy``'s ``raw_action``), not the joint command. The
  joint command (``recons_actions``) is what gets sent to ``env.step``;
  PPO never sees it.

We do not subclass ``rsl_rl.algorithms.PPO``: rsl_rl 5.2's PPO assumes the
policy is an ``MLPModel`` instance with a built-in distribution, while
``DownStreamPolicy`` keeps the actor / critic / std as bare ``nn.Module``
attributes (mirrors the reference implementation). Reusing ``rsl_rl``'s PPO
would require a wrapper layer that's larger than this file's hand-rolled
loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from mjlab.tasks.motion_prior.rl.policies.downstream_policy import DownStreamPolicy


@dataclass
class DownStreamPpoCfg:
  """Hyperparameters for ``DownStreamPPO``."""

  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  clip_param: float = 0.2
  gamma: float = 0.99
  lam: float = 0.95
  value_loss_coef: float = 1.0
  entropy_coef: float = 0.005
  learning_rate: float = 1.0e-3
  max_grad_norm: float = 1.0
  use_clipped_value_loss: bool = True
  schedule: str = "adaptive"  # {"adaptive", "fixed"}
  desired_kl: float = 0.01


class _DownStreamRolloutStorage:
  """Minimal rollout buffer for the downstream PPO loop.

  Separately holds ``policy_obs`` / ``prop_obs`` / ``critic_obs`` so the
  per-step pull during ``update()`` matches what ``DownStreamPolicy.act``
  needs without a TensorDict round-trip.
  """

  def __init__(
    self,
    num_envs: int,
    num_steps: int,
    policy_obs_shape: tuple[int, ...],
    prop_obs_shape: tuple[int, ...],
    critic_obs_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    device: str | torch.device,
  ) -> None:
    self.num_envs = num_envs
    self.num_steps = num_steps
    self.device = device

    self.policy_obs = torch.zeros(num_steps, num_envs, *policy_obs_shape, device=device)
    self.prop_obs = torch.zeros(num_steps, num_envs, *prop_obs_shape, device=device)
    self.critic_obs = torch.zeros(num_steps, num_envs, *critic_obs_shape, device=device)
    self.actions = torch.zeros(num_steps, num_envs, *action_shape, device=device)
    self.rewards = torch.zeros(num_steps, num_envs, device=device)
    self.dones = torch.zeros(num_steps, num_envs, device=device, dtype=torch.float32)
    self.values = torch.zeros(num_steps, num_envs, device=device)
    self.actions_log_prob = torch.zeros(num_steps, num_envs, device=device)
    self.mu = torch.zeros(num_steps, num_envs, *action_shape, device=device)
    self.sigma = torch.zeros(num_steps, num_envs, *action_shape, device=device)

    self.returns = torch.zeros(num_steps, num_envs, device=device)
    self.advantages = torch.zeros(num_steps, num_envs, device=device)

    self.step = 0

  def add(
    self,
    *,
    policy_obs: torch.Tensor,
    prop_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    actions_log_prob: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
  ) -> None:
    if self.step >= self.num_steps:
      raise RuntimeError("rollout buffer overflow; call clear() first")
    i = self.step
    self.policy_obs[i] = policy_obs
    self.prop_obs[i] = prop_obs
    self.critic_obs[i] = critic_obs
    self.actions[i] = actions
    self.rewards[i] = rewards.float()
    self.dones[i] = dones.float()
    self.values[i] = values.flatten()
    self.actions_log_prob[i] = actions_log_prob.flatten()
    self.mu[i] = mu
    self.sigma[i] = sigma
    self.step += 1

  def clear(self) -> None:
    self.step = 0

  def compute_returns(
    self, last_values: torch.Tensor, gamma: float, lam: float
  ) -> None:
    """GAE-λ over the (num_steps, num_envs) buffer."""
    advantage = torch.zeros(self.num_envs, device=self.device)
    for t in reversed(range(self.num_steps)):
      next_values = (
        last_values.flatten() if t == self.num_steps - 1 else self.values[t + 1]
      )
      mask = 1.0 - self.dones[t]
      delta = self.rewards[t] + gamma * next_values * mask - self.values[t]
      advantage = delta + gamma * lam * advantage * mask
      self.returns[t] = advantage + self.values[t]
    # Normalize advantages over the whole rollout (per the rsl_rl default).
    adv = self.returns - self.values
    self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

  def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
    n_total = self.num_steps * self.num_envs
    batch_size = n_total // num_mini_batches
    indices = torch.randperm(n_total, device=self.device)

    # Flatten (T, N, ...) -> (T*N, ...).
    flat = {
      "policy_obs": self.policy_obs.reshape(n_total, *self.policy_obs.shape[2:]),
      "prop_obs": self.prop_obs.reshape(n_total, *self.prop_obs.shape[2:]),
      "critic_obs": self.critic_obs.reshape(n_total, *self.critic_obs.shape[2:]),
      "actions": self.actions.reshape(n_total, *self.actions.shape[2:]),
      "values": self.values.reshape(n_total),
      "advantages": self.advantages.reshape(n_total),
      "returns": self.returns.reshape(n_total),
      "actions_log_prob": self.actions_log_prob.reshape(n_total),
      "mu": self.mu.reshape(n_total, *self.mu.shape[2:]),
      "sigma": self.sigma.reshape(n_total, *self.sigma.shape[2:]),
    }
    for _ in range(num_epochs):
      for mb in range(num_mini_batches):
        idx = indices[mb * batch_size : (mb + 1) * batch_size]
        yield {k: v[idx] for k, v in flat.items()}


class DownStreamPPO:
  """PPO over a frozen motion-prior backbone."""

  def __init__(
    self,
    policy: DownStreamPolicy,
    *,
    cfg: DownStreamPpoCfg | None = None,
    device: str | torch.device = "cpu",
  ) -> None:
    self.policy = policy.to(device)
    self.device = device
    self.cfg = cfg or DownStreamPpoCfg()
    self.learning_rate = self.cfg.learning_rate

    trainable = [p for p in policy.parameters() if p.requires_grad]
    self.optimizer = optim.Adam(trainable, lr=self.learning_rate)

    self.storage: _DownStreamRolloutStorage | None = None

  def init_storage(
    self,
    num_envs: int,
    num_steps_per_env: int,
    policy_obs_shape: tuple[int, ...],
    prop_obs_shape: tuple[int, ...],
    critic_obs_shape: tuple[int, ...],
  ) -> None:
    self.storage = _DownStreamRolloutStorage(
      num_envs=num_envs,
      num_steps=num_steps_per_env,
      policy_obs_shape=policy_obs_shape,
      prop_obs_shape=prop_obs_shape,
      critic_obs_shape=critic_obs_shape,
      action_shape=(self.policy.latent_dim,),
      device=self.device,
    )

  # ---------- rollout ---------- #

  def act(
    self,
    policy_obs: torch.Tensor,
    prop_obs: torch.Tensor,
    critic_obs: torch.Tensor,
  ) -> torch.Tensor:
    """Sample one transition. Returns ``recons_actions`` for env.step."""
    assert self.storage is not None
    recons_actions, raw_action = self.policy.act(policy_obs, prop_obs)
    log_prob = self.policy.get_actions_log_prob(raw_action).detach()
    value = self.policy.evaluate(critic_obs).detach()
    mu = self.policy.action_mean.detach()
    sigma = self.policy.action_std.detach()

    self._pending = dict(
      policy_obs=policy_obs.detach(),
      prop_obs=prop_obs.detach(),
      critic_obs=critic_obs.detach(),
      actions=raw_action.detach(),
      values=value,
      actions_log_prob=log_prob,
      mu=mu,
      sigma=sigma,
    )
    return recons_actions.detach()

  def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    assert self.storage is not None
    self.storage.add(rewards=rewards, dones=dones, **self._pending)

  def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
    assert self.storage is not None
    with torch.no_grad():
      last_values = self.policy.evaluate(last_critic_obs)
    self.storage.compute_returns(last_values, self.cfg.gamma, self.cfg.lam)

  # ---------- update ---------- #

  def update(self) -> dict[str, float]:
    assert self.storage is not None
    cfg = self.cfg
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    n_steps = 0

    for batch in self.storage.mini_batch_generator(
      cfg.num_mini_batches, cfg.num_learning_epochs
    ):
      # Re-evaluate actor / critic at current parameters.
      self.policy.update_distribution(batch["policy_obs"])
      log_prob_batch = self.policy.get_actions_log_prob(batch["actions"])
      mu_batch = self.policy.action_mean
      sigma_batch = self.policy.action_std
      entropy_batch = self.policy.entropy
      value_batch = self.policy.evaluate(batch["critic_obs"]).flatten()

      # Adaptive learning-rate via KL between old and new distribution.
      if cfg.schedule == "adaptive" and cfg.desired_kl > 0:
        with torch.no_grad():
          kl = torch.sum(
            torch.log(sigma_batch / (batch["sigma"] + 1e-5) + 1e-5)
            + (batch["sigma"].pow(2) + (batch["mu"] - mu_batch).pow(2))
            / (2.0 * sigma_batch.pow(2))
            - 0.5,
            dim=-1,
          )
          kl_mean = kl.mean()
          if kl_mean > cfg.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < cfg.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate

      ratio = torch.exp(log_prob_batch - batch["actions_log_prob"])
      surrogate = -batch["advantages"] * ratio
      surrogate_clipped = -batch["advantages"] * torch.clamp(
        ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if cfg.use_clipped_value_loss:
        value_clipped = batch["values"] + (value_batch - batch["values"]).clamp(
          -cfg.clip_param, cfg.clip_param
        )
        value_losses = (value_batch - batch["returns"]).pow(2)
        value_losses_clipped = (value_clipped - batch["returns"]).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch["returns"] - value_batch).pow(2).mean()

      loss = (
        surrogate_loss
        + cfg.value_loss_coef * value_loss
        - cfg.entropy_coef * entropy_batch.mean()
      )

      self.optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(
        [p for p in self.policy.parameters() if p.requires_grad],
        cfg.max_grad_norm,
      )
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      n_steps += 1

    self.storage.clear()
    inv = 1.0 / max(n_steps, 1)
    return {
      "loss/surrogate": mean_surrogate_loss * inv,
      "loss/value": mean_value_loss * inv,
      "loss/entropy": mean_entropy * inv,
      "schedule/learning_rate": self.learning_rate,
    }
