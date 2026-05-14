"""PPO + AMP discriminator joint optimization.

Extends rsl_rl 5.2's ``PPO`` by:
  - holding a discriminator with its own optimizer (kept separate from the
    PPO optimizer so weight decay and learning rate can be tuned per-side);
  - drawing two extra streams each minibatch — policy AMP samples from a
    ReplayBuffer and expert samples from an AMPLoader — and minimizing the
    LSGAN objective + gradient penalty on the discriminator;
  - extending ``save`` / ``load`` to round-trip the discriminator weights,
    its optimizer state, and the AMP input normalizer.

The runner side is responsible for:
  - inserting policy AMP transitions into ``self.amp_storage``;
  - calling ``predict_amp_reward`` and lerp-blending into env rewards before
    invoking ``process_env_step``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.algorithms.ppo import PPO

from mjlab.tasks.amp_velocity.rl.discriminator import Discriminator
from mjlab.tasks.amp_velocity.rl.motion_loader import AMPLoader
from mjlab.tasks.amp_velocity.rl.normalizer import Normalizer
from mjlab.tasks.amp_velocity.rl.replay_buffer import ReplayBuffer


class AMPPPO(PPO):
  """PPO with a co-trained AMP discriminator.

  Args:
    discriminator: The AMP discriminator module.
    amp_data: The expert motion dataset (``AMPLoader``).
    amp_normalizer: Running-stats normalizer for AMP features.
    amp_storage: Replay buffer holding policy-side AMP (s, s') pairs.
    discriminator_lr: Adam learning rate for the discriminator optimizer.
    discriminator_weight_decay: Weight decay for the discriminator optimizer.
    grad_pen_lambda: Coefficient on the LSGAN gradient-penalty term.
    *args, **kwargs: Forwarded to ``PPO.__init__``.
  """

  def __init__(
    self,
    *args: Any,
    discriminator: Discriminator,
    amp_data: AMPLoader,
    amp_normalizer: Normalizer,
    amp_storage: ReplayBuffer,
    discriminator_lr: float = 1e-4,
    discriminator_weight_decay: float = 1e-2,
    grad_pen_lambda: float = 10.0,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.discriminator = discriminator.to(self.device)
    self.amp_data = amp_data
    self.amp_normalizer = amp_normalizer
    self.amp_storage = amp_storage
    self.grad_pen_lambda = grad_pen_lambda
    self.disc_optimizer = torch.optim.Adam(
      self.discriminator.parameters(),
      lr=discriminator_lr,
      weight_decay=discriminator_weight_decay,
    )

  # ------------------------------------------------------------------ #
  # Discriminator training                                             #
  # ------------------------------------------------------------------ #

  def _mini_batch_size(self) -> int:
    """Per-minibatch sample count, matching PPO's storage minibatch size."""
    return (
      self.storage.num_envs * self.storage.num_transitions_per_env
    ) // self.num_mini_batches

  def update(self) -> dict[str, float]:
    """Run PPO update and co-train the discriminator over the same number
    of mini-batches.

    Returns a merged loss dict with PPO keys (``value``, ``surrogate``,
    ``entropy``, optional ``rnd``/``symmetry``) plus AMP keys
    (``amp_disc``, ``amp_grad_pen``, ``amp_policy_pred``, ``amp_expert_pred``).
    """
    # Run base PPO update first so its mini_batch_generator advances exactly
    # ``num_learning_epochs * num_mini_batches`` steps. Reuse the same count
    # for the discriminator so the two sides see matched compute budgets.
    loss_dict = super().update()

    n_updates = self.num_learning_epochs * self.num_mini_batches
    batch_size = self._mini_batch_size()

    amp_policy_gen = self.amp_storage.feed_forward_generator(n_updates, batch_size)
    amp_expert_gen = self.amp_data.feed_forward_generator(n_updates, batch_size)

    mean_disc_loss = 0.0
    mean_grad_pen = 0.0
    mean_policy_pred = 0.0
    mean_expert_pred = 0.0
    effective_updates = 0

    for (policy_state, policy_next_state), (
      expert_state,
      expert_next_state,
    ) in zip(amp_policy_gen, amp_expert_gen, strict=True):
      if self.amp_normalizer is not None:
        with torch.no_grad():
          policy_state_n = self.amp_normalizer.normalize_torch(
            policy_state, self.device
          )
          policy_next_state_n = self.amp_normalizer.normalize_torch(
            policy_next_state, self.device
          )
          expert_state_n = self.amp_normalizer.normalize_torch(
            expert_state, self.device
          )
          expert_next_state_n = self.amp_normalizer.normalize_torch(
            expert_next_state, self.device
          )
      else:
        policy_state_n = policy_state
        policy_next_state_n = policy_next_state
        expert_state_n = expert_state
        expert_next_state_n = expert_next_state

      policy_d = self.discriminator(
        torch.cat([policy_state_n, policy_next_state_n], dim=-1)
      )
      expert_d = self.discriminator(
        torch.cat([expert_state_n, expert_next_state_n], dim=-1)
      )

      # LSGAN: experts → +1, policy → -1.
      expert_loss = nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
      policy_loss = nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
      disc_loss = 0.5 * (expert_loss + policy_loss)

      grad_pen = self.discriminator.compute_grad_pen(
        expert_state_n, expert_next_state_n, lambda_=self.grad_pen_lambda
      )

      total = disc_loss + grad_pen

      self.disc_optimizer.zero_grad()
      total.backward()
      self.disc_optimizer.step()

      # Update normalizer with the *un-normalized* batch (numpy-side).
      if self.amp_normalizer is not None:
        self.amp_normalizer.update(policy_state.detach().cpu().numpy())
        self.amp_normalizer.update(expert_state.detach().cpu().numpy())

      mean_disc_loss += disc_loss.item()
      mean_grad_pen += grad_pen.item()
      mean_policy_pred += policy_d.mean().item()
      mean_expert_pred += expert_d.mean().item()
      effective_updates += 1

    denom = max(effective_updates, 1)
    loss_dict["amp_disc"] = mean_disc_loss / denom
    loss_dict["amp_grad_pen"] = mean_grad_pen / denom
    loss_dict["amp_policy_pred"] = mean_policy_pred / denom
    loss_dict["amp_expert_pred"] = mean_expert_pred / denom
    return loss_dict

  # ------------------------------------------------------------------ #
  # Train / eval mode                                                  #
  # ------------------------------------------------------------------ #

  def train_mode(self) -> None:
    super().train_mode()
    self.discriminator.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.discriminator.eval()

  # ------------------------------------------------------------------ #
  # Checkpointing                                                      #
  # ------------------------------------------------------------------ #

  def save(self) -> dict:
    """Augment the base PPO checkpoint with AMP-side state."""
    sd = super().save()
    sd["discriminator_state_dict"] = self.discriminator.state_dict()
    sd["disc_optimizer_state_dict"] = self.disc_optimizer.state_dict()
    sd["amp_normalizer"] = {
      "mean": self.amp_normalizer.mean,
      "var": self.amp_normalizer.var,
      "count": self.amp_normalizer.count,
    }
    return sd

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    """Load PPO + discriminator + normalizer state.

    Missing AMP keys are tolerated (e.g., loading a pure-PPO checkpoint as a
    warm-start) — only a warning is printed.
    """
    out = super().load(loaded_dict, load_cfg, strict)
    if "discriminator_state_dict" in loaded_dict:
      self.discriminator.load_state_dict(
        loaded_dict["discriminator_state_dict"], strict=strict
      )
    elif strict:
      print(
        "[AMPPPO] checkpoint missing 'discriminator_state_dict' — "
        "discriminator left at init."
      )
    if "disc_optimizer_state_dict" in loaded_dict:
      self.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
    if "amp_normalizer" in loaded_dict:
      stats = loaded_dict["amp_normalizer"]
      self.amp_normalizer.mean = stats["mean"]
      self.amp_normalizer.var = stats["var"]
      self.amp_normalizer.count = stats["count"]
    return out
