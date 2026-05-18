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
from mjlab.tasks.amp_velocity.rl.discriminator_multi import DiscriminatorMulti
from mjlab.tasks.amp_velocity.rl.motion_loader import AMPLoader
from mjlab.tasks.amp_velocity.rl.motion_loader_joint import AMPJointLoader
from mjlab.tasks.amp_velocity.rl.normalizer import Normalizer
from mjlab.tasks.amp_velocity.rl.replay_buffer import ReplayBuffer
from mjlab.tasks.amp_velocity.rl.replay_buffer_multi import ReplayBufferMulti


class AMPPPO(PPO):
  """PPO with a co-trained AMP discriminator.

  Supports two AMP variants distinguished by the discriminator type:

  - :class:`Discriminator` + :class:`AMPLoader` + :class:`ReplayBuffer`:
    classic AMP_mjlab (s, s') path. Body-space features.
  - :class:`DiscriminatorMulti` + :class:`AMPJointLoader` +
    :class:`ReplayBufferMulti`: telebotM2-style K-frame stack path.
    Joint-space features.

  The choice is implicit in the objects passed in; ``update`` dispatches via
  ``isinstance`` checks. All other behaviour (checkpointing, train/eval mode,
  PPO update) is shared.

  Args:
    discriminator: AMP discriminator (single-frame pair or K-frame stack).
    amp_data: Expert dataset matching the discriminator variant.
    amp_normalizer: Running-stats normalizer for AMP features.
    amp_storage: Replay buffer matching the discriminator variant.
    discriminator_lr: Adam learning rate for the discriminator optimizer.
    discriminator_weight_decay: Weight decay for the discriminator optimizer.
    grad_pen_lambda: Coefficient on the LSGAN gradient-penalty term.
    *args, **kwargs: Forwarded to ``PPO.__init__``.
  """

  def __init__(
    self,
    *args: Any,
    discriminator: Discriminator | DiscriminatorMulti,
    amp_data: AMPLoader | AMPJointLoader,
    amp_normalizer: Normalizer,
    amp_storage: ReplayBuffer | ReplayBufferMulti,
    discriminator_lr: float = 1e-4,
    discriminator_weight_decay: float = 1e-2,
    discriminator_head_weight_decay: float | None = None,
    grad_pen_lambda: float = 10.0,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.discriminator = discriminator.to(self.device)
    self.amp_data = amp_data
    self.amp_normalizer = amp_normalizer
    self.amp_storage = amp_storage
    self.grad_pen_lambda = grad_pen_lambda
    self._is_multi = isinstance(discriminator, DiscriminatorMulti)
    if self._is_multi:
      assert isinstance(amp_data, AMPJointLoader), (
        "DiscriminatorMulti requires AMPJointLoader as amp_data; got "
        f"{type(amp_data).__name__}"
      )
      assert isinstance(amp_storage, ReplayBufferMulti), (
        "DiscriminatorMulti requires ReplayBufferMulti as amp_storage; got "
        f"{type(amp_storage).__name__}"
      )
    # Split disc params into trunk vs. output head and give the head a much
    # larger weight decay. This is the AMP_mjlab trick that keeps the LSGAN
    # logits from running away to ±∞ — without it the disc saturates, expert
    # / policy preds pin near ±1, and the AMP reward collapses to ~0. If
    # ``discriminator_head_weight_decay`` is None, both groups share
    # ``discriminator_weight_decay`` (i.e. the old single-group behaviour).
    head_wd = (
      discriminator_head_weight_decay
      if discriminator_head_weight_decay is not None
      else discriminator_weight_decay
    )
    self.disc_optimizer = torch.optim.Adam(
      [
        {
          "params": self.discriminator.trunk.parameters(),
          "weight_decay": discriminator_weight_decay,
          "name": "amp_trunk",
        },
        {
          "params": self.discriminator.amp_linear.parameters(),
          "weight_decay": head_wd,
          "name": "amp_head",
        },
      ],
      lr=discriminator_lr,
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

    for policy_sample, expert_sample in zip(
      amp_policy_gen, amp_expert_gen, strict=True
    ):
      disc_loss, grad_pen, policy_d, expert_d = self._disc_step(
        policy_sample, expert_sample
      )

      mean_disc_loss += disc_loss
      mean_grad_pen += grad_pen
      mean_policy_pred += policy_d
      mean_expert_pred += expert_d
      effective_updates += 1

    denom = max(effective_updates, 1)
    loss_dict["amp_disc"] = mean_disc_loss / denom
    loss_dict["amp_grad_pen"] = mean_grad_pen / denom
    loss_dict["amp_policy_pred"] = mean_policy_pred / denom
    loss_dict["amp_expert_pred"] = mean_expert_pred / denom
    return loss_dict

  def _disc_step(
    self,
    policy_sample,
    expert_sample,
  ) -> tuple[float, float, float, float]:
    """Run one discriminator gradient step, dispatching by variant.

    Body variant (pair): ``policy_sample`` / ``expert_sample`` are each
    ``(state, next_state)`` tuples; discriminator takes the concatenation.

    Joint variant (stack): each is a single ``(B, K, d)`` tensor; discriminator
    takes the flattened stack.

    Returns the per-step scalars used for logging.
    """
    if self._is_multi:
      assert isinstance(self.discriminator, DiscriminatorMulti)
      policy_states = policy_sample
      expert_states = expert_sample

      if self.amp_normalizer is not None:
        with torch.no_grad():
          policy_states_n = self.amp_normalizer.normalize_torch(
            policy_states, self.device
          )
          expert_states_n = self.amp_normalizer.normalize_torch(
            expert_states, self.device
          )
      else:
        policy_states_n = policy_states
        expert_states_n = expert_states

      policy_d = self.discriminator(policy_states_n.flatten(1))
      expert_d = self.discriminator(expert_states_n.flatten(1))

      expert_loss = nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
      policy_loss = nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
      disc_loss = 0.5 * (expert_loss + policy_loss)

      grad_pen = self.discriminator.compute_grad_pen(
        expert_states_n, lambda_=self.grad_pen_lambda
      )

      total = disc_loss + grad_pen
      self.disc_optimizer.zero_grad()
      total.backward()
      self.disc_optimizer.step()

      if self.amp_normalizer is not None:
        # Feed already-normalized tensors back into the normalizer, matching
        # AMP_mjlab's behavior. This causes (mean, var) to drift toward (0, 1)
        # asymptotically so normalization stays near-identity once warm — which
        # empirically stabilizes LSGAN training by keeping disc inputs in a
        # consistent range rather than chasing a moving policy/expert mixture.
        # Flatten K-frame stacks to (B*K, d) before updating running stats.
        self.amp_normalizer.update(
          policy_states_n.detach().reshape(-1, policy_states_n.shape[-1]).cpu().numpy()
        )
        self.amp_normalizer.update(
          expert_states_n.detach().reshape(-1, expert_states_n.shape[-1]).cpu().numpy()
        )

      return (
        disc_loss.item(),
        grad_pen.item(),
        policy_d.mean().item(),
        expert_d.mean().item(),
      )

    # Body variant: (s, s') pair.
    assert isinstance(self.discriminator, Discriminator)
    policy_state, policy_next_state = policy_sample
    expert_state, expert_next_state = expert_sample

    if self.amp_normalizer is not None:
      with torch.no_grad():
        policy_state_n = self.amp_normalizer.normalize_torch(policy_state, self.device)
        policy_next_state_n = self.amp_normalizer.normalize_torch(
          policy_next_state, self.device
        )
        expert_state_n = self.amp_normalizer.normalize_torch(expert_state, self.device)
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

    if self.amp_normalizer is not None:
      # See joint-variant branch above: feed already-normalized tensors back
      # so running stats drift toward (0, 1) and disc inputs stay stable.
      self.amp_normalizer.update(policy_state_n.detach().cpu().numpy())
      self.amp_normalizer.update(expert_state_n.detach().cpu().numpy())

    return (
      disc_loss.item(),
      grad_pen.item(),
      policy_d.mean().item(),
      expert_d.mean().item(),
    )

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
