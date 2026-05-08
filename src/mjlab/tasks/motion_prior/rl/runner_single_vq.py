"""Single-teacher VQ-VAE motion-prior distillation runner.

Single-env counterpart to :class:`MotionPriorVQOnPolicyRunner`. Subclasses
:class:`MotionPriorSingleOnPolicyRunner` so the rollout loop /
``learn`` / episode-stats plumbing is reused; only policy / algorithm
construction, the per-step action accessor, the per-epoch loss call,
logging, and save/load are VQ-specific.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from tensordict import TensorDict

from mjlab.tasks.motion_prior.rl.algorithms.distillation_motion_prior_single_vq import (
  DistillationMotionPriorSingleVQ,
  DistillationSingleVQLossCfg,
)
from mjlab.tasks.motion_prior.rl.policies.motion_prior_single_encoder_vq_policy import (
  MotionPriorSingleEncoderVQPolicy,
)
from mjlab.tasks.motion_prior.rl.runner_single import (
  MotionPriorSingleOnPolicyRunner,
)


def _t(td: TensorDict, key: str) -> torch.Tensor:
  return cast(torch.Tensor, td[key])


class MotionPriorSingleVQOnPolicyRunner(MotionPriorSingleOnPolicyRunner):
  """Single-env VQ-VAE motion-prior distillation runner."""

  policy: MotionPriorSingleEncoderVQPolicy  # type: ignore[assignment]
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
    """Build the single-encoder VQ policy + algorithm from ``train_cfg``."""
    teacher_policy_path = train_cfg["teacher_policy_path"]
    policy_cfg = train_cfg.get("policy", {})

    self.policy = MotionPriorSingleEncoderVQPolicy(
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
      num_code=int(policy_cfg.get("num_code", 2048)),
      code_dim=int(policy_cfg.get("code_dim", 64)),
      ema_decay=float(policy_cfg.get("ema_decay", 0.99)),
      activation=str(policy_cfg.get("activation", "elu")),
      device=device,
    )
    # AR(1) reshape uses code_dim (raw encoder output width) for VQ.
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

  # --------------------------------------------------------------------- #
  # Rollout / epoch step (override only the bits that change for VQ)      #
  # --------------------------------------------------------------------- #

  def _policy_step_action(self, obs: TensorDict) -> torch.Tensor:
    """VQ ``forward`` returns
    ``(student_act, q, enc, mp_code, commit, perplexity)`` — student
    action is the 1st element."""
    sa_step, *_ = self.policy.forward(
      _t(obs, "student"), _t(obs, "teacher_t"), training=False
    )
    return sa_step

  def _epoch_step(
    self,
    rollout: dict[str, torch.Tensor],
    cur_iter_num: int,
  ) -> dict[str, float]:
    n_envs = self.env.num_envs
    T = self.num_steps_per_env
    Z = self._latent_z_dims  # = code_dim for VQ

    sa, q, enc, mp_code, commit, perplexity = self.policy.forward(
      rollout["prop_obs"], rollout["teacher_obs"], training=True
    )
    assert commit is not None  # ``training=True`` always returns a real commit.

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

  # --------------------------------------------------------------------- #
  # Logging                                                               #
  # --------------------------------------------------------------------- #

  # VQ shares the same console / writer log layout as the single VAE;
  # the differences are the loss-dict keys (``loss/commit``, ``loss/mp``,
  # ``perplexity``), which are surfaced uniformly via the shared
  # ``_log_iteration`` body inherited from ``MotionPriorSingleOnPolicyRunner``.

  # --------------------------------------------------------------------- #
  # Save / load                                                           #
  # --------------------------------------------------------------------- #

  def save(self, path: str, infos: dict | None = None) -> None:
    """Persist trainable VQ submodules + quantizer buffers + iter.

    Frozen teacher is NOT saved (re-loaded from ``teacher_policy_path`` on
    construction).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
      "encoder": self.policy.encoder.state_dict(),
      "decoder": self.policy.decoder.state_dict(),
      "motion_prior": self.policy.motion_prior.state_dict(),
      # Quantizer state_dict carries codebook + code_sum + code_count
      # buffers (registered in EMAQuantizer).
      "quantizer": self.policy.quantizer.state_dict(),
      "optimizer": self.alg.optimizer.state_dict(),
      "iter": self.current_learning_iteration,
      "infos": infos or {},
    }
    torch.save(state, path)
    print(f"[MotionPriorSingleVQ] saved checkpoint to {path}")

  def get_inference_policy(
    self,
    device: str | torch.device | None = None,
    path: str | None = None,
  ):
    """Return a callable ``(obs_td) -> action`` running encoder or deploy."""
    import os

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
        # deploy path: motion_prior(prop) → quantize → decode
        mp_code = self.policy.motion_prior_inference(prop)
        q = self.policy.quantizer_inference(mp_code)
        return self.policy.decoder_inference(prop, q)

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
    self.policy.decoder.load_state_dict(state["decoder"], strict=strict)
    self.policy.motion_prior.load_state_dict(state["motion_prior"], strict=strict)
    self.policy.quantizer.load_state_dict(state["quantizer"], strict=strict)
    if "optimizer" in state and state["optimizer"]:
      self.alg.optimizer.load_state_dict(state["optimizer"])
    self.current_learning_iteration = int(state.get("iter", 0))
    return state.get("infos", {})
