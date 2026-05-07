"""Multi-motion tracking PPO runner.

Thin subclass of :class:`mjlab.rl.MjlabOnPolicyRunner` whose only purpose
is to accept the ``registry_name`` kwarg that ``mjlab.scripts.train``
forwards to all tracking-style tasks (the single-motion
:class:`mjlab.tasks.tracking.rl.MotionTrackingOnPolicyRunner` does the
same). For multi-motion the W&B motion-artifact registry doesn't apply
— motion data comes from a directory glob via ``--motion-path`` — so we
simply store the kwarg and ignore it.

ONNX export uses :class:`MjlabOnPolicyRunner`'s policy-only path. The
single-motion runner's ``_OnnxMotionModel`` (which bundles ``motion.joint_pos``
etc. into the ONNX graph) doesn't apply: with a clip pool there's no
single ``motion`` to embed, and deploy code reads clips from a separate
data path.
"""

from __future__ import annotations

from rsl_rl.env.vec_env import VecEnv

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper


class MultiMotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    # Accepted for ``train.py`` API compatibility; not used by multi-motion
    # since motion data comes from a directory glob, not a W&B artifact.
    self.registry_name = registry_name
