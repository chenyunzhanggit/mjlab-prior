"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

# CNN encoder for the 21x33 height-scan, followed by a Linear(32, 128) + ELU
# projection. Output: 128-d latent concatenated with prop obs before the actor
# MLP. The same encoder is reused by the critic via share_cnn_encoders=True.
_HEIGHT_CNN_CFG = {
  "output_channels": [16, 32],
  "kernel_size": [5, 3],
  "stride": [2, 2],
  "padding": "zeros",
  "activation": "elu",
  "max_pool": False,
  "global_pool": "avg",  # -> 32-d
  "proj_dim": 128,  # Linear(32, 128) + ELU
  "proj_activation": "elu",
}
_HEIGHT_MODEL_CLS = "mjlab.rl.cnn_proj:CNNProjModel"


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name=_HEIGHT_MODEL_CLS,
      cnn_cfg=_HEIGHT_CNN_CFG,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      class_name=_HEIGHT_MODEL_CLS,
      cnn_cfg=_HEIGHT_CNN_CFG,
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      share_cnn_encoders=True,
    ),
    obs_groups={
      "actor": ("actor", "height"),
      "critic": ("critic", "height"),
    },
    experiment_name="g1_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
