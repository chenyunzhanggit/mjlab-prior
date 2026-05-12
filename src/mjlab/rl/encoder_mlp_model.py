"""Actor/critic model with per-group MLP encoders alongside CNN encoders.

Extends rsl_rl's :class:`CNNModel` with an additional ``mlp_cfg`` channel: any
observation group named in ``mlp_cfg`` is routed through its own
``rsl_rl.modules.MLP`` encoder (instead of being flattened straight into the
1D latent), and the resulting per-group latents are concatenated with the 1D
and CNN latents before the trunk MLP.

Designed for the depth-loco history use case: ``lower_actor_history`` is a
``[B, T, D]`` tensor (5-step rolling buffer of proprioceptive obs) which we
flatten over time and compress with a small MLP into a fixed-size latent.

Resolved from cfg via class_name
``"mjlab.rl.encoder_mlp_model:EncoderMLPModel"``.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.modules import CNN, MLP, HiddenState
from tensordict import TensorDict


class EncoderMLPModel(CNNModel):
  """CNNModel + per-group MLP encoders for arbitrary-shape observation groups."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
    cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    mlp_cfg: dict[str, dict] | None = None,
  ) -> None:
    """Initialize.

    Args:
      mlp_cfg: ``{group_name: {hidden_dims, output_dim, activation,
        flatten_time}}``. Each named group bypasses the 1D concat path and
        instead is fed to its own MLP encoder. Use ``flatten_time=True`` for
        ``[B, T, D]`` history-style inputs.
    """
    self._mlp_cfg_raw: dict[str, dict] = mlp_cfg or {}
    # Snapshot encoder-routed group shapes BEFORE _get_obs_dim filters them
    # out of the 1D pool. Keys: group name -> tuple(non-batch dims).
    self._mlp_obs_shapes: dict[str, tuple[int, ...]] = {}
    self._mlp_obs_groups_ordered: list[str] = []
    for g in obs_groups.get(obs_set, []):
      if g in self._mlp_cfg_raw:
        self._mlp_obs_shapes[g] = tuple(obs[g].shape[1:])
        self._mlp_obs_groups_ordered.append(g)

    # CNNModel.__init__ requires at least one 2D group. Synthesise one if the
    # caller has neither cnn nor 2D obs (rare, but possible for pure-encoder
    # configs); otherwise just delegate.
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
      cnn_cfg=cnn_cfg,
      cnns=cnns,
    )

    # Build the per-group MLP encoders. Total latent contribution is summed
    # into ``self.encoder_mlp_latent_dim`` and added to the trunk input dim.
    encoders: dict[str, nn.Module] = {}
    self.encoder_mlp_latent_dim = 0
    self._encoder_flatten_time: dict[str, bool] = {}
    for g in self._mlp_obs_groups_ordered:
      shape = self._mlp_obs_shapes[g]
      cfg = dict(self._mlp_cfg_raw[g])
      flatten_time = cfg.pop("flatten_time", True)
      hidden = cfg.pop("hidden_dims", [128])
      out_dim = int(cfg.pop("output_dim", 64))
      act = cfg.pop("activation", "elu")
      if flatten_time:
        in_dim = int(torch.tensor(shape).prod().item())
      else:
        # No flatten_time: assume [B, D] input.
        if len(shape) != 1:
          raise ValueError(
            f"EncoderMLPModel: group '{g}' has shape {shape}; with "
            f"flatten_time=False it must be 1D (post-batch)."
          )
        in_dim = shape[0]
      mlp = MLP(in_dim, out_dim, list(hidden), activation=act)
      encoders[g] = mlp
      self._encoder_flatten_time[g] = flatten_time
      self.encoder_mlp_latent_dim += out_dim

    self.encoder_mlps = nn.ModuleDict(encoders)

    # Re-create the trunk MLP with the corrected input dim (parent already
    # built one with the wrong dim because it didn't know about our encoders
    # at the time of super().__init__). We re-use mlp_output_dim discovered
    # from the existing self.mlp.
    if encoders:
      old_mlp = self.mlp
      last_layer = old_mlp[-1]  # type: ignore[index]
      assert isinstance(last_layer, nn.Linear)
      mlp_out_dim = int(last_layer.out_features)
      new_input_dim = self._get_latent_dim()
      new_mlp = MLP(new_input_dim, mlp_out_dim, list(hidden_dims), activation)
      if self.distribution is not None:
        self.distribution.init_mlp_weights(new_mlp)
      self.mlp = new_mlp

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    """Override CNNModel._get_obs_dim.

    Behaviour changes:
    - Groups listed in ``mlp_cfg`` are excluded from both 1D and 2D pools and
      handled separately by the per-group MLP encoders.
    - Tolerates ND inputs (e.g. ``[B, T, D]`` history) for those groups; the
      encoder's ``flatten_time=True`` setting will fold the time axis.
    """
    active_obs_groups = obs_groups[obs_set]
    obs_dim_1d = 0
    obs_groups_1d: list[str] = []
    obs_dims_2d: list[tuple[int, int]] = []
    obs_channels_2d: list[int] = []
    obs_groups_2d: list[str] = []

    encoder_groups = set(self._mlp_cfg_raw.keys())

    for g in active_obs_groups:
      if g in encoder_groups:
        # Encoder-routed group: skip here, handled by per-group MLP.
        continue
      shape = obs[g].shape
      if len(shape) == 4:  # [B, C, H, W]
        obs_groups_2d.append(g)
        obs_dims_2d.append(shape[2:4])
        obs_channels_2d.append(shape[1])
      elif len(shape) == 2:  # [B, D]
        obs_groups_1d.append(g)
        obs_dim_1d += shape[-1]
      else:
        raise ValueError(
          f"EncoderMLPModel: group '{g}' has shape {tuple(shape)}; only 2D "
          f"(flat) or 4D (image) groups are supported in the trunk path. "
          f"Add '{g}' to mlp_cfg for higher-dim inputs."
        )

    self.obs_dims_2d = obs_dims_2d
    self.obs_channels_2d = obs_channels_2d
    self.obs_groups_2d = obs_groups_2d
    return obs_groups_1d, obs_dim_1d

  def _get_latent_dim(self) -> int:
    """Trunk-MLP input dim = 1D + CNN latent + sum of encoder MLP latents."""
    base = super()._get_latent_dim()
    # Use getattr so this also works during the parent __init__ call before
    # encoder_mlp_latent_dim has been assigned (in which case base is correct).
    return base + getattr(self, "encoder_mlp_latent_dim", 0)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Concatenate [1D + CNN + encoder MLP] latents."""
    base_latent = super().get_latent(obs, masks, hidden_state)
    if not self._mlp_obs_groups_ordered:
      return base_latent
    enc_latents: list[torch.Tensor] = []
    for g in self._mlp_obs_groups_ordered:
      x = obs[g]
      if self._encoder_flatten_time[g]:
        x = x.reshape(x.shape[0], -1)
      enc_latents.append(self.encoder_mlps[g](x))
    return torch.cat([base_latent, *enc_latents], dim=-1)

  def as_jit(self) -> nn.Module:
    return _TorchEncoderMLPModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxEncoderMLPModel(self, verbose)


class _TorchEncoderMLPModel(nn.Module):
  """JIT-export wrapper for EncoderMLPModel."""

  def __init__(self, model: EncoderMLPModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnns = nn.ModuleList(
      [copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d]
    )
    self.encoder_mlps = nn.ModuleList(
      [copy.deepcopy(model.encoder_mlps[g]) for g in model._mlp_obs_groups_ordered]
    )
    self._encoder_flatten = [
      bool(model._encoder_flatten_time[g]) for g in model._mlp_obs_groups_ordered
    ]
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(
    self,
    obs_1d: torch.Tensor,
    obs_2d: list[torch.Tensor],
    obs_enc: list[torch.Tensor],
  ) -> torch.Tensor:
    latent_1d = self.obs_normalizer(obs_1d)
    parts: list[torch.Tensor] = [latent_1d]
    for i, cnn in enumerate(self.cnns):
      parts.append(cnn(obs_2d[i]))
    for i, enc in enumerate(self.encoder_mlps):
      x = obs_enc[i]
      if self._encoder_flatten[i]:
        x = x.reshape(x.shape[0], -1)
      parts.append(enc(x))
    out = self.mlp(torch.cat(parts, dim=-1))
    return self.deterministic_output(out)

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxEncoderMLPModel(nn.Module):
  """ONNX-export wrapper for EncoderMLPModel."""

  def __init__(self, model: EncoderMLPModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnns = nn.ModuleList(
      [copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d]
    )
    self.encoder_mlps = nn.ModuleList(
      [copy.deepcopy(model.encoder_mlps[g]) for g in model._mlp_obs_groups_ordered]
    )
    self._encoder_flatten = [
      bool(model._encoder_flatten_time[g]) for g in model._mlp_obs_groups_ordered
    ]
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

    self.obs_groups_2d = list(model.obs_groups_2d)
    self.obs_dims_2d = list(model.obs_dims_2d)
    self.obs_channels_2d = list(model.obs_channels_2d)
    self.obs_dim_1d = model.obs_dim
    self._enc_groups = list(model._mlp_obs_groups_ordered)
    self._enc_shapes = [model._mlp_obs_shapes[g] for g in self._enc_groups]

  def forward(
    self,
    obs_1d: torch.Tensor,
    *rest: torch.Tensor,
  ) -> torch.Tensor:
    n2 = len(self.cnns)
    obs_2d = list(rest[:n2])
    obs_enc = list(rest[n2:])
    latent_1d = self.obs_normalizer(obs_1d)
    parts: list[torch.Tensor] = [latent_1d]
    for i, cnn in enumerate(self.cnns):
      parts.append(cnn(obs_2d[i]))
    for i, enc in enumerate(self.encoder_mlps):
      x = obs_enc[i]
      if self._encoder_flatten[i]:
        x = x.reshape(x.shape[0], -1)
      parts.append(enc(x))
    out = self.mlp(torch.cat(parts, dim=-1))
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    dummy_1d = torch.zeros(1, self.obs_dim_1d)
    dummy_2d = []
    for i in range(len(self.obs_groups_2d)):
      h, w = self.obs_dims_2d[i]
      c = self.obs_channels_2d[i]
      dummy_2d.append(torch.zeros(1, c, h, w))
    dummy_enc = [torch.zeros(1, *shape) for shape in self._enc_shapes]
    return (dummy_1d, *dummy_2d, *dummy_enc)

  @property
  def input_names(self) -> list[str]:
    return ["obs", *self.obs_groups_2d, *self._enc_groups]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


# Suppress the "unused" import warning; CNN re-exported for convenience callers.
_ = CNN
