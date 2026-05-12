"""CNN encoder with an extra MLP projection head.

Wraps ``rsl_rl.modules.CNN`` so the flattened/pooled feature is projected to a
fixed-size latent via a single ``Linear + activation`` before being concatenated
with 1D observations.

Use case: low-resolution height-scan or similar small images where you want a
larger fused latent (e.g. 128) than the bare ``global_pool="avg"`` output would
give (e.g. 32 channels).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import CNN
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


class CNNWithProjection(nn.Module):
  """``rsl_rl.modules.CNN`` followed by a ``Linear + activation`` projection.

  Mirrors the ``CNN`` interface so :class:`CNNProjModel` can drop it into
  ``CNNModel.cnns`` unchanged. ``output_channels`` is always ``None`` (the
  output is already flattened); ``output_dim`` is the projected dimension.
  """

  def __init__(
    self,
    input_dim: tuple[int, int],
    input_channels: int,
    proj_dim: int,
    proj_activation: str = "elu",
    **cnn_kwargs: Any,
  ) -> None:
    super().__init__()
    # The base CNN must produce a flat vector for the Linear head.
    cnn_kwargs.setdefault("flatten", True)
    self.cnn = CNN(
      input_dim=input_dim,
      input_channels=input_channels,
      **cnn_kwargs,
    )
    cnn_out_dim = self.cnn.output_dim
    if not isinstance(cnn_out_dim, int):
      raise ValueError(
        "CNNWithProjection requires a flattened CNN output; got "
        f"output_dim={cnn_out_dim}. Use global_pool='avg'/'max' or flatten=True."
      )
    self.proj = nn.Sequential(
      nn.Linear(cnn_out_dim, proj_dim),
      resolve_nn_activation(proj_activation),
    )
    self._output_dim = int(proj_dim)

  @property
  def output_dim(self) -> int:
    return self._output_dim

  @property
  def output_channels(self) -> None:
    return None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.proj(self.cnn(x))


class CNNProjModel(CNNModel):
  """CNN model whose 2D encoders include a Linear projection head.

  Drop-in replacement for ``rsl_rl.models.CNNModel``. Each 2D obs group is
  encoded by a :class:`CNNWithProjection` instead of a bare ``CNN``. The
  per-group projection dimension is read from ``cnn_cfg["proj_dim"]``
  (with optional ``proj_activation``, default ``"elu"``).
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    cnn_cfg: dict[str, dict] | dict[str, Any],
    cnns: nn.ModuleDict | None = None,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
  ) -> None:
    self._get_obs_dim(obs, obs_groups, obs_set)

    if cnns is not None:
      if set(cnns.keys()) != set(self.obs_groups_2d):
        raise ValueError(
          "The 2D observations must be identical for all models sharing CNN encoders."
        )
      print(
        "Sharing CNN encoders between models, the CNN "
        "configurations of the receiving model are ignored."
      )
      _cnns: dict[str, nn.Module] | nn.ModuleDict = cnns
    else:
      if not all(isinstance(v, dict) for v in cnn_cfg.values()):
        cnn_cfg = {group: cnn_cfg for group in self.obs_groups_2d}
      assert len(cnn_cfg) == len(self.obs_groups_2d), (
        "The number of CNN configurations must match the "
        "number of 2D observation groups."
      )
      _cnns = {}
      for idx, obs_group in enumerate(self.obs_groups_2d):
        group_cfg = dict(cnn_cfg[obs_group])
        proj_dim = group_cfg.pop("proj_dim", None)
        proj_activation = group_cfg.pop("proj_activation", activation)
        if proj_dim is None:
          raise ValueError(
            f"CNNProjModel requires 'proj_dim' in cnn_cfg for group '{obs_group}'."
          )
        _cnns[obs_group] = CNNWithProjection(
          input_dim=self.obs_dims_2d[idx],
          input_channels=self.obs_channels_2d[idx],
          proj_dim=int(proj_dim),
          proj_activation=proj_activation,
          **group_cfg,
        )

    self.cnn_latent_dim = 0
    for cnn in _cnns.values():
      if cnn.output_channels is not None:
        raise ValueError(
          "The output of the CNN must be flattened before passing it to the MLP."
        )
      self.cnn_latent_dim += int(cnn.output_dim)  # type: ignore[arg-type]

    MLPModel.__init__(
      self,
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    if isinstance(_cnns, nn.ModuleDict):
      self.cnns = _cnns
    else:
      self.cnns = nn.ModuleDict(_cnns)
