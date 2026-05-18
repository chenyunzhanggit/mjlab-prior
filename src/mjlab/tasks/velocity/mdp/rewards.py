from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error  # + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error  # + xy_error
  return torch.exp(-ang_vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      # Project terrain normal into body frame. When aligned with the terrain surface
      # this should be (0, 0, 1); XY measures tilt.
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_clearance requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights  # [B, F]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, F]
  delta = torch.abs(foot_height - target_height)  # [B, F]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      f"feet_swing_height requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stuck_penalty(
  env: ManagerBasedRlEnv,
  vel_threshold: float = 0.1,
  cmd_threshold: float = 0.1,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize being stuck: commanded to move but velocity along the command
  direction is too small (or negative).

  Returns 1.0 when (a) the projection of body-frame planar velocity onto the
  commanded body-frame planar velocity is below ``vel_threshold`` (i.e. the
  base is not making progress along the command direction -- this includes
  moving backwards or sliding sideways) and (b) the commanded planar speed
  exceeds ``cmd_threshold``. Apply a negative weight to turn into a penalty.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  base_vel_xy = asset.data.root_link_lin_vel_b[:, :2]  # [B, 2]
  cmd_xy = command[:, :2]  # [B, 2]
  cmd_norm = cmd_xy.norm(dim=-1)  # [B]
  # Unit command direction; safe when commanded speed < cmd_threshold (the
  # `active` mask below zeroes the result anyway).
  cmd_dir = cmd_xy / cmd_norm.clamp(min=1e-6).unsqueeze(-1)
  proj = (base_vel_xy * cmd_dir).sum(dim=-1)  # signed scalar [B]
  active = cmd_norm > cmd_threshold
  stuck = (proj < vel_threshold) & active
  return stuck.float()


def cheat_penalty(
  env: ManagerBasedRlEnv,
  yaw_threshold: float = 1.0,
  level_threshold: int = -1,
  command_name: str = "twist",
  cmd_threshold: float = 0.1,
  mode: str = "body_frame",
  terrain_classes: tuple[int, ...] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize moving in a direction that deviates from the commanded direction.

  Two modes:

  - ``"body_frame"`` (default, mjlab-native): compares body-frame actual velocity
    direction against body-frame commanded direction. Works for any cmd / yaw
    setup but cannot catch slow accumulating yaw drift.

  - ``"world_heading"`` (MoRE-style): penalizes ``|heading_w| > yaw_threshold``,
    i.e. the robot's world-frame heading deviates from +x. Only applied to envs
    flagged as ``is_forward_env`` by the velocity command (or matching
    ``terrain_classes``), which require the cmd to be forward-only and reset
    yaw to be locked at 0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  if mode == "body_frame":
    cmd_xy = command[:, :2]
    cmd_speed = torch.norm(cmd_xy, dim=1)
    active = cmd_speed > cmd_threshold

    base_vel_b = asset.data.root_link_lin_vel_b[:, :2]
    actual_dir = torch.atan2(base_vel_b[:, 1], base_vel_b[:, 0])
    desired_dir = torch.atan2(cmd_xy[:, 1], cmd_xy[:, 0])
    diff = actual_dir - desired_dir
    err = torch.atan2(torch.sin(diff), torch.cos(diff))
    cheat = (torch.abs(err) > yaw_threshold) & active
  elif mode == "world_heading":
    heading = asset.data.heading_w
    cheat = torch.abs(heading) > yaw_threshold
    if terrain_classes:
      terrain = env.scene.terrain
      env_class = getattr(terrain, "env_terrain_class", None) if terrain else None
      if env_class is None:
        raise ValueError(
          "cheat_penalty(mode='world_heading') with terrain_classes requires the "
          "scene terrain to expose 'env_terrain_class'."
        )
      forced_classes = torch.tensor(
        terrain_classes, device=cheat.device, dtype=env_class.dtype
      )
      mask = (env_class.unsqueeze(-1) == forced_classes).any(dim=-1)
      cheat = cheat & mask
    else:
      cmd_term = env.command_manager.get_term(command_name)
      is_forward = getattr(cmd_term, "is_forward_env", None)
      if is_forward is None:
        raise ValueError(
          f"cheat_penalty(mode='world_heading') requires the '{command_name}' "
          f"command term to expose 'is_forward_env'."
        )
      cheat = cheat & is_forward
  else:
    raise ValueError(
      f"Unknown cheat_penalty mode {mode!r}; expected 'body_frame' or 'world_heading'."
    )
  if level_threshold >= 0:
    terrain = env.scene.terrain
    if terrain is not None and getattr(terrain, "terrain_levels", None) is not None:
      gate = terrain.terrain_levels > level_threshold
      cheat = cheat & gate

  return cheat.float()


def feet_edge_penalty(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  contact_sensor_name: str,
  height_range_threshold: float = 0.04,
  level_threshold: int = 3,
) -> torch.Tensor:
  """Penalize feet landing on stair edges.

  For each foot in contact, computes the per-ray clearance range under the foot
  and flags the foot as on-edge when the range exceeds
  ``height_range_threshold`` (i.e. half of the foot is over a step, half over
  a tread). Sum across feet; pair with a negative weight.
  """
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"feet_edge_penalty requires a TerrainHeightSensor, got "
    f"{type(height_sensor).__name__}"
  )
  contact_sensor = env.scene[contact_sensor_name]
  assert isinstance(contact_sensor, ContactSensor), (
    f"feet_edge_penalty requires a ContactSensor, got {type(contact_sensor).__name__}"
  )

  raw = height_sensor.data
  # Reconstruct per-ray heights regardless of sensor reduction.
  F = height_sensor.num_frames
  N = height_sensor.num_rays_per_frame
  B = raw.distances.shape[0]
  frame_z = raw.frame_pos_w[:, :, 2]  # [B, F]
  hit_z = raw.hit_pos_w[:, :, 2].view(B, F, N)  # [B, F, N]
  heights = frame_z.unsqueeze(-1) - hit_z  # [B, F, N]

  dists = raw.distances.view(B, F, N)
  normal_z = raw.normals_w[:, :, 2].view(B, F, N)
  miss = dists < 0
  backface = (dists >= 0) & (normal_z < 0)
  invalid = miss | backface
  heights = torch.where(invalid, torch.zeros_like(heights), heights)

  per_foot_range = heights.max(dim=-1).values - heights.min(dim=-1).values  # [B, F]
  on_edge = per_foot_range > height_range_threshold  # [B, F]

  found = contact_sensor.data.found
  assert found is not None, "feet_edge_penalty needs contact sensor 'found' field."
  in_contact = found > 0  # [B, F_contact]
  if in_contact.shape[1] != on_edge.shape[1]:
    raise ValueError(
      f"feet_edge_penalty: contact sensor has {in_contact.shape[1]} primaries "
      f"but height sensor has {on_edge.shape[1]} frames; they must match."
    )
  fired = on_edge & in_contact  # [B, F]
  reward = torch.sum(fired.float(), dim=1)
  if level_threshold >= 0:
    terrain = env.scene.terrain
    if terrain is not None and getattr(terrain, "terrain_levels", None) is not None:
      gate = (terrain.terrain_levels > level_threshold).float()
      reward = reward * gate
  return reward


def foothold_penalty(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  contact_sensor_name: str,
  epsilon: float = 0.03,
  normalize: bool = True,
  level_threshold: int = -1,
  terrain_classes: tuple[int, ...] | None = None,
) -> torch.Tensor:
  """BeamDojo-style foothold penalty.

  Samples N points under each foot via a TerrainHeightSensor; for each foot in
  contact, counts samples whose terrain clearance exceeds ``epsilon`` (i.e. the
  sample is over a void / unsafe area). Returns shape ``[B]``; pair with a
  negative weight via :class:`RewardTermCfg`.

  Args:
    terrain_classes: Optional tuple of ``terrain_class`` IDs. When given, the
      penalty is masked to envs whose ``env_terrain_class`` is in the tuple.
      Use this to restrict the reward to terrains where it makes physical
      sense (e.g. stairs). On slopes the sensor's tilted clearance pattern
      yields false positives even when the foot is firmly planted.
  """
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    f"foothold_penalty requires a TerrainHeightSensor, got "
    f"{type(height_sensor).__name__}"
  )
  contact_sensor = env.scene[contact_sensor_name]
  assert isinstance(contact_sensor, ContactSensor), (
    f"foothold_penalty requires a ContactSensor, got {type(contact_sensor).__name__}"
  )

  raw = height_sensor.data
  F = height_sensor.num_frames
  N = height_sensor.num_rays_per_frame
  B = raw.distances.shape[0]

  frame_z = raw.frame_pos_w[:, :, 2]  # [B, F]
  hit_z = raw.hit_pos_w[:, :, 2].view(B, F, N)  # [B, F, N]
  heights = frame_z.unsqueeze(-1) - hit_z  # [B, F, N]

  dists = raw.distances.view(B, F, N)
  normal_z = raw.normals_w[:, :, 2].view(B, F, N)
  miss = dists < 0
  backface = (dists >= 0) & (normal_z < 0)
  # Miss => no terrain at this sample => treat as unsafe foothold.
  # Backface => ray origin inside terrain => sample is supported, mark safe.
  unsafe = (heights > epsilon) | miss
  unsafe = unsafe & ~backface  # [B, F, N]

  per_foot_unsafe = unsafe.float().sum(dim=-1)  # [B, F]
  if normalize:
    per_foot_unsafe = per_foot_unsafe / float(N)

  found = contact_sensor.data.found
  assert found is not None, "foothold_penalty needs contact sensor 'found' field."
  in_contact = (found > 0).float()  # [B, F_contact]
  if in_contact.shape[1] != per_foot_unsafe.shape[1]:
    raise ValueError(
      f"foothold_penalty: contact sensor has {in_contact.shape[1]} primaries "
      f"but height sensor has {per_foot_unsafe.shape[1]} frames; they must match."
    )

  reward = (per_foot_unsafe * in_contact).sum(dim=1)  # [B]

  if terrain_classes:
    terrain = env.scene.terrain
    env_class = getattr(terrain, "env_terrain_class", None) if terrain else None
    if env_class is None:
      raise ValueError(
        "foothold_penalty(terrain_classes=...) requires the scene terrain to "
        "expose 'env_terrain_class'. Tag your sub_terrains with terrain_class."
      )
    forced = torch.tensor(terrain_classes, device=reward.device, dtype=env_class.dtype)
    mask = (env_class.unsqueeze(-1) == forced).any(dim=-1).float()
    reward = reward * mask

  if level_threshold >= 0:
    terrain = env.scene.terrain
    if terrain is not None and getattr(terrain, "terrain_levels", None) is not None:
      gate = (terrain.terrain_levels > level_threshold).float()
      reward = reward * gate
  return reward


class feet_contact_singlefoot:
  """CReF Table I "Feet contact" reward.

  Expression (paper):
      r_feet_contact = I_stand + (1 - I_stand) * I_single^{0, 0.2s}

  Where:
    I_stand           = 1{ |u_xy_cmd|_2 + |w_yaw_cmd| < u_s }
    I_single^{0,0.2s} = 1 if a single-foot contact has occurred within the
                        last ``history_seconds``, else 0.

  Intuition: when commanded to stand, give a constant +1 baseline so the
  policy can collect non-zero return without locomoting. When commanded to
  move, only reward time intervals where the gait actually transitions
  through a single-support phase. This directly suppresses the
  shuffle / double-support attractor on stairs (both feet always on the same
  step => I_single never fires).

  Args:
    sensor_name: Contact sensor whose ``found`` per-foot tells us how many
      feet touch the ground each step. Must contain both feet as primaries.
    command_name: Velocity command term name (default ``"twist"``).
    stand_speed_thresh: ``u_s`` — total commanded speed below this counts
      as standing.
    history_seconds: ``0.2 s`` window length for ``I_single``.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._sensor_name: str = cfg.params["sensor_name"]
    self._command_name: str = cfg.params.get("command_name", "twist")
    self._stand_speed_thresh: float = cfg.params.get("stand_speed_thresh", 0.1)
    self._history_seconds: float = cfg.params.get("history_seconds", 0.2)

    self._history_steps: int = max(1, int(round(self._history_seconds / env.step_dt)))
    self._single_history = torch.zeros(
      env.num_envs, self._history_steps, device=env.device, dtype=torch.bool
    )
    self._head: int = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    stand_speed_thresh: float = 0.1,
    history_seconds: float = 0.2,
  ) -> torch.Tensor:
    del sensor_name, command_name, stand_speed_thresh, history_seconds  # use self.*

    contact_sensor: ContactSensor = env.scene[self._sensor_name]
    found = contact_sensor.data.found
    assert found is not None, (
      "feet_contact_singlefoot needs the contact sensor's 'found' field."
    )
    in_contact = found > 0  # [B, F_feet]
    n_in_contact = in_contact.sum(dim=1)  # [B]
    is_single = n_in_contact == 1  # [B]

    self._single_history[:, self._head] = is_single
    self._head = (self._head + 1) % self._history_steps

    single_in_window = self._single_history.any(dim=1)  # [B]

    command = env.command_manager.get_command(self._command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    is_stand = total_command < self._stand_speed_thresh  # [B]

    reward = is_stand.float() + (~is_stand).float() * single_in_window.float()
    return reward

  def reset(self, env_ids: torch.Tensor) -> None:
    self._single_history[env_ids] = False
