"""Observation-group builders for motion-prior tasks.

Three groups are exposed:

* ``student``    — proprioceptive obs only (history_length=4). Same schema
                   in both flat (teacher_a) and rough (teacher_b) envs so a
                   single student policy can run on either. ``extra_terms``
                   is the hook for later adding privileged signals
                   (height_scan, foot contact, etc.) without restructuring.
* ``teacher_a`` (+ ``teacher_a_history``) — Teleopit ``General-Tracking-G1``
                   actor obs schema (166-dim, 10 terms). The history group
                   shares the same terms with ``history_length=10,
                   flatten_history_dim=False`` for the Conv1D path.
* ``teacher_b`` (+ ``teacher_b_height``) — mjlab ``Velocity-Rough-Unitree-G1``
                   actor obs schema (May 2026 CNN+MLP architecture). The 1D
                   group ``teacher_b`` carries proprio + twist command
                   (99-dim, 7 terms); the 2D group ``teacher_b_height`` is
                   the 11x17 height scan fed to ``CNNProjModel``'s CNN
                   encoder. teacher_b has no history path.

The teacher schemas mirror the obs that produced the supplied checkpoints
(prior.md task #1). Changing them silently breaks teacher inference.
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.motion_prior import mdp
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

STUDENT_HISTORY_LENGTH = 4
"""History length applied to every student term (matches motionprior's StudentObsCfg)."""

TEACHER_A_HISTORY_LENGTH = 10
"""History length for teacher_a's Conv1D path (Teleopit cfg.actor_history_length)."""

HEIGHT_SCAN_MAX_DISTANCE = 5.0
"""Default ``terrain_scan`` ``max_distance`` (matches velocity_env_cfg.py)."""


def make_student_height_scan_term(
  sensor_name: str = "terrain_scan",
  history_length: int = 1,
  max_distance: float = HEIGHT_SCAN_MAX_DISTANCE,
) -> ObservationTermCfg:
  """Height-scan student term, matching ``make_velocity_env_cfg``'s actor scan.

  Caller must ensure the env's scene registers a raycast sensor with
  ``sensor_name`` (e.g. ``terrain_scan``). Pair with ``extra_terms`` of
  ``make_student_obs_group``. Switching to depth later means swapping this
  factory for one that emits a depth tensor in its own obs group.

  ``history_length`` defaults to 1 (no temporal stacking) to match
  teacher_b — teacher_b's actor sees a single 187-dim scan slice, so
  stacking history on the student side adds parameters without giving
  the distillation target any new signal. Override only if you later
  introduce a teacher whose action depends on terrain history.
  """
  return ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": sensor_name},
    noise=Unoise(n_min=-0.1, n_max=0.1),
    scale=1.0 / max_distance,
    history_length=history_length,
  )


def make_student_depth_obs_group(
  sensor_name: str = "camera",
  data_type: str = "distance_to_image_plane_noised",
  enable_corruption: bool = False,
) -> ObservationGroupCfg:
  """Depth-image student obs group (``[B, 1, H, W]``).

  The sensor's noise pipeline (distance Gaussian, normalization, dropout,
  crop+resize) already runs inside the camera, so corruption is disabled at
  the obs-manager level by default — adding 1-D Unoise on a (1, H, W) tensor
  would scramble the image. Kept as its own group because the 4-D shape is
  incompatible with the flat 1-D student concat.
  """
  return ObservationGroupCfg(
    terms={
      "depth_image": ObservationTermCfg(
        func=envs_mdp.depth_image,
        params={
          "sensor_cfg": SceneEntityCfg(sensor_name),
          "data_type": data_type,
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )


def make_student_obs_group(
  history_length: int = STUDENT_HISTORY_LENGTH,
  enable_corruption: bool = True,
  extra_terms: dict[str, ObservationTermCfg] | None = None,
) -> ObservationGroupCfg:
  """Build the proprioceptive student observation group.

  ``extra_terms`` lets callers append privileged signals (e.g. height_scan)
  without forking this builder. They are concatenated **after** the
  proprioceptive terms, so ordering stays stable for callers that don't use
  extras.
  """
  terms: dict[str, ObservationTermCfg] = {
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
      history_length=history_length,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
      history_length=history_length,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=history_length,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.0, n_max=1.0),
      history_length=history_length,
    ),
    "actions": ObservationTermCfg(
      func=envs_mdp.last_action,
      history_length=history_length,
    ),
  }
  if extra_terms:
    for k, v in extra_terms.items():
      if k in terms:
        raise ValueError(f"extra_terms key '{k}' collides with default student term")
      terms[k] = v
  return ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )


def _teacher_a_terms(command_name: str) -> dict[str, ObservationTermCfg]:
  """Teleopit General-Tracking-G1 actor obs (166-dim).

  Mirrors prior.md "Teleopit Teacher 实情":
    base tracking actor (without ``motion_anchor_pos_b`` / ``base_lin_vel``)
    + ``_VELCMD_ACTOR_TERMS`` (projected_gravity + 3 ref signals).
  """
  return {
    "command": ObservationTermCfg(
      func=mdp.generated_commands, params={"command_name": command_name}
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": command_name},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ref_base_lin_vel_b": ObservationTermCfg(
      func=mdp.ref_base_lin_vel_b, params={"command_name": command_name}
    ),
    "ref_base_ang_vel_b": ObservationTermCfg(
      func=mdp.ref_base_ang_vel_b, params={"command_name": command_name}
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
      func=mdp.ref_projected_gravity_b, params={"command_name": command_name}
    ),
  }


def make_teacher_a_obs_groups(
  command_name: str = "motion",
  history_length: int = TEACHER_A_HISTORY_LENGTH,
  enable_corruption: bool = True,
) -> tuple[ObservationGroupCfg, ObservationGroupCfg]:
  """Build the (teacher_a, teacher_a_history) observation groups.

  Both groups share term definitions; the history group adds a temporal
  axis at index 1 (no flattening) so the Conv1D encoder can consume it.
  """
  terms = _teacher_a_terms(command_name)
  actor = ObservationGroupCfg(
    terms=terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )
  history = ObservationGroupCfg(
    terms=deepcopy(terms),
    concatenate_terms=True,
    enable_corruption=enable_corruption,
    history_length=history_length,
    flatten_history_dim=False,
  )
  return actor, history


def make_teacher_b_obs_groups(
  twist_command_name: str = "twist",
  height_scan_sensor_name: str = "terrain_scan",
  height_grid: tuple[int, int] = (11, 17),
  enable_corruption: bool = True,
) -> tuple[ObservationGroupCfg, ObservationGroupCfg]:
  """Build the (teacher_b, teacher_b_height) observation groups.

  Mirrors the current ``Mjlab-Velocity-Rough-Unitree-G1`` actor obs schema
  used by ``CNNProjModel``:

  * ``teacher_b``         — 1D proprioceptive group (99-dim for G1 rough):
                            base_lin_vel + base_ang_vel + projected_gravity
                            + joint_pos + joint_vel + actions + twist_cmd
  * ``teacher_b_height``  — 2D height-scan group, ``[B, 1, H, W]`` (default
                            ``11x17`` for the 1.6 m x 1.0 m @ 0.1 m grid),
                            normalized by ``max_distance=5.0`` to match
                            training.

  The split mirrors the velocity rough rl_cfg's ``obs_groups={"actor":
  ("actor", "height"), ...}``; ``CNNProjModel`` reads them as a flat MLP
  input and a CNN input respectively.
  """
  height_h, height_w = height_grid

  prop_terms: dict[str, ObservationTermCfg] = {
    "base_lin_vel": ObservationTermCfg(
      func=velocity_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=velocity_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=velocity_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=velocity_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=velocity_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=velocity_mdp.last_action),
    "command": ObservationTermCfg(
      func=velocity_mdp.generated_commands,
      params={"command_name": twist_command_name},
    ),
  }
  height_terms: dict[str, ObservationTermCfg] = {
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan_2d,
      params={
        "sensor_name": height_scan_sensor_name,
        "height": height_h,
        "width": height_w,
      },
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1.0 / HEIGHT_SCAN_MAX_DISTANCE,
    ),
  }
  prop_group = ObservationGroupCfg(
    terms=prop_terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )
  height_group = ObservationGroupCfg(
    terms=height_terms,
    concatenate_terms=True,
    enable_corruption=enable_corruption,
  )
  return prop_group, height_group
