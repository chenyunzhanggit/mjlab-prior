"""Live 2D depth-image overlay for the perception passing task's
forward-LiDAR sensor.

This is a thin wrapper around ``mjlab.scripts.play``:

  1. Monkey-patches :class:`ViserPlayViewer` to peek at the
     ``pelvis_forward_lidar`` raycast sensor every ``--depth-stride`` steps
     and redraw a matplotlib window with the current (Ny × Nx) depth
     image — the actual flattened input the policy is consuming.
  2. Then forwards CLI args straight to :func:`mjlab.scripts.play.main`.

So you get both visualisations at once:

  * **3D rays / hits** — viser viewer (already on because the perception
    env_cfg sets ``debug_vis=True`` on the LiDAR; toggle off in viser's
    "Sensor debug viz" panel if it's too cluttered).
  * **2D depth image** — this matplotlib window. The X axis spans the
    pelvis-frame forward offset of each grid cell (±0.75 m), the Y axis
    spans the pelvis-frame lateral offset (±0.5 m). Brighter = closer.

Usage::

  uv run python scripts/play_perception_depth_viz.py \\
    Mjlab-Football-Passing-Perception-Unitree-G1 \\
    --checkpoint-file /path/to/.../model_XXXX.pt \\
    --motion-prior-ckpt-path /path/to/motion_prior_ckpt/model_XXXX.pt \\
    --num-envs 1 \\
    --viewer viser

All CLI args after the task name are forwarded verbatim to
``mjlab.scripts.play``. The depth-window-specific flags
(``--sensor-name``, ``--depth-stride``, ``--depth-cmap``) are parsed and
stripped before forwarding.
"""

from __future__ import annotations

import sys

import matplotlib

# Pick a GUI backend that doesn't clash with viser's tornado loop. TkAgg
# is what mpl falls back to anyway on a fresh interpreter; we make it
# explicit so headless environments fail loudly.
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mjlab.scripts import play as play_mod  # noqa: E402
from mjlab.viewer.viser.viewer import ViserPlayViewer  # noqa: E402


# ---------------------------------------------------------------------------
# Argparse — strip our flags before tyro sees them.
# ---------------------------------------------------------------------------


def _pop_arg(argv: list[str], name: str, default: str) -> str:
  """Extract ``--name VALUE`` (or ``--name=VALUE``) from argv and return the value."""
  i = 0
  while i < len(argv):
    a = argv[i]
    if a == name:
      if i + 1 >= len(argv):
        raise ValueError(f"Missing value after {name}")
      v = argv[i + 1]
      del argv[i : i + 2]
      return v
    if a.startswith(name + "="):
      v = a.split("=", 1)[1]
      del argv[i]
      return v
    i += 1
  return default


# ---------------------------------------------------------------------------
# Depth-window state — lazy init so we don't spawn the window until the
# first step has data.
# ---------------------------------------------------------------------------


class _DepthWindow:
  def __init__(self, sensor_name: str, cmap: str = "turbo_r") -> None:
    self.sensor_name = sensor_name
    self.cmap = cmap
    self._fig = None
    self._im = None
    self._ax = None
    self._initialised = False
    self._ny: int | None = None
    self._nx: int | None = None

  def _ensure_initialised(self, depth_2d: np.ndarray, max_distance: float) -> None:
    if self._initialised:
      return
    self._ny, self._nx = depth_2d.shape
    plt.ion()
    self._fig, self._ax = plt.subplots(figsize=(6, 4.5))
    self._fig.canvas.manager.set_window_title(
      f"{self.sensor_name} live depth ({self._ny}×{self._nx})"
    )
    self._im = self._ax.imshow(
      depth_2d,
      vmin=0.0,
      vmax=max_distance,
      cmap=self.cmap,
      aspect="auto",
      origin="lower",
    )
    self._ax.set_xlabel("pelvis-frame X cell (forward ±)")
    self._ax.set_ylabel("pelvis-frame Y cell (lateral ±)")
    cbar = self._fig.colorbar(self._im, ax=self._ax)
    cbar.set_label("ray distance [m]  (close → bright)")
    self._fig.tight_layout()
    self._initialised = True

  def update(self, depth_2d: np.ndarray, max_distance: float) -> None:
    self._ensure_initialised(depth_2d, max_distance)
    assert self._im is not None and self._fig is not None
    self._im.set_data(depth_2d)
    # Keep colorbar pinned to [0, max_distance]; don't auto-scale.
    self._fig.canvas.draw_idle()
    # ``flush_events`` is needed for the TkAgg backend to actually
    # repaint without a blocking ``plt.pause``.
    self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Monkey-patch ViserPlayViewer to inject the depth-window callback.
# ---------------------------------------------------------------------------


def _install_depth_hook(sensor_name: str, stride: int, cmap: str) -> None:
  """Wrap ``ViserPlayViewer._execute_step`` so every ``stride``-th
  successful step reads the raycast distances and pushes them to the
  matplotlib window."""
  window = _DepthWindow(sensor_name=sensor_name, cmap=cmap)
  orig = ViserPlayViewer._execute_step
  state = {"n": 0}

  def patched(self):
    ok = orig(self)
    if not ok:
      return ok
    state["n"] += 1
    if state["n"] % stride != 0:
      return ok
    env = self.env.unwrapped
    sensor = env.scene.sensors.get(sensor_name)
    if sensor is None:
      return ok
    # ``sensor.data.distances``: [B, N_total] where N_total = Nx * Ny for a
    # single-frame grid pattern. We always look at env 0 for the live
    # visualisation.
    distances = sensor.data.distances[0].detach().cpu().numpy()
    max_d = float(sensor.cfg.max_distance)
    # Misses come back as -1; clamp to max_distance so the colormap
    # treats "no hit" as "far".
    distances = np.where(distances < 0, max_d, distances)

    # GridPatternCfg builds rays with ``meshgrid(x, y, indexing='xy')``
    # then flattens row-major, so the natural reshape is
    # ``(num_y, num_x)``. For a (size=(1.5, 1.0), resolution=0.1) grid
    # that's (11, 16). We recover the shape from the local-offset cache.
    offsets = sensor._local_offsets  # [N_per_frame, 3]
    assert offsets is not None, "RayCastSensor not initialised yet"
    x_vals = offsets[:, 0].unique()
    nx = int(x_vals.numel())
    n_total = distances.shape[0]
    ny = n_total // nx
    depth_2d = distances.reshape(ny, nx)
    window.update(depth_2d, max_distance=max_d)
    return ok

  ViserPlayViewer._execute_step = patched
  print(
    f"[play+depth-viz] hooked ViserPlayViewer._execute_step "
    f"(sensor='{sensor_name}', stride={stride})"
  )


# ---------------------------------------------------------------------------
# Entrypoint — strip our flags, install hook, delegate to play.main().
# ---------------------------------------------------------------------------


def main() -> None:
  argv = list(sys.argv)
  sensor_name = _pop_arg(argv, "--sensor-name", "pelvis_forward_lidar")
  stride = int(_pop_arg(argv, "--depth-stride", "1"))
  cmap = _pop_arg(argv, "--depth-cmap", "turbo_r")
  sys.argv = argv

  _install_depth_hook(sensor_name=sensor_name, stride=stride, cmap=cmap)
  play_mod.main()


if __name__ == "__main__":
  main()
