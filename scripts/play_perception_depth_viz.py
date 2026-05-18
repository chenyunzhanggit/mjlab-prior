"""Live 2D depth-image overlay for the perception passing task's
forward LiDAR sensor.

This is a thin wrapper around ``mjlab.scripts.play``:

  1. Monkey-patches :class:`ViserPlayViewer` to peek at the
     ``pelvis_forward_lidar`` raycast sensor every ``--depth-stride`` steps
     and push the depth array to a child process.
  2. Then forwards CLI args straight to :func:`mjlab.scripts.play.main`.

Why a subprocess: viser uses its own background threads + tornado event
loop. Running an interactive GUI loop in the same process competes with
that and intermittently crashes.

Why **not** matplotlib for the depth window: matplotlib 3.10 on Python
3.13 has a known FreeType bug — drawing axis tick labels can hit
``Could not set the fontsize (invalid ppem value; error code 0x97)``
because the figure's DPI transiently goes to zero. Even an
isolated subprocess can trip it. We sidestep matplotlib's figure /
text rendering entirely: the child process colour-maps the depth
array with ``matplotlib.colormaps[cmap]`` (a pure numpy function — no
figure, no text, no FreeType) and then renders the resulting RGB
image into a plain ``tkinter`` window via ``PIL.ImageTk``. That's the
same plumbing every image viewer uses; it doesn't touch matplotlib's
buggy code path at all.

You get two visualisations:

  * **3D rays / hits** — viser viewer (sidebar → Sensor debug viz).
  * **2D depth image** — a plain Tk window showing the (Ny × Nx) depth
    image, colour-mapped with the chosen cmap (default ``turbo``: close →
    blue, far → red). Window can be closed at any time without stopping
    the simulation.

Usage::

  uv run python scripts/play_perception_depth_viz.py \\
    Mjlab-Football-Passing-Perception-Unitree-G1 \\
    --checkpoint-file /path/to/.../model_XXXX.pt \\
    --motion-prior-ckpt-path /path/to/motion_prior_ckpt/model_XXXX.pt \\
    --num-envs 1 \\
    --viewer viser

All CLI args after the task name are forwarded verbatim to
``mjlab.scripts.play``. The depth-window-specific flags
(``--sensor-name``, ``--depth-stride``, ``--depth-cmap``,
``--depth-window-size``) are parsed and stripped before forwarding.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import queue as queue_mod
import sys

import numpy as np


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
# Colormap helper — pure-numpy LUT lookup; never touches a Figure.
# ---------------------------------------------------------------------------


def _get_cmap_fn(name: str):
  """Return a callable ``cmap_fn(values_in_[0,1]) -> RGBA in [0,1]``.

  Uses matplotlib's colormap registry but **never** creates a Figure /
  Axes / Text. The FreeType bug only triggers when matplotlib actually
  draws text, which we don't do here.
  """
  try:
    from matplotlib import colormaps  # matplotlib >= 3.5

    return colormaps[name]
  except (ImportError, KeyError):
    import matplotlib.cm

    return matplotlib.cm.get_cmap(name)


# ---------------------------------------------------------------------------
# Child-process viewer loop: tkinter + PIL, no matplotlib figure.
# ---------------------------------------------------------------------------


def _depth_viewer_process(
  q: "mp.Queue",
  max_distance: float,
  cmap: str,
  sensor_name: str,
  window_size: int,
) -> None:
  """Receive depth arrays from ``q`` and display them in a Tk window.

  ``q.put(None)`` signals shutdown.
  """
  import tkinter as tk

  from PIL import Image, ImageTk

  cmap_fn = _get_cmap_fn(cmap)

  # Wait for the first frame so we know the image shape.
  while True:
    try:
      first = q.get(timeout=1.0)
    except queue_mod.Empty:
      continue
    if first is None:
      return
    first = np.asarray(first, dtype=np.float32)
    break

  H, W = first.shape
  # Upscale small depth images so they're visible. ``window_size`` caps
  # the longer edge; we keep aspect ratio and round to integer scale for
  # nearest-neighbour upscaling (sharp pixels, no interpolation blur).
  scale = max(1, window_size // max(H, W))
  disp_w, disp_h = W * scale, H * scale

  root = tk.Tk()
  root.title(f"{sensor_name} depth (live)")
  root.resizable(False, False)

  # Top label: shape / scale info.
  info = tk.Label(
    root,
    text=(
      f"{sensor_name}  |  shape {H}×{W}  |  display {disp_h}×{disp_w} "
      f"(×{scale})  |  cmap={cmap}  |  range 0–{max_distance:.1f} m"
    ),
    font=("TkFixedFont", 10),
  )
  info.pack(side="top", fill="x")

  img_label = tk.Label(root)
  img_label.pack()

  photo_ref: list[ImageTk.PhotoImage | None] = [None]

  def render(frame_2d: np.ndarray) -> None:
    """Map depth → RGB → PhotoImage and update the label."""
    depth_norm = np.clip(frame_2d / max_distance, 0.0, 1.0)
    rgba = cmap_fn(depth_norm)  # [H, W, 4] in [0, 1]
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
      img = img.resize((disp_w, disp_h), Image.NEAREST)
    photo = ImageTk.PhotoImage(img)
    photo_ref[0] = photo  # Keep reference so Tk doesn't GC the image.
    img_label.configure(image=photo)

  render(first)

  stopped = [False]

  def _on_close():
    stopped[0] = True
    try:
      root.quit()
    except Exception:
      pass

  root.protocol("WM_DELETE_WINDOW", _on_close)

  def poll():
    """Drain the queue (keeping only the latest frame) and schedule the next tick."""
    if stopped[0]:
      return
    latest: np.ndarray | None = None
    try:
      while True:
        item = q.get_nowait()
        if item is None:
          stopped[0] = True
          root.quit()
          return
        latest = item
    except queue_mod.Empty:
      pass
    if latest is not None:
      try:
        render(np.asarray(latest, dtype=np.float32))
      except Exception as e:  # noqa: BLE001
        print(f"[depth-viewer] render error: {e}", file=sys.stderr)
    root.after(30, poll)

  root.after(30, poll)
  try:
    root.mainloop()
  except Exception:
    pass


# ---------------------------------------------------------------------------
# Parent-side depth window manager.
# ---------------------------------------------------------------------------


class _DepthWindow:
  """Lazy-spawned Tk subprocess. First ``update()`` call starts the child;
  further calls push frames through a bounded queue (drops on
  back-pressure)."""

  def __init__(self, sensor_name: str, cmap: str, window_size: int) -> None:
    self.sensor_name = sensor_name
    self.cmap = cmap
    self.window_size = window_size
    self._q: mp.Queue | None = None
    self._proc: mp.Process | None = None

  def _ensure_started(self, max_distance: float) -> None:
    if self._proc is not None:
      return
    # ``spawn`` is essential — fork would inherit the parent's CUDA /
    # Warp / viser handles and segfault as soon as Tk touches them.
    ctx = mp.get_context("spawn")
    self._q = ctx.Queue(maxsize=2)
    self._proc = ctx.Process(
      target=_depth_viewer_process,
      args=(self._q, max_distance, self.cmap, self.sensor_name, self.window_size),
      daemon=True,
    )
    self._proc.start()
    atexit.register(self.shutdown)
    print(
      f"[play+depth-viz] depth-viewer subprocess started "
      f"(pid={self._proc.pid}, cmap={self.cmap}, window={self.window_size}px)"
    )

  def update(self, depth_2d: np.ndarray, max_distance: float) -> None:
    self._ensure_started(max_distance)
    assert self._q is not None
    if self._proc is not None and not self._proc.is_alive():
      return  # Viewer was closed by the user — silently no-op.
    try:
      self._q.put_nowait(depth_2d)
    except queue_mod.Full:
      pass  # Drop frame.

  def shutdown(self) -> None:
    if self._q is not None:
      try:
        self._q.put_nowait(None)
      except (queue_mod.Full, ValueError):
        pass
    if self._proc is not None and self._proc.is_alive():
      self._proc.join(timeout=1.0)
      if self._proc.is_alive():
        self._proc.terminate()


# ---------------------------------------------------------------------------
# Monkey-patch ViserPlayViewer to inject the depth-window callback.
# ---------------------------------------------------------------------------


def _install_depth_hook(
  sensor_name: str, stride: int, cmap: str, window_size: int
) -> None:
  """Wrap ``ViserPlayViewer._execute_step`` so every ``stride``-th
  successful step pushes the sensor distances to the depth-viewer
  subprocess."""
  from mjlab.viewer.viser.viewer import ViserPlayViewer

  window = _DepthWindow(
    sensor_name=sensor_name, cmap=cmap, window_size=window_size
  )
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
    # ``sensor.data.distances``: [B, N_total]. Take env 0.
    distances = sensor.data.distances[0].detach().cpu().numpy()
    max_d = float(sensor.cfg.max_distance)
    distances = np.where(distances < 0, max_d, distances)

    # Infer (Ny, Nx) from the local-offset cache (works for GridPattern,
    # PinholeCamera and any future pattern that lays out via
    # ``meshgrid(x, y, indexing="xy")`` + row-major flatten).
    offsets = sensor._local_offsets
    assert offsets is not None, "RayCastSensor not initialised yet"
    nx = int(offsets[:, 0].unique().numel())
    n_total = distances.shape[0]
    ny = n_total // nx
    if ny * nx != n_total:
      ny, nx = 1, n_total
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
  sensor_name = _pop_arg(argv, "--sensor-name", "pelvis_forward_camera")
  stride = int(_pop_arg(argv, "--depth-stride", "1"))
  cmap = _pop_arg(argv, "--depth-cmap", "turbo")
  window_size = int(_pop_arg(argv, "--depth-window-size", "640"))
  sys.argv = argv

  _install_depth_hook(
    sensor_name=sensor_name,
    stride=stride,
    cmap=cmap,
    window_size=window_size,
  )

  from mjlab.scripts import play as play_mod

  play_mod.main()


if __name__ == "__main__":
  main()
