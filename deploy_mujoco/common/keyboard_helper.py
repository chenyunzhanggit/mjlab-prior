"""
Terminal raw-mode keyboard controller for SE(2) velocity commands.

Increment style (mirrors IsaacLab play_vel DebugKeyboardWithOneHot):
  Each key tap adds a fixed step to the corresponding axis.
  Velocity persists until L zeros it or the opposite key reduces it.
  OS key-repeat is suppressed via per-key debounce (0.25 s).

Key bindings:
  Up         : vx += step_vx
  Down       : vx -= step_vx
  Z          : wz += step_wz
  X          : wz -= step_wz
  L          : zero all commands
  R          : call on_reset callback
  Ctrl-C     : SIGINT (exit)
"""

import os
import atexit
import select
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np


class KeyboardVelocityCommand:
    """
    Increment-style velocity command reader from terminal stdin.

    Args:
        step_vx:   vx change per keypress (m/s).
        step_wz:   wz change per keypress (rad/s).
        max_vx:    clamp |vx| to this value.
        max_wz:    clamp |wz| to this value.
        debounce:  seconds to ignore OS key-repeat after each press.
        on_reset:  Optional zero-argument callback for the R key.
    """

    _ARROW_CODE = {"A": "up", "B": "down", "C": "right", "D": "left"}

    # key → (axis-index, sign)
    _KEY_AXIS = {
        "up":   (0, +1),
        "down": (0, -1),
        "z":    (2, +1),
        "x":    (2, -1),
    }

    def __init__(
        self,
        step_vx: float = 0.1,
        step_wz: float = 0.1,
        max_vx: float = 1.0,
        max_wz: float = 1.0,
        debounce: float = 0.25,
        on_reset=None,
    ):
        self._step     = np.array([step_vx, 0.0, step_wz], dtype=np.float32)
        self._max      = np.array([max_vx,  0.0, max_wz],  dtype=np.float32)
        self._debounce = debounce
        self._on_reset = on_reset

        self._lock     = threading.Lock()
        self._command  = np.zeros(3, dtype=np.float32)
        self._last_key: dict[str, float] = {}
        self._running  = True

        self._fd       = sys.stdin.fileno()
        self._old_tty  = termios.tcgetattr(self._fd)
        atexit.register(self._restore_tty)

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

        print("[Keyboard] Increment-mode terminal keyboard started.")
        print(f"  Up / Down : vx += / -= {step_vx:.2f}  (clamped to ±{max_vx:.2f})")
        print(f"  Z / X     : wz += / -= {step_wz:.2f}  (clamped to ±{max_wz:.2f})")
        print("  L         : zero all commands")
        print("  R         : reset simulation")
        print("  Ctrl-C    : exit")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def command(self) -> np.ndarray:
        """Current [vx, vy, wz] as float32 numpy array (copy)."""
        with self._lock:
            return self._command.copy()

    def reset(self) -> None:
        with self._lock:
            self._command[:] = 0.0

    def stop(self) -> None:
        self._running = False
        self._restore_tty()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _restore_tty(self) -> None:
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_tty)
        except Exception:
            pass

    def _apply(self, key: str) -> None:
        now = time.time()

        if key == "l":
            with self._lock:
                self._command[:] = 0.0
            self._print_cmd("zeroed")
            return

        if key == "r":
            if self._on_reset:
                self._on_reset()
            return

        entry = self._KEY_AXIS.get(key)
        if entry is None:
            return

        # Suppress OS key-repeat: ignore same key within debounce window
        if now - self._last_key.get(key, 0.0) < self._debounce:
            return
        self._last_key[key] = now

        idx, sign = entry
        with self._lock:
            self._command[idx] = float(np.clip(
                self._command[idx] + sign * self._step[idx],
                -self._max[idx], self._max[idx],
            ))
        self._print_cmd()

    def _print_cmd(self, label: str = "") -> None:
        with self._lock:
            cmd = self._command.copy()
        tag = f"  {label}" if label else ""
        print(
            f"\r[cmd] vx={cmd[0]:+.2f}  vy={cmd[1]:+.2f}  wz={cmd[2]:+.2f}{tag}          ",
            end="", flush=True,
        )

    def _read_loop(self) -> None:
        try:
            tty.setcbreak(self._fd)
            while self._running:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue

                ch = os.read(self._fd, 1)
                try:
                    ch = ch.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                if ch == "\x03":                  # Ctrl-C
                    os.kill(os.getpid(), signal.SIGINT)
                    break

                elif ch == "\x1b":                # ESC — maybe arrow key
                    r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r2:
                        continue
                    bracket = os.read(self._fd, 1).decode("utf-8", errors="ignore")
                    if bracket != "[":
                        continue
                    r3, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r3:
                        continue
                    code = os.read(self._fd, 1).decode("utf-8", errors="ignore")
                    key = self._ARROW_CODE.get(code)
                    if key:
                        self._apply(key)

                else:
                    self._apply(ch.lower())

        except Exception as e:
            print(f"\r[Keyboard] error: {e}", flush=True)
        finally:
            self._restore_tty()
