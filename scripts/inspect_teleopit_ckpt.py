"""One-shot diagnostic for Teleopit teacher checkpoint and ONNX graph.

Reads ~/project/Teleopit/track.pt + track.onnx and prints:
  - all top-level keys of the .pt file
  - actor / critic state_dict shapes (sorted)
  - obs_normalizer running stats shape (= actor obs dim)
  - ONNX input/output names + shapes (= ground-truth obs schema)

Throw-away script. Not imported by anything.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

TELEOPIT_ROOT = Path(os.path.expanduser("~/project/Teleopit"))
PT_PATH = TELEOPIT_ROOT / "track.pt"
ONNX_PATH = TELEOPIT_ROOT / "track.onnx"


def section(title: str) -> None:
  print("\n" + "=" * 78)
  print(f"  {title}")
  print("=" * 78)


def inspect_pt() -> dict:
  section(f"Loading {PT_PATH}")
  ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
  print(f"top-level keys: {list(ckpt.keys())}")
  return ckpt


def summarize_state_dict(state_dict: dict, name: str) -> None:
  section(f"state_dict summary: {name}  (n_tensors={len(state_dict)})")
  by_prefix: dict[str, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
  for k, v in state_dict.items():
    prefix = k.split(".")[0]
    shape = tuple(v.shape) if hasattr(v, "shape") else ()
    by_prefix[prefix].append((k, shape))

  for prefix, items in by_prefix.items():
    print(f"\n[{prefix}]  ({len(items)} tensors)")
    for k, s in items:
      print(f"  {k:<70s} {s}")


def inspect_onnx() -> None:
  section(f"ONNX graph: {ONNX_PATH}")
  model = onnx.load(str(ONNX_PATH))
  print("--- inputs ---")
  for inp in model.graph.input:
    dims = [
      d.dim_value if d.dim_value > 0 else d.dim_param
      for d in inp.type.tensor_type.shape.dim
    ]
    print(f"  {inp.name:<30s} shape={dims}")
  print("--- outputs ---")
  for out in model.graph.output:
    dims = [
      d.dim_value if d.dim_value > 0 else d.dim_param
      for d in out.type.tensor_type.shape.dim
    ]
    print(f"  {out.name:<30s} shape={dims}")

  section("ONNX runtime smoke test (zeros input)")
  sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
  feeds = {}
  for inp in sess.get_inputs():
    shape = [1 if (isinstance(d, str) or d is None or d <= 0) else d for d in inp.shape]
    feeds[inp.name] = torch.zeros(*shape).numpy()
    print(f"  feed[{inp.name}] shape={shape}")
  outs = sess.run(None, feeds)
  for o, val in zip(sess.get_outputs(), outs, strict=False):
    print(f"  out[{o.name}] shape={val.shape}  norm={float((val**2).sum() ** 0.5):.6f}")


def main() -> None:
  ckpt = inspect_pt()

  # rsl_rl 5.x mjlab teacher ckpts split actor / critic into separate dicts.
  for key in ("actor_state_dict", "critic_state_dict", "model_state_dict"):
    if key in ckpt and isinstance(ckpt[key], dict):
      summarize_state_dict(ckpt[key], key)

  # Try to surface env_state (obs_groups, obs_dims, etc.) — useful schema info.
  if "infos" in ckpt and isinstance(ckpt["infos"], dict):
    section("infos['env_state'] (training metadata)")
    env_state = ckpt["infos"].get("env_state", {})
    if isinstance(env_state, dict):
      for k, v in env_state.items():
        if isinstance(v, (int, float, str, list, tuple)):
          print(f"  {k}: {v}")
        elif isinstance(v, dict):
          print(f"  {k}: dict keys={list(v.keys())}")
        else:
          print(f"  {k}: type={type(v).__name__}")

  for k in ("optimizer_state_dict", "iter"):
    if k in ckpt:
      v = ckpt[k]
      print(f"\n[{k}] type={type(v).__name__}", end="")
      if isinstance(v, int):
        print(f" value={v}")
      elif isinstance(v, dict):
        print(f" keys={list(v.keys())[:8]}{'...' if len(v) > 8 else ''}")
      else:
        print()

  inspect_onnx()


if __name__ == "__main__":
  main()
