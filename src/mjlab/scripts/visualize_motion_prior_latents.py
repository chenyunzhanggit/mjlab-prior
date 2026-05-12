"""Visualize VQ-VAE motion-prior latents collected by
``collect_motion_prior_latents.py``.

Produces four PNGs side-by-side in ``--out-dir``:

  * ``latent_tsne.png``        — 2-D t-SNE of raw encoder outputs, colored
                                 by encoder (A=flat, B=rough).
  * ``latent_pca.png``         — first two PCA components, same coloring.
  * ``latent_per_dim_hist.png``— per-dimension marginal histograms (8×8 grid
                                 for code_dim=64).
  * ``codebook_usage.png``     — codebook-index frequency histogram per
                                 encoder, plus the overlap.

Usage::

  uv run python -m mjlab.scripts.visualize_motion_prior_latents \\
      --npz /tmp/mp_latents.npz --out-dir /tmp/mp_latent_plots
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass(frozen=True)
class VisConfig:
  npz: str
  """Path to the ``.npz`` produced by ``collect_motion_prior_latents.py``."""
  out_dir: str = "/tmp/mp_latent_plots"

  max_points_per_encoder: int = 5000
  """Cap samples per encoder for t-SNE (it scales O(N²))."""
  tsne_perplexity: float = 30.0
  pca_random_state: int = 0
  tsne_random_state: int = 0
  use_quantized: bool = False
  """If True, run t-SNE/PCA on post-quantization codes (``q_*``) instead of
  the raw encoder outputs (``enc_*``)."""


def _load(npz_path: Path) -> dict[str, np.ndarray]:
  data = np.load(npz_path)
  return {k: data[k] for k in data.files}


def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
  if len(x) <= n:
    return x
  idx = rng.choice(len(x), size=n, replace=False)
  return x[idx]


def plot_tsne(
  z_a: np.ndarray, z_b: np.ndarray, out: Path, perplexity: float, seed: int
) -> None:
  from sklearn.manifold import TSNE

  n = min(len(z_a), len(z_b))
  z_a, z_b = z_a[:n], z_b[:n]
  Z = np.concatenate([z_a, z_b], axis=0)
  print(f"[vis] t-SNE on {Z.shape}, perplexity={perplexity}")
  emb = TSNE(
    n_components=2,
    perplexity=perplexity,
    init="pca",
    random_state=seed,
  ).fit_transform(Z)

  plt.figure(figsize=(7, 6))
  plt.scatter(emb[:n, 0], emb[:n, 1], s=5, alpha=0.45, label="encoder_a (flat)")
  plt.scatter(emb[n:, 0], emb[n:, 1], s=5, alpha=0.45, label="encoder_b (rough)")
  plt.legend(loc="best")
  plt.title(f"t-SNE of latent z (N={n} per encoder)")
  plt.tight_layout()
  plt.savefig(out, dpi=150)
  plt.close()
  print(f"[vis] wrote {out}")


def plot_pca(z_a: np.ndarray, z_b: np.ndarray, out: Path, seed: int) -> None:
  from sklearn.decomposition import PCA

  Z = np.concatenate([z_a, z_b], axis=0)
  pca = PCA(n_components=2, random_state=seed).fit(Z)
  pa = pca.transform(z_a)
  pb = pca.transform(z_b)
  ev = pca.explained_variance_ratio_

  plt.figure(figsize=(7, 6))
  plt.scatter(pa[:, 0], pa[:, 1], s=5, alpha=0.45, label="encoder_a (flat)")
  plt.scatter(pb[:, 0], pb[:, 1], s=5, alpha=0.45, label="encoder_b (rough)")
  plt.xlabel(f"PC1 ({ev[0] * 100:.1f}% var)")
  plt.ylabel(f"PC2 ({ev[1] * 100:.1f}% var)")
  plt.legend(loc="best")
  plt.title("PCA of latent z")
  plt.tight_layout()
  plt.savefig(out, dpi=150)
  plt.close()
  print(f"[vis] wrote {out}")


def plot_per_dim_hist(z_a: np.ndarray, z_b: np.ndarray, out: Path) -> None:
  d = z_a.shape[1]
  cols = 8
  rows = (d + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 1.6))
  axes = np.atleast_2d(axes)
  for i in range(rows * cols):
    ax = axes[i // cols, i % cols]
    if i < d:
      ax.hist(z_a[:, i], bins=50, alpha=0.5, density=True, label="A")
      ax.hist(z_b[:, i], bins=50, alpha=0.5, density=True, label="B")
      ax.set_title(f"z[{i}]", fontsize=7)
      ax.tick_params(labelsize=5)
    else:
      ax.axis("off")
  axes[0, 0].legend(fontsize=6, loc="upper right")
  fig.suptitle("Per-dimension marginals of latent z")
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"[vis] wrote {out}")


def plot_codebook_usage(
  idx_a: np.ndarray,
  idx_b: np.ndarray,
  num_code: int,
  out: Path,
) -> None:
  count_a = np.bincount(idx_a, minlength=num_code).astype(np.float64)
  count_b = np.bincount(idx_b, minlength=num_code).astype(np.float64)
  freq_a = count_a / count_a.sum().clip(min=1)
  freq_b = count_b / count_b.sum().clip(min=1)

  used_a = (count_a > 0).sum()
  used_b = (count_b > 0).sum()
  used_both = ((count_a > 0) & (count_b > 0)).sum()

  # Sort codes by combined frequency for readability.
  order = np.argsort(-(freq_a + freq_b))
  freq_a_s, freq_b_s = freq_a[order], freq_b[order]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
  x = np.arange(num_code)
  ax1.bar(x, freq_a_s, width=1.0, alpha=0.6, label=f"A (used {used_a}/{num_code})")
  ax1.bar(x, freq_b_s, width=1.0, alpha=0.6, label=f"B (used {used_b}/{num_code})")
  ax1.set_yscale("log")
  ax1.set_xlabel("code index (sorted by combined frequency)")
  ax1.set_ylabel("frequency (log)")
  ax1.set_title("Per-code usage")
  ax1.legend(loc="upper right", fontsize=8)

  ax2.scatter(freq_a, freq_b, s=8, alpha=0.5)
  m = max(freq_a.max(), freq_b.max())
  ax2.plot([0, m], [0, m], "k--", lw=0.8)
  ax2.set_xscale("log")
  ax2.set_yscale("log")
  ax2.set_xlabel("freq(A)")
  ax2.set_ylabel("freq(B)")
  ax2.set_title(f"A vs B per code (overlap: {used_both}/{num_code})")
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  print(f"[vis] wrote {out}")
  print(
    f"[vis] codebook stats: |A used|={used_a}, |B used|={used_b}, "
    f"|A ∩ B|={used_both}, |total codes|={num_code}"
  )


def print_summary(z_a: np.ndarray, z_b: np.ndarray) -> None:
  def stats(z: np.ndarray, name: str) -> None:
    var = z.var(axis=0)
    active = int((var > 1e-3).sum())
    print(
      f"[vis] {name}: shape={z.shape}, mean‖z‖={np.linalg.norm(z, axis=1).mean():.3f}, "
      f"mean(var)={var.mean():.4f}, active_dims(var>1e-3)={active}/{z.shape[1]}"
    )

  stats(z_a, "encoder_a")
  stats(z_b, "encoder_b")


def main(cfg: VisConfig) -> None:
  out_dir = Path(cfg.out_dir).expanduser()
  out_dir.mkdir(parents=True, exist_ok=True)
  data = _load(Path(cfg.npz).expanduser())

  z_a = data["q_a"] if cfg.use_quantized else data["enc_a"]
  z_b = data["q_b"] if cfg.use_quantized else data["enc_b"]
  idx_a, idx_b = data["idx_a"], data["idx_b"]
  codebook = data["codebook"]
  num_code = codebook.shape[0]

  print_summary(z_a, z_b)

  rng = np.random.default_rng(0)
  z_a_sub = _subsample(z_a, cfg.max_points_per_encoder, rng)
  z_b_sub = _subsample(z_b, cfg.max_points_per_encoder, rng)

  plot_tsne(
    z_a_sub,
    z_b_sub,
    out_dir / "latent_tsne.png",
    cfg.tsne_perplexity,
    cfg.tsne_random_state,
  )
  plot_pca(z_a_sub, z_b_sub, out_dir / "latent_pca.png", cfg.pca_random_state)
  plot_per_dim_hist(z_a, z_b, out_dir / "latent_per_dim_hist.png")
  plot_codebook_usage(idx_a, idx_b, num_code, out_dir / "codebook_usage.png")


if __name__ == "__main__":
  main(tyro.cli(VisConfig))
