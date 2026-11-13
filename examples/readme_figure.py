"""Regenerate docs/reconstruction.png, the figure at the top of the README, from a synthetic object.

python examples/readme_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import qoraphase as qp

N, M, STARTS = 64, 24, 8

# A triangular object (not centrosymmetric, so the twin is not also a solution
# of the support) with a smooth amplitude dip and a phase bump.
y, x = np.indices((M, M)) - (M - 1) / 2
inside = np.indices((M, M)).sum(0) < M
block = (1 - 0.5 * np.exp(-(x**2 + y**2) / 20)) * np.exp(
    1j * 1.2 * np.exp(-((x + 4) ** 2 + (y + 4) ** 2) / 30)
)
obj = np.zeros((N, N), complex)
obj[N // 2 - M // 2 : N // 2 + M // 2, N // 2 - M // 2 : N // 2 + M // 2] = block * inside

intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj), norm="ortho"))) ** 2

ensemble = qp.reconstruct(intensity, starts=STARTS)  # no support: derived from the data
best = ensemble.best
rec = np.asarray(qp.align_to(best.object, obj))
print(f"best of {STARTS} starts: Fourier error {best.error:.1e}, {best.metrics.size} iterations")
print(f"relative error vs. true object: {np.linalg.norm(rec - obj) / np.linalg.norm(obj):.1e}")

crop = (slice(N // 2 - 16, N // 2 + 16),) * 2
inside_rec = np.abs(rec) > 0.1 * np.abs(rec).max()
panels = [
    (np.abs(obj)[crop], "true amplitude", "gray", None),
    (
        np.where(np.abs(obj) > 0, np.angle(obj), np.nan)[crop],
        "true phase",
        "viridis",
        (0, 1.2),
    ),
    (np.log10(intensity + 1e-6), "log10 intensity (input)", "magma", (-4, None)),
    (np.abs(rec)[crop], "reconstructed amplitude", "gray", None),
    (
        np.where(inside_rec, np.angle(rec), np.nan)[crop],
        "reconstructed phase",
        "viridis",
        (0, 1.2),
    ),
]

# Both phase panels use the same physical scale (radians). Give each a
# labeled colorbar so the values are interpretable even when axes are hidden.
fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.2), layout="constrained")
for ax, (img, title, cmap, lim) in zip(axes, panels, strict=False):
    vmin, vmax = lim or (None, None)
    artist = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    if title.endswith("phase"):
        bar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 0.4, 0.8, 1.2])
        bar.set_label("Phase (rad)", fontsize=9)
        bar.ax.tick_params(labelsize=8)

out = Path(__file__).resolve().parents[1] / "docs" / "reconstruction.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}")
