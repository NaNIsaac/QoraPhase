"""Figures for the lab-recipe comparison from the result files of ``bcdi_lab.py``.

    ROCKING_XRD_SIM=<checkout> uv run --python 3.14+gil --with scipy \
        python benchmarks/bcdi_lab_figures.py CASES_DIR --results DIR [DIR ...] --out docs/benchmarks         [--slices CASE]

CASES_DIR holds the generated cases (``truth_det.npy`` per case). Every ``<case>.json`` in the
result directories is merged per case, later directories winning, so a run with ``--keep`` or
``--own-consensus`` adds its columns to the full run of the same case. Writes
``bcdi-lab-spread.png``, the phase error of every start, and with ``--slices CASE``
``bcdi-lab-slices.png``, a central slice of that case's kept objects.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bcdi_lab import load_scorer

INK = "#3b4252"
MEDIAN = "#000000"
OFF_SCALE = 0.45
PHASE_LIMIT = 1.2
CUT = 0.3
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

ROWS = (
    ("pass1_scores", "Lab pass 1 (20 starts)"),
    ("pass2_lab.scores", "Lab pass 2, support as lab (3)"),
    ("pass2_twin_checked.scores", "Lab pass 2, twin-checked support (3)"),
    ("default_scores", "QoraPhase default, 720 it (20)"),
    ("default_1000.scores", "QoraPhase default, 1000 it (20)"),
    ("own_consensus_shrinkwrap.scores", "Own consensus, default recipe (20)"),
    ("own_consensus_fixed.scores", "Own consensus, fixed support (20)"),
)

SLICES = (
    ("default_best", "QoraPhase best"),
    ("default_consensus", "QoraPhase consensus"),
    ("default_1000_consensus", "QoraPhase 1000 it, consensus"),
    ("own_consensus_fixed_best", "Own consensus, fixed, best"),
    ("lab_lab", "Lab pass 2, lab rule"),
    ("lab_twin_checked", "Lab pass 2, twin-checked, lab rule"),
)


def title(case):
    match = re.fullmatch(r".*_strain([\d.]+)_ph(inf|[\de]+)_s\d+", case)
    dose = (
        "noiseless"
        if match[2] == "inf"
        else f"Poisson, 10{match[2][2:].translate(SUPERSCRIPT)} photons"
    )
    return f"peak phase {match[1]} rad, {dose}"


def lookup(result, path):
    for key in path.split("."):
        if key not in result:
            return None
        result = result[key]
    return [row["phase_rmse_rad"] for row in result]


def merged(directories):
    results = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            results[result["case"]] = results.get(result["case"], {}) | result
    return results


def spread_figure(results, path):
    cases = sorted(results, key=lambda c: (c.split("_ph")[0], "inf" in c))
    columns = min(len(cases), 2)
    rows = -(-len(cases) // columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4.4 * columns + 2.6, 3.4 * rows), sharey=True, squeeze=False
    )
    for ax in axes.flat[len(cases) :]:
        ax.set_visible(False)
    rng = np.random.default_rng(0)
    for ax, case in zip(axes.flat, cases, strict=False):
        for y, (key, _) in enumerate(ROWS):
            values = lookup(results[case], key)
            if values is None:
                ax.text(0.01, y, "not run", va="center", fontsize=8, color="0.5")
                continue
            values = np.asarray(values)
            shown = np.minimum(values, OFF_SCALE)
            jitter = rng.uniform(-0.18, 0.18, len(values))
            ax.scatter(shown, y + jitter, s=14, color=INK, alpha=0.75, linewidths=0)
            ax.plot([np.median(values)] * 2, [y - 0.32, y + 0.32], color=MEDIAN, lw=2)
            for v, j in zip(values, jitter, strict=True):
                if v > OFF_SCALE:
                    ax.annotate(
                        f"{v:.2f}",
                        (OFF_SCALE, y + j),
                        xytext=(-4, 6),
                        textcoords="offset points",
                        ha="right",
                        fontsize=7,
                    )
        ax.set_xlim(0, OFF_SCALE + 0.01)
        ax.set_xlabel("phase RMSE (rad)")
        ax.set_title(title(case), fontsize=9)
        ax.grid(axis="x", color="0.9")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_yticks(range(len(ROWS)), [label for _, label in ROWS])
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def crop(truth, margin=3):
    inside = np.argwhere(np.abs(truth) > CUT * np.abs(truth).max())
    low, high = inside.min(0) - margin, inside.max(0) + margin + 1
    return tuple(slice(max(lo, 0), hi) for lo, hi in zip(low, high, strict=True))


def slice_figure(truth, objects, detframe, case, path):
    centre = round(np.average(np.arange(truth.shape[0]), weights=np.abs(truth).sum((1, 2))))
    window = crop(truth)[1:]
    panels = [("Truth", truth)]
    panels += [
        (label, detframe.align(objects[key], truth)) for key, label in SLICES if key in objects
    ]
    fig, axes = plt.subplots(
        2, len(panels), figsize=(1.9 * len(panels) + 0.9, 3.9), squeeze=False, layout="constrained"
    )
    for column, (label, obj) in enumerate(panels):
        plane = obj[centre][window]
        amplitude = np.abs(plane) / np.abs(obj).max()
        phase = np.ma.masked_where(amplitude < CUT, np.angle(plane))
        top = axes[0][column].imshow(
            amplitude, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest"
        )
        bottom = axes[1][column].imshow(
            phase, cmap="coolwarm", vmin=-PHASE_LIMIT, vmax=PHASE_LIMIT, interpolation="nearest"
        )
        axes[0][column].set_title(label, fontsize=8)
        for ax in axes[:, column]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0][0].set_ylabel("|object|, scaled", fontsize=8)
    axes[1][0].set_ylabel("phase (rad)", fontsize=8)
    fig.colorbar(top, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
    fig.colorbar(bottom, ax=axes[1].tolist(), shrink=0.8, pad=0.01)
    fig.suptitle(f"{title(case)}: frame {centre} of {truth.shape[0]}", fontsize=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", type=Path)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slices", metavar="CASE", help="case for the slice figure")
    args = parser.parse_args()
    results = merged(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    spread_figure(results, args.out / "bcdi-lab-spread.png")
    if args.slices:
        objects = {}
        for directory in args.results:
            for path in directory.glob(f"{args.slices}-*.npz"):
                with np.load(path) as saved:
                    objects |= dict(saved)
        detframe, _ = load_scorer()
        truth = np.load(args.cases / args.slices / "truth_det.npy")
        slice_figure(truth, objects, detframe, args.slices, args.out / "bcdi-lab-slices.png")


if __name__ == "__main__":
    main()
