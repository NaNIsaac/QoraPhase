"""QoraPhase on a measured 3D BCDI dataset against a reconstruction the lab MATLAB engine made of it.

    uv run --python 3.14+gil --with scipy --with mat-io python benchmarks/measured.py \
        --data <prepared intensity .mat> --reference <reconstruction prefix> --label "dataset A"

The data and the reference stay outside the repository: ``--data`` is the engine's prepared
intensity (``array``), ``--reference`` the prefix of its ``-AMP.rec``, ``-PH.rec`` and ``-SUP.rec``
files (saved gpuArrays). Only summary numbers are written, to ``docs/benchmarks/measured.md``:
no images, arrays or scan parameters.

A measured dataset has no ground truth, so the comparison is between the two results:

- orientation: the reference is tried under every combination of axis flips, and the one with
  the lowest Fourier residual against the data is kept, each after undoing a phase ramp (the
  engine may reorient and remove the ramp on saving);
- mutual error: each QoraPhase start registered onto the reference (shift, global phase, twin,
  complex scale), as in the ladder;
- Fourier residual of every result against the same data, after a least-squares scale;
- spread: the mutual error among QoraPhase's own starts, for scale.
"""

import argparse
import itertools
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import qoraphase as qp

sys.path.insert(0, str(Path(__file__).parent))
from ladder import centered_fft, environment, score
from legacy import box_support, matched_recipe

OUT = Path(__file__).parents[1] / "docs" / "benchmarks" / "measured.md"


def load_intensity(path):
    from scipy.io import loadmat

    return np.asarray(loadmat(path)["array"], float)


def load_rec(path):
    """The largest array inside a saved gpuArray."""
    import matio

    found = []

    def walk(x):
        if isinstance(x, np.ndarray) and x.dtype == object:
            for y in x.flat:
                walk(y)
        elif isinstance(x, np.ndarray):
            found.append(x)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)

    walk(matio.load_from_mat(str(path))["array"].properties)
    return np.asarray(max(found, key=lambda a: a.size), float)


def residual(obj, amplitude):
    """Normalised Fourier modulus error after the least-squares real scale."""
    computed = np.abs(centered_fft(obj))
    c = (computed * amplitude).sum() / (computed**2).sum()
    return float(np.sqrt(((c * computed - amplitude) ** 2).sum() / (amplitude**2).sum()))


def oriented(reference, amplitude):
    """*reference* under the axis flips and phase ramp that best fit the data, and that fit."""
    flips = [
        axes
        for r in range(reference.ndim + 1)
        for axes in itertools.combinations(range(reference.ndim), r)
    ]
    fits = []
    for axes in flips:
        moved, ramp = recentred(np.flip(reference, axes) if axes else reference, amplitude)
        fits.append((residual(moved, amplitude), axes, ramp, moved))
    fit, axes, ramp, moved = min(fits, key=lambda f: f[0])
    return moved, axes, ramp, fit


def recentred(reference, amplitude):
    """*reference* with the phase ramp that moves its diffraction onto the data's.

    The engine may remove a phase ramp before saving, which shifts the computed pattern off
    the measured one; the integer shift is found by cross-correlating the two moduli.
    """
    computed = np.abs(centered_fft(reference))
    corr = np.fft.ifftn(np.fft.fftn(amplitude) * np.conj(np.fft.fftn(computed))).real
    shift = np.unravel_index(int(np.argmax(corr)), corr.shape)
    shift = tuple(
        int(s) if s <= n // 2 else int(s) - n for s, n in zip(shift, corr.shape, strict=True)
    )
    pattern = np.roll(centered_fft(reference), shift, axis=tuple(range(reference.ndim)))
    moved = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(pattern), norm="ortho"))
    return moved, shift


def run(problem, recipe, starts):
    t = time.perf_counter()
    ens = problem.reconstruct(recipe, starts=starts, seed=0)
    objects = [np.asarray(r.object) for r in ens]
    return objects, (time.perf_counter() - t) / starts


def describe(name, objects, reference, amplitude, seconds):
    fits = [residual(o, amplitude) for o in objects]
    mutual = [score(o, reference)[0] for o in objects]
    twins = sum(score(o, reference)[1] for o in objects)
    best = objects[int(np.argmin(fits))]
    own = [score(a, b)[0] for a, b in itertools.combinations(objects, 2)]
    return (
        f"| {name} | {len(objects)} | {statistics.median(fits):.3f} ({min(fits):.3f}) "
        f"| {score(best, reference)[0]:.3f} | {statistics.median(mutual):.3f} | {twins} "
        f"| {statistics.median(own):.3f} | {seconds:.0f} |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--label", required=True, help="how the report names the dataset")
    parser.add_argument("--starts", type=int, default=10)
    args = parser.parse_args()

    intensity = load_intensity(args.data)
    amplitude = np.sqrt(np.clip(intensity, 0, None))
    reference = load_rec(f"{args.reference}-AMP.rec") * np.exp(
        1j * load_rec(f"{args.reference}-PH.rec")
    )
    reference, flips, ramp, reference_fit = oriented(reference, amplitude)
    print(f"reference: flips {flips}, ramp {ramp}, residual {reference_fit:.3f}", flush=True)

    rows = []
    shape = intensity.shape
    for name, problem, recipe in (
        (
            "QoraPhase, matched",
            qp.CDI(intensity, support=box_support(shape)),
            matched_recipe(shape),
        ),
        ("QoraPhase, default", qp.CDI(intensity), qp.default_recipe()),
    ):
        objects, seconds = run(problem, recipe, args.starts)
        rows.append(describe(name, objects, reference, amplitude, seconds))
        print(rows[-1], flush=True)

    flipped = "none" if not flips else ", ".join(f"axis {a}" for a in flips)
    lines = [
        f"## {args.label}",
        "",
        *environment(),
        f"- {args.starts} starts per recipe, keys split from seed 0; 3D, CPU",
        (
            f"- lab MATLAB reference: one saved reconstruction (shrinkwrap run); Fourier residual "
            f"{reference_fit:.3f} after flipping {flipped} and restoring a removed phase ramp "
            f"(pattern shift {', '.join(map(str, ramp))} px)"
        ),
        "",
        (
            "| engine | starts | Fourier residual, median (best) | mutual error, best run vs "
            "reference | mutual error, median | twin | spread among own starts, median "
            "| s per start, incl. compile (reference) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    header = [
        "# Measured data",
        "",
        (
            "Generated by `benchmarks/measured.py` (method in its docstring). Summary numbers only; the "
            "data, the reference reconstruction and the scan are not published."
        ),
        "",
    ]
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else "\n".join(header)
    marker = f"## {args.label}\n"
    if marker in old:
        before, _, rest = old.partition(marker)
        after = rest.partition("\n## ")
        old = before + (("## " + after[2]) if after[1] else "")
    OUT.write_text(old.rstrip("\n") + "\n\n" + "\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
