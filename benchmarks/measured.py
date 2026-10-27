"""QoraPhase on a measured 3D BCDI dataset against a reconstruction the lab MATLAB engine made of it.

    uv run --python 3.14+gil --with scipy --with mat-io python benchmarks/measured.py \
        --data <prepared intensity .mat> --reference <reconstruction prefix> --label "dataset A"

The data and the reference stay outside the repository: ``--data`` is the engine's prepared
intensity (``array``), ``--reference`` the prefix of its ``-AMP.rec``, ``-PH.rec`` and ``-SUP.rec``
files (saved gpuArrays). Only summary numbers are written, to ``docs/benchmarks/measured.md``:
no images, arrays or scan parameters. ``--keep`` saves QoraPhase's results to a local ``.npz``
outside the repository and reuses them on the next run, so the comparison can change without
phasing again.

``--lab`` adds the lab engine's own spread: a folder of ``lab-<seed>.mat`` files that
``benchmarks/measured_driver.m`` writes by rerunning the engine with the reference's saved
parameters, one seed each (the engine is not in this repository; put its folders on the MATLAB
path first). Each run is cut to its final support, as the reference is, and compared like a
QoraPhase start; a last table compares the two engines' runs with each other.

A measured dataset has no ground truth, so the comparison is between the two results:

- orientation: the reference is tried under every combination of axis flips, and the one with
  the lowest Fourier residual against the data is kept, each after restoring a phase ramp (the
  engine may reorient and remove the ramp on saving), then cut to its saved support, since the
  saved array is the iterate before the support step;
- Fourier residual of every result against the same data, after a least-squares scale;
- mutual error, built up one alignment at a time so each one's share shows: shift to 1/10 px,
  global phase, twin and complex scale, as in the ladder; then a linear phase ramp to 1/10 px of
  the pattern; then only inside the reference support or where either amplitude exceeds 0.1 of
  its maximum; then the amplitudes alone in that region;
- spread: the same errors among QoraPhase's own starts, for scale.
"""

import argparse
import itertools
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import qoraphase as qp
from qoraphase.postprocess import _register_and_apply, register, twin

sys.path.insert(0, str(Path(__file__).parent))
from ladder import centered_fft, environment
from legacy import box_support, matched_recipe

OUT = Path(__file__).parents[1] / "docs" / "benchmarks" / "measured.md"
UPSAMPLE = 10
LEVEL = 0.1
METRICS = (
    "complex, whole array: shift, phase, twin, scale",
    "+ linear phase ramp",
    "+ only inside support or amplitude > 0.1 max",
    "amplitude only, same region",
)


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


def load_lab(folder):
    """The engine's reruns cut to their supports, its median seconds and median chi."""
    from scipy.io import loadmat

    runs = [loadmat(p) for p in sorted(Path(folder).glob("lab-*.mat"))]
    if not runs:
        raise SystemExit(f"no lab-*.mat in {folder}")
    objects = [np.asarray(r["iterate"] * r["support"], complex) for r in runs]
    seconds = statistics.median(float(r["seconds"].squeeze()) for r in runs)
    chi = statistics.median(float(r["chi"].squeeze()) for r in runs)
    return objects, seconds, chi


def residual(obj, amplitude):
    """Normalised Fourier modulus error after the least-squares real scale."""
    computed = np.abs(centered_fft(obj))
    c = (computed * amplitude).sum() / (computed**2).sum()
    return float(np.sqrt(((c * computed - amplitude) ** 2).sum() / (amplitude**2).sum()))


def ramped(obj, modulus):
    """*obj* times the linear phase ramp that moves its pattern onto *modulus*, and the shift."""
    shift, _ = register(modulus, np.abs(centered_fft(obj)), UPSAMPLE)
    shift = tuple(float(s) for s in shift)
    grids = np.meshgrid(*(np.arange(n) - n // 2 for n in obj.shape), indexing="ij")
    phase = sum(s * g / n for s, g, n in zip(shift, grids, obj.shape, strict=True))
    return obj * np.exp(2j * np.pi * phase), shift


def oriented(reference, support, amplitude):
    """*reference* and *support* under the flips and ramp that best fit the data, the flips, the
    pattern shift and the fit."""
    fits = []
    for r in range(reference.ndim + 1):
        for axes in itertools.combinations(range(reference.ndim), r):
            moved, shift = ramped(np.flip(reference, axes), amplitude)
            fits.append((residual(moved, amplitude), axes, shift, moved))
    fit, axes, shift, moved = min(fits, key=lambda f: f[0])
    return moved, np.flip(support, axes), axes, shift, fit


def aligned(obj, reference, ramp):
    """*obj* or its twin, optionally ramped, registered and scaled onto *reference*."""
    best = None
    for is_twin, candidate in ((False, obj), (True, np.asarray(twin(obj)))):
        if ramp:
            candidate, _ = ramped(candidate, np.abs(centered_fft(reference)))
        moved = np.asarray(_register_and_apply(candidate, reference, UPSAMPLE)[0])
        moved = moved * np.vdot(moved, reference) / np.vdot(moved, moved)
        err = np.linalg.norm(moved - reference) / np.linalg.norm(reference)
        if best is None or err < best[0]:
            best = (err, is_twin, moved)
    return best


def errors(obj, reference, support):
    """The four mutual errors of METRICS, and whether the twin matched."""
    plain, is_twin, _ = aligned(obj, reference, ramp=False)
    ramp_err, _, moved = aligned(obj, reference, ramp=True)
    region = support | (np.abs(reference) > LEVEL * np.abs(reference).max())
    region |= np.abs(moved) > LEVEL * np.abs(moved).max()
    a, b = moved[region], reference[region]
    inside = np.linalg.norm(a * np.vdot(a, b) / np.vdot(a, a) - b) / np.linalg.norm(b)
    a, b = np.abs(a), np.abs(b)
    amp = np.linalg.norm(a * (a @ b) / (a @ a) - b) / np.linalg.norm(b)
    return [float(plain), float(ramp_err), float(inside), float(amp)], is_twin


def run(problem, recipe, starts):
    t = time.perf_counter()
    ens = problem.reconstruct(recipe, starts=starts, seed=0)
    objects = [np.asarray(r.object) for r in ens]
    return objects, (time.perf_counter() - t) / starts


def own_support(obj):
    return np.abs(obj) > LEVEL * np.abs(obj).max()


def summary(name, objects, reference, support, amplitude, seconds):
    """A fit row, and per metric the best-fitting start against the reference, the median start
    and the median among the starts themselves."""
    fits = [residual(o, amplitude) for o in objects]
    against = [errors(o, reference, support) for o in objects]
    best = against[int(np.argmin(fits))][0]
    twins = sum(t for _, t in against)
    own = [errors(a, b, own_support(b))[0] for a, b in itertools.combinations(objects, 2)]
    fit = (
        f"| {name} | {len(objects)} | {statistics.median(fits):.3f} ({min(fits):.3f}) "
        f"| {twins} | {seconds:.0f} |"
    )
    cells = [
        (best[m], statistics.median(e[m] for e, _ in against), statistics.median(o[m] for o in own))
        for m in range(len(METRICS))
    ]
    return fit, cells


def between(objects, others, amplitude):
    """Per metric, the best-fitting run of *objects* against the best-fitting one of *others*,
    and the median over all pairs; *others* is the side registered onto."""
    best = min(objects, key=lambda o: residual(o, amplitude))
    other = min(others, key=lambda o: residual(o, amplitude))
    top = errors(best, other, own_support(other))[0]
    pairs = [errors(a, b, own_support(b))[0] for a in objects for b in others]
    return [(top[m], statistics.median(p[m] for p in pairs)) for m in range(len(METRICS))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--label", required=True, help="how the report names the dataset")
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--keep", help="local .npz for QoraPhase's results, reused when present")
    parser.add_argument("--lab", help="folder of lab-<seed>.mat reruns from measured_driver.m")
    args = parser.parse_args()

    intensity = load_intensity(args.data)
    amplitude = np.sqrt(np.clip(intensity, 0, None))
    reference = load_rec(f"{args.reference}-AMP.rec") * np.exp(
        1j * load_rec(f"{args.reference}-PH.rec")
    )
    support = load_rec(f"{args.reference}-SUP.rec") > 0.5
    iterate, support, flips, shift, iterate_fit = oriented(reference, support, amplitude)
    # The engine saves its last iterate after the Fourier step, before the support step, so it
    # matches the data nearly by construction; QoraPhase returns support-projected objects.
    reference = iterate * support
    outside = float((np.abs(iterate[~support]) ** 2).sum() / (np.abs(iterate) ** 2).sum())
    reference_fit = residual(reference, amplitude)
    print(
        f"reference: flips {flips}, shift {shift}, residual {iterate_fit:.4f} as saved, "
        f"{reference_fit:.4f} in its support, {outside:.3f} of the energy outside",
        flush=True,
    )

    shape = intensity.shape
    recipes = {
        "QoraPhase, matched": (
            qp.CDI(intensity, support=box_support(shape)),
            matched_recipe(shape),
        ),
        "QoraPhase, default": (qp.CDI(intensity), qp.default_recipe()),
    }
    kept = dict(np.load(args.keep)) if args.keep and Path(args.keep).exists() else {}
    results = {}
    for i, (name, (problem, recipe)) in enumerate(recipes.items()):
        if f"objects{i}" in kept:
            results[name] = (list(kept[f"objects{i}"]), float(kept[f"seconds{i}"]))
        else:
            results[name] = run(problem, recipe, args.starts)
    if args.keep and not kept:
        np.savez(
            args.keep,
            **{f"objects{i}": np.stack(r[0]) for i, r in enumerate(results.values())},
            **{f"seconds{i}": r[1] for i, r in enumerate(results.values())},
        )

    lab = None
    if args.lab:
        objects, seconds, chi = load_lab(args.lab)
        lab = f"lab MATLAB, {len(objects)} seeds"
        results[lab] = (objects, seconds)

    fits, columns = [], []
    for name, (objects, seconds) in results.items():
        fit, cells = summary(name, objects, reference, support, amplitude, seconds)
        fits.append(fit)
        columns.append(cells)
        print(fit, cells, flush=True)

    cross = []
    if lab:
        for name in results:
            if name == lab:
                continue
            cells = between(results[name][0], results[lab][0], amplitude)
            print(name, "vs", lab, cells, flush=True)
            cross.append(f"| {name} | " + " | ".join(f"{t:.3f} ({p:.3f})" for t, p in cells) + " |")
        spread = [
            (
                f"- lab MATLAB reruns (`benchmarks/measured_driver.m`): the reference's saved "
                f"parameters, MATLAB `rng` seeds 0 to {len(results[lab][0]) - 1}, the engine's GPU "
                f"path as in the saved run; engine chi median {chi:.4f}; each run cut to its final "
                f"support; its time column is per run on the GPU, not comparable with the CPU rows"
            ),
        ]
    else:
        spread = ["- the lab code's own spread over seeds is not measured yet: it needs MATLAB"]

    flipped = "none" if not flips else ", ".join(f"axis {a}" for a in flips)
    lines = [
        f"## {args.label}",
        "",
        *environment(),
        f"- {len(next(iter(results.values()))[0])} starts per recipe, keys split from seed 0; 3D, CPU",
        (
            f"- lab MATLAB reference: one saved reconstruction, a single run (no genetic algorithm, "
            f"no averaging), on the same prepared intensity; oriented by flipping {flipped} and "
            f"restoring a removed phase ramp (pattern shift "
            f"{', '.join(f'{s:.1f}' for s in shift)} px)"
        ),
        (
            f"- the saved array is the iterate after the Fourier step, before the support step: "
            f"Fourier residual {iterate_fit:.3f} as saved, with {outside:.1%} of its energy outside "
            f"its saved support; cut to that support, as compared below, {reference_fit:.3f}"
        ),
        *spread,
        "",
        "| engine | starts | Fourier residual, median (best) | twin vs reference | s per start, incl. compile |",
        "|---|---:|---:|---:|---:|",
        *fits,
        "",
        "Mutual error: the best-fitting start against the reference (median start in brackets), and the median among the engine's own starts.",
        "",
        "| alignment | "
        + " | ".join(f"{n}: vs reference | {n}: own starts" for n in results)
        + " |",
        "|---|" + "---:|" * (2 * len(results)),
        *(
            f"| {metric} | "
            + " | ".join(f"{c[m][0]:.3f} ({c[m][1]:.3f}) | {c[m][2]:.3f}" for c in columns)
            + " |"
            for m, metric in enumerate(METRICS)
        ),
        "",
    ]
    if cross:
        lines += [
            (
                "Between the engines: each QoraPhase recipe's best-fitting start against the best-"
                "fitting lab run (median over all pairs in brackets), registered onto the lab run."
            ),
            "",
            "| QoraPhase recipe | " + " | ".join(METRICS) + " |",
            "|---|" + "---:|" * len(METRICS),
            *cross,
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
