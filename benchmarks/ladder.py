"""Benchmark ladder on synthetic 2D data, CPU: how often and how well the default recipe recovers
a known object as the measurement gets worse.

    uv run python benchmarks/ladder.py            # writes docs/benchmarks/
    uv run python benchmarks/ladder.py --quick    # 4 seeds, a smoke run

Stage 1 is noiseless, stage 2 adds Poisson noise and a support that is not the true one. Every
case runs the same seeds with the same iteration budget (720, the length of ``default_recipe``).

Each start is scored against the true object after resolving the three CDI ambiguities: shift
(to 0.1 pixel), global phase and twin. Both the start and its twin are registered onto the truth
and the smaller error is kept; how many starts matched as the twin is counted separately. The
real-space error is ``|c a - t| / |t|`` over the whole array, with ``c`` the least-squares complex
factor (global phase and the photon-count scale). The Fourier residual is the library's own
normalised modulus error, and is also given relative to the noise floor: the residual the true
object itself has against the noisy data.

Output: ``docs/benchmarks/ladder.md`` (environment, seeds, table) and ``ladder.png``.
"""

import argparse
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

import qoraphase as qp
from qoraphase.postprocess import _register_and_apply, twin

OUT = Path(__file__).parents[1] / "docs" / "benchmarks"
N, M = 128, 40
OBJECT_SEED, NOISE_SEED = 11, 23
SEEDS = 20
UPSAMPLE = 10
SUCCESS = 0.1
# Okabe-Ito; each support also has its own marker and line style, so colour is never the only cue.
STYLE = {
    "exact": ("#0072B2", "o", "-"),
    "dilated 2 px": ("#E69F00", "s", "--"),
    "autocorrelation + shrinkwrap": ("#009E73", "^", ":"),
}
PEAKS = (1e4, 1e6, 1e8)


def smooth(rng, shape, width):
    """Gaussian-filtered white noise scaled to [-1, 1]."""
    k = [np.fft.fftfreq(n) for n in shape]
    kernel = np.exp(-2 * (np.pi * width) ** 2 * sum(np.square(g) for g in np.meshgrid(*k)))
    field = np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal(shape)) * kernel))
    return field / np.abs(field).max()


def make_object():
    """A complex object (amplitude 0.4 to 1, phase within +-1 rad) on a triangular support, which
    is not centrosymmetric, so the twin is a different object."""
    rng = np.random.default_rng(OBJECT_SEED)
    block = (0.7 + 0.3 * smooth(rng, (M, M), 3)) * np.exp(1j * smooth(rng, (M, M), 5))
    block = block * (np.indices((M, M)).sum(0) < M)
    obj = np.zeros((N, N), complex)
    lo = N // 2 - M // 2
    obj[lo : lo + M, lo : lo + M] = block
    return obj


def centered_fft(x):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), norm="ortho"))


def dilate(mask, px):
    out = mask.copy()
    for _ in range(px):
        grown = out.copy()
        for axis in (0, 1):
            grown |= np.roll(out, 1, axis) | np.roll(out, -1, axis)
        out = grown
    return out


def fixed_support_recipe():
    """``default_recipe`` with the shrinkwrap passes replaced by plain RAAR: the same 720
    iterations, for a support that must not change."""
    return qp.Recipe(qp.RAAR(50), qp.detwin(qp.HIO(20)), qp.RAAR(50), qp.RAAR(400), qp.ER(200))


def score(obj, truth):
    """(real-space error, matched as twin) with shift, global phase, scale and twin resolved."""
    best = None
    for is_twin, candidate in ((False, obj), (True, twin(obj))):
        moved, _ = _register_and_apply(candidate, truth, UPSAMPLE)
        a = np.asarray(moved)
        c = np.vdot(a, truth) / np.vdot(a, a)
        err = float(np.linalg.norm(c * a - truth) / np.linalg.norm(truth))
        if best is None or err < best[0]:
            best = (err, is_twin)
    return best


@dataclass
class Case:
    stage: int
    peak: float | None
    support: str
    errors: list[float]
    twins: int
    fourier: list[float]
    floor: float
    first_s: float
    warm_s: float

    @property
    def success(self):
        return sum(e < SUCCESS for e in self.errors)


def run_case(stage, peak, support_name, obj, seeds):
    clean = np.abs(centered_fft(obj)) ** 2
    if peak is None:
        data, scale, floor = clean, 1.0, 0.0
    else:
        scale = peak / clean.max()
        data = np.random.default_rng(NOISE_SEED).poisson(clean * scale).astype(float)
        floor = float(np.sqrt(((np.sqrt(clean * scale) - np.sqrt(data)) ** 2).sum() / data.sum()))
    true_support = np.abs(obj) > 0
    if support_name == "exact":
        problem, recipe = qp.CDI(data, support=true_support), fixed_support_recipe()
    elif support_name == "dilated 2 px":
        problem, recipe = qp.CDI(data, support=dilate(true_support, 2)), fixed_support_recipe()
    else:
        problem, recipe = qp.CDI(data), qp.default_recipe()

    def once():
        t = time.perf_counter()
        ens = problem.reconstruct(recipe, starts=len(seeds), seed=seeds[0])
        objects = [np.asarray(r.object) for r in ens]
        return time.perf_counter() - t, ens, objects

    first, _, _ = once()
    warm, ens, objects = once()
    truth = obj * np.sqrt(scale)
    scored = [score(o, truth) for o in objects]
    return Case(
        stage,
        peak,
        support_name,
        [e for e, _ in scored],
        sum(t for _, t in scored),
        [float(r.error) for r in ens],
        floor,
        first,
        warm,
    )


def quartiles(values):
    return np.percentile(values, [50, 25, 75])


def environment():
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return [
        f"- run {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}, commit {commit or 'unknown'}",
        (
            f"- Python {platform.python_version()}, JAX {jax.__version__}, "
            f"jaxlib {version('jaxlib')}, NumPy {np.__version__}, qoraphase {version('qoraphase')}"
        ),
        f"- device {jax.devices()[0]}, x64 {'on' if jax.config.jax_enable_x64 else 'off'}",
        f"- {platform.platform()}, processor {platform.processor() or platform.machine()}",
    ]


def table(cases, seeds):
    lines = [
        (
            "| stage | peak counts | support | success | twin | real-space error, median (IQR) "
            "| Fourier residual, median | residual / noise floor | first call s | warm s |"
        ),
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        med, q1, q3 = quartiles(c.errors)
        fmed = float(np.median(c.fourier))
        ratio = f"{fmed / c.floor:.2f}" if c.floor else "n/a"
        peak = "noiseless" if c.peak is None else f"{c.peak:.0e}"
        lines.append(
            f"| {c.stage} | {peak} | {c.support} | {c.success}/{len(seeds)} | {c.twins} "
            f"| {med:.2e} ({q1:.2e} to {q3:.2e}) | {fmed:.2e} | {ratio} "
            f"| {c.first_s:.1f} | {c.warm_s:.1f} |"
        )
    return lines


def figure(cases, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    noisy = [c for c in cases if c.peak is not None]
    for name, (colour, marker, line) in STYLE.items():
        mine = [c for c in noisy if c.support == name]
        peaks = [c.peak for c in mine]
        med = [np.median(c.errors) for c in mine]
        lo = [np.percentile(c.errors, 25) for c in mine]
        hi = [np.percentile(c.errors, 75) for c in mine]
        left.errorbar(
            peaks,
            med,
            yerr=[np.subtract(med, lo), np.subtract(hi, med)],
            color=colour,
            marker=marker,
            linestyle=line,
            capsize=3,
            label=name,
        )
        right.plot(
            peaks,
            [c.success / len(c.errors) for c in mine],
            color=colour,
            marker=marker,
            linestyle=line,
            label=name,
        )
    left.axhline(SUCCESS, color="#999999", linestyle="-.", linewidth=1, label="success threshold")
    left.set(xscale="log", yscale="log", xlabel="peak counts", ylabel="real-space error")
    left.set_title("Stage 2: error after shift, phase and twin (median, IQR)")
    right.set(xscale="log", ylim=(-0.05, 1.05), xlabel="peak counts", ylabel="success rate")
    right.set_title(f"Stage 2: starts with error < {SUCCESS:g}")
    left.legend(fontsize=8)
    right.legend(fontsize=8)
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="4 seeds")
    args = parser.parse_args()
    seeds = list(range(4 if args.quick else SEEDS))
    obj = make_object()
    cases = []
    for support in STYLE:
        cases.append(run_case(1, None, support, obj, seeds))
        print(table(cases[-1:], seeds)[-1], flush=True)
    for peak in PEAKS:
        for support in STYLE:
            cases.append(run_case(2, peak, support, obj, seeds))
            print(table(cases[-1:], seeds)[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    figure(cases, OUT / "ladder.png")
    text = [
        "# Benchmark ladder",
        "",
        (
            f"Generated by `benchmarks/ladder.py`{' --quick' if args.quick else ''}; the method is in "
            "its docstring. Synthetic data only: these numbers say nothing about measured scans."
        ),
        "",
        *environment(),
        (
            f"- object {N}x{N}, {M}-pixel triangular support, object seed {OBJECT_SEED}, "
            f"Poisson seed {NOISE_SEED}"
        ),
        (
            f"- starts: `CDI.reconstruct(..., starts={len(seeds)}, seed=0)`, i.e. keys split from seed 0 "
            f"(start indices {seeds[0]} to {seeds[-1]}); 720 iterations each"
        ),
        (
            f"- success: real-space error < {SUCCESS:g}. This is an arbitrary threshold, not derived "
            "from the data; the median and IQR are given so another cut can be read off."
        ),
        "- twin: starts whose twin matched the truth better than the start itself",
        "- times: one vmapped call for all starts; a first call compiles only when its recipe is new to the run, so later rows show run time in both columns",
        "",
        *table(cases, seeds),
        "",
        "![ladder](ladder.png)",
        "",
    ]
    (OUT / "ladder.md").write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {OUT / 'ladder.md'}")


if __name__ == "__main__":
    sys.exit(main())
