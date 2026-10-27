"""Benchmark ladder on synthetic 2D data, CPU: how often and how well the default recipe recovers
a known object as the measurement gets worse.

    uv run --python 3.14+gil python benchmarks/ladder.py            # writes docs/benchmarks/
    uv run --python 3.14+gil python benchmarks/ladder.py --quick    # 4 seeds, a smoke run

``+gil`` keeps uv off a free-threaded Python, for which jaxlib has no wheel; on a machine with
both builds of 3.14, uv on Windows otherwise picked the free-threaded one.

Stage 1 is noiseless, stage 2 adds Poisson noise and a support that is not the true one, stage 3
adds a beamstop, detector gaps and dead pixels, and a stronger phase. Every
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
# Stage 3, at the middle count level of stage 2.
STAGE3_PEAK = 1e6
BEAMSTOP = 4
GAP, GAP_OFFSET, DEAD, MASK_SEED = 3, 12, 0.005, 31
STRONG_PHASE = 2.5
CONDITIONS = (
    "beamstop",
    "gaps + dead pixels",
    f"phase ±{STRONG_PHASE:g} rad",
    "all three",
)


def smooth(rng, shape, width):
    """Gaussian-filtered white noise scaled to [-1, 1]."""
    k = [np.fft.fftfreq(n) for n in shape]
    kernel = np.exp(-2 * (np.pi * width) ** 2 * sum(np.square(g) for g in np.meshgrid(*k)))
    field = np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal(shape)) * kernel))
    return field / np.abs(field).max()


def make_object(phase=1.0):
    """A complex object (amplitude 0.4 to 1, phase within +-``phase`` rad) on a triangular
    support, which is not centrosymmetric, so the twin is a different object."""
    rng = np.random.default_rng(OBJECT_SEED)
    block = (0.7 + 0.3 * smooth(rng, (M, M), 3)) * np.exp(1j * phase * smooth(rng, (M, M), 5))
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


def beamstop():
    """Unusable pixels: a disc of radius BEAMSTOP around the zero frequency."""
    y, x = np.indices((N, N)) - N // 2
    return x**2 + y**2 <= BEAMSTOP**2


def gaps():
    """Unusable pixels: a horizontal and a vertical module gap, GAP pixels wide and off centre,
    plus DEAD of the pixels at random."""
    mask = np.zeros((N, N), bool)
    lo = N // 2 + GAP_OFFSET
    mask[lo : lo + GAP, :] = True
    mask[:, lo : lo + GAP] = True
    return mask | (np.random.default_rng(MASK_SEED).random((N, N)) < DEAD)


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
    condition: str
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


def run_case(stage, peak, support_name, obj, seeds, condition="", mask=None):
    """``mask`` marks unusable pixels: they read 0 and are left out of the fit and the floor."""
    clean = np.abs(centered_fft(obj)) ** 2
    usable = np.ones(clean.shape, bool) if mask is None else ~mask
    if peak is None:
        data, scale, floor = clean, 1.0, 0.0
    else:
        scale = peak / clean.max()
        data = np.random.default_rng(NOISE_SEED).poisson(clean * scale).astype(float)
        misfit = (np.sqrt(clean * scale) - np.sqrt(data)) ** 2
        floor = float(np.sqrt(misfit[usable].sum() / data[usable].sum()))
    data = np.where(usable, data, 0.0)
    true_support = np.abs(obj) > 0
    if support_name == "exact":
        support, recipe = true_support, fixed_support_recipe()
    elif support_name == "dilated 2 px":
        support, recipe = dilate(true_support, 2), fixed_support_recipe()
    else:
        support, recipe = None, qp.default_recipe()
    problem = qp.CDI(data, support=support, mask=mask)

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
        condition,
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
            "| stage | peak counts | condition | support | success | twin "
            "| real-space error, median (IQR) | Fourier residual, median | residual / noise floor "
            "| first call s | warm s |"
        ),
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cases:
        med, q1, q3 = quartiles(c.errors)
        fmed = float(np.median(c.fourier))
        ratio = f"{fmed / c.floor:.2f}" if c.floor else "n/a"
        peak = "noiseless" if c.peak is None else f"{c.peak:.0e}"
        lines.append(
            f"| {c.stage} | {peak} | {c.condition or '-'} | {c.support} "
            f"| {c.success}/{len(seeds)} | {c.twins} "
            f"| {med:.2e} ({q1:.2e} to {q3:.2e}) | {fmed:.2e} | {ratio} "
            f"| {c.first_s:.1f} | {c.warm_s:.1f} |"
        )
    return lines


def figure(cases, path):
    fig, (left, right, third) = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    noisy = [c for c in cases if c.stage == 2]
    hard = [c for c in cases if c.stage == 3]
    for k, (name, (colour, marker, line)) in enumerate(STYLE.items()):
        mine = [c for c in hard if c.support == name]
        third.plot(
            np.arange(len(mine)) + (k - 1) * 0.08,
            [c.success / len(c.errors) for c in mine],
            color=colour,
            marker=marker,
            linestyle=line,
            label=name,
        )
    third.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=15, fontsize=8)
    third.set(ylim=(-0.05, 1.05), ylabel="success rate")
    third.set_title(f"Stage 3 at {STAGE3_PEAK:.0e} peak counts")
    third.legend(fontsize=8)
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
    strong = make_object(STRONG_PHASE)
    stage3 = dict(
        zip(
            CONDITIONS,
            ((obj, beamstop()), (obj, gaps()), (strong, None), (strong, beamstop() | gaps())),
            strict=True,
        )
    )
    for condition, (target, mask) in stage3.items():
        for support in STYLE:
            cases.append(run_case(3, STAGE3_PEAK, support, target, seeds, condition, mask))
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
        (
            f"- stage 3 at {STAGE3_PEAK:.0e} peak counts: beamstop = disc of radius {BEAMSTOP} px "
            f"at zero frequency; gaps = a horizontal and a vertical {GAP}-px gap {GAP_OFFSET} px "
            f"off centre plus {DEAD:.1%} dead pixels (seed {MASK_SEED}); phase = the same object "
            f"with phase within ±{STRONG_PHASE:g} rad. Masked pixels read 0 and are left out of "
            "the fit and of the noise floor; the real-space error still covers the whole array."
        ),
        (
            "- reproduce: `uv run --python 3.14+gil python benchmarks/ladder.py`; with a "
            "free-threaded 3.14 also installed, a plain `uv run` on Windows picked it and jaxlib has "
            "no wheel for it"
        ),
        "- twin: starts whose twin matched the truth better than the start itself",
        "- times: one vmapped call for all starts; a first call compiles only when its recipe is new to the run, so later rows show run time in both columns",
        "",
        *table(cases, seeds),
        "",
        "![ladder](ladder.png)",
        "",
        "## Not compared",
        "",
        (
            "PyNX: the `pynx` package on PyPI is an unrelated project, and ESRF PyNX "
            "(gitlab.esrf.fr/favre/PyNX) depends on `pyvkfft`, which needs an OpenCL or CUDA "
            "toolkit to build and has no Windows wheel. A comparison on a Linux machine with a GPU "
            "toolkit would be separate work."
        ),
        "",
        "## Public measured data",
        "",
        (
            "Stage 4, in [cxidb.md](cxidb.md) (`benchmarks/cxidb.py`): CXIDB 3, H. N. Chapman et "
            "al., Nat. Phys. 2, 839 (2006), doi:10.1038/nphys461, data doi:10.11577/1096905; and "
            "CXIDB 28, S. Marchesini et al., Phys. Rev. B 68, 140101(R) (2003), "
            "doi:10.1103/PhysRevB.68.140101, data doi:10.11577/1172615. Neither published "
            "reconstruction is reproduced: see #38. Physics check needed."
        ),
        "",
        "## Lab MATLAB code",
        "",
        (
            "On one measured 3D dataset (`benchmarks/measured.py`, numbers in "
            "[measured.md](measured.md)), against one saved reconstruction from the lab MATLAB code: "
            "Fourier residual 0.315 for that reconstruction cut to its support (it was saved before the "
            "support step), 0.310 for the best of 10 QoraPhase starts; mutual error between the two "
            "0.235 (0.179 on amplitudes alone), against a median of 0.099 (0.079) among QoraPhase's "
            "own starts. The lab code's own spread is not measured yet. Physics check needed: these "
            "are the numbers only, read by the owner."
        ),
        "",
        (
            "On the synthetic stages (`benchmarks/legacy.py`) the comparison has not run: it needs a "
            "MATLAB installation."
        ),
        "",
    ]
    (OUT / "ladder.md").write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {OUT / 'ladder.md'}")


if __name__ == "__main__":
    sys.exit(main())
