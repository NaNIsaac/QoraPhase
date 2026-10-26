"""QoraPhase against a lab MATLAB phasing engine on the ladder's synthetic data (stages 1 and 2).

    LEGACY_MATLAB_ENGINE=<engine folder> uv run --python 3.14+gil python benchmarks/legacy.py

The engine is not in this repository and is not named here; ``benchmarks/legacy_driver.m`` runs
it through ``matlab -batch`` (``MATLAB`` names the executable when it is not on PATH), CPU only.
Both sides get the same object, Poisson draws and counts as ``benchmarks/ladder.py``, 20 seeds
each.

QoraPhase runs a recipe built to follow the MATLAB schedule (``matched_recipe``): the same ER and
HIO switch points, beta, quadrant cut at iteration 3, half-size box start support, shrinkwrap
windows, threshold and sigma, and the same widening Gaussian detector mask. What could not be
matched exactly is listed in the report. ``default_recipe`` with an autocorrelation support is a
third column for reference.

The criterion is agreement of the results, not speed: the mutual error between the two engines'
reconstructions (after shift, global phase and twin, as in the ladder), each one's error against
the truth, and the Fourier residual, all computed here the same way for both. Time is a
reference column only.

Output: ``docs/benchmarks/legacy.md``.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import qoraphase as qp
from qoraphase.recipes.recipe import Recipe, Stage

sys.path.insert(0, str(Path(__file__).parent))
from ladder import (
    NOISE_SEED,
    OBJECT_SEED,
    PEAKS,
    SUCCESS,
    N,
    centered_fft,
    environment,
    make_object,
    score,
)

OUT = Path(__file__).parents[1] / "docs" / "benchmarks"
SEEDS = list(range(20))
ITERATIONS = 620
# Iterations where the engine switches between ER (first) and HIO.
SWITCH = (20, 200, 220, 400, 420, 600)
BETA = 0.9
TWIN_AT = 3
THRESHOLD = 0.15
ENGINE_DIRS = ("phasing", "mod_proj", "other_useful_funcs", "GA")


def ramp(n, last, first):
    """``n`` values from ``first`` to ``last``, as the engine's schedules are written."""
    return np.linspace(first, last, n) if n > 1 else np.array([last])


def schedule():
    """Per iteration: (algorithm, shrinkwrap on, shrinkwrap sigma, detector mask width)."""
    sw_switch = set(np.round(ramp(50, ITERATIONS, 1)).astype(int))
    sigma = np.concatenate([ramp(ITERATIONS // 2, 1, 2), np.ones(ITERATIONS - ITERATIONS // 2)])
    width = np.concatenate([ramp(ITERATIONS // 2, 1, 0.7), np.ones(ITERATIONS - ITERATIONS // 2)])
    er, shrink, out = True, False, []
    for it in range(1, ITERATIONS + 1):
        er ^= it in SWITCH
        shrink ^= it in sw_switch
        out.append(("ER" if er else "HIO", shrink, sigma[it - 1], width[it - 1]))
    return out


def detector_mask(width, shape):
    """The engine's low-to-high resolution mask: a Gaussian of std ``width`` times the array."""
    if width >= 1:
        return None
    axes = np.indices(shape) - np.array(shape).reshape(-1, *[1] * len(shape)) // 2
    scale = (np.array(shape) * width).reshape(-1, *[1] * len(shape))
    return np.exp(-0.5 * ((axes / scale) ** 2).sum(0))


@dataclass(frozen=True)
class Masked(Stage):
    """Run ``body`` against the measured amplitude times ``mask``."""

    body: Recipe
    mask: jnp.ndarray

    def run(self, state, problem):
        dimmed = qp.CDI(
            amplitude=problem.amplitude * self.mask,
            mask=problem.mask,
            support=problem.support,
            model=problem.model,
        )
        return self.body.run(state, dimmed)


@dataclass(frozen=True)
class Quadrant(Stage):
    """Run ``body`` with the support cut to the first half of the first two axes (a quadrant in
    2D, a quarter column in 3D), then restore it."""

    body: Recipe

    def run(self, state, problem):
        n0, n1 = state.support.shape[:2]
        keep = jnp.zeros(state.support.shape, bool).at[: n0 // 2, : n1 // 2].set(True)
        after, metrics = self.body.run(state._replace(support=state.support & keep), problem)
        return after._replace(support=state.support), metrics


jax.tree_util.register_dataclass(Masked, data_fields=["body", "mask"], meta_fields=[])
jax.tree_util.register_dataclass(Quadrant, data_fields=["body"], meta_fields=[])


def step(alg, n):
    return qp.ER(n) if alg == "ER" else qp.HIO(n, BETA)


def matched_recipe(shape=(N, N)):
    """The MATLAB schedule as QoraPhase stages. Runs of iterations with the same algorithm and
    shrinkwrap state become one stage; sigma and mask width take their mean over the run."""
    plan = schedule()

    def key(i):
        # The cut iteration stands alone; the mask schedule ends halfway.
        return (*plan[i][:2], i == TWIN_AT - 1, i < ITERATIONS // 2)

    stages, it = [], 0
    while it < ITERATIONS:
        end = it + 1
        while end < ITERATIONS and key(end) == key(it):
            end += 1
        alg, shrink, cut, _ = key(it)
        sigma = float(np.mean([p[2] for p in plan[it:end]]))
        width = float(np.mean([p[3] for p in plan[it:end]]))
        if cut:
            # The engine shrinkwraps first, then updates against that support cut to a quadrant.
            parts = [qp.Shrinkwrap(THRESHOLD, sigma)] if shrink else []
            body = Recipe(*parts, Quadrant(Recipe(step(alg, 1))))
        elif shrink:
            body = Recipe(qp.repeat(end - it, qp.Shrinkwrap(THRESHOLD, sigma), step(alg, 1)))
        else:
            body = Recipe(step(alg, end - it))
        mask = detector_mask(width, shape)
        stages.append(body if mask is None else Masked(body, jnp.asarray(mask, jnp.float32)))
        it = end
    return Recipe(*stages)


def box_support(shape=(N, N)):
    """The engine's start support: a centred box half the array along every axis."""
    support = np.zeros(shape, bool)
    support[tuple(slice(n // 2 - n // 4, n // 2 - n // 4 + n // 2) for n in shape)] = True
    return support


def residual(obj, amplitude):
    """Normalised Fourier modulus error, as the library reports it."""
    computed = np.abs(centered_fft(obj))
    return float(np.sqrt(((computed - amplitude) ** 2).sum() / (amplitude**2).sum()))


def cases(obj):
    clean = np.abs(centered_fft(obj)) ** 2
    yield "noiseless", clean, 1.0
    for peak in PEAKS:
        scale = peak / clean.max()
        counts = np.random.default_rng(NOISE_SEED).poisson(clean * scale).astype(float)
        yield f"{peak:.0e}", counts, scale


def run_matlab(work, engine):
    exe = os.environ.get("MATLAB") or shutil.which("matlab")
    if not exe:
        raise SystemExit("matlab not found: set MATLAB to the executable")
    dirs = ";".join(str(Path(engine) / d) for d in ENGINE_DIRS)
    here = Path(__file__).parent
    command = (
        f"addpath('{here}'); addpath('{dirs}'); "
        f"legacy_driver('{work}', {N}, [{' '.join(map(str, SEEDS))}])"
    )
    subprocess.run([exe, "-batch", command], check=True)
    version = subprocess.run(
        [exe, "-batch", "disp(version)"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return version


def read_rec(path):
    raw = np.fromfile(path, dtype="<f8")
    return (raw[: N * N] + 1j * raw[N * N :]).reshape(N, N)


@dataclass
class Engine:
    objects: list
    seconds: list

    def scored(self, truth, amplitude):
        errors = [score(o, truth) for o in self.objects]
        fourier = [residual(o, amplitude) for o in self.objects]
        return errors, fourier


def run_qoraphase(data, support, recipe):
    problem = qp.CDI(data, support=support)
    problem.reconstruct(recipe, starts=len(SEEDS), seed=0)  # compiles
    t = time.perf_counter()
    ens = problem.reconstruct(recipe, starts=len(SEEDS), seed=0)
    objects = [np.asarray(r.object) for r in ens]
    per_start = (time.perf_counter() - t) / len(SEEDS)
    return Engine(objects, [per_start] * len(SEEDS))


def best(engine, amplitude):
    return min(engine.objects, key=lambda o: residual(o, amplitude))


def summary(errors, fourier, twins, seconds):
    med = statistics.median(errors)
    return (
        f"{sum(e < SUCCESS for e in errors)}/{len(errors)} | {twins} | {med:.2e} "
        f"| {statistics.median(fourier):.2e} | {statistics.median(seconds):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--work", help="folder for the exchange files (default: a temp folder)")
    parser.add_argument("--starts", type=int, default=len(SEEDS), help="seeds per engine")
    args = parser.parse_args()
    SEEDS[:] = range(args.starts)
    engine = os.environ.get("LEGACY_MATLAB_ENGINE")
    if not engine:
        raise SystemExit("set LEGACY_MATLAB_ENGINE to the engine folder")
    work = Path(args.work or tempfile.mkdtemp(prefix="qoraphase-legacy-"))
    work.mkdir(parents=True, exist_ok=True)

    obj = make_object()
    data = {name: (counts, scale) for name, counts, scale in cases(obj)}
    for name, (counts, _) in data.items():
        np.sqrt(counts).astype("<f8").tofile(work / f"{name}.amp")
    matlab_version = run_matlab(work, engine)

    recipe = matched_recipe()
    rows, mutual = [], []
    for name, (counts, scale) in data.items():
        amplitude = np.sqrt(counts)
        truth = obj * np.sqrt(scale)
        times = [float(x) for x in (work / f"{name}.times").read_text().split()]
        legacy = Engine([read_rec(work / f"{name}-{s}.rec") for s in SEEDS], times)
        matched = run_qoraphase(counts, box_support(), recipe)
        default = run_qoraphase(counts, None, qp.default_recipe())
        for label, eng in (
            ("lab MATLAB", legacy),
            ("QoraPhase, matched", matched),
            ("QoraPhase, default", default),
        ):
            scored, fourier = eng.scored(truth, amplitude)
            errors = [e for e, _ in scored]
            twins = sum(t for _, t in scored)
            rows.append(f"| {name} | {label} | {summary(errors, fourier, twins, eng.seconds)} |")
            print(rows[-1], flush=True)
        a, b = best(legacy, amplitude), best(matched, amplitude)
        pairs = [score(x, y)[0] for x in legacy.objects for y in matched.objects]
        mutual.append(
            f"| {name} | {score(a, b)[0]:.2e} | {statistics.median(pairs):.2e} "
            f"| {sum(p < SUCCESS for p in pairs)}/{len(pairs)} |"
        )
        print(mutual[-1], flush=True)

    text = [
        "# QoraPhase and a lab MATLAB engine",
        "",
        (
            "Generated by `benchmarks/legacy.py` (method in its docstring). The MATLAB engine is the "
            "lab's ER/HIO phasing code; it is not in this repository, so this report is reproducible "
            "only where that code is available. Synthetic data only."
        ),
        "",
        *environment(),
        f"- MATLAB {matlab_version}, CPU (the engine's GPU path off)",
        (
            f"- object and counts as in ladder.md: object seed {OBJECT_SEED}, Poisson seed "
            f"{NOISE_SEED}; MATLAB `rng(seed)` and QoraPhase keys split from seed 0, seeds "
            f"{SEEDS[0]} to {SEEDS[-1]}; {ITERATIONS} iterations each"
        ),
        f"- success: error against the truth < {SUCCESS:g} (arbitrary threshold, as in ladder.md)",
        "- the MATLAB result is its last iterate cut to its final support",
        "",
        "## Agreement between the two engines",
        "",
        (
            "Mutual error: one reconstruction registered onto the other (shift, global phase, twin, "
            "complex scale), as against the truth. Best = the run with the lowest Fourier residual "
            "on each side."
        ),
        "",
        "| data | best vs best | median over all pairs | pairs < threshold |",
        "|---|---:|---:|---:|",
        *mutual,
        "",
        "## Each engine against the truth",
        "",
        (
            "| data | engine | success | twin | error, median | Fourier residual, median "
            "| s per run (reference) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## What the matched recipe does not reproduce exactly",
        "",
        "| item | lab MATLAB | QoraPhase matched |",
        "|---|---|---|",
        (
            "| shrinkwrap sigma and detector mask width | change every iteration | mean over each "
            "run of iterations (at most 13; sigma within 0.02 px, width within 0.006) |"
        ),
        "| detector mask centre | (n-1)/2, between pixels | n/2, the zero frequency here |",
        "| shrinkwrap threshold test | smoothed >= 0.15 max | blurred > 0.15 max |",
        (
            "| random start | MATLAB `rand` phases | JAX keys; same construction (data with random "
            "phase, back-transformed, cut to the support) |"
        ),
        "| precision | float64 | float32 (x64 off) |",
        "| time | one run at a time, MATLAB FFT | 20 starts in one vmapped call, divided by 20 |",
        "",
    ]
    (OUT / "legacy.md").write_text("\n".join(text), encoding="utf-8")
    print(f"wrote {OUT / 'legacy.md'}; exchange files in {work}")


if __name__ == "__main__":
    main()
