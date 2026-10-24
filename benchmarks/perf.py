"""Where 2D and 3D reconstructions spend their time, on synthetic data, on CPU.

    uv run python benchmarks/perf.py            # full run, Markdown on stdout
    uv run python benchmarks/perf.py --quick    # small sizes, under a minute

Five sections: end-to-end ``qp.reconstruct`` with the default recipe (cold vs
warm), the cost of each primitive inside one iteration, a NumPy reference for
RAAR, the cost of each stage of the default recipe, and multi-start scaling.
Every timed call ends in ``block_until_ready`` or a host read, because JAX
dispatch is asynchronous.
"""

import argparse
import os
import platform
import statistics
import subprocess
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import qoraphase as qp
from qoraphase.algorithms.er import ER
from qoraphase.algorithms.hio import HIO
from qoraphase.algorithms.raar import RAAR
from qoraphase.core.problem import CDI
from qoraphase.core.state import initial_state
from qoraphase.operators.modulus import modulus_project
from qoraphase.operators.shrinkwrap import Shrinkwrap, gaussian_blur
from qoraphase.operators.support import support_project

RECIPE_ITERATIONS = 50 + 20 + 50 + 8 * 50 + 200  # qp.default_recipe()


def label(shape):
    return "x".join(map(str, shape))


def synthetic(shape, fill=0.5):
    """Noiseless intensity of a smooth complex object inside a centred ellipsoid.

    Same family as examples/bench_multistart.py: a Gaussian-dimpled amplitude
    with a smooth phase bump, zero outside an ellipsoid spanning ``fill`` of
    each axis.  Deterministic.
    """
    grids = np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij")
    pairs = list(zip(grids, shape, strict=True))
    radius2 = sum((g / (fill * n / 2)) ** 2 for g, n in pairs)
    width2 = sum((g / (0.25 * n)) ** 2 for g, n in pairs)
    shifted2 = sum(((g + 0.1 * n) / (0.3 * n)) ** 2 for g, n in pairs)
    obj = (1 - 0.5 * np.exp(-width2)) * np.exp(1j * 1.2 * np.exp(-shifted2)) * (radius2 < 1)
    wave = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))
    return np.abs(wave) ** 2


def clock(fn, repeats):
    """Seconds per call after one untimed warm-up call."""
    jax.block_until_ready(fn())
    out = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        out.append(time.perf_counter() - t)
    return out


def med(xs):
    return statistics.median(xs)


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |", flush=True)
    print()


def end_to_end(shapes, big):
    rows = []
    for shape in shapes:
        intensity = synthetic(shape)
        n = int(np.prod(shape))
        t = time.perf_counter()
        float(qp.reconstruct(intensity, seed=0).error)
        cold = time.perf_counter() - t
        warm = []
        for _ in range(2 if n >= big else 4):
            t = time.perf_counter()
            r = qp.reconstruct(intensity, seed=0)
            float(r.error)
            warm.append(time.perf_counter() - t)
        w = med(warm)
        rows.append(
            (
                label(shape),
                f"{n:,}",
                f"{cold:.2f}",
                f"{w:.2f}",
                f"{min(warm):.2f}-{max(warm):.2f}",
                f"{cold - w:.2f}",
                f"{RECIPE_ITERATIONS / w:.1f}",
                f"{float(r.error):.1e}",
            )
        )
    table(
        [
            "shape",
            "voxels",
            "cold s",
            "warm median s",
            "warm min-max s",
            "cold - warm s",
            "iterations/s",
            "final error",
        ],
        rows,
    )


def make_problem(shape):
    problem = CDI(synthetic(shape))
    return problem, initial_state(jax.random.key(0), problem)


def primitive_times(shape, repeats):
    problem, state = make_problem(shape)
    model = problem.model
    x = state.obj
    cases = {
        "fftn + ifftn (raw, no shifts)": (lambda a: jnp.fft.ifftn(jnp.fft.fftn(a)), x),
        "forward + adjoint (with shifts)": (lambda a: model.adjoint(model.forward(a)), x),
        "modulus_project": (lambda a: modulus_project(a, problem), x),
        "support_project": (lambda a: support_project(a, problem.support), x),
        "gaussian_blur (2 FFTs)": (lambda a: gaussian_blur(jnp.abs(a), 1.0), x),
        "RAAR step": (lambda s: RAAR(1, 0.9).step(s, problem), state),
        "HIO step": (lambda s: HIO(1, 0.9).step(s, problem), state),
        "ER step": (lambda s: ER(1).step(s, problem), state),
        "Shrinkwrap": (lambda s: Shrinkwrap().run(s, problem), state),
    }
    times = {}
    for name, (fn, arg) in cases.items():
        compiled = jax.jit(fn)
        times[name] = med(clock(partial(compiled, arg), repeats))
    return times


def primitives(shapes, repeats):
    rows = []
    for shape in shapes:
        times = primitive_times(shape, repeats)
        for name, t in times.items():
            share = 100 * t / times["RAAR step"]
            rows.append((label(shape), name, f"{t * 1e3:.3f}", f"{share:.0f}%"))
    table(["shape", "operation", "ms per call", "% of RAAR step"], rows)


def numpy_raar(obj, amplitude, support, n, beta=0.9):
    """RAAR in NumPy with the same maths as qoraphase.RAAR (no mask, coherent)."""

    def fwd(a):
        return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(a), norm="ortho"))

    def adj(a):
        return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(a), norm="ortho"))

    x = obj
    for _ in range(n):
        wave = fwd(x)
        pm = adj(amplitude * wave / (np.abs(wave) + 1e-12))
        rm = 2.0 * pm - x
        rs_rm = 2.0 * np.where(support, rm, 0) - rm
        x = 0.5 * beta * (rs_rm + x) + (1.0 - beta) * pm
    return x


def final_object(stage, problem, state):
    return stage.run(state, problem)[0].obj


def numpy_row(shape, iters):
    intensity = synthetic(shape)
    problem, state = make_problem(shape)
    x0 = np.asarray(state.obj).astype(np.complex128)
    support = np.asarray(problem.support)
    run_jax = jax.jit(partial(final_object, RAAR(iters, 0.9), problem))
    tj = med(clock(partial(run_jax, state), 3))
    tn = []
    for _ in range(3):
        t = time.perf_counter()
        ref = numpy_raar(x0, np.sqrt(intensity), support, iters)
        tn.append(time.perf_counter() - t)
    diff = np.linalg.norm(np.asarray(run_jax(state)) - ref) / np.linalg.norm(ref)
    return (
        label(shape),
        f"{iters / med(tn):.1f}",
        f"{iters / tj:.1f}",
        f"{med(tn) / tj:.2f}",
        f"{diff:.1e}",
    )


def numpy_comparison(shapes, iters):
    rows = [numpy_row(shape, iters) for shape in shapes]
    table(
        ["shape", "NumPy complex128 it/s", "JAX complex64 it/s", "JAX / NumPy", "relative diff"],
        rows,
    )


def stage_rows(shape, repeats):
    problem, state = make_problem(shape)
    parts = {
        "RAAR(50)": (RAAR(50), 50),
        "detwin(HIO(20))": (qp.detwin(HIO(20)), 20),
        "RAAR(50), second": (RAAR(50), 50),
        "8 x [RAAR(50) + Shrinkwrap]": (qp.repeat(8, RAAR(50), Shrinkwrap()), 400),
        "ER(200)": (ER(200), 200),
    }
    timed = []
    for name, (stage, its) in parts.items():
        fn = jax.jit(partial(final_object, stage, problem))
        timed.append((name, its, med(clock(partial(fn, state), repeats))))
    total = sum(t for _, _, t in timed)
    whole = jax.jit(partial(final_object, qp.default_recipe(), problem))
    tw = med(clock(partial(whole, state), repeats))
    rows = [
        (label(shape), name, its, f"{t:.3f}", f"{1e3 * t / its:.2f}", f"{100 * t / total:.0f}%")
        for name, its, t in timed
    ]
    rows.append((label(shape), "sum of stages", RECIPE_ITERATIONS, f"{total:.3f}", "", "100%"))
    rows.append((label(shape), "whole recipe, one jit", RECIPE_ITERATIONS, f"{tw:.3f}", "", ""))
    return rows


def stages(shapes, repeats):
    rows = [row for shape in shapes for row in stage_rows(shape, repeats)]
    table(["shape", "stage", "iterations", "s", "ms/iteration", "share"], rows)


def start_many(intensity, starts):
    ens = qp.reconstruct(intensity, starts=starts, seed=0)
    return ens.error if starts == 1 else jnp.asarray([r.error for r in ens])


def multistart(shapes, counts, repeats):
    rows = []
    for shape in shapes:
        intensity = synthetic(shape)
        for s in counts:
            ts = clock(partial(start_many, intensity, s), repeats)
            rows.append((label(shape), s, f"{med(ts):.2f}", f"{med(ts) / s:.2f}"))
    table(["shape", "starts", "warm median s", "s per start"], rows)


def environment():
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            cpu = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    print(f"- CPU: {cpu}, logical cores: {os.cpu_count()}")
    print(f"- OS: {platform.platform()}")
    print(f"- Python {platform.python_version()}, JAX {jax.__version__}, NumPy {np.__version__}")
    print(f"- JAX devices: {jax.devices()}; float32 / complex64 (x64 off)")
    print(f"- default recipe: {RECIPE_ITERATIONS} iterations")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    quick = parser.parse_args().quick

    if quick:
        s2, s3, micro3 = [(64, 64), (128, 128)], [(24,) * 3, (32,) * 3], [(32,) * 3]
        mid2, mid3 = s2, s3
    else:
        s2 = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
        s3 = [(32,) * 3, (64,) * 3, (128,) * 3]
        micro3 = s3 + [(256,) * 3]
        mid2, mid3 = s2[:4], s3[:2]

    print("## Environment\n")
    environment()
    print("## End to end: qp.reconstruct, default recipe, one start\n")
    end_to_end(s2 + s3, 1 << 20)
    print("## Cost of the primitives inside one iteration\n")
    primitives(s2 + micro3, 5 if quick else 10)
    print("## NumPy reference vs JAX, RAAR only, 20 iterations\n")
    numpy_comparison(mid2 + mid3, 20)
    print("## Stages of the default recipe\n")
    stages(mid2 + mid3, 3)
    print("## Multi-start under vmap\n")
    multistart([(64, 64)] if quick else [(128, 128), (32,) * 3], (1, 4, 8), 2)


if __name__ == "__main__":
    main()
