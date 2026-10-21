"""Time multi-start on CPU and count how many starts converge.

Same synthetic object and noiseless intensity as examples/readme_figure.py, no support
given.  Prints a Markdown table for the README.

python examples/bench_multistart.py
"""

import platform
import time

import jax
import numpy as np

import qoraphase as qp

N, M = 64, 24
STARTS = (1, 8, 32)
REPEATS = 3  # warm timings: best of 3
# Final errors here split cleanly: converged runs end at 1e-7..1e-5, stalled
# ones at 7e-3 and above.  readme_figure.py's best run ends at 1.5e-7.
THRESHOLD = 1e-3

y, x = np.indices((M, M)) - (M - 1) / 2
inside = np.indices((M, M)).sum(0) < M
block = (1 - 0.5 * np.exp(-(x**2 + y**2) / 20)) * np.exp(
    1j * 1.2 * np.exp(-((x + 4) ** 2 + (y + 4) ** 2) / 30)
)
obj = np.zeros((N, N), complex)
obj[N // 2 - M // 2 : N // 2 + M // 2, N // 2 - M // 2 : N // 2 + M // 2] = block * inside
intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj), norm="ortho"))) ** 2


def vmapped(s):
    ens = qp.reconstruct(intensity, starts=s)
    return [ens.error] if s == 1 else [r.error for r in ens]  # float() syncs


def looped(s):
    return [qp.reconstruct(intensity, seed=i).error for i in range(s)]


def timed(fn, s):
    t = time.perf_counter()
    errors = fn(s)
    return time.perf_counter() - t, errors


def best_of(fn, s):
    runs = [timed(fn, s) for _ in range(REPEATS)]
    return min(t for t, _ in runs), runs[0][1]


def converged(errors):
    return f"{sum(e < THRESHOLD for e in errors)}/{len(errors)}"


print(
    f"CPU: {platform.processor() or platform.machine()} | "
    f"JAX {jax.__version__} ({jax.default_backend()}) | "
    f"Python {platform.python_version()}"
)
print(f"{N}x{N} intensity, default recipe, no support; converged = Fourier error < {THRESHOLD:g}")
print()
print(
    "| starts | first call (incl. compile) | vmap, warm | loop of starts=1, warm | converged (vmap) | converged (loop) |"
)
print("|---:|---:|---:|---:|---:|---:|")
for s in STARTS:
    first, _ = timed(vmapped, s)  # each starts value is a new shape: compiles once
    t_vmap, e_vmap = best_of(vmapped, s)
    t_loop, e_loop = best_of(looped, s)
    print(
        f"| {s} | {first:.2f} s | {t_vmap:.2f} s | {t_loop:.2f} s | {converged(e_vmap)} | {converged(e_loop)} |",
        flush=True,
    )
