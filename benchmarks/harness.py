"""Repeatable timing of QoraPhase iterations on whatever JAX device is present.

    uv run python benchmarks/harness.py                       # default cases, Markdown
    uv run python benchmarks/harness.py --quick               # small sizes
    uv run python benchmarks/harness.py --json run.jsonl      # also keep the raw numbers
    uv run python benchmarks/harness.py --compare a.jsonl b.jsonl   # b relative to a

For every (algorithm, shape) case it reports, separately:

- compile time: ``jax.jit(...).lower(...).compile()``, the one-off XLA cost;
- run time: ``--repeat`` timed calls of the compiled function after ``--warmup`` untimed
  calls, each ending in ``block_until_ready`` because JAX dispatch is asynchronous.
  The spread (min, quartiles, max) is printed with the median, so a noisy machine shows;
- throughput: voxel-iterations per second from the median;
- peak device memory, when the device reports it (``memory_stats()``, GPUs do; the CPU
  backend does not, and the column then says so instead of guessing).

The wall time after ``block_until_ready`` is the only clock: there is no separate
device-side timer, so on a GPU it includes launch and synchronisation overhead.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from perf import final_object, label, make_problem

from qoraphase.algorithms.er import ER
from qoraphase.algorithms.hio import HIO
from qoraphase.algorithms.raar import RAAR

ALGORITHMS = {"RAAR": lambda n: RAAR(n, 0.9), "HIO": lambda n: HIO(n, 0.9), "ER": ER}


@dataclass
class Result:
    algorithm: str
    shape: str
    iterations: int
    device: str
    compile_s: float
    times_s: list[float]
    peak_bytes: int | None

    @property
    def median(self) -> float:
        return statistics.median(self.times_s)

    def quartiles(self) -> tuple[float, float]:
        if len(self.times_s) < 2:
            return self.times_s[0], self.times_s[0]
        q = statistics.quantiles(self.times_s, n=4, method="inclusive")
        return q[0], q[2]


def peak_bytes(device) -> int | None:
    stats = device.memory_stats()
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


def run_case(algorithm, shape, iterations, warmup, repeat) -> Result:
    problem, state = make_problem(shape)
    fn = jax.jit(partial(final_object, ALGORITHMS[algorithm](iterations), problem))
    t = time.perf_counter()
    compiled = fn.lower(state).compile()
    compile_s = time.perf_counter() - t
    for _ in range(warmup):
        jax.block_until_ready(compiled(state))
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        jax.block_until_ready(compiled(state))
        times.append(time.perf_counter() - t)
    device = jax.devices()[0]
    return Result(
        algorithm, label(shape), iterations, str(device), compile_s, times, peak_bytes(device)
    )


def voxels(shape_label: str) -> int:
    return int(np.prod([int(n) for n in shape_label.split("x")]))


def mib(n: int | None) -> str:
    return "not reported by device" if n is None else f"{n / 2**20:.0f}"


def print_results(results: list[Result]):
    print(
        "| algorithm | shape | iterations | compile s | median s | quartiles s | min-max s"
        " | ms/iteration | Mvoxel-it/s | peak device MiB so far |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        q1, q3 = r.quartiles()
        throughput = voxels(r.shape) * r.iterations / r.median / 1e6
        print(
            f"| {r.algorithm} | {r.shape} | {r.iterations} | {r.compile_s:.2f}"
            f" | {r.median:.4f} | {q1:.4f}-{q3:.4f} | {min(r.times_s):.4f}-{max(r.times_s):.4f}"
            f" | {1e3 * r.median / r.iterations:.3f} | {throughput:.1f} | {mib(r.peak_bytes)} |",
            flush=True,
        )
    print()


def load(path: str) -> dict[tuple[str, str], Result]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = Result(**json.loads(line))
            out[(r.algorithm, r.shape)] = r
    return out


def overlap(b: Result, a: Result) -> str:
    """Whether the min-max ranges of two runs overlap: then the difference of the medians
    is not distinguishable from run-to-run noise on that machine."""
    return "yes" if min(a.times_s) <= max(b.times_s) and min(b.times_s) <= max(a.times_s) else "no"


def compare(before_path: str, after_path: str):
    before, after = load(before_path), load(after_path)
    print(f"Median run time of {after_path} relative to {before_path} (below 1 is faster).\n")
    print(
        "| algorithm | shape | before median s | after median s | after / before"
        " | spread before | spread after | ranges overlap |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for key in sorted(before.keys() & after.keys()):
        b, a = before[key], after[key]
        spread = [f"{min(r.times_s):.4f}-{max(r.times_s):.4f}" for r in (b, a)]
        print(
            f"| {key[0]} | {key[1]} | {b.median:.4f} | {a.median:.4f}"
            f" | {a.median / b.median:.3f} | {spread[0]} | {spread[1]} | {overlap(b, a)} |"
        )
    missing = before.keys() ^ after.keys()
    if missing:
        print(f"\nCases in only one file: {sorted(missing)}")


def environment():
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    print(
        f"- commit {commit or 'unknown'}, Python {platform.python_version()}, JAX {jax.__version__}"
    )
    print(f"- devices: {jax.devices()}; x64 {'on' if jax.config.jax_enable_x64 else 'off'}")
    print(f"- OS: {platform.platform()}, processor: {platform.processor() or 'unknown'}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="small sizes")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    parser.add_argument("--iterations", type=int, default=50, help="iterations per timed call")
    parser.add_argument("--warmup", type=int, default=2, help="untimed calls after compiling")
    parser.add_argument("--repeat", type=int, default=10, help="timed calls per case")
    parser.add_argument("--json", help="append one JSON line per case to this file")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.quick:
        shapes = [(64, 64), (128, 128), (32,) * 3]
    else:
        shapes = [(128, 128), (512, 512), (1024, 1024), (64,) * 3, (128,) * 3]

    print("## Environment\n")
    environment()
    results = []
    for shape in shapes:
        for algorithm in args.algorithms:
            r = run_case(algorithm, shape, args.iterations, args.warmup, args.repeat)
            results.append(r)
            if args.json:
                with open(args.json, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(r)) + "\n")
    print(
        f"## Iterations: {args.iterations} per call, {args.warmup} warm-up, {args.repeat} timed\n"
    )
    print_results(results)


if __name__ == "__main__":
    main()
