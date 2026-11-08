"""The lab's MATLAB BCDI engine itself on the cases of bcdi_lab.py, scored the same way.

    ROCKING_XRD_SIM=<checkout> uv run --python 3.14+gil --with scipy \
        python benchmarks/bcdi_lab_matlab.py CASE_DIR [CASE_DIR ...] --out DIR --engine ENGINE_DIR

ENGINE_DIR is the lab's MATLAB folder holding BCDI3d2.m and unring (not part of this repository,
also read from BCDI_LAB_ENGINE); MATLAB is ``matlab`` on PATH or BCDI_LAB_MATLAB. CASE_DIR is a
case folder of ``bench.make_cases``, as for bcdi_lab.py.

Per case, as the lab's driver runs it: pass 1, 20 seeds on the cut amplitude from the engine's
automatic support with shrinkwrap; the 20 final supports shifted onto the first (cross-correlation
peak to 1/10 px, clamped to 5 px; the lab's L1 fit needs a toolbox not installed here), once as
they are and once with a twin check, then de-ringed with the engine's unring, summed with the
first counted twice and cut at 20 % (bcdi_lab_sum.m); pass 2, 3 seeds on the raw amplitude with
each of the two supports fixed. Every seed is its own MATLAB call (bcdi_lab_seed.m).

The run is a list of steps, each with one output file under ``DIR/<case>/``; a step whose file
exists is skipped, so a stopped run goes on where it stopped. A seed is the unit of work: when
the file named by TEST_QUEUE_PREEMPT appears, the running step finishes, and the script exits 75
before the next one (test-queue.ps1 -Preemptible queues it again), so a waiting run waits at most
one seed.

Output: ``DIR/<case>/scores.json`` with every score, the engine's own seed choice beside the one
recomputed from the stored errors, the wall time of every seed, and the environment.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat

from qoraphase.postprocess import _fourier_shift, flip_about_origin, register

sys.path.insert(0, str(Path(__file__).parent))
import bcdi_lab as bl  # noqa: E402

PASS1_SEEDS = 20
PASS2_SEEDS = 3
YIELD = 75


class Preempted(Exception):
    pass


def preempt_requested():
    flag = os.environ.get("TEST_QUEUE_PREEMPT")
    return bool(flag) and Path(flag).exists()


def quoted(path):
    return "'" + str(path).replace("'", "''") + "'"


def matlab(args, call):
    here = Path(__file__).parent
    command = f"addpath({quoted(args.engine)}); addpath({quoted(here)}); {call}"
    code = subprocess.run([args.matlab, "-batch", command]).returncode
    if code:
        raise SystemExit(f"MATLAB exited {code}: {call}")


def produce(path, make):
    """Run *make* into a part file and move it to *path*, so a killed step leaves no output."""
    if path.exists():
        return
    if preempt_requested():
        raise Preempted
    part = path.with_name(path.stem + ".part" + path.suffix)
    part.unlink(missing_ok=True)
    print(time.strftime("%H:%M:%S"), "making", path, flush=True)
    make(part)
    part.replace(path)


def export_input(case, part):
    raw = np.sqrt(np.clip(np.load(case / "frames.npy").astype(np.float32), 0, None))
    raw = raw.astype(np.float64)
    savemat(part, {"raw": raw, "cut": np.where(raw <= 1, 0, raw)}, do_compression=True)


def seed_file(work, stage, seed):
    return work / stage / f"seed-{seed:02d}.mat"


def run_seed(args, work, stage, seed, support):
    (work / stage).mkdir(exist_ok=True)
    pass_name = "pass1" if stage == "pass1" else "pass2"
    produce(
        seed_file(work, stage, seed),
        lambda part: matlab(
            args,
            f"bcdi_lab_seed({quoted(work / 'input.mat')}, {quoted(part)}, '{pass_name}', {seed}, "
            f"{quoted(support or '')}, {args.threads}, {str(args.quick).lower()})",
        ),
    )


def load_stage(work, stage, seeds):
    runs = [loadmat(seed_file(work, stage, s)) for s in range(1, seeds + 1)]
    return {
        "objects": [r["obj"].astype(np.complex64) for r in runs],
        "supports": [r["sup"].astype(np.float32) for r in runs],
        "errors": np.stack([r["err"].ravel() for r in runs]),
        "best_e": [float(r["best_e"].ravel()[0]) for r in runs],
        "wall": [float(r["wall"].ravel()[0]) for r in runs],
    }


def align(supports, twin_check):
    reference = jnp.asarray(supports[0], jnp.float32)
    stack, flips = [], 0
    for s in supports:
        candidates = [jnp.asarray(s, jnp.float32)]
        if twin_check:
            candidates.append(flip_about_origin(candidates[0]))
        best, cost, chosen = None, np.inf, 0
        for i, c in enumerate(candidates):
            shift, _ = register(c, reference, upsample=10)
            shift = np.clip(np.asarray(shift, float), -bl.SHIFT_BOUND, bl.SHIFT_BOUND)
            # not clipped to [0, 1]: the ringing is left for the engine's unring
            moved = np.real(np.asarray(_fourier_shift(c, -shift)))
            distance = np.abs(np.clip(moved, 0, 1) - np.asarray(reference)).sum()
            if distance < cost:
                best, cost, chosen = moved, distance, i
        flips += chosen
        stack.append(best.astype(np.float32))
    return np.stack(stack, axis=-1), flips


def write_aligned(work, seeds, twin_check, part):
    aligned, flips = align(load_stage(work, "pass1", seeds)["supports"], twin_check)
    savemat(part, {"aligned": aligned, "flipped": flips})


def score(case, work, args, part):
    detframe, sim_commit = bl.load_scorer()
    frames = np.load(case / "frames.npy").astype(np.float32)
    truth = np.load(case / "truth_det.npy")
    q_norm = json.loads((case / "meta.json").read_text())["Q_norm_invA"]
    eligible = bl.schedule(True)["eligible"]
    ortho = np.sqrt(frames.size)
    result = {"case": case.name, "shape": list(frames.shape), "quick": args.quick}
    for name, seeds in (
        ("pass1", args.seeds),
        ("pass2_lab", PASS2_SEEDS),
        ("pass2_twin", PASS2_SEEDS),
    ):
        s = load_stage(work, name, seeds)
        rows = bl.scores(detframe, s["objects"], truth, q_norm)
        own = int(np.argmin(s["best_e"]))
        stage = {
            "scores": rows,
            "median": bl.summary_of(rows),
            "engine_choice": own,
            "chosen": rows[own],
            "best_error": s["best_e"],
            "seed_seconds": s["wall"],
            "support_voxels": [int(v.sum()) for v in s["supports"]],
            "prtf": bl.prtf_resolution([o * ortho for o in s["objects"]], frames),
        }
        if not args.quick:
            errors = np.where(eligible, s["errors"], np.inf)
            stage["recomputed_choice"] = int(errors.min(axis=1).argmin())
            stage["best_iteration"] = (errors.argmin(axis=1) + 1).tolist()
        result[name] = stage
        print(
            name,
            "median phase",
            round(stage["median"]["phase_rmse_rad"], 3),
            "chosen seed",
            own + 1,
            "phase",
            round(rows[own]["phase_rmse_rad"], 3),
            flush=True,
        )
    for v in ("lab", "twin"):
        sup = loadmat(work / f"sup_{v}.mat")
        result[f"support_{v}"] = {
            "voxels": int(sup["voxels"].ravel()[0]),
            "max_sum": float(sup["max_sum"].ravel()[0]),
            "flipped": int(loadmat(work / f"aligned_{v}.mat")["flipped"].ravel()[0]),
        }
    version = subprocess.run(
        [args.matlab, "-batch", "disp(version)"], capture_output=True, text=True
    )
    result["environment"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "matlab": version.stdout.strip().splitlines()[-1] if version.stdout.strip() else None,
        "matlab_threads": args.threads,
        "qoraphase": bl.short_commit(Path(__file__).parents[1]),
        "rocking_xrd_sim": sim_commit,
    }
    part.write_text(json.dumps(result, indent=1), encoding="utf-8")


def run_case(case, args):
    work = args.out / case.name
    work.mkdir(parents=True, exist_ok=True)
    produce(work / "input.mat", lambda part: export_input(case, part))
    for seed in range(1, args.seeds + 1):
        run_seed(args, work, "pass1", seed, None)
    for v, twin in (("lab", False), ("twin", True)):
        produce(
            work / f"aligned_{v}.mat",
            lambda part, twin=twin: write_aligned(work, args.seeds, twin, part),
        )
        produce(
            work / f"sup_{v}.mat",
            lambda part, v=v: matlab(
                args,
                f"bcdi_lab_sum({quoted(work / f'aligned_{v}.mat')}, {quoted(part)}, {args.threads})",
            ),
        )
    for v in ("lab", "twin"):
        for seed in range(1, PASS2_SEEDS + 1):
            run_seed(args, work, f"pass2_{v}", seed, work / f"sup_{v}.mat")
    produce(work / "scores.json", lambda part: score(case, work, args, part))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=os.environ.get("BCDI_LAB_ENGINE"))
    parser.add_argument("--matlab", default=os.environ.get("BCDI_LAB_MATLAB", "matlab"))
    parser.add_argument(
        "--threads", type=int, default=1, help="maxNumCompThreads of every MATLAB call"
    )
    parser.add_argument("--seeds", type=int, default=PASS1_SEEDS)
    parser.add_argument(
        "--quick", action="store_true", help="4 iterations a block, to check the chain end to end"
    )
    args = parser.parse_args()
    if not args.engine or not (args.engine / "BCDI3d2.m").exists():
        raise SystemExit("--engine (or BCDI_LAB_ENGINE) must be the folder holding BCDI3d2.m")
    try:
        for case in args.cases:
            run_case(case, args)
    except Preempted:
        print(time.strftime("%H:%M:%S"), "preempted; the next run goes on from here", flush=True)
        sys.exit(YIELD)


if __name__ == "__main__":
    main()
