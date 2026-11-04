"""QoraPhase's default recipe against a lab MATLAB BCDI recipe, on simulated BCDI data.

    ROCKING_XRD_SIM=<checkout> uv run --python 3.14+gil --with scipy \
        python benchmarks/bcdi_lab.py CASE_DIR [CASE_DIR ...] --out DIR

CASE_DIR is a case folder written by ``bench.make_cases`` of rocking_xrd_sim (private; it also
scores the results, so ROCKING_XRD_SIM points at its checkout): ``frames.npy`` (intensity),
``truth_det.npy`` and ``meta.json``.

The lab recipe is not run in MATLAB here. It is re-implemented in JAX from the lab's 3D BCDI code,
two passes as its driver runs them:

1. support estimate: amplitude with values <= 1 cut to 0, autocorrelation start support (blur
   sigma 5, 5 % of the range), 20 seeds of ER 20 / HIO 180 alternating (1000 iterations,
   beta 0.9), shrinkwrap every 20 iterations of a block in the first six blocks (blur sigma 2 of
   the amplitude after the Fourier step, threshold min + t (max - min), t from 0.10 to 0.15, AND
   the start support), and a Gaussian Q window on the data over the same 600 iterations (std
   0.8 to 2 times the half array); the 20 final supports are shifted onto the first, summed and
   cut at 20 % of the maximum;
2. phasing: the raw amplitude, that support fixed, the same schedule without shrinkwrap, 3 seeds,
   the one with the lowest best-so-far error kept; the result is its last iterate.

The lab's start phase is constant along the third axis; here it is random in every voxel. The
support shift is the cross-correlation peak to 1/10 px instead of the lab's L1 fit, the shifted
support is clipped to [0, 1] instead of de-ringed, and the first support is counted once. The
support sum is also run with a twin check, which the lab's lacks. The best-so-far error is kept
two ways: as the lab computes it, against the windowed data while the Q window is on, and
against the unwindowed data for every iterate; a seed is chosen by each.

QoraPhase runs ``default_recipe()`` from its autocorrelation support, 20 starts, and reports
``Ensemble.best`` and ``Ensemble.consensus``.

Output: ``DIR/<case>.json`` with every score, and ``DIR/summary.md``.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

import qoraphase as qp
from qoraphase.postprocess import _fourier_shift, flip_about_origin, register

BETA = 0.9
BLOCKS = (20, 180, 20, 180, 20, 180, 20, 180, 20, 180, 20)
SHRINK_FACTOR = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0, 0, 0, 0, 0)
LOWRES = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
SHRINK_SIGMA, SHRINK_RATE, SHRINK_EVERY = 2.0, 0.1, 20
WINDOW = (0.8, 2.0)
MIN_ITERATION = 5
PASS1_SEEDS, PASS2_SEEDS = 20, 3
CONSENSUS_CUT = 0.2
SHIFT_BOUND = 5


def fftc(x):
    return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(x)))


def ifftc(x):
    return jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(x)))


def schedule(shrink):
    """Per iteration: HIO or not, shrinkwrap now, its threshold, Q window on, window width, and
    whether the iteration may set the best-so-far error."""
    hio, now, level, lowres, width, eligible = [], [], [], [], [], []
    total = sum(n * f for n, f in zip(BLOCKS, LOWRES, strict=True))
    done, windowed = 0, 0
    for block, n in enumerate(BLOCKS):
        is_hio = block % 2 == 1
        for k in range(1, n + 1):
            done += 1
            hio.append(is_hio)
            now.append(bool(shrink and SHRINK_FACTOR[block]) and k % SHRINK_EVERY == 0)
            level.append(SHRINK_FACTOR[block] * SHRINK_RATE)
            lowres.append(bool(LOWRES[block]))
            width.append(WINDOW[0] + windowed / total * (WINDOW[1] - WINDOW[0]))
            windowed += LOWRES[block]
            # The lab code tests the block-local count in HIO blocks and the global one elsewhere.
            eligible.append((k if is_hio else done) > MIN_ITERATION)
    return {
        "hio": jnp.array(hio),
        "now": jnp.array(now),
        "level": jnp.array(level, jnp.float32),
        "lowres": jnp.array(lowres),
        "width": jnp.array(width, jnp.float32),
        "eligible": np.array(eligible),
    }


def q_window(shape, width):
    """MATLAB ``fspecial3('gaussian', shape, ceil((shape+1)/2) * width)`` over its maximum: centred at
    (n-1)/2, between pixels for even n."""
    w = jnp.ones(shape, jnp.float32)
    for axis, n in enumerate(shape):
        x = jnp.arange(n, dtype=jnp.float32) - (n - 1) / 2
        sigma = np.ceil((n + 1) / 2) * width
        profile = jnp.exp(-0.5 * (x / sigma) ** 2)
        w = w * profile.reshape([-1 if a == axis else 1 for a in range(len(shape))])
    return w / w.max()


def replicate_blur(a, sigma):
    """``imgaussfilt3(a, sigma)``: kernel 2 ceil(2 sigma) + 1, replicate edges."""
    radius = int(np.ceil(2 * sigma))
    x = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (x / sigma) ** 2)
    weights = weights / weights.sum()
    for axis, n in enumerate(a.shape):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (radius, radius)
        padded = jnp.pad(a, pad, mode="edge")
        a = sum(
            float(w) * jax.lax.slice_in_dim(padded, j, j + n, axis=axis)
            for j, w in enumerate(weights)
        )
    return a


def auto_support(amplitude):
    """The lab's start support: autocorrelation blurred by 5 px, cut at 5 % of its range."""
    auto = np.abs(np.asarray(fftc(jnp.asarray(amplitude**2, jnp.complex64))))
    cut = gaussian_filter(auto, 5, mode="nearest", truncate=2.0)
    return cut >= cut.min() + 0.05 * (cut.max() - cut.min())


def lab_run(key, amplitude, start_support, plan):
    """One seed of the lab engine (ER and HIO blocks only). Returns the last iterate, the last
    support and the error of every iteration."""
    shape = amplitude.shape
    phase = jax.random.uniform(key, shape, minval=-jnp.pi, maxval=jnp.pi)
    r = ifftc(amplitude * jnp.exp(1j * phase)) * start_support
    k = fftc(r)
    first = jnp.where(plan["lowres"][0], q_window(shape, WINDOW[0]), 1.0)
    k = amplitude * first * jnp.exp(1j * jnp.angle(k))

    def step(carry, it):
        k, buffer, support = carry
        r = ifftc(k)

        def wrap(support):
            blurred = replicate_blur(jnp.abs(r), SHRINK_SIGMA)
            low, high = blurred.min(), blurred.max()
            return (blurred >= low + it["level"] * (high - low)) & start_support

        # The schedule is shared by all seeds, so under vmap these predicates stay unbatched
        # and lax.cond skips the work instead of computing both branches.
        support = jax.lax.cond(it["now"], wrap, lambda s: s, support)
        window = jnp.where(it["lowres"], q_window(shape, it["width"]), 1.0)
        target = amplitude * window
        r = jnp.where(it["hio"], jnp.where(support, r, buffer - BETA * r), r * support)
        k = fftc(r)
        inside = jnp.abs(jax.lax.cond(it["hio"], lambda: fftc(r * support), lambda: k))
        windowed = jnp.sum(jnp.abs(target - inside)) / jnp.sum(target)
        # The lab's error is against the windowed data, so its values from the window phase are
        # on another scale; the same L1 error against the unwindowed data is kept beside it.
        full = jnp.sum(jnp.abs(amplitude - inside)) / jnp.sum(amplitude)
        return (target * jnp.exp(1j * jnp.angle(k)), r, support), (windowed, full)

    xs = {name: plan[name] for name in ("hio", "now", "level", "lowres", "width")}
    (_, last, support), errors = jax.lax.scan(step, (k, r, start_support), xs)
    return last, support, errors


lab_batch = jax.jit(jax.vmap(lab_run, in_axes=(0, None, None, None)))


def run_lab(amplitude, start_support, shrink, seeds, seed, batch):
    plan = schedule(shrink)
    keys = jax.random.split(jax.random.key(seed), seeds)
    amplitude = jnp.asarray(amplitude, jnp.float32)
    start_support = jnp.asarray(start_support)
    objects, supports, windowed, full = [], [], [], []
    # The lab transforms are unnormalised and QoraPhase's are orthonormal; rescale so that
    # qp.prtf compares the lab objects with the data on the same footing.
    ortho = np.sqrt(amplitude.size)
    for i in range(0, seeds, batch):
        o, s, (w, f) = lab_batch(keys[i : i + batch], amplitude, start_support, plan)
        objects += list(np.asarray(o) * ortho)
        supports += list(np.asarray(s))
        windowed += list(np.asarray(w))
        full += list(np.asarray(f))
    in_window = np.asarray(plan["lowres"])
    best = {}
    for rule, errors in (("lab", np.array(windowed)), ("unwindowed", np.array(full))):
        masked = np.where(plan["eligible"], errors, np.inf)
        where = masked.argmin(axis=1)
        best[rule] = {
            "error": masked.min(axis=1).tolist(),
            "iteration": (where + 1).tolist(),
            "in_window": in_window[where].tolist(),
            "final_error": errors[:, -1].tolist(),
        }
    return objects, supports, best


def consensus_support(supports, twin_check):
    """Shift every support onto the first, sum, and cut at 20 % of the maximum.

    The lab fits a sub-pixel shift within 5 px by an L1 cost and removes the ringing of the
    shifted binary (``unring``); here the shift is the cross-correlation peak to 1/10 px, clamped
    to 5 px, and the real part is clipped to [0, 1]. With ``twin_check`` the point-reflected
    support is also tried and the one with the smaller L1 distance kept.
    """
    reference = jnp.asarray(supports[0], jnp.float32)
    total = np.zeros(reference.shape)
    flips = 0
    for s in supports:
        candidates = [jnp.asarray(s, jnp.float32)]
        if twin_check:
            candidates.append(flip_about_origin(candidates[0]))
        best, cost = None, np.inf
        for i, c in enumerate(candidates):
            shift, _ = register(c, reference, upsample=10)
            shift = np.clip(np.asarray(shift, float), -SHIFT_BOUND, SHIFT_BOUND)
            moved = np.clip(np.real(np.asarray(_fourier_shift(c, -shift))), 0, 1)
            distance = np.abs(moved - np.asarray(reference)).sum()
            if distance < cost:
                best, cost, chosen = moved, distance, i
        flips += chosen
        total += best
    return total >= CONSENSUS_CUT * total.max(), flips


def load_scorer():
    root = os.environ.get("ROCKING_XRD_SIM")
    if not root:
        raise SystemExit("set ROCKING_XRD_SIM to a rocking_xrd_sim checkout")
    sys.path.insert(0, root)
    from bench import detframe

    return detframe, short_commit(root)


def short_commit(checkout):
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def locked(name, lock):
    """Version of *name* pinned in uv.lock, recorded beside the one that ran: a run outside the
    lock, or on a free-threaded Python for which the pinned jaxlib has no Windows wheel, gets
    other versions."""
    match = re.search(rf'name = "{name}"\r?\nversion = "([^"]+)"', lock.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def environment(sim_commit, seed):
    here = Path(__file__).parents[1]
    commit = short_commit(here)
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "jax_in_uv_lock": locked("jax", here / "uv.lock"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "qoraphase": f"{version('qoraphase')} at {commit}",
        "rocking_xrd_sim": sim_commit,
        "device": str(jax.devices()[0]),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "seed": seed,
    }


def scores(detframe, objects, truth, q_norm):
    return [detframe.score(np.asarray(o), truth, q_norm) for o in objects]


def summary_of(rows):
    keys = ("phase_rmse_rad", "amplitude_nrmse", "support_iou", "complex_nrmse")
    return {k: float(np.median([r[k] for r in rows])) for k in keys}


def prtf_resolution(objects, intensity):
    """Crossing of 1/e in cycles per pixel (0 when the first shell is already below it) and the
    curve, rounded."""
    curve = qp.prtf(list(objects), intensity, upsample=10)
    return {
        "crossing": float(curve.crossing),
        "value": [round(float(v), 3) for v in np.asarray(curve.value)],
    }


def run_case(case, detframe, args):
    frames = np.load(case / "frames.npy").astype(np.float32)
    truth = np.load(case / "truth_det.npy")
    meta = json.loads((case / "meta.json").read_text())
    q_norm = meta["Q_norm_invA"]
    raw = np.sqrt(np.clip(frames, 0, None))
    cut = np.where(raw <= 1, 0, raw)
    out = {"case": case.name, "shape": list(frames.shape), "seeds": args.seeds}

    t = time.perf_counter()
    start = auto_support(cut)
    p1 = run_lab(cut, start, True, args.seeds, args.seed, args.batch)
    jax.block_until_ready(p1[0][-1])
    out["pass1_seconds"] = time.perf_counter() - t
    objects, supports, best = p1
    out["start_support_voxels"] = int(start.sum())
    out["pass1_support_voxels"] = [int(s.sum()) for s in supports]
    out["pass1_best"] = best
    out["pass1_scores"] = scores(detframe, objects, truth, q_norm)
    out["pass1_prtf_px"] = prtf_resolution(objects, frames)

    for twin_check in (False, True):
        label = "twin_checked" if twin_check else "lab"
        fixed, flips = consensus_support(supports, twin_check)
        t = time.perf_counter()
        p2 = run_lab(raw, fixed, False, args.pass2, args.seed + 1, args.batch)
        seconds = time.perf_counter() - t
        objects2, _, best2 = p2
        rows = scores(detframe, objects2, truth, q_norm)
        chosen = {rule: int(np.argmin(b["error"])) for rule, b in best2.items()}
        out[f"pass2_{label}"] = {
            "support_voxels": int(fixed.sum()),
            "flipped_supports": flips,
            "seconds": seconds,
            "best": best2,
            "chosen": chosen,
            "chosen_score": {rule: rows[i] for rule, i in chosen.items()},
            "scores": rows,
            "prtf": prtf_resolution(objects2, frames),
        }

    problem = qp.CDI(frames)
    t = time.perf_counter()
    ens = problem.reconstruct(None, starts=args.seeds, seed=args.seed, batch=args.batch)
    jax.block_until_ready(ens[0].object)
    out["default_seconds"] = time.perf_counter() - t
    rows = scores(detframe, [r.object for r in ens], truth, q_norm)
    out["default_scores"] = rows
    out["default_error"] = [r.error for r in ens]
    out["default_best"] = rows[ens.ranking[0]]
    out["default_consensus"] = rows[
        min(range(len(ens)), key=ens.spread.__getitem__) if len(ens) >= 3 else ens.ranking[0]
    ]
    out["default_support_voxels"] = [int(np.asarray(r.support).sum()) for r in ens]
    out["default_prtf_px"] = prtf_resolution([r.object for r in ens], frames)
    out["pass1_median"] = summary_of(out["pass1_scores"])
    out["default_median"] = summary_of(rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=PASS1_SEEDS)
    parser.add_argument("--pass2", type=int, default=PASS2_SEEDS)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seeds % args.batch or args.pass2 > args.batch and args.pass2 % args.batch:
        raise SystemExit("--seeds and --pass2 must be multiples of --batch, or --pass2 <= --batch")
    detframe, sim_commit = load_scorer()
    args.out.mkdir(parents=True, exist_ok=True)
    env = environment(sim_commit, args.seed)
    print(json.dumps(env), flush=True)
    for case in args.cases:
        result = run_case(case, detframe, args)
        result["environment"] = env
        (args.out / f"{case.name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(
            case.name,
            json.dumps({k: result[k] for k in result if k.endswith(("median", "seconds"))}),
            flush=True,
        )


if __name__ == "__main__":
    main()
