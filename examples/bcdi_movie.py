"""Film a 3D reconstruction of a simulated BCDI rocking scan, one frame every 10 iterations.

python examples/bcdi_movie.py CASE_DIR [--out DIR]

The data are synthetic.  CASE_DIR is a case folder written by
``bench.make_cases`` of rocking_xrd_sim (a private NaNIsaac repository, the
BCDI forward simulator): ``frames.npy``, the (frame, row, col) intensity of a
faceted, strained particle around its Bragg peak, sampled on the Ewald sphere
at every rocking angle with Poisson noise, and ``truth_det.npy``, the true
object on the grid conjugate to those indices.  The case on nanisaac.github.io
came from rocking_xrd_sim f445fe3, run from its checkout with
``uv run --python 3.13 --with numpy,scipy,xrayutilities``:

    from pathlib import Path
    from bench.make_cases import write_case
    write_case(Path(CASE_DIR), "asymmetric", 350.0, 1.2, 1e5, 0, 224, 128, 0.3)

an asymmetric 350 nm particle with 1.2 rad of strain phase, 1e5 photons at the
peak, 61 frames of 128 x 128 pixels at 5.55 keV (peak memory about 1.4 GB).

QoraPhase runs the default recipe from STARTS random starts with no support,
one start at a time in 10-iteration chunks, and the movie follows the start
that ``Ensemble.consensus`` picks.  Each frame shows
amplitude (top) and phase (bottom) on three central cuts through the particle
-- fixed frame, fixed row, fixed column -- after aligning the run to the truth
(shift, twin, global phase and phase ramp, none of which the data determine).
The grid is the detector-conjugate one, sheared, not the lab frame.

Writes DIR/frames/*.png, DIR/poster.png (the last frame) and, when ffmpeg is on
PATH, DIR/bcdi_movie.mp4.
"""

import argparse
import dataclasses
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import qoraphase as qp
from qoraphase.core.state import initial_state
from qoraphase.recipes.recipe import Detwin, Iterative, Repeat

STARTS, SEED, CHUNK, FPS, HOLD, HALF = 4, 0, 10, 8, 2.0, 16
PAPER, INK, MUTED = "#e8e8e3", "#181715", "#63625c"


def chunks(stage) -> list:
    """*stage* as a flat list of stages of at most CHUNK iterations, run in order.

    A detwin restores the support it cut, so a detwin over two halves is the
    detwin over the whole; everything else carries its state across chunks.
    """
    match stage:
        case qp.Recipe():
            return [c for s in stage.stages for c in chunks(s)]
        case Repeat():
            return chunks(stage.body) * stage.n
        case Detwin():
            return [Detwin(qp.Recipe(c), stage.axis) for c in chunks(stage.body)]
        case Iterative() if stage.n > CHUNK:
            if stage.n % CHUNK:
                raise ValueError(f"{stage} does not split into chunks of {CHUNK}")
            return [dataclasses.replace(stage, n=CHUNK)] * (stage.n // CHUNK)
        case _:
            return [stage]


def align(obj: np.ndarray, truth: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """*obj* moved onto *truth*: shift, twin and global phase, then the linear phase ramp."""
    obj = np.asarray(qp.align_to(obj, truth))
    idx = np.argwhere(inside).astype(float)
    design = np.hstack([np.ones((idx.shape[0], 1)), idx])
    weight = np.abs(obj[inside]) * np.abs(truth[inside])
    diff = np.angle(obj[inside] * np.conj(truth[inside]))
    for _ in range(3):  # refit on the residual, unwrapped about the last fit
        coef, *_ = np.linalg.lstsq(design * weight[:, None], diff * weight, rcond=None)
        fitted = design @ coef
        diff = fitted + np.angle(np.exp(1j * (diff - fitted)))
    grid = np.stack(np.indices(obj.shape), -1)
    return obj * np.exp(-1j * (coef[0] + grid @ coef[1:]))


def cuts(volume: np.ndarray, centre) -> tuple:
    f, r, c = centre
    window = [slice(n - HALF, n + HALF) for n in centre]
    return (
        volume[f, window[1], window[2]],
        volume[window[0], r, window[2]],
        volume[window[0], window[1], c],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, default=Path("bcdi_movie"))
    args = parser.parse_args()
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()

    frames = np.load(args.case / "frames.npy")
    truth = np.load(args.case / "truth_det.npy")
    problem = qp.CDI(frames)
    keys = jax.random.split(jax.random.key(SEED), STARTS)
    stages = chunks(qp.default_recipe())
    runners = {stage: jax.jit(stage.run) for stage in stages}

    def run(key, on_chunk=None) -> qp.Reconstruction:
        # One start at a time, not vmapped as in qp.reconstruct: batching rounds
        # differently, and over 720 iterations that is enough to end elsewhere.
        state = initial_state(key, problem)
        errors = []
        if on_chunk:
            on_chunk(0, state.obj, None)
        for stage in stages:
            state, metrics = runners[stage](state, problem)
            if metrics.size:
                errors.append(float(metrics[-1]))
                if on_chunk:
                    on_chunk(int(state.iteration), state.obj, errors[-1])
        return qp.Reconstruction(state.obj, state.support, np.asarray(errors))

    ensemble = qp.Ensemble(tuple(run(key) for key in keys), problem)
    pick = next(i for i, r in enumerate(ensemble.runs) if r is ensemble.consensus)

    amp_true = np.abs(truth)
    inside = amp_true > 0.5 * amp_true.max()
    scored = amp_true > 0.3 * amp_true.max()
    final = align(ensemble.consensus.object, truth, inside)
    phase_err = np.angle(final[scored] * np.conj(truth[scored]))
    print(f"qoraphase {version('qoraphase')}, start {pick} of {STARTS} (consensus)")
    print("Fourier error per start:", " ".join(f"{r.error:.3f}" for r in ensemble.runs))
    print(f"phase RMSE inside 30% of the true amplitude: {np.sqrt(np.mean(phase_err**2)):.3f} rad")

    centre = tuple(round(c) for c in np.argwhere(scored).mean(0))
    amp_max = np.abs(final).max()
    amp_map = LinearSegmentedColormap.from_list("paper_ink", [PAPER, INK])
    phase_map = plt.get_cmap("cividis").with_extremes(bad=PAPER)

    width, height, side, gap = 960, 540, 230, 10
    left = (width - 3 * side - 2 * gap) / 2
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=PAPER)
    panels = []
    for row, (cmap, lim) in enumerate(((amp_map, (0, amp_max)), (phase_map, (-1.2, 1.2)))):
        for col in range(3):
            x, y = left + col * (side + gap), height - 15 - (row + 1) * side - row * gap
            ax = fig.add_axes((x / width, y / height, side / width, side / height))
            ax.set_axis_off()
            panels.append(
                ax.imshow(np.zeros((2 * HALF, 2 * HALF)), cmap=cmap, vmin=lim[0], vmax=lim[1],
                          interpolation="nearest")
            )  # fmt: skip
    label = fig.text(left / width, 14 / height, "", color=MUTED, fontsize=20, family="monospace")

    written = []

    def draw(iteration, obj, err) -> None:
        aligned = align(obj, truth, inside)
        amp = np.abs(aligned)
        phase = np.where(amp > 0.2 * amp.max(), np.angle(aligned), np.nan)
        for panel, img in zip(panels, cuts(amp, centre) + cuts(phase, centre), strict=True):
            panel.set_data(img)
        text = f"iteration {iteration:>3}"
        label.set_text(text if err is None else f"{text}   Fourier error {err:.3f}")
        written.append(frames_dir / f"{len(written):04d}.png")
        fig.savefig(written[-1], facecolor=PAPER)

    filmed = run(keys[pick], draw)
    if not np.array_equal(np.asarray(filmed.object), np.asarray(ensemble.consensus.object)):
        raise RuntimeError("the filmed run is not the consensus run")
    shutil.copyfile(written[-1], args.out / "poster.png")
    print(f"wrote {len(written)} frames and {args.out / 'poster.png'}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not on PATH: no movie")
        return
    movie = args.out / "bcdi_movie.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS),
         "-i", str(frames_dir / "%04d.png"),
         "-vf", f"tpad=stop_mode=clone:stop_duration={HOLD}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
         str(movie)],
        check=True,
    )  # fmt: skip
    print(f"wrote {movie} ({movie.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
