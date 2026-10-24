"""Film a 3D reconstruction of a simulated BCDI rocking scan the way BCDI results are shown.

uv run --group examples python examples/bcdi_movie.py CASE_DIR [--out DIR]

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
that ``Ensemble.consensus`` picks.  The movie has three parts, turning slowly
all the way through:

1. the measured intensity before any reconstruction, as nested isosurfaces of
   log10 counts, with three cuts through the peak below;
2. the reconstruction every 10 iterations: the isosurface at ISO of the
   maximum amplitude, coloured by the phase, which in Bragg CDI is the
   displacement field projected on the Bragg vector (phase = Q . u), with
   three phase cuts below;
3. the final reconstruction beside the true particle, drawn the same way.

Each frame aligns the run to the truth first (shift, twin, global phase and
phase ramp, none of which the data determine).  Surfaces are drawn on the
object grid interpolated UPSAMPLE times by zero-padding its spectrum, which
smooths the facets without adding information.  The grid is the
detector-conjugate one, sheared, not the lab frame.

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
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

import qoraphase as qp
from qoraphase.core.state import initial_state
from qoraphase.recipes.recipe import Detwin, Iterative, Repeat

STARTS, SEED, CHUNK, FPS = 4, 0, 10, 12
INTRO, OUTRO, TURN, ELEV = 36, 36, 1.5, 22  # frames, frames, degrees per frame, degrees
ISO, PHASE_LIM, UPSAMPLE, MARGIN, SCAN_MARGIN, HALF = 0.4, 1.2, 2, 3, 6, 16
LEVELS = ((2.5, 1), (3.5, 1), (4.3, 1))  # log10 counts and the marching-cubes step for each
PAPER, INK, MUTED = "#e8e8e3", "#181715", "#63625c"
PHASE_MAP = plt.get_cmap("coolwarm")
INK_MAP = LinearSegmentedColormap.from_list("paper_ink", [PAPER, INK])
LIGHT = LightSource(azdeg=315, altdeg=35)


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


def upsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """*volume* on a grid *factor* times finer, by zero-padding its centred spectrum."""
    spectrum = np.fft.fftshift(np.fft.fftn(volume))
    # zero frequency must land where ifftshift expects it on the finer grid, or odd axes gain a ramp
    pad = [
        (factor * n // 2 - n // 2, (factor - 1) * n - (factor * n // 2 - n // 2))
        for n in volume.shape
    ]
    return np.fft.ifftn(np.fft.ifftshift(np.pad(spectrum, pad))) * factor**volume.ndim


def particle(obj: np.ndarray, box: tuple) -> Poly3DCollection:
    """The ISO amplitude surface of *obj* inside *box*, each face coloured by its phase."""
    fine = np.pad(upsample(obj[box], UPSAMPLE), 1)  # closed surfaces even where they touch the box
    amp = np.abs(fine)
    verts, faces, normals, _ = marching_cubes(amp, ISO * amp.max())
    # read the phase one voxel inside the surface, where the amplitude carries it, not at the rim
    inner = np.clip(np.rint(verts - normals).astype(int), 0, np.array(amp.shape) - 1)
    values = fine[tuple(inner.T)][faces].sum(1)  # complex mean: no averaging across a wrap
    colours = PHASE_MAP(Normalize(-PHASE_LIM, PHASE_LIM)(np.angle(values)))
    return Poly3DCollection(
        verts[faces],
        facecolors=colours,
        linewidths=0,
        antialiased=False,
        shade=True,
        lightsource=LIGHT,
    )


def intensity_meshes(log: np.ndarray) -> list[tuple[np.ndarray, tuple]]:
    """Nested isosurfaces of log10 counts, outermost and faintest first, as (triangles, RGBA)."""
    tones = INK_MAP(np.linspace(0.35, 0.95, len(LEVELS)))
    meshes = []
    for (level, step), tone, alpha in zip(LEVELS, tones, (0.12, 0.3, 0.95), strict=True):
        verts, faces, *_ = marching_cubes(log, level, step_size=step)
        meshes.append((verts[faces], (*tone[:3], alpha)))
    return meshes


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
    peak = np.unravel_index(np.argmax(frames), frames.shape)
    lo = np.argwhere(amp_true > 0.1 * amp_true.max()).min(0) - MARGIN
    hi = np.argwhere(amp_true > 0.1 * amp_true.max()).max(0) + MARGIN + 1
    box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    extent = (np.array(hi) - lo) * UPSAMPLE + 2
    log = np.log10(frames + 1)
    # Framed on the outermost level, ignoring the stray noise voxels that also cross it
    lit = np.argwhere(log > LEVELS[0][0])
    first, last = np.percentile(lit, [0.5, 99.5], axis=0).astype(int)
    view = tuple(
        slice(max(a - SCAN_MARGIN, 0), b + SCAN_MARGIN + 1)
        for a, b in zip(first, last, strict=True)
    )
    scan = np.pad(log[view], 1)
    scan_meshes = intensity_meshes(scan)

    width, height = 960, 540
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=PAPER)

    def axes3d(x, y, w, h, shape, zoom):
        ax = fig.add_axes((x / width, y / height, w / width, h / height), projection="3d")
        ax.set_facecolor(PAPER)
        ax.set_axis_off()
        ax.set(xlim=(0, shape[0]), ylim=(0, shape[1]), zlim=(0, shape[2]))
        ax.set_box_aspect(shape, zoom=zoom)
        return ax

    scan_ax = axes3d(140, 125, 680, 410, scan.shape, 1.9)
    recon_ax = axes3d(10, 130, 440, 400, extent, 1.45)
    truth_ax = axes3d(440, 130, 440, 400, extent, 1.45)
    titles = [fig.text((x + 220) / width, 505 / height, s, ha="center", color=MUTED, fontsize=13)
              for x, s in ((10, "QoraPhase"), (440, "truth"))]  # fmt: skip
    bar_ax = fig.add_axes((890 / width, 190 / height, 12 / width, 270 / height))
    bar = fig.colorbar(plt.cm.ScalarMappable(Normalize(-PHASE_LIM, PHASE_LIM), PHASE_MAP),
                       cax=bar_ax, ticks=[-PHASE_LIM, 0, PHASE_LIM])  # fmt: skip
    bar.ax.set_title("phase\n(rad)", color=MUTED, fontsize=10)
    bar.ax.tick_params(colors=MUTED, labelsize=10)
    bar.outline.set_visible(False)
    side, gap = 108, 8
    cut_panels = []
    for col in range(3):
        ax = fig.add_axes(
            ((20 + col * (side + gap)) / width, 12 / height, side / width, side / height)
        )
        ax.set_axis_off()
        cut_panels.append(ax.imshow(np.zeros((2 * HALF, 2 * HALF)), interpolation="nearest"))
    label = fig.text(380 / width, 58 / height, "", color=MUTED, fontsize=18, family="monospace")

    written = []

    def save(meshes: dict, cut_images, cmap, lim, text) -> None:
        azim = -60 + TURN * len(written)
        for ax in (scan_ax, recon_ax, truth_ax):
            for coll in list(ax.collections):
                coll.remove()
            ax.set_visible(ax in meshes)
            ax.view_init(ELEV, azim)
        for ax, colls in meshes.items():
            for coll in colls:
                ax.add_collection3d(coll)
        titles[0].set_visible(recon_ax in meshes)
        titles[1].set_visible(truth_ax in meshes)
        bar_ax.set_visible(recon_ax in meshes)
        for panel, img in zip(cut_panels, cut_images, strict=True):
            panel.set(data=img, cmap=cmap, clim=lim)
        label.set_text(text)
        written.append(frames_dir / f"{len(written):04d}.png")
        fig.savefig(written[-1], facecolor=PAPER)

    log_cuts = cuts(log, peak)
    for _ in range(INTRO):
        surfaces = [Poly3DCollection(tris, facecolors=rgba, linewidths=0, shade=True, lightsource=LIGHT)
                    for tris, rgba in scan_meshes]  # fmt: skip
        save({scan_ax: surfaces}, log_cuts, INK_MAP, (0, 5), "diffraction intensity, log scale")

    phase_map = PHASE_MAP.with_extremes(bad=PAPER)

    def phase_cuts(aligned):
        amp = np.abs(aligned)
        return cuts(np.where(amp > 0.2 * amp.max(), np.angle(aligned), np.nan), centre)

    shown = []

    def draw(iteration, obj, err) -> None:
        aligned = align(obj, truth, inside)
        text = f"iteration {iteration:>3}"
        text = text if err is None else f"{text}   Fourier error {err:.3f}"
        save({recon_ax: [particle(aligned, box)]}, phase_cuts(aligned), phase_map,
             (-PHASE_LIM, PHASE_LIM), text)  # fmt: skip
        shown[:] = [aligned, text]

    filmed = run(keys[pick], draw)
    if not np.array_equal(np.asarray(filmed.object), np.asarray(ensemble.consensus.object)):
        raise RuntimeError("the filmed run is not the consensus run")
    aligned, text = shown
    for _ in range(OUTRO):
        save({recon_ax: [particle(aligned, box)], truth_ax: [particle(truth, box)]},
             phase_cuts(aligned), phase_map, (-PHASE_LIM, PHASE_LIM), text)  # fmt: skip
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
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart",
         str(movie)],
        check=True,
    )  # fmt: skip
    print(f"wrote {movie} ({movie.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
