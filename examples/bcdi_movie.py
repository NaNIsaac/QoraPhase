"""Film a simulated BCDI measurement and its QoraPhase reconstruction, in the order of the experiment.

uv run --group examples python examples/bcdi_movie.py CASE_DIR [--out DIR] [--stills F,F,...]

The data are synthetic.  CASE_DIR is a case folder written by
``bench.make_cases`` of rocking_xrd_sim (a private NaNIsaac repository, the
BCDI forward simulator): ``frames.npy``, the (frame, row, col) intensity of a
faceted, strained particle around its Bragg peak, sampled on the Ewald sphere
at every rocking angle with Poisson noise; ``truth_det.npy``, the true object
on the grid conjugate to those indices; and ``meta.json``, whose linear fit
q - Q = q0 + B k maps detector indices k to the lab frame.  The case on
nanisaac.github.io came from rocking_xrd_sim f445fe3, run from its checkout
with ``uv run --python 3.13 --with numpy,scipy,xrayutilities``:

    from pathlib import Path
    from bench.make_cases import write_case
    write_case(Path(CASE_DIR), "asymmetric", 350.0, 1.2, 1e6, 0, 224, 128, 0.3)

an asymmetric 350 nm particle bounded by 15 facets, with 1.2 rad of strain
phase, 1e6 photons at the peak (six decades down to single counts), 61 frames
of 128 x 128 pixels at 5.55 keV (about 15 s, peak memory 1.2 GB).

QoraPhase runs the default recipe from STARTS random starts with no support,
one start at a time and one iteration per call, so that every iteration can be
filmed.  The movie, 22 s at 30 fps, follows the start that
``Ensemble.consensus`` picks, in five shots that cross-fade into each other:

1. 0-3 s, the setup: the beam, the particle and the detector, with the frame
   the detector records at each rocking angle;
2. 2.5-6 s, the 61 frames stacked along the rocking angle, in detector pixels;
3. 6-10 s, the stack sheared onto the lab frame, where it is the intensity in
   reciprocal space: each facet throws a streak along its normal, with fringes
   whose spacing gives the particle size;
4. 9.5-18 s, the reconstruction from its random start, as the ISO amplitude
   surface coloured by phase (in Bragg CDI the phase is Q . u, the
   displacement projected on the Bragg vector), on a time axis that gives
   more frames to the iterations that change the object most (see ``pace``);
5. 18-22 s, beside the true particle, then a fade back to the first frame so
   that the movie loops.

The camera turns at TURN degrees per frame through every shot.  Shots 2 to 5
are drawn in one frame of reference, the lab frame turned so that the detector
axes lie as close to its axes as a rotation allows (QR of B), with Q pointing
up; real space is drawn on that frame through x = 2 pi B^-T m / N, so a facet
faces the way its streak points.  Every iteration is moved onto the truth by
the twin choice and shift of the final object and centred on its own amplitude,
so the picture does not jump between iterations; the global phase and the
phase ramp, which the data do not determine, are refitted every iteration.
Captions are not in the movie: the shot times above are where
nanisaac.github.io shows them.

The reconstruction is cached in DIR/run.npz; delete it to run again.  Writes
DIR/bcdi_movie.mp4 (needs ffmpeg on PATH) and DIR/poster.png, or with
--stills only DIR/stills/<frame>.png.
"""

import argparse
import dataclasses
import json
import shutil
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes
from vtkmodules.vtkRenderingCore import vtkCoordinate

import qoraphase as qp
from qoraphase.core.state import initial_state
from qoraphase.postprocess import twin
from qoraphase.recipes.recipe import Detwin, Iterative, Repeat

STARTS, SEED, FPS, W, H = 4, 0, 30, 960, 540
ISO, PHASE_LIM, UPSAMPLE, MARGIN = 0.4, 1.2, 2, 10
TURN, ELEV = 0.2, 18.0  # degrees per frame, degrees
LOG_MAX, FADE, LEAD, RUN = 6.0, 15, 15, 240  # log10 counts; frames
PACE = 0.6  # share of the recovery's frames paced by how far the object moves, the rest by log(iteration)
SHOTS = {
    "setup": (0, 90),
    "stack": (75, 180),
    "reciprocal": (180, 300),
    "recovery": (285, 285 + LEAD + RUN),
    "compare": (285 + LEAD + RUN, 660),
}
LENGTH = 660
PAPER, INK, MUTED, BEAM = "#e8e8e3", "#181715", "#63625c", "#4b5e86"
PHASE_MAP = plt.get_cmap("coolwarm")
INK_MAP = LinearSegmentedColormap.from_list("paper_ink", [PAPER, "#8f8e87", INK])
VOLUME_MAP = LinearSegmentedColormap.from_list("volume", ["#a9adb6", "#5b6273", INK])


def chunks(stage) -> list:
    """*stage* as a flat list of one-iteration stages and shrinkwraps, run in order.

    A detwin restores the support it cut, so a detwin over each iteration is the
    detwin over the whole; everything else carries its state across chunks.
    """
    match stage:
        case qp.Recipe():
            return [c for s in stage.stages for c in chunks(s)]
        case Repeat():
            return chunks(stage.body) * stage.n
        case Detwin():
            return [Detwin(qp.Recipe(c), stage.axis) for c in chunks(stage.body)]
        case Iterative() if stage.n > 1:
            return [dataclasses.replace(stage, n=1)] * stage.n
        case _:
            return [stage]


def labels(stage, note: str = "") -> list[str]:
    """What runs at each iteration of *stage*, for the line under the iteration count."""
    match stage:
        case qp.Recipe():
            return [label for s in stage.stages for label in labels(s, note)]
        case Repeat():
            if any(isinstance(s, qp.Shrinkwrap) for s in stage.body.stages):
                note = ", shrinkwrap tightening the support"
            return labels(stage.body, note) * stage.n
        case Detwin():
            return labels(stage.body, " on half the support, to pick one twin")
        case Iterative():
            return [type(stage).__name__ + note] * stage.n
        case _:
            return []


def remove_ramp(obj: np.ndarray, truth: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """*obj* less the global phase and linear phase ramp that best move it onto *truth*."""
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


def placement(obj, truth: np.ndarray) -> tuple[bool, tuple]:
    """Twin choice and whole-pixel shift that put *obj* onto *truth*."""
    found = []
    for flipped in (False, True):
        shift, value = qp.register(twin(obj) if flipped else obj, truth)
        found.append((abs(complex(value)), flipped, tuple(round(float(s)) for s in shift)))
    _, flipped, shift = max(found)
    return flipped, shift


def place(obj, flipped: bool, shift: tuple) -> np.ndarray:
    obj = np.asarray(twin(obj) if flipped else obj)
    return np.roll(obj, tuple(-s for s in shift), axis=(0, 1, 2))


def upsample(volume: np.ndarray, factor: int) -> np.ndarray:
    """*volume* on a grid *factor* times finer, by zero-padding its centred spectrum."""
    spectrum = np.fft.fftshift(np.fft.fftn(volume))
    # zero frequency must land where ifftshift expects it on the finer grid, or odd axes gain a ramp
    pad = [
        (factor * n // 2 - n // 2, (factor - 1) * n - (factor * n // 2 - n // 2))
        for n in volume.shape
    ]
    return np.fft.ifftn(np.fft.ifftshift(np.pad(spectrum, pad))) * factor**volume.ndim


def pace(records: np.ndarray, frames: int, weight: float = PACE) -> np.ndarray:
    """Iteration shown at each of *frames* frames, fractional between two recorded ones.

    The frames are spread evenly over a mix of two clocks: the logarithm of the
    iteration count, so that the early iterations are not over in a moment, and
    the distance the object travels from one iteration to the next, so that the
    iterations where it changes most (the detwin, the first shrinkwraps) get the
    frames they need to dissolve into each other instead of jumping.
    """
    steps = np.sqrt((np.abs(np.diff(records, axis=0)) ** 2).sum((1, 2, 3)))
    travel = np.concatenate([[0.0], np.cumsum(steps)])
    count = np.log1p(np.arange(len(records)) / 2)
    clock = weight * travel / travel[-1] + (1 - weight) * count / count[-1]
    return np.interp(np.linspace(0, 1, frames), clock, np.arange(len(records)))


def compute(case: Path, cache: Path) -> dict:
    frames = np.load(case / "frames.npy")
    truth = np.load(case / "truth_det.npy")
    problem = qp.CDI(frames)
    keys = jax.random.split(jax.random.key(SEED), STARTS)
    stages = chunks(qp.default_recipe())
    runners = {stage: jax.jit(stage.run) for stage in stages}

    amp_true = np.abs(truth)
    inside = amp_true > 0.5 * amp_true.max()
    scored = amp_true > 0.3 * amp_true.max()
    # wide enough for the early iterates, which fill the looser support of the autocorrelation
    lo = np.maximum(np.argwhere(amp_true > 0.1 * amp_true.max()).min(0) - MARGIN, 0)
    hi = np.minimum(np.argwhere(amp_true > 0.1 * amp_true.max()).max(0) + MARGIN + 1, truth.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))

    def run(key, film=None) -> qp.Reconstruction:
        # One start at a time, not vmapped as in qp.reconstruct: batching rounds
        # differently, and over 720 iterations that is enough to end elsewhere.
        state = initial_state(key, problem)
        errors = []
        if film:
            film(0, state.obj)
        for stage in stages:
            state, metrics = runners[stage](state, problem)
            if metrics.size:
                errors.append(float(metrics[-1]))
                if film:
                    film(int(state.iteration), state.obj)
        return qp.Reconstruction(state.obj, state.support, np.asarray(errors))

    ensemble = qp.Ensemble(tuple(run(key) for key in keys), problem)
    pick = next(i for i, r in enumerate(ensemble.runs) if r is ensemble.consensus)
    flipped, shift = placement(ensemble.consensus.object, truth)

    every = np.empty((len(labels(qp.default_recipe())) + 1, *truth[box].shape), np.complex64)

    def film(iteration, obj):
        every[iteration] = remove_ramp(place(obj, flipped, shift)[box], truth[box], inside[box])

    filmed = run(keys[pick], film)
    if not np.array_equal(np.asarray(filmed.object), np.asarray(ensemble.consensus.object)):
        raise RuntimeError("the filmed run is not the consensus run")
    final = remove_ramp(place(filmed.object, flipped, shift), truth, inside)
    phase_err = np.angle(final[scored] * np.conj(truth[scored]))
    rmse = float(np.sqrt(np.mean(phase_err**2)))
    print(f"qoraphase {version('qoraphase')}, start {pick} of {STARTS} (consensus)")
    print("Fourier error per start:", " ".join(f"{r.error:.3f}" for r in ensemble.runs))
    print(f"phase RMSE inside 30% of the true amplitude: {rmse:.3f} rad")

    shown = pace(every, RUN)
    iterations = np.union1d(np.floor(shown), np.ceil(shown)).astype(int)
    data = {
        "shown": shown,
        "iterations": iterations,
        "records": every[iterations],
        "errors": np.asarray(filmed.metrics, np.float32),
        "truth": truth[box].astype(np.complex64),
        "rmse": np.array(rmse),
    }
    np.savez_compressed(cache, **data)
    return data


def frame_of_reference(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation from lab (x, y, z) to the drawing frame, and B there (upper triangular).

    *B* has rows (z, y, x), the axis order of the simulator, and columns
    (frame, row, col).
    """
    rotation, triangle = np.linalg.qr(B[::-1])
    if np.linalg.det(rotation) < 0:  # a proper rotation, or the picture comes out mirrored
        rotation[:, -1] *= -1
        triangle[-1] *= -1
    return rotation.T, triangle


class Camera:
    """The turning camera of every shot; a shot sets its heading, where it looks and from how far."""

    def __init__(self, up: np.ndarray):
        self.up = up / np.linalg.norm(up)
        side = np.cross(self.up, [1.0, 0, 0])
        if np.linalg.norm(side) < 0.1:
            side = np.cross(self.up, [0, 1.0, 0])
        self.e1 = side / np.linalg.norm(side)
        self.e2 = np.cross(self.up, self.e1)

    def facing(self, direction) -> float:
        """Azimuth, in degrees, of a camera that looks back along *direction*."""
        return float(np.degrees(np.arctan2(direction @ self.e2, direction @ self.e1)))

    def rightward(self, direction) -> float:
        """Azimuth, in degrees, at which *direction* points to the right of the screen."""
        return float(np.degrees(np.arctan2(-(direction @ self.e1), direction @ self.e2)))

    def view(self, azimuth: float) -> np.ndarray:
        """Unit vector from the focus to the camera."""
        az, el = np.radians(azimuth), np.radians(ELEV)
        return np.cos(el) * (np.cos(az) * self.e1 + np.sin(az) * self.e2) + np.sin(el) * self.up

    def aim(self, plotter: pv.Plotter, azimuth: float, focus, distance: float) -> None:
        view = self.view(azimuth)
        focus = np.asarray(focus, float)
        plotter.camera_position = [tuple(focus + distance * view), tuple(focus), tuple(self.up)]
        plotter.camera.view_angle = 30.0
        plotter.reset_camera_clipping_range()


def plotter(size=(W, H)) -> pv.Plotter:
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background(PAPER)
    p.enable_anti_aliasing("ssaa")
    return p


def image(p: pv.Plotter) -> np.ndarray:
    return np.asarray(p.screenshot(return_img=True), dtype=np.float32)[..., :3] / 255


def project(p: pv.Plotter, point) -> tuple[float, float]:
    """Pixel (x from the left, y from the top) where *point* landed in *p*'s last render."""
    c = vtkCoordinate()
    c.SetCoordinateSystemToWorld()
    c.SetValue(*map(float, point))
    x, y = c.GetComputedDoubleDisplayValue(p.renderer)
    return x, p.window_size[1] - y


def particle(obj: np.ndarray, to_display: np.ndarray) -> pv.PolyData:
    """The ISO amplitude surface of *obj*, in the drawing frame about its own centre, coloured by phase."""
    weight = np.abs(obj) ** 2
    centre = (np.stack(np.indices(obj.shape), -1) * weight[..., None]).sum((0, 1, 2)) / weight.sum()
    fine = np.pad(upsample(obj, UPSAMPLE), 1)  # closed surfaces even where they touch the box
    amp = np.abs(fine)
    # a high percentile, not the maximum, which one hot voxel can move between iterations
    verts, faces, normals, _ = marching_cubes(amp, ISO * np.percentile(np.abs(obj), 99.5))
    points = ((verts - 1) / UPSAMPLE - centre) @ to_display.T
    mesh = pv.PolyData(points, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    # read the phase one voxel inside the surface, where the amplitude carries it, not at the rim
    inner = np.clip(np.rint(verts - normals).astype(int), 0, np.array(amp.shape) - 1)
    phase = np.angle(fine[tuple(inner.T)])
    colours = PHASE_MAP(Normalize(-PHASE_LIM, PHASE_LIM)(phase))[:, :3]
    mesh.point_data["rgb"] = np.round(255 * colours).astype(np.uint8)
    mesh.point_data["phase"] = phase
    return mesh.compute_normals(auto_orient_normals=True)


def best_heading(mesh: pv.PolyData, camera: Camera) -> float:
    """Azimuth at which the faces turned to the camera carry the widest spread of phase."""
    normals, phase = mesh.point_data["Normals"], mesh.point_data["phase"]
    azimuths = np.arange(0.0, 360.0, 5.0)
    spread = [np.std(phase[normals @ camera.view(az) > 0.3]) for az in azimuths]
    return float(azimuths[np.argmax(spread)])


def volume_on_grid(log: np.ndarray, A: np.ndarray, n: int = 150):
    """*log* (frame, row, col) drawn at y = A (k - N // 2), resampled onto an axis-aligned grid.

    Returns the grid and, for the stacking shot, the frame index at every grid point.
    """
    shape = np.array(log.shape)
    corners = np.array(np.meshgrid(*[[0, s - 1] for s in shape], indexing="ij")).reshape(3, -1)
    ends = A @ (corners - (shape // 2)[:, None])
    lo, hi = ends.min(1), ends.max(1)
    step = (hi - lo).max() / (n - 1)
    dims = np.ceil((hi - lo) / step).astype(int) + 1
    y = np.stack(np.meshgrid(*[lo[i] + step * np.arange(dims[i]) for i in range(3)], indexing="ij"))
    k = np.einsum("ij,j...->i...", np.linalg.inv(A), y) + (shape // 2)[:, None, None, None]
    grid = pv.ImageData(dimensions=dims, spacing=(step,) * 3, origin=lo)
    grid.point_data["log"] = map_coordinates(log, k, order=1, cval=0.0).ravel(order="F")
    return grid, k[0].ravel(order="F")


def streak_direction(grid: pv.ImageData, centre: np.ndarray, reach: float, up: np.ndarray):
    """Unit vector from *centre* along the brightest streak that points upward of the horizon."""
    golden = np.pi * (3 - np.sqrt(5))
    i = np.arange(4000)
    z = 1 - 2 * (i + 0.5) / i.size
    dirs = np.stack(
        [np.sqrt(1 - z**2) * np.cos(golden * i), np.sqrt(1 - z**2) * np.sin(golden * i), z], -1
    )
    dirs = dirs[dirs @ up > 0.25]
    radii = np.linspace(0.35, 0.8, 12) * reach
    points = centre + dirs[:, None, :] * radii[None, :, None]
    volume = grid.point_data["log"].reshape(grid.dimensions, order="F")
    idx = (points - np.asarray(grid.origin)) / np.asarray(grid.spacing)
    brightness = map_coordinates(volume, np.moveaxis(idx, -1, 0), order=1, cval=0.0).mean(1)
    return dirs[np.argmax(brightness)]


def add_intensity(p: pv.Plotter, grid: pv.ImageData, step: float):
    values = np.linspace(0, LOG_MAX, 128)
    opacity = np.interp(values, [0, 1.0, 1.6, 2.5, 4.0, LOG_MAX], [0, 0, 0.04, 0.3, 0.9, 1.0])
    return p.add_volume(grid, scalars="log", cmap=VOLUME_MAP, clim=(0.5, LOG_MAX), opacity=opacity,
                        opacity_unit_distance=step, show_scalar_bar=False, shade=False)  # fmt: skip


def figure():
    return plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=PAPER)


def picture_axes(fig, x, y, w, h):
    """An image axes at pixel box (x, y from the top, w, h)."""
    ax = fig.add_axes((x / W, 1 - (y + h) / H, w / W, h / H))
    ax.set_axis_off()
    return ax, ax.imshow(np.ones((2, 2, 3)), interpolation="none")


def place_axes(ax, x, y, w, h) -> None:
    ax.set_position((x / W, 1 - (y + h) / H, w / W, h / H))


def text(fig, x, y, s="", **kw):
    kw.setdefault("color", MUTED)
    kw.setdefault("fontsize", 12)
    kw.setdefault("zorder", 5)  # above the pictures, which are opaque
    return fig.text(x / W, 1 - y / H, s, **kw)


def move(artist, x, y) -> None:
    artist.set_position((x / W, 1 - y / H))


def colour_bar(fig, x, y, h, norm, cmap, ticks, labels, title):
    cax = fig.add_axes((x / W, 1 - (y + h) / H, 10 / W, h / H))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm, cmap), cax=cax, ticks=ticks)
    bar.ax.set_yticklabels(labels)
    bar.ax.set_title(title, color=MUTED, fontsize=10)
    bar.ax.tick_params(colors=MUTED, labelsize=9)
    bar.outline.set_visible(False)
    return cax


def counts_bar(fig, x, y, cmap=INK_MAP):
    ticks = list(range(0, int(LOG_MAX) + 1, 2))
    return colour_bar(fig, x, y, 200, Normalize(0, LOG_MAX), cmap, ticks,
                      [f"$10^{t}$" for t in ticks], "counts")  # fmt: skip


def draw(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32)[..., :3] / 255


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def render(case: Path, data: dict, out: Path, stills: list[int] | None = None) -> None:
    frames = np.load(case / "frames.npy").astype(np.float32)
    meta = json.loads((case / "meta.json").read_text())
    B = np.array(meta["detector_frame"]["B_invA_per_index"])
    shape = np.array(frames.shape)
    log = np.log10(frames + 1)
    turn, triangle = frame_of_reference(B)
    camera = Camera(turn @ [0.0, 0.0, 1.0])  # Q is along z
    # real space: x (z, y, x) = 2 pi B^-T m / N, reordered to (x, y, z) and turned
    to_real = turn @ (2 * np.pi * np.linalg.inv(B).T / shape)[::-1]
    truth_mesh = particle(data["truth"], to_real)
    size = np.ptp(np.asarray(truth_mesh.points), axis=0).max()

    # the setup, in units of the particle size and not to scale
    theta = np.radians(meta["two_theta_deg"] / 2)
    k_in = turn @ np.array([0.0, np.cos(theta), -np.sin(theta)])
    k_out = turn @ np.array([0.0, np.cos(theta), np.sin(theta)])
    gray = truth_mesh.scale(1 / size, inplace=False)
    gray.point_data["rgb"] = np.full((gray.n_points, 3), 176, np.uint8)
    plate = 3.0 * k_out
    setup_view = plotter((600, H))
    setup_view.add_mesh(gray, scalars="rgb", rgb=True, smooth_shading=True, specular=0.2)
    setup_view.add_mesh(pv.Tube(pointa=tuple(-3.4 * k_in), pointb=tuple(-0.75 * k_in), radius=0.04),
                        color=BEAM)  # fmt: skip
    setup_view.add_mesh(pv.Cone(center=tuple(-0.68 * k_in), direction=tuple(k_in), height=0.22,
                                radius=0.1), color=BEAM)  # fmt: skip
    setup_view.add_mesh(pv.Tube(pointa=tuple(0.6 * k_out), pointb=tuple(plate), radius=0.015),
                        color=BEAM, opacity=0.6)  # fmt: skip
    setup_view.add_mesh(pv.Plane(center=tuple(plate), direction=tuple(k_out), i_size=1.1, j_size=1.1),
                        color="#d6d6d0", show_edges=False, opacity=0.9)  # fmt: skip
    setup_view.add_mesh(pv.Plane(center=tuple(plate), direction=tuple(k_out), i_size=1.1, j_size=1.1),
                        style="wireframe", color=MUTED, line_width=1.2)  # fmt: skip
    setup_focus = (-3.4 * k_in + plate) / 3
    setup_heading = camera.rightward(k_out) - 30.0 - TURN * 45

    setup_fig = figure()
    _, setup_img = picture_axes(setup_fig, 0, 0, 600, H)
    det_ax = setup_fig.add_axes((640 / W, 1 - 400 / H, 260 / W, 260 / H))
    det_ax.set_xticks([])
    det_ax.set_yticks([])
    for spine in det_ax.spines.values():
        spine.set_color(MUTED)
    det_img = det_ax.imshow(log[0], cmap=INK_MAP, vmin=0, vmax=LOG_MAX, origin="lower",
                            interpolation="nearest")  # fmt: skip
    text(setup_fig, 640, 126, "detector, 128 x 128 pixels", fontsize=12)
    det_frame = text(setup_fig, 640, 430, "", fontsize=13, family="monospace")
    det_angle = text(setup_fig, 640, 456, "", fontsize=13, family="monospace")
    counts_bar(setup_fig, 918, 170)
    tags = {
        "beam": text(setup_fig, 0, 0, "X-ray beam", ha="left"),
        "particle": text(setup_fig, 0, 0, "crystal, 350 nm", ha="center", va="top"),
    }
    link = Line2D([], [], color=MUTED, lw=0.8, ls=(0, (3, 3)), transform=setup_fig.transFigure)
    setup_fig.add_artist(link)
    text(setup_fig, 40, 40, "the measurement", fontsize=15, color=INK)
    text(setup_fig, 40, 66, "the crystal is rocked through 0.6° and one frame is read at each step")
    omegas = np.asarray(meta["omega_deg"]) - meta["omega0_deg"]

    def setup(t: int, f: int) -> np.ndarray:
        j = min(shape[0] - 1, (shape[0] - 1) * t / 75)
        a, b = int(np.floor(j)), int(np.ceil(j))
        det_img.set_data((1 - (j - a)) * log[a] + (j - a) * log[b])
        camera.aim(setup_view, setup_heading + TURN * f, setup_focus, 7.5)
        setup_img.set_data(image(setup_view))
        x, y = project(setup_view, -1.6 * k_in)
        move(tags["beam"], x + 10, y - 40)
        x, y = project(setup_view, -0.7 * camera.up)
        move(tags["particle"], x, y + 4)
        x, y = project(setup_view, plate)
        link.set_data([x / W, 636 / W], [1 - y / H, 1 - 270 / H])
        det_frame.set_text(f"frame {round(j) + 1:2d} / {shape[0]}")
        det_angle.set_text(f"ω − ω_B = {omegas[round(j)]:+.2f}°")
        return draw(setup_fig)

    # the stack and reciprocal space share one plotter
    space = plotter()
    flat = np.diag(np.diag(triangle))
    stack_grid, frame_index = volume_on_grid(log, flat)
    step0 = stack_grid.spacing[0]
    full_log = stack_grid.point_data["log"].copy()
    peak = np.array(np.unravel_index(np.argmax(frames), frames.shape)) - shape // 2
    extent = np.abs(triangle @ np.diag(shape)).sum(1).max() / 2
    space_distance = 3.1 * extent
    space_heading = camera.facing(flat[:, 0]) + 14.0 - TURN * SHOTS["stack"][0]
    r, c = np.meshgrid(np.arange(shape[1]) - shape[1] // 2, np.arange(shape[2]) - shape[2] // 2,
                       indexing="ij")  # fmt: skip

    def newest(j: int) -> pv.StructuredGrid:
        k = np.stack([np.full(r.shape, j - shape[0] // 2), r, c])
        y = np.einsum("ij,j...->i...", flat, k)
        sheet = pv.StructuredGrid(*y)
        sheet["log"] = log[j].ravel(order="F")
        return sheet

    space_fig = figure()
    _, space_img = picture_axes(space_fig, 0, 0, W, H)
    space_title = text(space_fig, 40, 40, "", fontsize=15, color=INK)
    space_note = text(space_fig, 40, 66, "")
    space_frame = text(space_fig, 40, 500, "", fontsize=13, family="monospace")
    counts_bar(space_fig, 918, 170, VOLUME_MAP)
    callout = text(space_fig, 0, 0, "streak along a facet normal", color=INK, va="center")
    callout_line = Line2D([], [], color=INK, lw=0.8, transform=space_fig.transFigure)
    space_fig.add_artist(callout_line)
    shown_actors = []

    def show(*items) -> None:
        for actor in shown_actors:
            space.remove_actor(actor)
        shown_actors[:] = list(items)

    def stack(t: int, f: int) -> np.ndarray:
        j = min(shape[0] - 1, round((shape[0] - 1) * t / 80))
        stack_grid.point_data["log"] = np.where(frame_index <= j + 0.5, full_log, 0.0)
        sheet = newest(j)
        show(add_intensity(space, stack_grid, step0),
             space.add_mesh(sheet, scalars="log", cmap=INK_MAP, clim=(0, LOG_MAX), opacity=0.85,
                            show_scalar_bar=False, lighting=False))  # fmt: skip
        camera.aim(space, space_heading + TURN * f, flat @ peak, space_distance)
        space_img.set_data(image(space))
        space_title.set_text("what the detector records")
        space_note.set_text(
            "the frames stacked in the order they were read: a 3D intensity, with no phase"
        )
        space_frame.set_text(f"frame {j + 1:2d} / {shape[0]}")
        callout.set_visible(False)
        callout_line.set_visible(False)
        return draw(space_fig)

    lab_grid, _ = volume_on_grid(log, triangle)
    lab_peak = triangle @ peak
    streak = streak_direction(lab_grid, lab_peak, extent, camera.up)

    def reciprocal(t: int, f: int) -> np.ndarray:
        s = ease(t / 30)
        A = (1 - s) * flat + s * triangle
        if s < 1:
            show(add_intensity(space, volume_on_grid(log, A)[0], step0))
        elif t == 30 or not shown_actors:
            show(add_intensity(space, lab_grid, step0))
        camera.aim(space, space_heading + TURN * f, A @ peak, space_distance)
        space_img.set_data(image(space))
        space_title.set_text("reciprocal space" if s == 1 else "what the detector records")
        space_note.set_text("the same counts in the lab frame, Q up; the spacing of the fringes gives the size"
                            if s == 1 else "moved from detector pixels onto the lab frame")  # fmt: skip
        space_frame.set_text("")
        on = 40 <= t < 112
        callout.set_visible(on)
        callout_line.set_visible(on)
        if on:
            x, y = project(space, lab_peak + 0.62 * extent * streak)
            move(callout, x + 48, y)
            callout_line.set_data([(x + 6) / W, (x + 44) / W], [1 - y / H, 1 - y / H])
            fade = min(1.0, (t - 40) / 10, (112 - t) / 10)
            callout.set_alpha(fade)
            callout_line.set_alpha(fade)
        return draw(space_fig)

    obj_view = plotter((540, 540))
    errors, iterations = data["errors"], data["iterations"]
    names = labels(qp.default_recipe())
    total = len(errors)
    shown_at = data["shown"]
    meshes: dict[int, pv.PolyData] = {}
    obj_distance = 3.0 * size

    def mesh_at(i: int) -> pv.PolyData:
        if i not in meshes:
            meshes[i] = particle(data["records"][np.searchsorted(iterations, i)], to_real)
        return meshes[i]

    # turned to show the widest spread of phase while the two particles stand side by side
    middle = (SHOTS["compare"][0] + LENGTH) // 2
    obj_heading = best_heading(truth_mesh, camera) - TURN * middle

    def look(mesh, f: int, zoom: float = 1.0) -> np.ndarray:
        obj_view.clear_actors()
        obj_view.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.25,
                          specular_power=20)  # fmt: skip
        camera.aim(obj_view, obj_heading + TURN * f, np.zeros(3), zoom * obj_distance)
        return image(obj_view)

    obj_fig = figure()
    rec_ax, rec_img = picture_axes(obj_fig, 290, 0, 540, 540)
    truth_ax, truth_img = picture_axes(obj_fig, 455, 50, 440, 440)
    colour_bar(obj_fig, 910, 160, 220, Normalize(-PHASE_LIM, PHASE_LIM), PHASE_MAP,
               [-PHASE_LIM, 0, PHASE_LIM], [f"{-PHASE_LIM:g}", "0", f"{PHASE_LIM:g}"],
               "phase\n(rad)")  # fmt: skip
    obj_title = text(obj_fig, 40, 40, "", fontsize=15, color=INK)
    obj_note = text(obj_fig, 40, 66, "")
    count = text(obj_fig, 40, 412, "", fontsize=13, family="monospace")
    stage = text(obj_fig, 40, 436, "", fontsize=11)
    err_ax = obj_fig.add_axes((52 / W, 1 - 370 / H, 150 / W, 110 / H), facecolor=PAPER)
    err_ax.set_yscale("log")
    err_ax.set(xlim=(0, total), ylim=(0.8 * errors.min(), 1.25 * errors.max()), xticks=[0, total])
    for side in ("top", "right"):
        err_ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        err_ax.spines[side].set_color(MUTED)
    err_ax.tick_params(colors=MUTED, labelsize=8, which="both")
    err_ax.set_title("Fourier error", color=MUTED, fontsize=10, loc="left")
    err_line = err_ax.plot([], [], color=INK, lw=1.0)[0]
    err_dot = err_ax.plot([], [], "o", color=INK, ms=3)[0]
    titles = (text(obj_fig, 0, 46, "QoraPhase", ha="center", fontsize=13),
              text(obj_fig, 675, 46, "truth", ha="center", fontsize=13))  # fmt: skip
    rmse = float(data["rmse"])
    verdict = text(obj_fig, 465, 515, "", ha="center", fontsize=13, color=INK)

    def recovery(t: int, f: int) -> np.ndarray:
        x = 0.0 if t < LEAD else float(shown_at[min(t - LEAD, RUN - 1)])
        a, b = int(np.floor(x)), int(np.ceil(x))
        # back off while the early iterates fill the loose starting support, close in as it tightens
        zoom = 1 + 0.9 * (1 - ease((t - LEAD) / 90))
        picture = look(mesh_at(a), f, zoom)
        if b != a:
            picture = (b - x) * picture + (x - a) * look(mesh_at(b), f, zoom)
        rec_img.set_data(picture)
        place_axes(rec_ax, 290, 0, 540, 540)
        truth_ax.set_visible(False)
        for title in titles:
            title.set_visible(False)
        n = round(x)
        err_ax.set_visible(True)
        err_line.set_data(np.arange(1, n + 1), errors[:n])
        err_dot.set_data([n] if n else [], [errors[n - 1]] if n else [])
        obj_title.set_text("phase retrieval")
        obj_note.set_text("QoraPhase recovers the phase the detector cannot record")
        count.set_text(f"iteration {n:3d} / {total}")
        stage.set_text(names[n - 1] if n else "random start")
        verdict.set_text("")
        return draw(obj_fig)

    def compare(t: int, f: int) -> np.ndarray:
        s = ease(t / 20)
        place_axes(rec_ax, 290 + s * (15 - 290), s * 50, 540 - 100 * s, 540 - 100 * s)
        rec_img.set_data(look(mesh_at(total), f))
        fade = ease((t - 8) / 20)
        truth_ax.set_visible(fade > 0)
        truth_img.set_data(look(truth_mesh, f))
        truth_img.set_alpha(fade)
        titles[0].set_visible(True)
        titles[0].set_alpha(s)
        move(titles[0], 290 + s * (15 - 290) + (540 - 100 * s) / 2, 46)
        titles[1].set_visible(True)
        titles[1].set_alpha(fade)
        err_ax.set_visible(False)
        obj_title.set_text("")
        obj_note.set_text("")
        count.set_text("")
        stage.set_text("")
        verdict.set_text(f"phase error {rmse:.2f} rad rms inside the particle" if t >= 30 else "")
        verdict.set_alpha(ease((t - 30) / 15))
        return draw(obj_fig)

    painters = {"setup": setup, "stack": stack, "reciprocal": reciprocal,
                "recovery": recovery, "compare": compare}  # fmt: skip
    first = setup(0, 0)

    def frame(f: int) -> np.ndarray:
        active = [(k, f - a) for k, (a, b) in SHOTS.items() if a <= f < b]
        pictures = [(painters[k](t, f), t) for k, t in active]
        picture = pictures[0][0]
        if len(pictures) == 2:
            (old, _), (new, t) = pictures
            w = (t + 1) / (FADE + 1)
            picture = (1 - w) * old + w * new
        if f >= LENGTH - FADE:  # back to the first frame, so that the loop has no seam
            w = (f - (LENGTH - FADE) + 1) / FADE
            picture = (1 - w) * picture + w * first
        return np.clip(np.rint(255 * picture), 0, 255).astype(np.uint8)

    out.mkdir(parents=True, exist_ok=True)
    if stills:
        (out / "stills").mkdir(exist_ok=True)
        for f in stills:
            plt.imsave(out / "stills" / f"{f:03d}.png", frame(f))
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not on PATH")
    movie = out / "bcdi_movie.mp4"
    encoder = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(movie)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    for f in range(LENGTH):
        picture = frame(f)
        if f == SHOTS["compare"][0] + 60:
            plt.imsave(out / "poster.png", picture)
        encoder.stdin.write(picture.tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise SystemExit("ffmpeg failed")
    print(f"wrote {movie} ({movie.stat().st_size / 1e6:.2f} MB) and {out / 'poster.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, default=Path("bcdi_movie"))
    parser.add_argument("--stills", type=lambda s: [int(f) for f in s.split(",")],
                        help="write only these frames, as PNG")  # fmt: skip
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "run.npz"
    data = dict(np.load(cache)) if cache.exists() else compute(args.case, cache)
    render(args.case, data, args.out, args.stills)


if __name__ == "__main__":
    main()
