"""Ladder stage 4: QoraPhase on public measured 2D CDI patterns from CXIDB.

    uv run --python 3.14+gil --with h5py python benchmarks/cxidb.py --entry 3

Each entry is downloaded once into ``data/cxidb`` (ignored by git) from its CXIDB URL and checked
against the SHA-256 recorded here; the data are CC0, but they stay out of the repository. The
report, ``docs/benchmarks/cxidb.md``, and a figure of the reconstruction and its PRTF are written
for every entry run.

A measured pattern has no ground truth. What is reported: the Fourier residual against the
unmasked data, the spread among independent starts, and the phase-retrieval transfer function
(PRTF) of those starts with its 1/e crossing, which is how the paper of each entry states its
resolution. The paper's own image and curve are not reproduced here; the numbers it quotes are.
"""

import argparse
import hashlib
import itertools
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import qoraphase as qp
from qoraphase.postprocess import align_to

sys.path.insert(0, str(Path(__file__).parent))
from ladder import centered_fft, environment, score

ROOT = Path(__file__).parents[1]
OUT = ROOT / "docs" / "benchmarks"
DATA = ROOT / "data" / "cxidb"
SATURATED = 65535
DROP = 1 / np.e


@dataclass(frozen=True)
class Entry:
    number: int
    url: str
    sha256: str
    citation: str
    paper: str


ENTRIES = {
    3: Entry(
        3,
        "https://www.cxidb.org/data/3/cxidb-3.cxi",
        "62f1ad616f6085b35e48ead7f9d8daa094785d66e0dee03553c64a5fb2896708",
        (
            "H. N. Chapman et al., Femtosecond diffractive imaging with a soft-X-ray free-electron "
            "laser, Nat. Phys. 2, 839 (2006), doi:10.1038/nphys461; data CXIDB 3, "
            "doi:10.11577/1096905"
        ),
        (
            "Shrinkwrap with HIO, the support updated every 70 iterations at 0.2 of the maximum "
            "after a Gaussian blur going from 3 to 0.7 px FWHM by iteration 5000, 3000 to 4000 "
            "iterations, the mirror hole left free, no reality or positivity; the image is the "
            "average of 250 reconstructions, and its PRTF drops to 1/e at the edge of the CCD"
        ),
    ),
}


def fetch(entry, folder):
    path = folder / Path(entry.url).name
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"downloading {entry.url}", flush=True)
        # The server answers 403 to urllib's default User-Agent.
        request = urllib.request.Request(entry.url, headers={"User-Agent": "qoraphase-benchmark"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry.sha256:
        raise SystemExit(f"{path}: SHA-256 {digest}, expected {entry.sha256}")
    return path


def load(path):
    import h5py

    with h5py.File(path, "r") as f:
        return np.real(f["entry_1/image_1/data"][()]).astype(float)


def centre(intensity):
    """The centre of symmetry: where the pattern best matches itself turned by 180 degrees."""
    a = np.log1p(intensity)
    a -= a.mean()
    corr = np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(a)).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    # The autocorrelation of a with its inversion peaks at twice the centre, modulo the size.
    return tuple(
        min(
            ((p + k * n) / 2 for k in (0, 1)),
            key=lambda c: abs(c - n / 2),
        )
        for p, n in zip(peak, a.shape, strict=True)
    )


def prepare(raw):
    """Background-subtracted intensity, the unusable pixels, cropped about the centre of
    symmetry, and the centre and hole radius found."""
    border = np.concatenate(
        [raw[:64].ravel(), raw[-64:].ravel(), raw[:, :64].ravel(), raw[:, -64:].ravel()]
    )
    background, noise = np.median(border), np.std(border)
    intensity = np.clip(raw - background, 0, None)
    saturated = raw >= SATURATED
    saturated |= np.roll(saturated, 1, 0) | np.roll(saturated, -1, 0)
    saturated |= np.roll(saturated, 1, 1) | np.roll(saturated, -1, 1)
    cy, cx = centre(np.where(saturated, 0, intensity))
    yy, xx = np.indices(raw.shape)
    r = np.hypot(yy - cy, xx - cx)
    # The shadow of the mirror hole: the radius where the ring median rises clear of the noise,
    # plus its soft edge.
    radius = next(
        lo for lo in range(1, 200) if np.median(intensity[(r >= lo) & (r < lo + 1)]) > 10 * noise
    )
    hole = r < radius + 3
    iy, ix = round(cy), round(cx)
    half = (min(iy, raw.shape[0] - iy), min(ix, raw.shape[1] - ix))
    crop = (slice(iy - half[0], iy + half[0]), slice(ix - half[1], ix + half[1]))
    mask = (saturated | hole)[crop]
    return np.where(mask, 0, intensity[crop]), mask, (cy, cx), radius


def loose_support(intensity, sigma=12, factor=3):
    """Where the blurred autocorrelation stands clear of its median level.

    With the mirror hole masked, the autocorrelation is high-passed and its peak is the bright
    cross of the window-frame streaks, so the usual threshold at a fraction of the maximum keeps
    only that cross, which shrinkwrap cannot grow out of. Its outline above the background level
    still has the extent of the object's autocorrelation.
    """
    from qoraphase.operators.shrinkwrap import gaussian_blur

    ac = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(intensity))))
    blurred = np.asarray(gaussian_blur(ac, sigma))
    return blurred > factor * np.median(blurred)


def paper_recipe(iterations=3500, every=70, group=5):
    """HIO with the paper's shrinkwrap schedule, then ER. The blur steps down once per *group*
    support updates, which keeps the compiled recipe short."""
    stages = [qp.HIO(every, 0.9)]
    cycles = iterations // every - 1
    for first in range(0, cycles, group):
        fwhm = max(0.7, 3 - 2.3 * (first + 1) * every / 5000)
        n = min(group, cycles - first)
        stages.append(qp.repeat(n, qp.Shrinkwrap(0.2, fwhm / 2.3548), qp.HIO(every, 0.9)))
    return qp.Recipe(*stages, qp.ER(200))


def residual(obj, amplitude, usable):
    computed = np.abs(centered_fft(obj))[usable]
    measured = amplitude[usable]
    c = (computed * measured).sum() / (computed**2).sum()
    return float(np.linalg.norm(c * computed - measured) / np.linalg.norm(measured))


def run(entry, starts, iterations):
    raw = load(fetch(entry, DATA))
    intensity, mask, (cy, cx), radius = prepare(raw)
    amplitude = np.sqrt(intensity)
    usable = ~mask
    support = loose_support(intensity)
    print(
        f"entry {entry.number}: {raw.shape} -> {intensity.shape}, centre ({cy:.1f}, {cx:.1f}), "
        f"hole radius {radius} px, {mask.sum()} masked, start support {support.sum()} px",
        flush=True,
    )
    problem = qp.CDI(intensity, support=support, mask=mask)
    rows, results = [], {}
    for name, recipe in (
        ("paper schedule", paper_recipe(iterations)),
        ("HIO, start support kept", qp.Recipe(qp.HIO(iterations, 0.9), qp.ER(200))),
        ("QoraPhase default", qp.default_recipe()),
    ):
        t = time.perf_counter()
        ens = problem.reconstruct(recipe, starts=starts, seed=0)
        objects = [np.asarray(r.object) for r in ens]
        seconds = (time.perf_counter() - t) / starts
        fits = [residual(o, amplitude, usable) for o in objects]
        order = np.argsort(fits)
        best = objects[order[0]]
        aligned = [best] + [np.asarray(align_to(objects[i], best, 10)) for i in order[1:]]
        curve = qp.prtf(aligned, intensity, mask, align=False)
        own = [score(a, b)[0] for a, b in itertools.combinations(objects, 2)]
        edge = 0.5
        rows.append(
            f"| {name} | {starts} "
            f"| {statistics.median(fits):.3f} ({min(fits):.3f}) | {statistics.median(own):.3f} "
            f"| {curve.crossing / edge:.2f} | {seconds:.0f} |"
        )
        results[name] = (np.mean(aligned, axis=0), curve)
        print(rows[-1], flush=True)
    return rows, results, intensity.shape, (cy, cx), radius, int(mask.sum())


def figure(entry, results, path):
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4.4 * (len(results) + 1), 4.4))
    styles = {
        "paper schedule": ("#0072B2", "-", "o"),
        "HIO, start support kept": ("#009E73", ":", "^"),
        "QoraPhase default": ("#D55E00", "--", "s"),
    }
    for ax, (name, (mean, _)) in zip(axes[:-1], results.items(), strict=True):
        a = np.abs(mean)
        ax.imshow(a, cmap="gray_r", vmax=np.percentile(a, 99.9))
        ax.set_title(f"{name}: mean of aligned starts, |object|")
        ax.set_axis_off()
    ax = axes[-1]
    for name, (_, curve) in results.items():
        color, line, marker = styles[name]
        f, v = np.asarray(curve.frequency), np.asarray(curve.value)
        ax.plot(
            f / 0.5,
            v,
            color=color,
            linestyle=line,
            marker=marker,
            markevery=max(1, len(f) // 12),
            label=name,
        )
    ax.axhline(DROP, color="0.4", linestyle=":", label="1/e")
    ax.axvline(1.0, color="0.6", linestyle="-.", label="edge of the CCD (paper's crossing)")
    ax.set_xlabel("spatial frequency / frequency at the edge")
    ax.set_ylabel("PRTF")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.suptitle(f"CXIDB {entry.number}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--entry", type=int, choices=sorted(ENTRIES), required=True)
    parser.add_argument("--starts", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=3500)
    args = parser.parse_args()
    entry = ENTRIES[args.entry]
    rows, results, shape, (cy, cx), radius, masked = run(entry, args.starts, args.iterations)
    image = OUT / f"cxidb-{entry.number}.png"
    figure(entry, results, image)
    lines = [
        f"## CXIDB {entry.number}",
        "",
        f"- {entry.citation}",
        f"- the paper: {entry.paper}",
        *environment(),
        (
            f"- preparation: background (median of a 64 px border) subtracted, saturated pixels "
            f"and their neighbours masked, the mirror hole masked to {radius + 3} px around the "
            f"centre of symmetry ({cy:.1f}, {cx:.1f}); cropped to {shape[0]} x {shape[1]} about "
            f"it, {masked} pixels masked"
        ),
        (
            f"- paper schedule: {args.iterations} HIO iterations (beta 0.9) with the support "
            f"updated every 70, and 200 ER; HIO, start support kept: the same iterations with no "
            f"support update; QoraPhase default: `default_recipe()`, 720 iterations"
        ),
        (
            "- start support for all three: the blurred autocorrelation (sigma 12 px) above 3 times "
            "its median (`loose_support`), not a fraction of its maximum, which here keeps only "
            "the cross of the window-frame streaks"
        ),
        "- PRTF over the aligned starts (shift to 1/10 px, phase, twin), crossing of 1/e as a fraction of the frequency at the edge of the CCD",
        "",
        "| recipe | starts | Fourier residual, median (best) | mutual error among starts, median | PRTF 1/e crossing / edge | s per start, incl. compile |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"![CXIDB {entry.number}]({image.name})",
        "",
        "Physics check needed: the numbers are reported without interpretation.",
        "",
    ]
    report = OUT / "cxidb.md"
    header = "# Public measured data (CXIDB)\n\nGenerated by `benchmarks/cxidb.py` (method in its docstring).\n"
    old = report.read_text(encoding="utf-8") if report.exists() else header
    marker = f"## CXIDB {entry.number}\n"
    if marker in old:
        before, _, rest = old.partition(marker)
        after = rest.partition("\n## ")
        old = before + (("## " + after[2]) if after[1] else "")
    report.write_text(old.rstrip("\n") + "\n\n" + "\n".join(lines), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
