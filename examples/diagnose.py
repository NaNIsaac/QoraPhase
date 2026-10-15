"""Run QoraPhase against a real measurement and report whether to believe it.

    python examples/diagnose.py scan.npz --key intensity --mask-key mask

The library never touches files, so the loading lives here.  What the
script adds on top of ``qp.diagnose`` is the check that synthetic data can never
provide: whether independent random starts, once aligned, actually agree with
each other.  A low Fourier error only says a run fits the data; agreement across
starts is what says the answer is determined.

    --no-run    stop after the pre-flight report
    --plot out.png   save amplitude/phase panels and the convergence curves
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import qoraphase as qp


def load(path: Path, key: str | None):
    """Read one array from .npy, .npz, .mat-style HDF5 or .h5/.cxi."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as archive:
            names = list(archive.files)
            if key is None:
                if len(names) != 1:
                    raise SystemExit(f"{path} holds {names}; pick one with --key")
                key = names[0]
            return archive[key]
    if suffix in (".h5", ".hdf5", ".cxi", ".nxs"):
        try:
            import h5py
        except ImportError:
            raise SystemExit("reading HDF5 needs h5py: uv pip install h5py") from None
        if key is None:
            raise SystemExit(f"--key is required for {suffix} files")
        with h5py.File(path, "r") as handle:
            return handle[key][()]
    raise SystemExit(f"unsupported file type: {suffix}")


def agreement(ensemble):
    """Pairwise similarity between runs, after resolving the ambiguities.

    Each run is aligned onto the first, then compared by normalised overlap.
    Runs that solved the same problem land on the same answer; runs that found
    different local minima do not, however good their individual error looks.
    """
    reference = np.asarray(ensemble[0].object)
    aligned = [np.asarray(qp.align_to(run.object, ensemble[0].object)) for run in ensemble]
    scores = []
    for i, a in enumerate(aligned):
        for b in aligned[i + 1 :]:
            overlap = abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
            scores.append(float(overlap))
    return np.array(scores) if scores else np.array([1.0])


def plot(result, ensemble, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 4))
    panels = figure.subplots(1, 3)
    result.plot(axes=panels[:2] if result.object.ndim == 2 else None)
    for run in ensemble:
        panels[2].semilogy(np.asarray(run.metrics), lw=0.8)
    panels[2].set_title("Fourier error")
    panels[2].set_xlabel("iteration")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    print(f"\nwrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--key", help="dataset name inside .npz / .h5")
    parser.add_argument("--mask", type=Path, help="mask file; True marks bad pixels")
    parser.add_argument("--mask-key")
    parser.add_argument("--support", type=Path)
    parser.add_argument("--support-key")
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args(argv)

    intensity = load(args.path, args.key)
    mask = None if args.mask is None else load(args.mask, args.mask_key).astype(bool)
    support = (
        None if args.support is None else load(args.support, args.support_key).astype(bool)
    )

    report = qp.diagnose(intensity, mask=mask, support=support)
    print(report)
    if args.no_run:
        return 0
    if report.warnings:
        print("\ncontinuing anyway -- read the warnings above before trusting this")

    print(f"\nreconstructing, {args.starts} starts ...")
    ensemble = qp.reconstruct(
        intensity,
        support=support,
        mask=mask,
        starts=args.starts,
        batch=args.batch,
        seed=args.seed,
    )
    if args.starts == 1:
        ensemble = qp.Ensemble((ensemble,))

    errors = np.array([run.error for run in ensemble])
    scores = agreement(ensemble)
    print(f"final error   best {errors.min():.3e}   worst {errors.max():.3e}")
    print(f"run agreement min {scores.min():.3f}   median {np.median(scores):.3f}")
    if scores.min() < 0.9:
        print(
            "!! the starts did not converge on the same object. A low error on a "
            "single run means it fits the data, not that the answer is determined."
        )

    if args.plot:
        plot(ensemble.best, ensemble, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
