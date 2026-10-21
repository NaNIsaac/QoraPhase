"""End to end on a noisy synthetic measurement, through the default user path.

Known object -> |FFT|^2 scaled to photon counts -> Poisson noise -> ``qp.reconstruct`` with
the default recipe, an autocorrelation support and shrinkwrap -> metrics. Seeds are fixed, so
the run is deterministic; the metrics are pinned in ``golden/end_to_end.json`` with
tolerances that allow for float32 differences between platforms and JAX versions.

Regenerate the golden file after an intended change with::

    QORAPHASE_REGEN_GOLDEN=1 uv run pytest tests/test_end_to_end.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import centered_fft, complex_similarity, make_data, similarity

import qoraphase as qp

GOLDEN = Path(__file__).parent / "golden" / "end_to_end.json"
PEAK_COUNTS = 1e6  # photons in the brightest voxel
SHAPE, SIZE, OBJECT_SEED, NOISE_SEED, RUN_SEED, STARTS = (24, 24, 24), 8, 3, 7, 0, 4


@pytest.fixture(scope="module")
def run():
    obj, true_support, clean = make_data(shape=SHAPE, m=SIZE, seed=OBJECT_SEED)
    scale = PEAK_COUNTS / clean.max()
    counts = np.random.default_rng(NOISE_SEED).poisson(clean * scale).astype(float)
    best = qp.reconstruct(counts, recipe=qp.default_recipe(), starts=STARTS, seed=RUN_SEED).best

    # the error the true object itself has against the noisy counts: the noise floor
    true_amplitude = np.abs(centered_fft(obj)) * np.sqrt(scale)
    floor = np.sqrt(((true_amplitude - np.sqrt(counts)) ** 2).sum() / counts.sum())
    return {
        "total_counts": float(counts.sum()),
        "noise_floor": float(floor),
        "error": float(best.error),
        "similarity": float(similarity(best.object, obj)),
        "complex_similarity": float(complex_similarity(best.object, obj)),
        "support_fraction": float(np.mean(np.asarray(best.support))),
        "true_support_fraction": float(true_support.mean()),
    }


def test_matches_the_golden_values(run):
    if os.environ.get("QORAPHASE_REGEN_GOLDEN") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        pytest.skip("golden file regenerated")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert run["total_counts"] == golden["total_counts"]  # numpy Poisson draw, exact
    assert run["noise_floor"] == pytest.approx(golden["noise_floor"], rel=1e-6)
    assert run["error"] == pytest.approx(golden["error"], rel=0.02)
    assert run["support_fraction"] == pytest.approx(golden["support_fraction"], rel=0.05)
    assert run["similarity"] > 0.999
    assert run["complex_similarity"] > 0.999


def test_converges_to_the_noise_floor(run):
    """Well above the floor is not converged; well below is fitting the noise."""
    assert 0.9 * run["noise_floor"] < run["error"] < 1.05 * run["noise_floor"]
    # Poisson: var(sqrt(I)) ~ 1/4, so the floor is about sqrt(voxels / 4 / total counts)
    estimate = np.sqrt(np.prod(SHAPE) / 4 / run["total_counts"])
    assert run["noise_floor"] == pytest.approx(estimate, rel=0.05)


def test_shrinkwrap_support_is_not_smaller_than_the_object(run):
    """Shrinkwrap may stay loose, but a support smaller than the object would cut into it."""
    assert (
        run["true_support_fraction"] <= run["support_fraction"] < 3 * run["true_support_fraction"]
    )
