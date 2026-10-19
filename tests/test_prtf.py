"""Phase retrieval transfer function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import centered_fft, make_data

import qoraphase as qp

RECIPE = qp.Recipe(qp.RAAR(400, beta=0.9), qp.ER(200))


@pytest.fixture(scope="module")
def converged():
    _, support, intensity = make_data(m=10, seed=3)
    ensemble = qp.reconstruct(intensity, support=support, recipe=RECIPE, starts=6)
    good = tuple(run for run in ensemble if run.error < 1e-3)
    assert len(good) >= 2
    return intensity, qp.Ensemble(good, ensemble.problem)


def test_converged_runs_transfer_every_frequency(converged):
    intensity, ensemble = converged

    result = qp.prtf(ensemble, intensity)

    assert np.all(np.asarray(result.value) > 0.99)
    assert np.isnan(result.crossing)


def test_prtf_needs_the_runs_aligned(converged):
    """Each run carries its own phase and position; unaligned, the mean is noise."""
    intensity, ensemble = converged

    raw = qp.prtf(ensemble, intensity, align=False)
    aligned = qp.prtf(ensemble, intensity)

    assert np.nanmean(raw.value) < 0.5 * np.nanmean(aligned.value)


def test_prtf_takes_a_plain_sequence_of_objects(converged):
    intensity, ensemble = converged
    objects = [run.object for run in ensemble]

    from_objects = qp.prtf(objects, intensity)

    assert np.all(np.asarray(from_objects.value) > 0.99)


def test_prtf_finds_the_frequency_where_the_runs_stop_agreeing():
    """Phases scrambled with a width that grows with frequency.

    A run with Fourier phase noise of standard deviation s keeps exp(-s^2 / 2) of
    its coherent amplitude under averaging, so with s = sqrt(2) at f = 0.25 the
    PRTF crosses 1/e there.
    """
    _, _, intensity = make_data(m=10, seed=3)
    spectrum = centered_fft(make_data(m=10, seed=3)[0])
    n = intensity.shape[0]
    f = np.hypot(*np.meshgrid(*[(np.arange(n) - n // 2) / n] * 2, indexing="ij"))
    rng = np.random.default_rng(0)
    runs = [
        qp.FFT().adjoint(jnp.asarray(spectrum * np.exp(1j * 4 * np.sqrt(2) * f * rng.normal(size=f.shape))))
        for _ in range(48)
    ]

    result = qp.prtf(runs, intensity, align=False)

    assert abs(result.crossing - 0.25) < 0.03
    assert abs(result.resolution - 4.0) < 0.5
    assert result.value[0] > 0.99


def test_masked_pixels_do_not_enter(converged):
    intensity, ensemble = converged
    mask = np.zeros(intensity.shape, bool)
    mask[10:14, 20:24] = True
    corrupted = np.where(mask, 1e6, intensity)

    clean = qp.prtf(ensemble, intensity, mask)
    dirty = qp.prtf(ensemble, corrupted, mask)

    assert np.allclose(clean.value, dirty.value, equal_nan=True)


def test_prtf_curve_runs_under_jit(converged):
    intensity, ensemble = converged
    aligned = jnp.stack([run.object for run in ensemble.align()])

    eager = qp.prtf_curve(aligned, intensity)
    compiled = jax.jit(qp.prtf_curve)(aligned, intensity)

    for a, b in zip(eager, compiled, strict=True):
        assert np.allclose(a, b, equal_nan=True)
