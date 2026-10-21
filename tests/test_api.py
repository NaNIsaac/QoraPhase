"""Public entry point behaviour."""

import numpy as np
import pytest

import qoraphase as qp
from conftest import make_data

FAST = qp.Recipe(qp.RAAR(20), qp.ER(10))


def test_single_start_returns_a_reconstruction(data):
    _, support, intensity = data
    result = qp.reconstruct(intensity, support=support, recipe=FAST)
    assert isinstance(result, qp.Reconstruction)
    assert result.object.shape == intensity.shape
    assert result.metrics.shape == (30,)


def test_multi_start_returns_an_ensemble(data):
    _, support, intensity = data
    ensemble = qp.reconstruct(intensity, support=support, recipe=FAST, starts=4)
    assert isinstance(ensemble, qp.Ensemble)
    assert len(ensemble) == 4


def test_amplitude_and_phase_derive_from_the_object(data):
    _, support, intensity = data
    result = qp.reconstruct(intensity, support=support, recipe=FAST)
    assert np.allclose(result.amplitude, np.abs(result.object))
    assert np.allclose(result.phase, np.angle(result.object))


def test_same_seed_reproduces_the_result(data):
    _, support, intensity = data
    kwargs = dict(support=support, recipe=FAST, starts=2)
    a = qp.reconstruct(intensity, seed=7, **kwargs)
    b = qp.reconstruct(intensity, seed=7, **kwargs)
    c = qp.reconstruct(intensity, seed=8, **kwargs)
    assert np.array_equal(np.asarray(a.best.object), np.asarray(b.best.object))
    assert not np.array_equal(np.asarray(a.best.object), np.asarray(c.best.object))


def test_starts_are_independent(data):
    """Each start must get its own key, not a shared one."""
    _, support, intensity = data
    ensemble = qp.reconstruct(intensity, support=support, recipe=FAST, starts=3)
    first = np.asarray(ensemble[0].object)
    assert not any(np.array_equal(first, np.asarray(run.object)) for run in ensemble[1:])


def test_batching_matches_unbatched(data):
    """Chunking multi-start for memory must not change the answer."""
    _, support, intensity = data
    kwargs = dict(support=support, recipe=FAST, starts=4, seed=1)
    whole = qp.reconstruct(intensity, batch=None, **kwargs)
    chunked = qp.reconstruct(intensity, batch=2, **kwargs)
    for a, b in zip(whole, chunked):
        assert np.allclose(np.asarray(a.object), np.asarray(b.object), atol=1e-5)


def test_runs_without_a_support(data):
    _, _, intensity = data
    result = qp.reconstruct(intensity, recipe=FAST)
    assert np.isfinite(result.error)


def test_default_recipe_runs(data):
    _, support, intensity = data
    result = qp.reconstruct(intensity, support=support)
    assert result.metrics.size == 50 + 20 + 50 + 8 * 50 + 200


def test_problem_object_api(data):
    """The researcher-facing path."""
    _, support, intensity = data
    problem = qp.CDI(intensity=intensity, support=support)
    result = problem.reconstruct(FAST)
    assert isinstance(result, qp.Reconstruction)
    assert np.allclose(problem.intensity, intensity, atol=1e-4)


def test_intensity_is_squared_amplitude(data):
    """The argument is an intensity, not an amplitude."""
    _, support, intensity = data
    problem = qp.CDI(intensity, support=support)
    assert np.allclose(np.asarray(problem.amplitude), np.sqrt(intensity), atol=1e-5)


def test_negative_intensity_is_clipped(data):
    _, support, intensity = data
    dented = intensity.copy()
    dented[0, 0] = -5.0
    problem = qp.CDI(dented, support=support)
    assert float(problem.amplitude[0, 0]) == 0.0
