"""Algorithm contracts and an end-to-end recovery check (spec 10, 14.1)."""

import jax.numpy as jnp
import numpy as np
import pytest

import qoraphase as qp
from conftest import flip_about_origin, make_data, similarity
from qoraphase.core.state import ReconState

ALGORITHMS = [qp.ER, qp.HIO, qp.RAAR]


def state_at(obj, support):
    return ReconState(
        obj=jnp.asarray(obj),
        support=jnp.asarray(support),
        iteration=jnp.int32(0),
        key=None,
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_true_solution_is_a_fixed_point(algorithm, data, problem):
    """Every algorithm must leave the exact solution alone.

    This is the sharpest non-statistical test available: it catches sign errors
    in the reflection formulas that a convergence test would only make slow.
    """
    obj, support, _ = data
    final, metrics = algorithm(20).run(state_at(obj, support), problem)
    assert np.allclose(final.obj, obj, atol=1e-4)
    assert float(metrics[-1]) < 1e-4


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_iteration_counter_advances(algorithm, data, problem):
    obj, support, _ = data
    final, metrics = algorithm(7).run(state_at(obj, support), problem)
    assert int(final.iteration) == 7
    assert metrics.shape == (7,)


def test_error_reduction_is_monotone(data, problem):
    """ER is guaranteed non-increasing in the Fourier error."""
    obj, support, _ = data
    start = jnp.asarray(obj) * 0.2 + 0.3
    _, metrics = qp.ER(40).run(state_at(start, support), problem)
    metrics = np.asarray(metrics)
    assert np.all(np.diff(metrics) <= 1e-6)
    assert metrics[-1] < metrics[0]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithms_reduce_the_error_from_a_random_start(algorithm, problem):
    import jax

    from qoraphase.core.state import initial_state

    start = initial_state(jax.random.key(0), problem)
    _, metrics = algorithm(200).run(start, problem)
    metrics = np.asarray(metrics)
    assert metrics[-1] < 0.5 * metrics[0]


RECOVERY = qp.Recipe(qp.RAAR(400, beta=0.9), qp.ER(200))


def test_recovers_a_known_object():
    """Full pipeline: intensity in, object out, compared up to CDI ambiguity."""
    obj, support, intensity = make_data(m=10, seed=3)
    ensemble = qp.reconstruct(intensity, support=support, recipe=RECOVERY, starts=8)

    assert ensemble.best.error < 1e-3
    assert similarity(ensemble.best.object, obj) > 0.95


def success_rate(support, intensity, starts=8, seed=1):
    ensemble = qp.reconstruct(
        intensity, support=support, recipe=RECOVERY, starts=starts, seed=seed
    )
    return sum(run.error < 1e-4 for run in ensemble) / starts


def test_centrosymmetric_support_collapses_the_success_rate():
    """Known limitation, not a regression (spec 14.1) -- and why multi-start is core.

    A support symmetric under ``r -> -r`` fits the object *and* its conjugate
    flip, so most starts stagnate in a mixture of the two.  Individual starts
    still occasionally succeed, which is exactly what a fleet of starts is for.
    Breaking the support symmetry restores near-certain convergence.
    """
    _, triangular, tri_intensity = make_data(m=10, seed=3)
    # Odd m so the box is exactly symmetric about the centre pixel.
    _, symmetric, sym_intensity = make_data(m=11, seed=3, centrosymmetric=True)

    assert not np.array_equal(triangular, flip_about_origin(triangular))
    assert np.array_equal(symmetric, flip_about_origin(symmetric))

    assert success_rate(triangular, tri_intensity) >= 0.875
    assert success_rate(symmetric, sym_intensity) <= 0.25
