import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_data

import qoraphase as qp
from qoraphase.core.state import initial_state
from qoraphase.metrics.errors import fourier_error
from qoraphase.operators.lowpass import _length, filtered_amplitude


def test_zero_sigma_changes_nothing(problem):
    plain = problem.reconstruct(qp.Recipe(qp.HIO(30), qp.ER(10)), seed=3)
    blurred = problem.reconstruct(qp.low_pass(qp.HIO(30), qp.ER(10), sigma=(0.0, 0.0)), seed=3)
    assert np.allclose(plain.object, blurred.object, atol=1e-4)
    assert np.allclose(plain.metrics, blurred.metrics, atol=1e-4)


def test_sigma_ramps_linearly_over_the_stage_and_holds_after():
    state = initial_state(jax.random.key(0), qp.CDI(make_data()[2]))
    state = state._replace(lowpass=(jnp.int32(5), jnp.float32(11), jnp.float32(2.0), jnp.float32(0.0)))

    def at(iteration):
        return float(state._replace(iteration=jnp.int32(iteration)).sigma)

    assert at(5) == pytest.approx(2.0)
    assert at(10) == pytest.approx(1.0)
    assert at(15) == pytest.approx(0.0)
    assert at(40) == pytest.approx(0.0)
    assert state.sigma is not None
    assert initial_state(jax.random.key(0), qp.CDI(make_data()[2])).sigma is None


def test_the_schedule_is_cleared_when_the_stage_ends(problem):
    state = initial_state(jax.random.key(0), problem)
    after, metrics = qp.low_pass(qp.RAAR(7), sigma=(1.0, 0.0)).run(state, problem)
    assert after.lowpass is None
    assert metrics.shape == (7,)
    assert int(after.iteration) == 7


def test_the_schedule_starts_from_where_the_iteration_is(problem):
    """Later stages in a recipe start their own ramp at their own first iteration."""
    state = initial_state(jax.random.key(0), problem)
    recipe = qp.Recipe(qp.ER(10), qp.low_pass(qp.ER(5), sigma=(3.0, 1.0)))
    after, metrics = recipe.run(state, problem)
    assert metrics.shape == (15,)
    assert after.lowpass is None


def test_metrics_are_against_the_blurred_data(problem):
    state = initial_state(jax.random.key(0), problem)
    _, plain = qp.ER(1).run(state, problem)
    _, blurred = qp.low_pass(qp.ER(1), sigma=2.0).run(state, problem)

    magnitude = jnp.abs(problem.model.forward(state.obj))
    expected = fourier_error(magnitude, filtered_amplitude(problem, 2.0), problem.mask)

    assert float(blurred[0]) == pytest.approx(float(expected), rel=1e-5)
    assert abs(float(blurred[0]) - float(plain[0])) > 1e-3


def test_masked_pixels_do_not_leak_into_the_blur():
    amplitude = jnp.ones((16, 16))
    mask = jnp.zeros((16, 16), bool).at[8, 8].set(True)
    problem = qp.CDI(amplitude=amplitude.at[8, 8].set(1e6), mask=mask, support=mask, model=qp.FFT())

    assert np.allclose(filtered_amplitude(problem, 1.5), 1.0, atol=1e-5)


def test_unmasked_data_with_no_blur_is_returned_unchanged(problem):
    assert np.allclose(filtered_amplitude(problem, 0.0), problem.amplitude, atol=1e-5)


def test_stage_lengths_follow_the_structure():
    recipe = qp.Recipe(
        qp.repeat(3, qp.RAAR(10), qp.Shrinkwrap()),
        qp.detwin(qp.HIO(5)),
        qp.phase_limit(-1, 1, qp.ER(4)),
        qp.partial_coherence(qp.RAAR(6), kernel=3),
    )
    assert _length(recipe) == 30 + 5 + 4 + 6


def contains(stage, kind):
    """Whether a stage of type *kind* appears anywhere inside *stage*."""
    if isinstance(stage, kind):
        return True
    children = getattr(stage, "stages", ())
    inner = [getattr(stage, name) for name in ("body", "algorithm") if hasattr(stage, name)]
    return any(contains(child, kind) for child in (*children, *inner))


def test_low_pass_is_reached_by_phase_limit_and_partial_coherence():
    limited = qp.phase_limit(-1, 1, qp.low_pass(qp.HIO(3)))
    assert isinstance(limited.stages[0], qp.LowPass)
    assert contains(limited, qp.PhaseLimited)

    coherent = qp.partial_coherence(qp.low_pass(qp.RAAR(3)), kernel=3)
    assert contains(coherent, qp.UpdateCoherence)


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        qp.low_pass(qp.RAAR(5), sigma=(-1.0, 0.0))


def test_runs_under_vmap_with_a_mask():
    _, support, intensity = make_data()
    mask = np.zeros(intensity.shape, bool)
    mask[15:18, 15:18] = True
    problem = qp.CDI(intensity, support=support, mask=mask)

    ensemble = problem.reconstruct(qp.low_pass(qp.RAAR(30), qp.ER(10), sigma=(1.5, 0.0)), starts=3)

    assert len(ensemble) == 3
    assert np.all(np.isfinite([run.error for run in ensemble]))


def test_it_does_not_stop_an_easy_problem_from_converging():
    """Only a floor, not a claim of benefit: the README table shows no consistent gain."""
    _, support, intensity = make_data(m=10, seed=3)
    problem = qp.CDI(intensity, support=support)
    recipe = qp.Recipe(
        qp.low_pass(qp.RAAR(400, beta=0.9), sigma=(1.0, 0.0)), qp.ER(200)
    )

    converged = sum(
        run.error < 1e-3 for seed in range(2) for run in problem.reconstruct(recipe, starts=8, seed=seed)
    )

    assert converged >= 14  # all 16 converged when this was measured


def test_one_generation_of_a_genetic_run_can_be_low_passed():
    _, support, intensity = make_data(m=10, seed=3)
    problem = qp.CDI(intensity, support=support)
    first = qp.Recipe(qp.low_pass(qp.RAAR(20), sigma=(1.0, 0.0)), qp.ER(10))
    later = qp.Recipe(qp.RAAR(20), qp.ER(10))

    ensemble = qp.evolve(problem, [first, later, later], starts=4)

    assert [len(g) for g in (*ensemble.history, ensemble)] == [4, 4, 4]
