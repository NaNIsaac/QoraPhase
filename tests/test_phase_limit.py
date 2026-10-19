import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qoraphase as qp
from qoraphase.core.state import initial_state


def test_the_full_range_changes_nothing(problem):
    plain = problem.reconstruct(qp.Recipe(qp.HIO(30), qp.ER(10)), seed=3)
    limited = problem.reconstruct(qp.phase_limit(-4.0, 4.0, qp.HIO(30), qp.ER(10)), seed=3)
    assert np.allclose(plain.object, limited.object, atol=1e-6)


def test_pixels_outside_the_range_are_cut_for_that_iteration(problem):
    state = initial_state(jax.random.key(0), problem)
    stage = qp.PhaseLimited(qp.ER(1), -0.3, 0.3)
    after, _ = stage.run(state, problem)

    outside = state.support & (jnp.abs(jnp.angle(state.obj)) > 0.3)
    assert bool(outside.any())
    assert bool(jnp.all(after.obj[outside] == 0))
    assert bool(jnp.all(after.support == state.support))


def test_every_nested_algorithm_is_limited_and_nothing_else():
    recipe = qp.phase_limit(-1, 1, qp.repeat(3, qp.RAAR(10), qp.Shrinkwrap()), qp.detwin(qp.HIO(5)))
    repeat, detwin = recipe.stages
    raar, shrinkwrap = repeat.body.stages
    assert isinstance(raar, qp.PhaseLimited) and isinstance(raar.algorithm, qp.RAAR)
    assert isinstance(shrinkwrap, qp.Shrinkwrap)
    assert isinstance(detwin.body.stages[0].algorithm, qp.HIO)


def test_metrics_keep_one_entry_per_iteration(problem):
    result = problem.reconstruct(qp.phase_limit(-1, 1, qp.RAAR(7)), seed=0)
    assert result.metrics.shape == (7,)


def test_an_empty_range_is_rejected():
    with pytest.raises(ValueError, match="low < high"):
        qp.phase_limit(0.5, -0.5, qp.ER(1))
