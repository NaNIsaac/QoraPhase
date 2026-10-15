"""Recipe composition."""

import jax
import numpy as np

import qoraphase as qp
from qoraphase.core.state import initial_state


def test_metrics_length_is_the_sum_of_stage_iterations(problem):
    state = initial_state(jax.random.key(0), problem)
    _, metrics = qp.Recipe(qp.RAAR(30), qp.ER(20)).run(state, problem)
    assert metrics.shape == (50,)


def test_shrinkwrap_contributes_no_metrics(problem):
    state = initial_state(jax.random.key(0), problem)
    _, metrics = qp.Recipe(qp.ER(10), qp.Shrinkwrap(), qp.ER(5)).run(state, problem)
    assert metrics.shape == (15,)


def test_repeat_multiplies_the_body(problem):
    state = initial_state(jax.random.key(0), problem)
    recipe = qp.Recipe(qp.repeat(4, qp.RAAR(5), qp.Shrinkwrap()))
    final, metrics = recipe.run(state, problem)
    assert metrics.shape == (20,)
    assert int(final.iteration) == 20


def test_shrinkwrap_inside_repeat_changes_the_support(problem):
    state = initial_state(jax.random.key(0), problem)
    recipe = qp.Recipe(qp.repeat(3, qp.RAAR(10), qp.Shrinkwrap(threshold=0.3)))
    final, _ = recipe.run(state, problem)
    assert not np.array_equal(np.asarray(final.support), np.asarray(problem.support))


def test_stage_order_is_execution_order(problem):
    """RAAR-then-ER and ER-then-RAAR are different computations."""
    state = initial_state(jax.random.key(0), problem)
    _, a = qp.Recipe(qp.RAAR(20), qp.ER(20)).run(state, problem)
    _, b = qp.Recipe(qp.ER(20), qp.RAAR(20)).run(state, problem)
    assert not np.allclose(np.asarray(a), np.asarray(b))


def test_empty_recipe_is_rejected(problem):
    import pytest

    with pytest.raises(ValueError, match="empty"):
        problem.reconstruct(qp.Recipe())
