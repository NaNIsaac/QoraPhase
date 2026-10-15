"""Pytree registration and JAX transform compatibility.

These are the tests that fail first if the pytree contract is broken -- which is
exactly what happens the moment a parameter becomes an array or a forward model
starts carrying coordinates.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qoraphase as qp
from qoraphase.core.state import initial_state

FAST = qp.Recipe(qp.RAAR(10), qp.ER(5))


def test_problem_round_trips_through_a_pytree(problem):
    leaves, treedef = jax.tree.flatten(problem)
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert len(leaves) == 3  # amplitude, mask, support
    assert np.array_equal(np.asarray(rebuilt.amplitude), np.asarray(problem.amplitude))
    assert rebuilt.model is problem.model  # meta field, kept static


def test_algorithm_parameters_are_leaves_and_counts_are_static():
    """beta is dynamic so schedules will not retrace; n is the scan length."""
    leaves, treedef = jax.tree.flatten(qp.RAAR(800, beta=0.9))
    assert leaves == [0.9]
    assert "800" in str(treedef)
    assert jax.tree.flatten(qp.ER(200))[0] == []


def test_recipe_is_a_pytree():
    recipe = qp.Recipe(qp.RAAR(10, beta=0.8), qp.Shrinkwrap(0.2, 1.0), qp.ER(5))
    leaves, treedef = jax.tree.flatten(recipe)
    assert leaves == [0.8, 0.2, 1.0]
    assert len(jax.tree.unflatten(treedef, leaves)) == 3


def test_problem_survives_a_jit_boundary(problem):
    @jax.jit
    def peek(p):
        return p.amplitude.sum()

    assert np.isclose(float(peek(problem)), float(problem.amplitude.sum()), rtol=1e-6)


def test_recipe_survives_a_jit_boundary(problem):
    state = initial_state(jax.random.key(0), problem)

    @jax.jit
    def go(p, r, s):
        return r.run(s, p)[1]

    assert np.allclose(np.asarray(go(problem, FAST, state)), np.asarray(FAST.run(state, problem)[1]), atol=1e-5)


def test_vmap_matches_a_python_loop(problem):
    keys = jax.random.split(jax.random.key(0), 3)

    def one(key):
        return FAST.run(initial_state(key, problem), problem)[1][-1]

    looped = np.asarray([float(one(k)) for k in keys])
    mapped = np.asarray(jax.vmap(one)(keys))
    assert np.allclose(looped, mapped, atol=1e-5)


def test_changing_beta_does_not_retrace(problem):
    """A traced parameter must reuse the compiled kernel."""
    state = initial_state(jax.random.key(0), problem)

    @jax.jit
    def go(p, r, s):
        return r.run(s, p)[1][-1]

    go(problem, qp.Recipe(qp.RAAR(10, beta=0.9)), state)
    before = go._cache_size()
    go(problem, qp.Recipe(qp.RAAR(10, beta=0.75)), state)
    assert go._cache_size() == before

    go(problem, qp.Recipe(qp.RAAR(11, beta=0.9)), state)
    assert go._cache_size() == before + 1  # iteration count is static


def test_unregistered_object_in_the_core_is_a_bug():
    """Sanity check that the registration is what makes this work at all."""

    class NotAPytree:
        pass

    with pytest.raises(TypeError):
        jax.jit(lambda x: x.amplitude)(NotAPytree())
