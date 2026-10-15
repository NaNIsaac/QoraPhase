"""Iteration state and initialisation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..operators.support import support_project


class ReconState(NamedTuple):
    """Everything that changes during iteration.

    Metrics are deliberately absent: the carry must be fixed-size, and metrics
    grow per iteration.  They come out of ``lax.scan`` as stacked outputs
    instead.
    """

    obj: jnp.ndarray  # complex, center-origin
    support: jnp.ndarray  # bool
    iteration: jnp.ndarray  # int32 scalar
    key: jax.Array  # PRNG key


def random_init(key: jax.Array, problem, support: jnp.ndarray) -> jnp.ndarray:
    """Measured amplitude with uniformly random phase, back-transformed."""
    phase = jax.random.uniform(
        key, problem.amplitude.shape, minval=0.0, maxval=2.0 * jnp.pi
    )
    obj = problem.model.adjoint(problem.amplitude * jnp.exp(1j * phase))
    return support_project(obj, support)


def initial_state(key: jax.Array, problem) -> ReconState:
    return ReconState(
        obj=random_init(key, problem, problem.support),
        support=problem.support,
        iteration=jnp.int32(0),
        key=key,
    )
