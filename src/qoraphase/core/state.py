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
    # Partial-coherence kernel, center-origin; None for a coherent model.
    coherence: jnp.ndarray | None = None
    # Low-pass schedule (start iteration, length, first sigma, last sigma); None
    # means the modulus constraint uses the data as measured.
    lowpass: tuple | None = None

    @property
    def sigma(self) -> jnp.ndarray | None:
        """Blur width of the data at the current iteration, or None."""
        if self.lowpass is None:
            return None
        start, length, first, last = self.lowpass
        progress = jnp.clip((self.iteration - start) / jnp.maximum(length - 1.0, 1.0), 0.0, 1.0)
        return first + (last - first) * progress


def random_init(key: jax.Array, problem, support: jnp.ndarray) -> jnp.ndarray:
    """Measured amplitude with uniformly random phase, back-transformed."""
    phase = jax.random.uniform(
        key, problem.amplitude.shape, minval=0.0, maxval=2.0 * jnp.pi
    )
    obj = problem.model.adjoint(problem.amplitude * jnp.exp(1j * phase))
    return support_project(obj, support)


def initial_state(key: jax.Array, problem, obj: jnp.ndarray | None = None) -> ReconState:
    """Start from *obj* (cut to the support) if given, else from random phases."""
    start = random_init(key, problem, problem.support) if obj is None else support_project(
        obj, problem.support
    )
    return ReconState(
        obj=start,
        support=problem.support,
        iteration=jnp.int32(0),
        key=key,
    )
