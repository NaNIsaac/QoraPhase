"""Support projection and autocorrelation support estimate."""

from dataclasses import dataclass

import jax.numpy as jnp


def support_project(obj: jnp.ndarray, support: jnp.ndarray) -> jnp.ndarray:
    """Zero everything outside the support."""
    return jnp.where(support, obj, 0)


@dataclass(frozen=True)
class SupportProjection:
    """Public handle for :func:`support_project`.

    v0.1 applies no real/positivity constraint -- the object is complex.  A
    ``constraint=`` argument belongs here when that changes.
    """

    def __call__(self, obj, support):
        return support_project(obj, support)


def half_support(support: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Keep the half of *support* below its centre of mass along *axis*.

    Used for detwinning: a support cut in half is no longer symmetric under
    ``r -> -r``, so it fits only one of the two conjugate solutions.  Cutting at
    the centre of mass rather than the array centre keeps the cut on the object
    even when the object sits off-centre.
    """
    others = tuple(i for i in range(support.ndim) if i != axis)
    coordinate = jnp.arange(support.shape[axis])
    weight = support.sum(axis=others)
    centre = jnp.sum(coordinate * weight) / jnp.maximum(jnp.sum(weight), 1)
    keep = jnp.expand_dims(coordinate < centre, others)
    return support & keep


def autocorrelation_support(intensity: jnp.ndarray, threshold: float = 0.1) -> jnp.ndarray:
    """Default support when the user supplies none.

    The autocorrelation is about twice the object size, so this is a loose
    starting support and relies on shrinkwrap to tighten it.  A measured support
    converges better; pass one when you have it.
    """
    ac = jnp.abs(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(intensity))))
    return ac > threshold * ac.max()
