"""Support projection and autocorrelation support estimate (spec 7, 10.2)."""

from dataclasses import dataclass

import jax.numpy as jnp


def support_project(obj: jnp.ndarray, support: jnp.ndarray) -> jnp.ndarray:
    """Zero everything outside the support."""
    return jnp.where(support, obj, 0)


@dataclass(frozen=True)
class SupportProjection:
    """Public handle for :func:`support_project`.

    v0.1 applies no real/positivity constraint -- the object is complex.  A
    ``constraint=`` argument belongs here when that changes (spec 10.2).
    """

    def __call__(self, obj, support):
        return support_project(obj, support)


def autocorrelation_support(intensity: jnp.ndarray, threshold: float = 0.1) -> jnp.ndarray:
    """Default support when the user supplies none (spec 7).

    The autocorrelation is about twice the object size, so this is a loose
    starting support and relies on shrinkwrap to tighten it.  A measured support
    converges better; pass one when you have it.
    """
    ac = jnp.abs(
        jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(intensity)))
    )
    return ac > threshold * ac.max()
