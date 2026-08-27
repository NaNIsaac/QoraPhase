"""Shrinkwrap support update (spec 12)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..recipes.recipe import NO_METRICS, Stage


def gaussian_blur(array: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Separable Gaussian blur applied in Fourier space.

    Works for any ndim.  ``sigma`` is in pixels; the transfer function is
    ``exp(-2 (pi sigma f)^2)`` with ``f`` in cycles/pixel, which is 1 at f=0 and
    therefore preserves the sum.
    """
    spectrum = jnp.fft.fftn(array)
    for axis, n in enumerate(array.shape):
        freq = jnp.fft.fftfreq(n)
        kernel = jnp.exp(-2.0 * (jnp.pi * sigma * freq) ** 2)
        broadcast = tuple(i for i in range(array.ndim) if i != axis)
        spectrum = spectrum * jnp.expand_dims(kernel, broadcast)
    return jnp.real(jnp.fft.ifftn(spectrum))


@dataclass(frozen=True)
class Shrinkwrap(Stage):
    """Re-derive the support from the current object magnitude.

    Produces no error metric, so it contributes nothing to the metric history.
    """

    threshold: float = 0.2
    sigma: float = 2.0

    def run(self, state, problem):
        blurred = gaussian_blur(jnp.abs(state.obj), self.sigma)
        support = blurred > self.threshold * blurred.max()
        return state._replace(support=support), NO_METRICS


jax.tree_util.register_dataclass(
    Shrinkwrap, data_fields=["threshold", "sigma"], meta_fields=[]
)
