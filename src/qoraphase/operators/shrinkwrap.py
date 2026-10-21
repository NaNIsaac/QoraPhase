"""Shrinkwrap support update."""

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

    The blurred amplitude is thresholded at ``threshold`` times a reference
    level, chosen by *method*:

    ``"max"``
        the largest blurred amplitude.  The classic Marchesini formulation, and
        the reference drifts down as the reconstruction sharpens.
    ``"rms"``
        the root-mean-square blurred amplitude inside the current support.  A
        bright outlier lifts this too -- it is inside the support after all --
        but diluted across every pixel rather than setting the reference on its
        own, so the support update is harder to destabilise.

    The two thresholds are on different scales and do not transfer: PyNX uses
    0.28 for ``"rms"``, and the classic Marchesini ``"max"`` value is 0.2.

    The defaults here are 0.3 / 1.0, not 0.2 / 2.0.  On a loose autocorrelation
    support -- the case that actually needs shrinkwrap -- 0.2 / 2.0 never
    converged in testing while 0.3 / 1.0 did, across every object tried.  That
    measurement is on synthetic data, so treat it as a starting point and not a
    law; these two numbers are the first thing to re-tune on a real dataset.

    Produces no error metric, so it contributes nothing to the metric history.
    """

    threshold: float = 0.3
    sigma: float = 1.0
    method: str = "max"

    def __post_init__(self):
        if self.method not in ("max", "rms"):
            raise ValueError(f"method must be 'max' or 'rms', got {self.method!r}")

    def run(self, state, problem):
        blurred = gaussian_blur(jnp.abs(state.obj), self.sigma)
        if self.method == "max":
            level = blurred.max()
        else:
            inside = jnp.where(state.support, blurred, 0.0)
            level = jnp.sqrt(jnp.sum(inside**2) / jnp.maximum(jnp.sum(state.support), 1))
        return state._replace(support=blurred > self.threshold * level), NO_METRICS


jax.tree_util.register_dataclass(
    Shrinkwrap, data_fields=["threshold", "sigma"], meta_fields=["method"]
)
