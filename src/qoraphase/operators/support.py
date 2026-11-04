"""Support projection and autocorrelation support estimate."""

from dataclasses import dataclass

import jax.numpy as jnp

from .shrinkwrap import gaussian_blur


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


def masked_autocorrelation_support(
    intensity: jnp.ndarray,
    mask: jnp.ndarray,
    factor: float = 3.0,
    sigma: float | None = None,
) -> jnp.ndarray:
    """Default support when the mask covers the centre of the pattern.

    With the lowest frequencies masked the autocorrelation is high-passed: its
    peak is whatever sharp structure is left, such as the streaks of a window
    frame, and a fraction of the maximum keeps only that (#38).  The full extent
    of the autocorrelation still stands out against the background, so the
    blurred magnitude is cut at *factor* times its median, which is the
    background level as long as the autocorrelation covers less than half the
    array.  *sigma* defaults to 1 % of the mean side, at least 1 pixel: much
    less leaves gaps in the high-passed autocorrelation (0.5 px fails a
    synthetic test, 1 to 8 px pass), and on the two CXIDB patterns the area
    changed by under 6 % for sigma from 8 to 24 px.
    """
    usable = jnp.where(mask, 0.0, intensity)
    ac = jnp.abs(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(usable))))
    if sigma is None:
        sigma = max(1.0, 0.01 * sum(intensity.shape) / intensity.ndim)
    blurred = gaussian_blur(ac, sigma)
    return blurred > factor * jnp.median(blurred)


def centre_masked(mask: jnp.ndarray) -> bool:
    """Whether the zero-frequency pixel, at index ``n // 2`` per axis, is masked."""
    return bool(mask[tuple(n // 2 for n in mask.shape)])
