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
    frame, and a fraction of the maximum keeps only that (#38).  Instead the
    blurred magnitude is cut at *factor* times its median, which is the
    background level as long as the autocorrelation covers less than half the
    array.  *sigma* defaults to 1 % of the mean side, at least 1 pixel.

    This was shown to work in a narrow regime only: on two CXIDB patterns with
    sigma 10 to 13 px (the area changed by under 6 % for sigma from 8 to 24 px),
    and on a synthetic 20 px triangle in a 128 px array with a 6 px hole, where
    0.5 px fails and 1 to 8 px contain the object.  It does not contain the
    object when the hole is small against the object: a 40 px triangle in a
    128 px array with a 4 px hole, as on the ladder benchmark, comes out
    smaller than the object, because removing the lowest frequencies leaves a
    near-zero ring inside the autocorrelation that the cut splits off.  The
    factor and sigma are not tuned against hole size (#38), and there is no 3D
    test: in 3D at low oversampling the autocorrelation can fill more than half
    the array and the median is no longer the background.

    Raises:
        ValueError: if no pixel passes the cut, for example when *sigma* is
            large enough to flatten the autocorrelation into the background.
    """
    usable = jnp.where(mask, 0.0, intensity)
    ac = jnp.abs(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(usable))))
    if sigma is None:
        sigma = max(1.0, 0.01 * sum(intensity.shape) / intensity.ndim)
    blurred = gaussian_blur(ac, sigma)
    support = blurred > factor * jnp.median(blurred)
    if not support.any():
        raise ValueError(
            f"empty support: no pixel of the blurred autocorrelation (sigma={sigma}) "
            f"exceeds {factor} times its median; pass a support or a smaller sigma"
        )
    return support


def centre_masked(mask: jnp.ndarray) -> bool:
    """Whether the zero-frequency pixel, at index ``n // 2`` per axis, is masked.

    Only that one pixel is checked: a beamstop that misses it, through a
    sub-pixel offset of the centre, falls back to the threshold rule.
    """
    return bool(mask[tuple(n // 2 for n in mask.shape)])
