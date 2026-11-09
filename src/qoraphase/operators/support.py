"""Support projection and autocorrelation support estimate."""

from dataclasses import dataclass

import jax
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
    factor: float = 2.0,
    sigma: float | None = None,
) -> jnp.ndarray:
    """Default support when the mask covers the centre of the pattern.

    With the lowest frequencies masked the autocorrelation is high-passed: its
    peak is whatever sharp structure is left, such as the streaks of a window
    frame, and a fraction of the maximum keeps only that (#38).  Instead the
    blurred magnitude is cut at *factor* times its median, which is the
    background level as long as the autocorrelation covers less than half the
    array.  *sigma* defaults to 2.3 % of the mean side, at least 1 pixel.

    The cut alone is not enough when the hole is small against the object:
    removing frequencies inside a radius r subtracts a smooth component about
    N / r wide, which leaves a near-zero ring inside the autocorrelation, and
    the cut splits the inside off.  An autocorrelation support holds the
    origin and reaches out along every direction, so every pixel with a cut
    pixel further out on its ray from the centre is added back.

    Measured on synthetic triangles and tetrahedra (12 to 80 px in 128 and
    256 px arrays, 16 to 24 px in 64 px cubes, holes of 2 to 16 px, noiseless
    and 1e4 to 1e6 peak photons): the estimate contains the object in every
    case, at 3.5 to 90 times its size and at most 38 % of the array. On the two
    CXIDB patterns of ``benchmarks/cxidb.py`` it holds all of the previous
    estimate (3 times the median at 1 % of the side, no ray fill) at about
    twice its area.  Not tested on a real 3D pattern; at low oversampling the
    autocorrelation can fill more than half the array and the median is no
    longer the background.

    Raises:
        ValueError: if no pixel passes the cut, for example when *sigma* is
            large enough to flatten the autocorrelation into the background.
    """
    usable = jnp.where(mask, 0.0, intensity)
    ac = jnp.abs(jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(usable))))
    if sigma is None:
        sigma = max(1.0, 0.023 * sum(intensity.shape) / intensity.ndim)
    blurred = gaussian_blur(ac, sigma)
    cut = blurred > factor * jnp.median(blurred)
    if not cut.any():
        raise ValueError(
            f"empty support: no pixel of the blurred autocorrelation (sigma={sigma}) "
            f"exceeds {factor} times its median; pass a support or a smaller sigma"
        )
    return fill_along_rays(cut)


def fill_along_rays(region: jnp.ndarray, step: float = 0.5) -> jnp.ndarray:
    """*region* and every pixel that has a *region* pixel further out on its ray from the
    centre (index ``n // 2`` per axis), sampled every *step* pixels."""
    shape = jnp.array(region.shape).reshape(-1, *[1] * region.ndim)
    centre = (shape // 2).astype(jnp.float32)
    offset = jnp.indices(region.shape, dtype=jnp.float32) - centre
    radius = jnp.maximum(jnp.sqrt(jnp.sum(offset**2, axis=0)), 1e-6)
    steps = int(jnp.ceil(jnp.sqrt(sum(n**2 for n in region.shape)) / 2 / step))

    def further(k, filled):
        position = jnp.rint(centre + offset * (1 + k * step / radius)).astype(jnp.int32)
        inside = jnp.all((position >= 0) & (position < shape), axis=0)
        index = tuple(jnp.where(inside, p, 0) for p in position)
        return filled | (inside & region[index])

    # the centre lies on every ray, and has no direction of its own to march along
    start = region.at[tuple(n // 2 for n in region.shape)].set(region.any())
    return jax.lax.fori_loop(1, steps + 1, further, start)


def centre_masked(mask: jnp.ndarray) -> bool:
    """Whether the zero-frequency pixel, at index ``n // 2`` per axis, is masked.

    Only that one pixel is checked: a beamstop that misses it, through a
    sub-pixel offset of the centre, falls back to the threshold rule.
    """
    return bool(mask[tuple(n // 2 for n in mask.shape)])
