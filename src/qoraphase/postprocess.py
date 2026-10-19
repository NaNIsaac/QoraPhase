"""Resolving the CDI ambiguities after a reconstruction.

A reconstruction is determined only up to a global phase, a translation, and a
conjugate flip.  Nothing during iteration can fix that -- all three are exact
symmetries of the constraints -- so it is dealt with here, once, on concrete
arrays.  These run on the host and are not part of any traced loop.

Phase-ramp removal is deliberately absent.  A linear phase across the object
comes from the Bragg peak sitting off the array centre, so removing it properly
needs the diffraction geometry -- which arrives with BCDI in v0.3.  Anything
short of that is a guess dressed up as a correction.
"""

import jax.numpy as jnp
import numpy as np


def flip_about_origin(x: jnp.ndarray) -> jnp.ndarray:
    """Reflect ``r -> -r`` for a center-origin array.

    Reversing the axes is not enough: with the origin at index ``n // 2`` the
    reflection maps ``i`` to ``(n - i) % n``, i.e. the reversal rolled by one.
    """
    return jnp.roll(jnp.flip(x), 1, axis=tuple(range(x.ndim)))


def twin(x: jnp.ndarray) -> jnp.ndarray:
    """The conjugate-flipped solution, indistinguishable from ``x`` in the data."""
    return jnp.conj(flip_about_origin(x))


def _cross_correlation(obj, reference):
    return jnp.fft.fftn(obj) * jnp.conj(jnp.fft.fftn(reference))


def _evaluate(spectrum, points):
    """Cross-correlation at arbitrary positions, by a matrix-multiply DFT.

    *spectrum* is the product ``F(obj) conj(F(reference))`` in FFT order and
    *points* holds, per axis, the (fractional) shifts to evaluate.  At integer
    shifts this equals ``ifftn(spectrum)``.  Signed frequencies keep the
    interpolation symmetric about zero.
    """
    out = spectrum
    for axis, shifts in enumerate(points):
        freq = jnp.fft.fftfreq(spectrum.shape[axis])
        kernel = jnp.exp(2j * jnp.pi * shifts[:, None] * freq[None, :])
        out = jnp.moveaxis(jnp.tensordot(kernel, out, axes=(1, axis)), 0, axis)
    return out / spectrum.size


def register(obj, reference, upsample: int = 1):
    """Shift and peak value that best match *obj* onto *reference*.

    Returns ``(shift, value)``: ``obj`` is *reference* displaced by ``+shift``
    (in pixels), and ``value`` is the complex correlation at that shift, whose
    angle is the global phase between the two.

    The whole-pixel peak comes from the FFT cross-correlation.  With
    ``upsample > 1`` it is then refined in decades around the peak, evaluating
    the correlation only on a small grid by matrix-multiply DFT (Guizar-Sicairos
    et al., Opt. Lett. 33, 156, 2008), down to a step of ``1 / upsample`` pixel
    or finer.
    """
    spectrum = _cross_correlation(obj, reference)
    correlation = jnp.fft.ifftn(spectrum)
    peak = jnp.unravel_index(jnp.argmax(jnp.abs(correlation)), correlation.shape)
    shift = np.array([int(p) for p in peak], dtype=float)
    # The peak index is in [0, n); report the equivalent signed shift.
    shift = np.where(shift > np.array(obj.shape) // 2, shift - np.array(obj.shape), shift)
    value = correlation[peak]

    step = 1.0
    while step * upsample > 1.0 + 1e-9:
        step /= 10
        offsets = np.arange(-10, 11) * step
        grid = [jnp.asarray(s + offsets) for s in shift]
        local = _evaluate(spectrum, grid)
        best = np.unravel_index(int(jnp.argmax(jnp.abs(local))), local.shape)
        shift = np.array([s + offsets[b] for s, b in zip(shift, best)])
        value = local[best]
    return tuple(float(s) for s in shift), value


def _fourier_shift(x, shift):
    """Displace *x* by *shift* pixels (periodic, any real values)."""
    spectrum = jnp.fft.fftn(x)
    for axis, delta in enumerate(shift):
        freq = jnp.fft.fftfreq(x.shape[axis])
        shape = [1] * x.ndim
        shape[axis] = -1
        spectrum = spectrum * jnp.exp(-2j * jnp.pi * delta * freq).reshape(shape)
    return jnp.fft.ifftn(spectrum)


def _register_and_apply(obj, reference, upsample):
    shift, value = register(obj, reference, upsample)
    whole = tuple(round(s) for s in shift)
    moved = jnp.roll(obj, tuple(-w for w in whole), axis=tuple(range(obj.ndim)))
    fraction = tuple(s - w for s, w in zip(shift, whole))
    if any(fraction):
        moved = _fourier_shift(moved, tuple(-f for f in fraction))
    return moved * jnp.exp(-1j * jnp.angle(value)), jnp.abs(value)


def align_to(obj: jnp.ndarray, reference: jnp.ndarray, upsample: int = 1) -> jnp.ndarray:
    """Transform *obj* into the equivalent solution closest to *reference*.

    The translation and global phase come from the cross-correlation peak, and
    the branch is chosen by taking whichever of ``obj`` and ``twin(obj)``
    correlates more strongly.

    This is a *relative* operation, and the twin part of it especially so: it
    adopts whatever branch *reference* is in.  It does not, and cannot, tell you
    which branch is physically correct -- both fit the measurement exactly.  What
    it does give you is a common representation, which is all that averaging an
    ensemble needs.

    With the default ``upsample=1`` the translation is to whole pixels.  A larger
    value locates the peak to about ``1 / upsample`` pixel (see :func:`register`)
    and applies the remainder as a Fourier shift, which is exact only for an
    object that does not reach the array edge.
    """
    candidates = [
        _register_and_apply(obj, reference, upsample),
        _register_and_apply(twin(obj), reference, upsample),
    ]
    return max(candidates, key=lambda c: float(c[1]))[0]
