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


def flip_about_origin(x: jnp.ndarray) -> jnp.ndarray:
    """Reflect ``r -> -r`` for a center-origin array.

    Reversing the axes is not enough: with the origin at index ``n // 2`` the
    reflection maps ``i`` to ``(n - i) % n``, i.e. the reversal rolled by one.
    """
    return jnp.roll(jnp.flip(x), 1, axis=tuple(range(x.ndim)))


def twin(x: jnp.ndarray) -> jnp.ndarray:
    """The conjugate-flipped solution, indistinguishable from ``x`` in the data."""
    return jnp.conj(flip_about_origin(x))


def _register(obj, reference):
    """Best (shift, global phase) match of *obj* onto *reference*.

    Returns the transformed object and the correlation peak height, which is how
    the caller decides between a candidate and its twin.
    """
    correlation = jnp.fft.ifftn(
        jnp.fft.fftn(obj) * jnp.conj(jnp.fft.fftn(reference))
    )
    peak = jnp.unravel_index(jnp.argmax(jnp.abs(correlation)), correlation.shape)
    value = correlation[peak]
    shifted = jnp.roll(
        obj, tuple(-int(p) for p in peak), axis=tuple(range(obj.ndim))
    )
    return shifted * jnp.exp(-1j * jnp.angle(value)), jnp.abs(value)


def align_to(obj: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    """Transform *obj* into the equivalent solution closest to *reference*.

    The translation and global phase come from the cross-correlation peak, and
    the branch is chosen by taking whichever of ``obj`` and ``twin(obj)``
    correlates more strongly.

    This is a *relative* operation, and the twin part of it especially so: it
    adopts whatever branch *reference* is in.  It does not, and cannot, tell you
    which branch is physically correct -- both fit the measurement exactly.  What
    it does give you is a common representation, which is all that averaging an
    ensemble needs.

    Alignment is to whole pixels.
    """
    # Note: integer-pixel registration. Sub-pixel needs peak interpolation,
    # which only matters once results are being averaged across scans.
    candidates = [_register(obj, reference), _register(twin(obj), reference)]
    return max(candidates, key=lambda c: float(c[1]))[0]
