"""Resolving the CDI ambiguities after a reconstruction.

A reconstruction is determined only up to a global phase, a translation, and a
conjugate flip.  Nothing during iteration can fix that -- all three are exact
symmetries of the constraints -- so it is dealt with here, once, on concrete
arrays.  These run on the host and are not part of any traced loop.

A linear phase across the object means its Fourier transform is centred off the
array centre.  :func:`remove_ramp` measures that offset from the data and cancels
it; it does not know the diffraction geometry, so it removes whatever ramp is
there, including one that carries physical information (strain, a displaced
Bragg peak).
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
        shift = np.array([s + offsets[b] for s, b in zip(shift, best, strict=False)])
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
    fraction = tuple(s - w for s, w in zip(shift, whole, strict=False))
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


def _spectral_peak(obj, pad):
    """Centre of mass of ``|F|^2`` of the zero-padded *obj*, per axis, in padded pixels."""
    shape = obj.shape
    padded_shape = tuple(pad * n for n in shape)
    before = [m // 2 - n // 2 for m, n in zip(padded_shape, shape, strict=False)]
    padded = jnp.pad(
        obj, [(b, m - n - b) for b, m, n in zip(before, padded_shape, shape, strict=False)]
    )
    intensity = jnp.abs(jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(padded)))) ** 2
    # An even-sized spectrum has a lone -m/2 sample with no +m/2 partner; left in,
    # it biases the centre of mass of even a ramp-free real object.
    for axis, m in enumerate(padded_shape):
        keep = jnp.abs(jnp.arange(m) - m // 2) < m / 2
        intensity = intensity * keep.reshape([-1 if a == axis else 1 for a in range(obj.ndim)])
    total = intensity.sum()
    return tuple(
        (
            intensity.sum(axis=tuple(a for a in range(obj.ndim) if a != axis))
            * (jnp.arange(m) - m // 2)
        ).sum()
        / total
        for axis, m in enumerate(padded_shape)
    )


def remove_ramp(obj, support=None, *, pad: int = 2, iterations: int = 3, zero_phase: bool = False):
    """Remove the linear phase ramp by centring the object's Fourier peak.

    The ramp is read off as the centre of mass of ``|F|^2`` of the zero-padded
    object (``pad`` times larger per axis, which sets how finely the offset is
    resolved), and cancelled with a phase ramp that is zero at the array centre.
    The offset is a real number, so this is a sub-pixel shift of the spectrum.
    The centre of mass is unbiased when the ramp-free spectrum is centrosymmetric
    (a real object); for a general complex object it is an estimate.

    A spectrum displaced from the centre is cut off asymmetrically by the array
    edge, which pulls the centre of mass towards zero.  Each pass removes most of
    the ramp and therefore most of that bias, so ``iterations`` passes are made.

    *support*, if given, restricts the measurement to the object (noise outside
    it would otherwise pull the peak); the returned array is not masked.  With
    ``zero_phase`` the global phase is then set so that the mean phase inside the
    support, weighted by amplitude, is zero.  Without a support the voxels above
    a tenth of the peak amplitude stand in for it.

    Traceable: usable under ``jit`` with ``pad``, ``iterations`` and
    ``zero_phase`` static.
    """
    inside = jnp.abs(obj) > 0.1 * jnp.abs(obj).max() if support is None else support
    positions = [
        (jnp.arange(n) - n // 2).reshape([-1 if a == axis else 1 for a in range(obj.ndim)])
        for axis, n in enumerate(obj.shape)
    ]

    out = obj
    for _ in range(iterations):
        peak = _spectral_peak(jnp.where(inside, out, 0), pad)
        phase = sum(q * x / (pad * n) for q, x, n in zip(peak, positions, obj.shape, strict=False))
        out = out * jnp.exp(-2j * jnp.pi * phase)

    if zero_phase:
        out = out * jnp.exp(-1j * jnp.angle(jnp.where(inside, out, 0).sum()))
    return out
