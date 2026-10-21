"""Forward models: object space <-> detector space."""

from dataclasses import dataclass
from functools import cache, reduce
from operator import mul

import jax.numpy as jnp
import numpy as np


@cache
def _axis_ramp(n: int):
    """``exp(2 pi i c j / n)`` for ``c = n // 2``, and the constant ``exp(-2 pi i c^2 / n)``.

    Rounded to 14 places so that the ramp of an even length is exactly +-1 and comes back
    real, which makes the multiplication a sign flip.
    """
    c = n // 2
    ramp = np.round(np.exp(2j * np.pi * ((c * np.arange(n)) % n) / n), 14)
    constant = np.round(np.exp(-2j * np.pi * ((c * c) % n) / n), 14)
    return (ramp.real if not ramp.imag.any() else ramp), constant


def _ramps(shape):
    """Input and output factors of the centered transform, one array per axis."""
    ramps, constant = [], 1.0
    for axis, n in enumerate(shape):
        ramp, c = _axis_ramp(n)
        ramps.append(ramp.reshape((n,) + (1,) * (len(shape) - axis - 1)))
        constant = constant * c
    out = [ramps[0] * constant, *ramps[1:]]
    return ramps, out


def _scale(array: jnp.ndarray, factors) -> jnp.ndarray:
    return reduce(mul, factors, array)


@dataclass(frozen=True)
class FFT:
    """Unitary centered FFT.

    All public arrays are center-origin: index ``n // 2`` is the origin, and users never
    call ``fftshift`` themselves.  ``adjoint(forward(x)) == x`` because ``norm="ortho"``
    makes the transform unitary.

    ``fftshift(fftn(ifftshift(x)))`` is evaluated without moving any data.  With ``c = n // 2``
    it is ``sum_j x_j exp(-2 pi i (j - c)(k - c) / n)``, which is the plain transform of
    ``x_j exp(2 pi i c j / n)`` times ``exp(2 pi i c k / n) exp(-2 pi i c^2 / n)``.  Both
    factors are separable per axis, and for even ``n`` they are +-1, so XLA fuses them into
    the neighbouring elementwise operations instead of materialising four rolled copies.
    """

    def forward(self, obj: jnp.ndarray) -> jnp.ndarray:
        ramps, out = _ramps(obj.shape)
        return _scale(jnp.fft.fftn(_scale(obj, ramps), norm="ortho"), out)

    def adjoint(self, wave: jnp.ndarray) -> jnp.ndarray:
        ramps, out = _ramps(wave.shape)
        ramps = [np.conj(r) for r in ramps]
        out = [np.conj(r) for r in out]
        return _scale(jnp.fft.ifftn(_scale(wave, ramps), norm="ortho"), out)
