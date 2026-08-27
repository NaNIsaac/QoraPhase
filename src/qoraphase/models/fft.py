"""Forward models: object space <-> detector space."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class FFT:
    """Unitary centered FFT (spec 9).

    All public arrays are center-origin (spec 4.1): index ``n // 2`` is the
    origin.  The ``ifftshift`` / ``fftshift`` sandwich is entirely internal, so
    users never call ``fftshift`` themselves.

    ``adjoint(forward(x)) == x`` because ``norm="ortho"`` makes the transform
    unitary.
    """

    def forward(self, obj: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(obj), norm="ortho"))

    def adjoint(self, wave: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(wave), norm="ortho"))
