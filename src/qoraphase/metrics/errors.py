"""Error metrics (spec 10.3)."""

import jax.numpy as jnp

EPS = 1e-12


def fourier_error(
    computed_magnitude: jnp.ndarray,
    amplitude: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Normalised Fourier-space error over unmasked pixels only.

        E_F = sqrt( sum_valid (|F(obj)| - amplitude)^2 / sum_valid amplitude^2 )

    Takes the already-computed magnitude rather than the object, so that
    ModulusProjection can hand over the FFT it has already paid for
    (spec 10.3 -- computing this separately would double the FFT cost).
    """
    valid = ~mask
    num = jnp.sum(jnp.where(valid, (computed_magnitude - amplitude) ** 2, 0.0))
    den = jnp.sum(jnp.where(valid, amplitude**2, 0.0))
    return jnp.sqrt(num / (den + EPS))
