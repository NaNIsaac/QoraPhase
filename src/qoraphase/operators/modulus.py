"""Modulus (Fourier magnitude) projection."""

from dataclasses import dataclass

import jax.numpy as jnp

from ..metrics.errors import fourier_error
from .coherence import blurred_magnitude

EPS = 1e-12


def modulus_project(obj: jnp.ndarray, problem, coherence=None):
    """Project onto the measured-magnitude set.

    Returns ``(projected_obj, fourier_error)``.  The error comes for free from
    the forward transform this projection already computed; a separate metric
    operator would double the FFT cost.

    Masked pixels keep their *computed* magnitude -- they are unconstrained, not
    zero.  With a *coherence* kernel the measurement is compared with the
    blurred model intensity, and the wave is rescaled by the same ratio.
    """
    wave = problem.model.forward(obj)
    magnitude = blurred_magnitude(wave, coherence)
    err = fourier_error(magnitude, problem.amplitude, problem.mask)
    ratio = jnp.where(problem.mask, 1.0, problem.amplitude / (magnitude + EPS))
    return problem.model.adjoint(ratio * wave), err


@dataclass(frozen=True)
class ModulusProjection:
    """Public handle for :func:`modulus_project`."""

    def __call__(self, obj, problem):
        return modulus_project(obj, problem)
