"""Modulus (Fourier magnitude) projection."""

from dataclasses import dataclass

import jax.numpy as jnp

from ..metrics.errors import fourier_error
from .coherence import blurred_magnitude
from .lowpass import filtered_amplitude

EPS = 1e-12


def modulus_project(obj: jnp.ndarray, problem, coherence=None, sigma=None):
    """Project onto the measured-magnitude set.

    Returns ``(projected_obj, fourier_error)``.  The error comes for free from
    the forward transform this projection already computed; a separate metric
    operator would double the FFT cost.

    Masked pixels keep their *computed* magnitude -- they are unconstrained, not
    zero.  With a *coherence* kernel the measurement is compared with the
    blurred model intensity, and the wave is rescaled by the same ratio.  With a
    *sigma* the measured amplitude is first low-passed to that width (see
    :func:`~qoraphase.low_pass`), and the error is against the filtered data.
    """
    amplitude = problem.amplitude if sigma is None else filtered_amplitude(problem, sigma)
    wave = problem.model.forward(obj)
    magnitude = blurred_magnitude(wave, coherence)
    err = fourier_error(magnitude, amplitude, problem.mask)
    target = jnp.where(problem.mask, magnitude, amplitude)
    return problem.model.adjoint(target * wave / (magnitude + EPS)), err


@dataclass(frozen=True)
class ModulusProjection:
    """Public handle for :func:`modulus_project`."""

    def __call__(self, obj, problem):
        return modulus_project(obj, problem)
