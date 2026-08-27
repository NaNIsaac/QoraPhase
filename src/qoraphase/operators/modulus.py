"""Modulus (Fourier magnitude) projection (spec 4.3, 10.1, 10.3)."""

from dataclasses import dataclass

import jax.numpy as jnp

from ..metrics.errors import fourier_error

EPS = 1e-12


def modulus_project(obj: jnp.ndarray, problem):
    """Project onto the measured-magnitude set.

    Returns ``(projected_obj, fourier_error)``.  The error comes for free from
    the forward transform this projection already computed; a separate metric
    operator would double the FFT cost (spec 10.3).

    Masked pixels keep their *computed* magnitude -- they are unconstrained, not
    zero (spec 4.3).
    """
    wave = problem.model.forward(obj)
    magnitude = jnp.abs(wave)
    err = fourier_error(magnitude, problem.amplitude, problem.mask)
    target = jnp.where(problem.mask, magnitude, problem.amplitude)
    return problem.model.adjoint(target * wave / (magnitude + EPS)), err


@dataclass(frozen=True)
class ModulusProjection:
    """Public handle for :func:`modulus_project`."""

    def __call__(self, obj, problem):
        return modulus_project(obj, problem)
