"""Plain NumPy reference of the iterative core, for checking other implementations.

:func:`run` executes a recipe of ER, HIO, RAAR, Shrinkwrap, ``repeat`` and ``detwin`` with the
same equations as the JAX stages, from a given starting object, in any complex dtype. It is
deliberately written without JAX and without the phase-ramp FFT of :class:`~qoraphase.FFT`, so
that agreement with it says something about both.

Phase retrieval iterations are not stable under rounding: HIO and RAAR amplify a 1e-7
difference to 1e-1 within about 50 iterations (complex64 against complex128 of this very
reference does the same). Compare iteration by iteration only over a few steps, and use
:func:`floor`, the distance between the complex64 and complex128 runs of this reference, as the
yardstick for what "the same up to single-precision rounding" means for a given case.

Partial coherence, low-pass filtering and phase limits are not covered.
"""

import numpy as np

from .algorithms.er import ER
from .algorithms.hio import HIO
from .algorithms.raar import RAAR
from .operators.shrinkwrap import Shrinkwrap
from .recipes.recipe import Detwin, Recipe, Repeat

EPS = 1e-12


def forward(obj):
    """Unitary centered transform with explicit shifts."""
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))


def adjoint(wave):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(wave), norm="ortho"))


def fourier_error(magnitude, amplitude, mask):
    valid = ~mask
    num = np.sum(np.where(valid, (magnitude - amplitude) ** 2, 0.0))
    den = np.sum(np.where(valid, amplitude**2, 0.0))
    return np.sqrt(num / (den + EPS))


def modulus_project(obj, amplitude, mask):
    """``(P_M obj, Fourier error of obj)``; masked pixels keep their computed magnitude."""
    wave = forward(obj)
    magnitude = np.abs(wave)
    target = np.where(mask, magnitude, amplitude)
    projected = adjoint(target * wave / (magnitude + EPS)).astype(obj.dtype)
    return projected, fourier_error(magnitude, amplitude, mask)


def support_project(obj, support):
    return np.where(support, obj, 0).astype(obj.dtype)


def gaussian_blur(array, sigma):
    spectrum = np.fft.fftn(array)
    for axis, n in enumerate(array.shape):
        kernel = np.exp(-2.0 * (np.pi * sigma * np.fft.fftfreq(n)) ** 2)
        spectrum = spectrum * np.expand_dims(
            kernel, tuple(i for i in range(array.ndim) if i != axis)
        )
    return np.real(np.fft.ifftn(spectrum))


def shrinkwrap(obj, support, stage):
    blurred = gaussian_blur(np.abs(obj), stage.sigma)
    if stage.method == "max":
        level = blurred.max()
    else:
        level = np.sqrt(np.sum(np.where(support, blurred, 0.0) ** 2) / max(support.sum(), 1))
    return blurred > stage.threshold * level


def half_support(support, axis=0):
    others = tuple(i for i in range(support.ndim) if i != axis)
    coordinate = np.arange(support.shape[axis])
    weight = support.sum(axis=others)
    centre = np.sum(coordinate * weight) / max(weight.sum(), 1)
    return support & np.expand_dims(coordinate < centre, others)


def _step(stage, obj, amplitude, support, mask):
    projected, err = modulus_project(obj, amplitude, mask)
    match stage:
        case ER():
            obj = support_project(projected, support)
        case HIO():
            obj = np.where(support, projected, obj - stage.beta * projected)
        case RAAR():
            reflected = 2 * projected - obj
            obj = 0.5 * stage.beta * (2 * support_project(reflected, support) - reflected + obj)
            obj = obj + (1 - stage.beta) * projected
    return obj.astype(projected.dtype), err


def _run(stage, obj, amplitude, support, mask, errors):
    match stage:
        case Recipe():
            for inner in stage.stages:
                obj, support = _run(inner, obj, amplitude, support, mask, errors)
        case Repeat():
            for _ in range(stage.n):
                obj, support = _run(stage.body, obj, amplitude, support, mask, errors)
        case Detwin():
            cut = half_support(support, stage.axis)
            obj, _ = _run(stage.body, obj, amplitude, cut, mask, errors)
        case Shrinkwrap():
            support = shrinkwrap(obj, support, stage)
        case ER() | HIO() | RAAR():
            for _ in range(stage.n):
                obj, err = _step(stage, obj, amplitude, support, mask)
                errors.append(err)
        case _:
            raise TypeError(f"{type(stage).__name__} has no NumPy reference")
    return obj, support


def run(recipe, intensity, support, init, *, mask=None, dtype=np.complex128):
    """Run *recipe* from *init* (cut to *support*) in *dtype*.

    Returns ``(object, support, errors)`` like the fields of a
    :class:`~qoraphase.Reconstruction`, with one Fourier error per iteration.
    """
    real = np.finfo(dtype).dtype
    amplitude = np.sqrt(np.clip(np.asarray(intensity, real), 0, None))
    support = np.asarray(support, bool)
    mask = np.zeros(amplitude.shape, bool) if mask is None else np.asarray(mask, bool)
    obj = support_project(np.asarray(init, dtype), support)
    errors = []
    obj, support = _run(recipe, obj, amplitude, support, mask, errors)
    return obj, support, np.asarray(errors, real)


def relative_distance(a, reference):
    """``||a - reference|| / ||reference||`` in float64."""
    a, reference = np.asarray(a, np.complex128), np.asarray(reference, np.complex128)
    return float(np.linalg.norm(a - reference) / np.linalg.norm(reference))


def floor(recipe, intensity, support, init, *, mask=None):
    """Distance of the complex64 run of this reference from its complex128 run.

    Single-precision rounding alone puts a correct complex64 implementation about this far
    from the complex128 result for this recipe, data and start.
    """
    exact = run(recipe, intensity, support, init, mask=mask)[0]
    single = run(recipe, intensity, support, init, mask=mask, dtype=np.complex64)[0]
    return relative_distance(single, exact)
