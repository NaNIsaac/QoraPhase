"""Partial coherence: the measured intensity is the coherent one blurred by a kernel.

With a partially coherent beam the detector records ``|F(obj)|^2 * K``, the
coherent intensity convolved with a coherence kernel ``K`` (Clark et al.,
Nat. Commun. 3, 993 (2012)).  Fitting a coherent model to such data fills the
missing contrast with spurious structure in the object.  Here ``K`` is estimated
during the run by Richardson-Lucy deconvolution, and the modulus constraint
compares the measurement with the blurred model instead.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from ..recipes.recipe import NO_METRICS, Detwin, Iterative, Recipe, Repeat, Stage
from .lowpass import LowPass

EPS = 1e-12


def _embed(kernel: jnp.ndarray, shape: tuple) -> jnp.ndarray:
    """A small center-origin kernel placed at the centre of a zero array of *shape*."""
    starts = tuple(n // 2 - k // 2 for n, k in zip(shape, kernel.shape))
    return lax.dynamic_update_slice(jnp.zeros(shape, kernel.dtype), kernel, starts)


def _crop(array: jnp.ndarray, size: tuple) -> jnp.ndarray:
    starts = tuple(n // 2 - k // 2 for n, k in zip(array.shape, size))
    return lax.dynamic_slice(array, starts, size)


def _flip(a: jnp.ndarray) -> jnp.ndarray:
    """``r -> -r`` about index ``n // 2``."""
    return jnp.roll(a[(slice(None, None, -1),) * a.ndim], 1, axis=tuple(range(a.ndim)))


def convolve(intensity: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Circular convolution of center-origin arrays; *kernel* may be smaller."""
    full = _embed(kernel, intensity.shape)
    shift = jnp.fft.ifftshift
    product = jnp.fft.fftn(shift(intensity)) * jnp.fft.fftn(shift(full))
    return jnp.real(jnp.fft.fftshift(jnp.fft.ifftn(product)))


def lucy_update(kernel, coherent, measured, mask, iterations: int):
    """Richardson-Lucy steps for ``measured = coherent * kernel``.

    The usual normalising sum is a constant here and drops out when the kernel
    is rescaled to unit sum after each step.  Masked pixels do not vote.
    """
    mirrored = _flip(coherent)

    def step(k, _):
        ratio = jnp.where(mask, 1.0, measured / (convolve(coherent, k) + EPS))
        k = k * _crop(convolve(ratio, mirrored), k.shape)
        return k / jnp.sum(k), None

    kernel, _ = lax.scan(step, kernel, None, length=iterations)
    return kernel


def blurred_magnitude(wave: jnp.ndarray, coherence) -> jnp.ndarray:
    """``sqrt(|wave|^2 * K)``, or ``|wave|`` when there is no kernel."""
    if coherence is None:
        return jnp.abs(wave)
    return jnp.sqrt(jnp.clip(convolve(jnp.abs(wave) ** 2, coherence), 0.0, None))


@dataclass(frozen=True)
class UpdateCoherence(Stage):
    """Re-estimate the coherence kernel from the current object.

    The model intensity is scaled to the measured total first, so the kernel
    carries only the shape of the blur.  Produces no error metric.
    """

    iterations: int = 20

    def run(self, state, problem):
        coherent = jnp.abs(problem.model.forward(state.obj)) ** 2
        valid = ~problem.mask
        measured = problem.intensity
        scale = jnp.sum(jnp.where(valid, measured, 0.0)) / (
            jnp.sum(jnp.where(valid, coherent, 0.0)) + EPS
        )
        kernel = lucy_update(
            state.coherence, coherent * scale, measured, problem.mask, self.iterations
        )
        return state._replace(coherence=kernel), NO_METRICS


@dataclass(frozen=True)
class PartialCoherence(Stage):
    """Run *body* with a coherence kernel, starting from a flat one.

    The kernel stays in the state afterwards, so later stages keep using it and
    the result reports it as ``Reconstruction.coherence``.
    """

    body: Recipe
    size: int | tuple

    def run(self, state, problem):
        if state.coherence is None:
            size = self.size
            if isinstance(size, int):
                size = (size,) * state.obj.ndim
            if len(size) != state.obj.ndim or any(
                k % 2 == 0 or k > n for k, n in zip(size, state.obj.shape)
            ):
                raise ValueError(
                    f"kernel {size} must be odd and fit inside the array {state.obj.shape}"
                )
            flat = jnp.full(size, 1.0 / math.prod(size))
            state = state._replace(coherence=flat)
        return self.body.run(state, problem)


def _with_updates(stage, iterations):
    match stage:
        case Iterative():
            return Recipe(UpdateCoherence(iterations), stage)
        case Recipe():
            return Recipe(*(_with_updates(s, iterations) for s in stage.stages))
        case Repeat():
            return Repeat(stage.n, _with_updates(stage.body, iterations))
        case Detwin():
            return Detwin(_with_updates(stage.body, iterations), stage.axis)
        case LowPass():
            return LowPass(_with_updates(stage.body, iterations), stage.sigma)
    return stage


def partial_coherence(*stages, kernel=7, lucy: int = 20) -> PartialCoherence:
    """``qp.partial_coherence(qp.repeat(8, qp.RAAR(50), qp.Shrinkwrap()), kernel=7)``

    Every iterative algorithm inside *stages* is preceded by a kernel update
    (*lucy* Richardson-Lucy steps), so the update interval is the length of the
    algorithm stages.  *kernel* is the kernel size in pixels, an odd int or one
    per axis; it must not exceed the array.  Start it after a few hundred
    coherent iterations: the first estimate comes from the current object.
    """
    if lucy < 1:
        raise ValueError(f"lucy must be at least 1, got {lucy}")
    body = _with_updates(Recipe(*stages), lucy)
    return PartialCoherence(body, kernel)


jax.tree_util.register_dataclass(UpdateCoherence, data_fields=[], meta_fields=["iterations"])
jax.tree_util.register_dataclass(PartialCoherence, data_fields=["body"], meta_fields=["size"])
