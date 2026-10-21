"""Low-pass resolution progression: fit blurred data first, the data itself last.

Early iterations see a Gaussian-smoothed version of the measured amplitude, and
the blur narrows to the raw data over the run.  This is cohere's
``lowpass_filter`` (``features.LowPassFilter``): the filter is applied to the
*data*, not to the computed wave, and the sigma ramps linearly with the
iteration number.

Why the data and not the wave: the modulus constraint is a projection onto
"amplitude equals the target".  Smoothing the target gives a different, smoother
constraint set that every algorithm projects onto unchanged, and the iterate is
never filtered, so nothing but the target differs between filtered and
unfiltered iterations.  Smoothing the wave instead would put a lossy operator in
the middle of the iteration, and it would not be a projection.

A Gaussian blur of the amplitude in the detector is a Gaussian envelope on the
object in real space.  Speckle contrast drops and the large-scale structure of
the pattern survives, so the early iterations are asked to explain less.  The
blur is in detector pixels, so a good sigma depends on the fringe spacing,
roughly the array size divided by the object size.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..recipes.recipe import Detwin, Iterative, Recipe, Repeat, Stage
from .shrinkwrap import gaussian_blur

EPS = 1e-6


def filtered_amplitude(problem, sigma) -> jnp.ndarray:
    """The measured amplitude blurred to *sigma* pixels, ignoring masked pixels.

    Normalised convolution: masked pixels (beamstop, hot pixels) are left out of
    the average instead of dragging their stored value into their neighbours.
    Where no unmasked pixel is in reach the measured value is kept.
    """
    valid = (~problem.mask).astype(problem.amplitude.dtype)
    weight = gaussian_blur(valid, sigma)
    blurred = gaussian_blur(problem.amplitude * valid, sigma) / jnp.maximum(weight, EPS)
    return jnp.where(weight > EPS, blurred, problem.amplitude)


def _length(stage) -> int:
    """Number of iterations *stage* runs, counted from its structure."""
    match stage:
        case Iterative():
            return stage.n
        case Recipe():
            return sum(_length(s) for s in stage.stages)
        case Repeat():
            return stage.n * _length(stage.body)
        case Detwin():
            return _length(stage.body)
    # Wrappers from other modules (phase limits, partial coherence, this one)
    # are recognised by what they hold, which keeps this module free of them.
    if hasattr(stage, "algorithm"):
        return _length(stage.algorithm)
    if hasattr(stage, "body"):
        return _length(stage.body)
    return 0


@dataclass(frozen=True)
class LowPass(Stage):
    """Run *body* with the data blurred by a sigma that ramps from ``sigma[0]`` to
    ``sigma[1]`` over the iterations of *body*, then go back to the data.

    The schedule lives in the iteration state, so the body can be any mix of
    algorithms, shrinkwrap, detwin, phase limits and partial coherence.  Error
    metrics inside the body are against the blurred data.
    """

    body: Recipe
    sigma: tuple

    def run(self, state, problem):
        schedule = (
            state.iteration,
            jnp.float32(_length(self.body)),
            jnp.float32(self.sigma[0]),
            jnp.float32(self.sigma[1]),
        )
        state, metrics = self.body.run(state._replace(lowpass=schedule), problem)
        return state._replace(lowpass=None), metrics


def low_pass(*stages, sigma=(2.0, 0.5)) -> LowPass:
    """``qp.low_pass(qp.RAAR(100), qp.Shrinkwrap(), sigma=(1.5, 0.0))``

    *sigma* is the blur width in detector pixels at the first and at the last
    iteration of *stages*, interpolated linearly in between; a single number is a
    constant blur.  A last sigma of 0 ends on the unblurred data.  Metrics from
    these iterations are against the blurred data.  To use it in one generation
    of :func:`~qoraphase.evolve` only, give that generation its own recipe.
    """
    first, last = (sigma, sigma) if isinstance(sigma, int | float) else sigma
    if first < 0 or last < 0:
        raise ValueError(f"sigma must not be negative, got {sigma}")
    return LowPass(Recipe(*stages), (float(first), float(last)))


jax.tree_util.register_dataclass(LowPass, data_fields=["body"], meta_fields=["sigma"])
