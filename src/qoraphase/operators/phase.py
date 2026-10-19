"""Phase constraint: confine the object's phase to a range during a run."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from ..recipes.recipe import Detwin, Iterative, Recipe, Repeat, Stage


@dataclass(frozen=True)
class PhaseLimited(Stage):
    """Run *algorithm* with its support narrowed, at every iteration, to the
    pixels whose current phase lies in ``[low, high]`` (radians).

    The narrowed support is used for that iteration only; the stored support
    is left alone, so shrinkwrap and later stages see the full one.  Phase is
    measured from zero, so a range around zero also pins the global phase.
    This is cohere's ``phc`` and the "phase constraint" of the legacy MATLAB
    code; a weakly strained crystal starts close to real, and keeping it there
    early on discourages solutions with wild phase.
    """

    algorithm: Iterative
    low: float
    high: float

    def run(self, state, problem):
        def step(s, _):
            phase = jnp.angle(s.obj)
            allowed = s.support & (phase >= self.low) & (phase <= self.high)
            after, err = self.algorithm.step(s._replace(support=allowed), problem)
            return after._replace(support=s.support), err

        return lax.scan(step, state, None, length=self.algorithm.n)


def _limit(stage, low, high):
    match stage:
        case Iterative():
            return PhaseLimited(stage, low, high)
        case Recipe():
            return Recipe(*(_limit(s, low, high) for s in stage.stages))
        case Repeat():
            return Repeat(stage.n, _limit(stage.body, low, high))
        case Detwin():
            return Detwin(_limit(stage.body, low, high), stage.axis)
    return stage


def phase_limit(low: float, high: float, *stages) -> Recipe:
    """``qp.phase_limit(-0.5, 0.5, qp.repeat(10, qp.HIO(50), qp.Shrinkwrap()))``

    Every iterative algorithm inside *stages*, however deeply nested, runs
    under :class:`PhaseLimited`; everything else runs unchanged.
    """
    if not low < high:
        raise ValueError(f"need low < high, got {low} and {high}")
    return _limit(Recipe(*stages), low, high)


jax.tree_util.register_dataclass(
    PhaseLimited, data_fields=["algorithm", "low", "high"], meta_fields=[]
)
