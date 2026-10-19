"""Stage protocol and declarative Recipe.

Code order is execution order.  No operator-overloading DSL.
"""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax

NO_METRICS = jnp.zeros((0,), jnp.float32)


class Stage:
    """Anything a Recipe can run.

    ``run(state, problem) -> (new_state, metrics)`` where ``metrics`` is a 1-D
    float array with one entry per iteration that produced one.  Stages that
    produce no error metric (Shrinkwrap) return an empty array.
    """

    def run(self, state, problem):
        raise NotImplementedError


@dataclass(frozen=True)
class Iterative(Stage):
    """A stage that repeats ``step`` ``n`` times under ``lax.scan``.

    ``n`` is the scan length and therefore static: changing it recompiles.
    """

    n: int

    def step(self, state, problem):
        raise NotImplementedError

    def run(self, state, problem):
        return lax.scan(
            lambda s, _: self.step(s, problem), state, None, length=self.n
        )


@dataclass(frozen=True, init=False)
class Recipe(Stage):
    """Ordered sequence of stages."""

    stages: tuple

    def __init__(self, *stages, **kwargs):
        # kwargs path is how pytree unflattening rebuilds us.
        object.__setattr__(self, "stages", tuple(kwargs.pop("stages", stages)))
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")

    def __len__(self) -> int:
        return len(self.stages)

    def run(self, state, problem):
        metrics = []
        for stage in self.stages:
            state, m = stage.run(state, problem)
            metrics.append(m)
        return state, jnp.concatenate(metrics) if metrics else NO_METRICS


@dataclass(frozen=True)
class Repeat(Stage):
    """Run ``body`` ``n`` times."""

    n: int
    body: Recipe

    def run(self, state, problem):
        state, metrics = lax.scan(
            lambda s, _: self.body.run(s, problem), state, None, length=self.n
        )
        return state, metrics.reshape(-1)


@dataclass(frozen=True)
class Detwin(Stage):
    """Run ``body`` against a support cut in half, then restore the support.

    The twin ``conj(obj(-r))`` fits the measurement exactly, so nothing in the
    data prefers one branch; a centrosymmetric support does not separate them
    either and the iteration stalls in a mixture.  Halving the support removes
    that symmetry for a few cycles and biases the solution into one branch, after
    which the full support is restored.

    Which branch it lands in follows from the cut, not from the data.
    """

    body: Recipe
    axis: int = 0

    def run(self, state, problem):
        from ..operators.support import half_support

        cut = half_support(state.support, self.axis)
        after, metrics = self.body.run(state._replace(support=cut), problem)
        return after._replace(support=state.support), metrics


def repeat(n: int, *stages) -> Repeat:
    """``qp.repeat(10, qp.RAAR(50), qp.Shrinkwrap())``"""
    return Repeat(n, Recipe(*stages))


def detwin(*stages, axis: int = 0) -> Detwin:
    """``qp.detwin(qp.HIO(20))`` -- break the twin symmetry, then carry on."""
    return Detwin(Recipe(*stages), axis)


jax.tree_util.register_dataclass(Recipe, data_fields=["stages"], meta_fields=[])
jax.tree_util.register_dataclass(Repeat, data_fields=["body"], meta_fields=["n"])
jax.tree_util.register_dataclass(Detwin, data_fields=["body"], meta_fields=["axis"])
