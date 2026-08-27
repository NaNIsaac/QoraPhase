"""Stage protocol and declarative Recipe (spec 11, 12).

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

    ``n`` is the scan length and therefore static: changing it recompiles
    (spec 13.1).
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


def repeat(n: int, *stages) -> Repeat:
    """``qp.repeat(10, qp.RAAR(50), qp.Shrinkwrap())``"""
    return Repeat(n, Recipe(*stages))


jax.tree_util.register_dataclass(Recipe, data_fields=["stages"], meta_fields=[])
jax.tree_util.register_dataclass(Repeat, data_fields=["body"], meta_fields=["n"])
