"""Error Reduction."""

from dataclasses import dataclass

import jax

from ..operators.modulus import modulus_project
from ..operators.support import support_project
from ..recipes.recipe import Iterative


@dataclass(frozen=True)
class ER(Iterative):
    """``x <- P_S P_M x``.  Monotone in the Fourier error, and slow."""

    def step(self, state, problem):
        projected, err = modulus_project(state.obj, problem, state.coherence, state.sigma)
        obj = support_project(projected, state.support)
        return state._replace(obj=obj, iteration=state.iteration + 1), err


jax.tree_util.register_dataclass(ER, data_fields=[], meta_fields=["n"])
