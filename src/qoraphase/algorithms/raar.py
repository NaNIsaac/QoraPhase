"""Relaxed Averaged Alternating Reflections."""

from dataclasses import dataclass

import jax

from ..operators.modulus import modulus_project
from ..operators.support import support_project
from ..recipes.recipe import Iterative


@dataclass(frozen=True)
class RAAR(Iterative):
    """``x <- (beta/2)(R_S R_M x + x) + (1 - beta) P_M x``

    with ``R_M = 2 P_M - I`` and ``R_S = 2 P_S - I``.  ``beta -> 1`` is pure
    reflection (fast, unstable), ``beta -> 0`` degenerates to modulus projection.
    """

    beta: float = 0.9

    def step(self, state, problem):
        x = state.obj
        pm, err = modulus_project(x, problem, state.coherence, state.sigma)
        rm = 2.0 * pm - x
        rs_rm = 2.0 * support_project(rm, state.support) - rm
        obj = 0.5 * self.beta * (rs_rm + x) + (1.0 - self.beta) * pm
        return state._replace(obj=obj, iteration=state.iteration + 1), err


jax.tree_util.register_dataclass(RAAR, data_fields=["beta"], meta_fields=["n"])
