"""Hybrid Input-Output (spec 10)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..operators.modulus import modulus_project
from ..recipes.recipe import Iterative


@dataclass(frozen=True)
class HIO(Iterative):
    """``x <- P_M x`` inside the support, ``x - beta P_M x`` outside."""

    beta: float = 0.9

    def step(self, state, problem):
        projected, err = modulus_project(state.obj, problem)
        obj = jnp.where(state.support, projected, state.obj - self.beta * projected)
        return state._replace(obj=obj, iteration=state.iteration + 1), err


jax.tree_util.register_dataclass(HIO, data_fields=["beta"], meta_fields=["n"])
