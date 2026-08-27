"""Public entry point (spec 6, 13, 21)."""

from functools import partial
from typing import Literal, overload

import jax
from jax import lax

from .algorithms.er import ER
from .algorithms.raar import RAAR
from .core.ensemble import Ensemble
from .core.problem import CDI
from .core.result import Reconstruction
from .core.state import initial_state
from .core.validate import check_run
from .operators.shrinkwrap import Shrinkwrap
from .recipes.recipe import Recipe, repeat


def default_recipe() -> Recipe:
    """Shrinkwrapped RAAR followed by ER (spec 6)."""
    return Recipe(repeat(10, RAAR(50), Shrinkwrap()), ER(200))


def _run_single(problem, recipe, key):
    state, metrics = recipe.run(initial_state(key, problem), problem)
    return state.obj, state.support, metrics


@partial(jax.jit, static_argnames=("batch",))
def _run_many(problem, recipe, keys, batch):
    fan = jax.vmap(_run_single, in_axes=(None, None, 0))
    if batch is None:
        return fan(problem, recipe, keys)
    starts = keys.shape[0]
    chunked = lax.map(
        lambda chunk: fan(problem, recipe, chunk),
        keys.reshape(starts // batch, batch),
    )
    return jax.tree.map(lambda a: a.reshape(starts, *a.shape[2:]), chunked)


def run(problem: CDI, recipe=None, *, starts=1, batch=None, seed=None):
    """Execute *recipe* on *problem*.  See :func:`reconstruct`."""
    recipe = default_recipe() if recipe is None else recipe
    check_run(recipe, starts, batch)

    keys = jax.random.split(jax.random.key(0 if seed is None else seed), starts)
    objects, supports, metrics = _run_many(problem, recipe, keys, batch)
    runs = tuple(
        Reconstruction(objects[i], supports[i], metrics[i]) for i in range(starts)
    )
    return runs[0] if starts == 1 else Ensemble(runs)


@overload
def reconstruct(intensity, *, starts: Literal[1] = 1, **kwargs) -> Reconstruction: ...
@overload
def reconstruct(intensity, *, starts: int, **kwargs) -> Ensemble: ...


def reconstruct(
    intensity,
    *,
    support=None,
    mask=None,
    model=None,
    recipe=None,
    starts: int = 1,
    batch: int | None = None,
    seed: int | None = None,
    support_threshold: float = 0.1,
) -> Reconstruction | Ensemble:
    """Array in, reconstruction out.

    Args:
        intensity: measured intensity, center-origin (spec 4.1), 2D or 3D.
        support: boolean, ``True`` inside the object.  ``None`` derives an
            autocorrelation support (spec 7).
        mask: boolean, ``True`` marks *unusable* pixels (spec 4.3).
        recipe: ``None`` uses :func:`default_recipe`.
        starts: number of random starts; >1 returns an :class:`Ensemble`.
        batch: starts run simultaneously under ``vmap``.  ``None`` runs them all
            at once, which is fastest but can exhaust GPU memory on large 3D
            data -- lower it if you hit OOM (spec 13.2).
        seed: PRNG seed; ``None`` means 0, not "random" (spec 16).
    """
    problem = CDI(
        intensity,
        support=support,
        mask=mask,
        model=model,
        support_threshold=support_threshold,
    )
    return run(problem, recipe, starts=starts, batch=batch, seed=seed)
