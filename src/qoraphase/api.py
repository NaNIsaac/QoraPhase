"""Public entry point."""

from functools import partial
from typing import Literal, overload

import jax
from jax import lax

from .algorithms.er import ER
from .algorithms.hio import HIO
from .algorithms.raar import RAAR
from .core.ensemble import Ensemble
from .core.problem import CDI
from .core.result import Reconstruction
from .core.state import initial_state
from .core.validate import check_run
from .operators.shrinkwrap import Shrinkwrap
from .recipes.recipe import Recipe, detwin, repeat


def default_recipe() -> Recipe:
    """RAAR, a detwin pass, then shrinkwrapped RAAR and ER.

    The detwin runs before shrinkwrap starts updating the support.  At the
    shrinkwrap settings shipped here the order makes little difference either
    way; it matters a great deal at looser settings, where a support that is
    still bloated gets cut in half and shrinkwrap -- which can only shrink --
    locks the fragment in.  Doing the cut first is simply the order with fewer
    ways to go wrong.
    """
    return Recipe(
        RAAR(50),
        detwin(HIO(20)),
        RAAR(50),
        repeat(8, RAAR(50), Shrinkwrap()),
        ER(200),
    )


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
    return runs[0] if starts == 1 else Ensemble(runs, problem)


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
        intensity: measured intensity, center-origin, 2D or 3D.
        support: boolean, ``True`` inside the object.  ``None`` derives an
            autocorrelation support.
        mask: boolean, ``True`` marks *unusable* pixels.
        recipe: ``None`` uses :func:`default_recipe`.
        starts: number of random starts; >1 returns an :class:`Ensemble`.
        batch: starts run simultaneously under ``vmap``.  ``None`` runs them all
            at once, which is fastest but can exhaust GPU memory on large 3D
            data -- lower it if you hit OOM.
        seed: PRNG seed; ``None`` means 0, not "random".
    """
    problem = CDI(
        intensity,
        support=support,
        mask=mask,
        model=model,
        support_threshold=support_threshold,
    )
    return run(problem, recipe, starts=starts, batch=batch, seed=seed)
