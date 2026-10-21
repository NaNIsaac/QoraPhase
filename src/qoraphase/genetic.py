"""Genetic algorithm over generations of reconstructions.

Several random starts are run, the best one is kept as the reference, every other
run is aligned to it and bred with it, and the next generation restarts from the
children instead of from random phases.  Breeding formulas and the idea follow
cohere's ``ga_utils`` / ``dvc_utils.breed``.
"""

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from .api import _run_many, default_recipe
from .core.ensemble import Ensemble
from .core.problem import CDI
from .core.result import Reconstruction
from .core.validate import check_run
from .recipes.recipe import Recipe


def _geometric(a, b):
    """Geometric mean of the amplitudes, arithmetic mean of the phases."""
    return jnp.sqrt(jnp.abs(a) * jnp.abs(b)) * jnp.exp(0.5j * (jnp.angle(a) + jnp.angle(b)))


def _geometric_phase_a(a, b):
    return jnp.sqrt(jnp.abs(a) * jnp.abs(b)) * jnp.exp(1j * jnp.angle(a))


def _arithmetic(a, b):
    return 0.5 * (a + b)


def _arithmetic_phase_a(a, b):
    return 0.5 * (jnp.abs(a) + jnp.abs(b)) * jnp.exp(1j * jnp.angle(a))


def _largest(a, b):
    return jnp.maximum(jnp.abs(a), jnp.abs(b)) * jnp.exp(0.5j * (jnp.angle(a) + jnp.angle(b)))


def _amplitude_b_phase_a(a, b):
    return jnp.abs(b) * jnp.exp(1j * jnp.angle(a))


def _clone(a, b):
    return a


# Names are cohere's, except "clone".  ``a`` is the best run, ``b`` the one being bred.
# "clone" ignores ``b``: every child is the aligned best run.  It is a control for
# separating what breeding adds from what aligning to the best run adds.
BREEDERS: dict[str, Callable] = {
    "sqrt_ab": _geometric,
    "sqrt_ab_pa": _geometric_phase_a,
    "avg_ab": _arithmetic,
    "avg_ab_pa": _arithmetic_phase_a,
    "max_ab": _largest,
    "b_pa": _amplitude_b_phase_a,
    "clone": _clone,
}


def breed(mode: str, alpha: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
    """Child of *alpha* (the best run) and *beta*, which must already be aligned.

    Alignment (position, twin branch, global phase) is what
    :meth:`Ensemble.align` does; breeding unaligned runs mixes unrelated phases.
    Both are first rotated by the same global phase so that the mean phase of
    *alpha* is zero, which keeps the phase averages away from the +-pi wrap.

    ``_pa`` modes keep the phase of *alpha* and mix only the amplitudes.
    """
    try:
        formula = BREEDERS[mode]
    except KeyError:
        raise ValueError(f"unknown breed mode {mode!r}; choose from {sorted(BREEDERS)}") from None
    rotation = jnp.exp(-1j * jnp.angle(alpha.sum()))
    return formula(alpha * rotation, beta * rotation)


_breed = breed


def _culls(cull, generations: int) -> list[int]:
    culls = [cull] * (generations - 1) if isinstance(cull, int) else list(cull)
    return culls[: generations - 1] + [0] * (generations - 1 - len(culls))


def evolve(
    problem: CDI,
    recipe=None,
    *,
    starts: int = 8,
    generations: int = 3,
    breed: str = "sqrt_ab",
    cull: int | Sequence[int] = 0,
    batch: int | None = None,
    seed: int | None = None,
    upsample: int = 10,
) -> Ensemble:
    """Run *recipe* for several generations, breeding with the best run in between.

    Generation 0 is ``starts`` random starts.  After each generation the
    ``cull[g]`` worst runs (by final error) are dropped, the rest are aligned to
    the best (:meth:`Ensemble.align`, to ``1 / upsample`` pixel) and bred with it,
    and the next generation runs *recipe* again from the children.  The best run
    is carried over unchanged.  The returned :class:`Ensemble` is the last
    generation; earlier ones are in its ``history``.

    Args:
        recipe: one recipe used for every generation, or a sequence with one per
            generation; ``None`` uses :func:`default_recipe`.
        generations: total generations, including the first.
        breed: a key of :data:`BREEDERS`.
        cull: an int (dropped after every generation) or a sequence indexed by
            generation; missing entries are 0.  The population must stay at one
            run or more, and ``batch`` must divide every generation's size.
        batch, seed: as in :func:`~qoraphase.reconstruct`.  The same seed gives
            the same result.

    Children restart on the problem's support, not on the shrinkwrapped one of
    the previous generation, and a partial-coherence kernel starts afresh.
    """
    if breed not in BREEDERS:
        raise ValueError(f"unknown breed mode {breed!r}; choose from {sorted(BREEDERS)}")
    if generations < 1:
        raise ValueError(f"generations must be >= 1, got {generations}")

    if recipe is None:
        recipe = default_recipe()
    recipes = list(recipe) if isinstance(recipe, (list, tuple)) else [recipe] * generations
    if len(recipes) != generations:
        raise ValueError(f"got {len(recipes)} recipes for {generations} generations")
    recipes = [r if isinstance(r, Recipe) else Recipe(r) for r in recipes]

    culls = _culls(cull, generations)
    sizes = [starts]
    for dropped in culls:
        sizes.append(sizes[-1] - dropped)
    if min(sizes) < 1:
        raise ValueError(f"cull leaves no runs: population sizes would be {sizes}")
    for r, size in zip(recipes, sizes, strict=True):
        check_run(r, size, batch)

    root = jax.random.key(0 if seed is None else seed)
    history: tuple = ()
    init = None
    for g in range(generations):
        # generation 0 draws exactly the keys a plain multi-start would
        keys = jax.random.split(root if g == 0 else jax.random.fold_in(root, g), sizes[g])
        objects, supports, metrics, coherence = _run_many(problem, recipes[g], keys, batch, init)
        ensemble = Ensemble(
            tuple(
                Reconstruction(
                    objects[i],
                    supports[i],
                    metrics[i],
                    None if coherence is None else coherence[i],
                )
                for i in range(sizes[g])
            ),
            problem,
            history,
        )
        if g == generations - 1:
            return ensemble

        survivors = ensemble.ranking[: sizes[g + 1]]
        kept = Ensemble(tuple(ensemble[i] for i in survivors), problem).align(upsample)
        alpha = kept[0].object
        init = jnp.stack([alpha] + [_breed(breed, alpha, run.object) for run in kept[1:]])
        history = (*history, ensemble)
