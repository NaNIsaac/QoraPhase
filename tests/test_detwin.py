"""Detwinning by cutting the support in half.

Follows the established recipe: a short run against a halved support, placed a
quarter of the way through the iterations, then the full support is restored.
"""

import jax
import jax.numpy as jnp
import numpy as np
from conftest import complex_similarity, flip_about_origin, make_data

import qoraphase as qp
from qoraphase.core.state import initial_state
from qoraphase.operators.support import half_support

BETA = dict(beta=0.9)
PLAIN = qp.Recipe(qp.RAAR(400, **BETA), qp.ER(200))
DETWIN = qp.Recipe(
    qp.RAAR(100, **BETA),
    qp.detwin(qp.HIO(20, **BETA)),
    qp.RAAR(280, **BETA),
    qp.ER(200),
)


def converged_count(support, intensity, recipe, seeds=(0, 1, 2), starts=16):
    """How many starts reached the solution, over a fixed set of seeds."""
    total = 0
    for seed in seeds:
        ensemble = qp.reconstruct(
            intensity, support=support, recipe=recipe, starts=starts, seed=seed
        )
        total += sum(run.error < 1e-4 for run in ensemble)
    return total


# --------------------------------------------------------------------------
# half_support
# --------------------------------------------------------------------------


def test_half_support_breaks_the_centrosymmetry():
    _, support, _ = make_data(m=11, seed=3, centrosymmetric=True)
    support = jnp.asarray(support)
    cut = np.asarray(half_support(support, axis=0))

    assert np.array_equal(np.asarray(support), flip_about_origin(np.asarray(support)))
    assert not np.array_equal(cut, flip_about_origin(cut))


def test_half_support_keeps_about_half_and_nothing_new():
    _, support, _ = make_data(m=11, seed=3, centrosymmetric=True)
    support = jnp.asarray(support)
    cut = np.asarray(half_support(support, axis=0))
    original = np.asarray(support)

    assert not (cut & ~original).any()  # a subset, never an expansion
    assert 0.3 < cut.sum() / original.sum() < 0.7


def test_half_support_cuts_at_the_centre_of_mass_not_the_array_centre():
    """An off-centre object must not be wiped out by the cut."""
    support = np.zeros((32, 32), bool)
    support[20:30, 20:30] = True  # entirely below the array centre
    cut = np.asarray(half_support(jnp.asarray(support), axis=0))

    assert cut.sum() > 0
    assert 0.3 < cut.sum() / support.sum() < 0.7


def test_half_support_follows_the_axis():
    _, support, _ = make_data(m=11, seed=3, centrosymmetric=True)
    support = jnp.asarray(support)
    rows = np.asarray(half_support(support, axis=0))
    cols = np.asarray(half_support(support, axis=1))
    assert not np.array_equal(rows, cols)


# --------------------------------------------------------------------------
# the Detwin stage
# --------------------------------------------------------------------------


def test_detwin_restores_the_full_support(problem):
    """The cut is temporary; later stages must see the support they expected."""
    state = initial_state(jax.random.key(0), problem)
    after, _ = qp.detwin(qp.HIO(5)).run(state, problem)
    assert np.array_equal(np.asarray(after.support), np.asarray(state.support))


def test_detwin_confines_the_object_while_it_runs(problem):
    """Inside the stage the object really is held to half the support."""
    state = initial_state(jax.random.key(0), problem)
    cut = np.asarray(half_support(state.support, 0))
    after, _ = qp.detwin(qp.ER(5)).run(state, problem)

    outside = np.asarray(after.obj)[np.asarray(state.support) & ~cut]
    assert np.allclose(outside, 0, atol=1e-6)


def test_detwin_reports_its_body_metrics(problem):
    state = initial_state(jax.random.key(0), problem)
    _, metrics = qp.detwin(qp.HIO(7), qp.ER(3)).run(state, problem)
    assert metrics.shape == (10,)


# --------------------------------------------------------------------------
# what it buys
# --------------------------------------------------------------------------


def test_detwin_rescues_starts_on_a_centrosymmetric_support():
    """The measured effect, on the case that stalls without it.

    Deterministic, not statistical: fixed seeds, fixed starts.  The gain is real
    but modest -- roughly 10% of starts converge instead of 2% -- because a
    centrosymmetric support is a genuinely bad position to be in.  Fixing the
    support is worth far more than any amount of detwinning.
    """
    _, support, intensity = make_data(m=11, seed=3, centrosymmetric=True)
    assert converged_count(support, intensity, DETWIN) > converged_count(support, intensity, PLAIN)


def test_detwin_does_not_disturb_a_well_determined_support():
    """It must be safe to leave in the default recipe."""
    obj, support, intensity = make_data(m=10, seed=3)
    assert converged_count(support, intensity, DETWIN) == 3 * 16


def test_shrinkwrap_settings_matter_more_than_detwin_placement():
    """What actually decided the default recipe.

    The first version of this recipe failed on data it should have solved
    exactly, and the cut looked like the culprit: detwin halves the support, and
    a shrinkwrap that can only shrink then locks in the fragment.  Re-measuring
    put the blame elsewhere.  With the classic 0.2 / 2.0 shrinkwrap the support
    stays bloated and nothing converges from an autocorrelation start, whatever
    the order; at 0.3 / 1.0 it converges either way.

    Kept as a warning against the parameters, not against the ordering.
    """
    _, _, intensity = make_data(shape=(64, 64), m=12, seed=1)

    def converged(shrinkwrap):
        recipe = qp.Recipe(
            qp.RAAR(50, **BETA),
            qp.detwin(qp.HIO(20, **BETA)),
            qp.RAAR(50, **BETA),
            qp.repeat(8, qp.RAAR(50), shrinkwrap),
            qp.ER(200),
        )
        ensemble = qp.reconstruct(intensity, recipe=recipe, starts=8, seed=0)
        return sum(run.error < 1e-4 for run in ensemble)

    assert converged(qp.Shrinkwrap(0.2, 2.0)) == 0
    assert converged(qp.Shrinkwrap(0.3, 1.0)) > 0
    assert qp.Shrinkwrap() == qp.Shrinkwrap(0.3, 1.0)  # and that is the default


def test_the_default_recipe_solves_a_clean_measurement():
    """The one-liner path has to work on data that is not hard."""
    obj, support, intensity = make_data(m=10, seed=3)
    result = qp.reconstruct(intensity, support=support, starts=6)
    assert result.best.error < 1e-4
    assert complex_similarity(result.best.object, obj) > 0.99


def test_the_default_recipe_finds_the_object_without_a_support():
    """And the harder one-liner: autocorrelation support, no user input.

    Only some starts get there, which is the point of running several.
    """
    # About 1 start in 5 converges here (measured on Linux and Windows). With 8
    # starts, "none did" is a ~20% outcome and platform FFT rounding decides it;
    # 32 starts make that ~0.3%.
    obj, _, intensity = make_data(m=10, seed=3)
    ensemble = qp.reconstruct(intensity, starts=32, seed=0)
    assert ensemble.best.error < 1e-4
    assert complex_similarity(ensemble.best.object, obj) > 0.99
