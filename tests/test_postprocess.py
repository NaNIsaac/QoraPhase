"""Resolving the CDI ambiguities after the fact."""

import jax.numpy as jnp
import numpy as np
import pytest

import qoraphase as qp
from conftest import complex_similarity, flip_about_origin, make_data, make_object

RECIPE = qp.Recipe(qp.RAAR(400, beta=0.9), qp.ER(200))


# --------------------------------------------------------------------------
# align_to
# --------------------------------------------------------------------------


def test_alignment_leaves_an_already_aligned_object_alone():
    obj = jnp.asarray(make_object())
    assert np.allclose(qp.align_to(obj, obj), obj, atol=1e-5)


def test_alignment_undoes_a_translation():
    obj = jnp.asarray(make_object())
    shifted = jnp.roll(obj, (3, -2), axis=(0, 1))
    assert np.allclose(qp.align_to(shifted, obj), obj, atol=1e-5)


def test_alignment_undoes_a_global_phase():
    obj = jnp.asarray(make_object())
    rotated = obj * jnp.exp(1.1j)
    assert np.allclose(qp.align_to(rotated, obj), obj, atol=1e-5)


def test_alignment_undoes_a_twin():
    obj = jnp.asarray(make_object())
    twin = jnp.conj(jnp.asarray(flip_about_origin(np.asarray(obj))))
    assert np.allclose(qp.align_to(twin, obj), obj, atol=1e-5)


def test_alignment_undoes_all_three_at_once():
    obj = jnp.asarray(make_object())
    twin = jnp.conj(jnp.asarray(flip_about_origin(np.asarray(obj))))
    mangled = jnp.roll(twin, (-4, 5), axis=(0, 1)) * jnp.exp(-2.3j)
    assert np.allclose(qp.align_to(mangled, obj), obj, atol=1e-5)


def test_alignment_reconciles_two_independent_runs():
    """Two starts that both converged disagree only by the ambiguities."""
    obj, support, intensity = make_data(m=10, seed=3)
    ensemble = qp.reconstruct(intensity, support=support, recipe=RECIPE, starts=4)
    reference = ensemble[0].object

    for run in ensemble[1:]:
        assert complex_similarity(qp.align_to(run.object, reference), reference) > 0.99


# --------------------------------------------------------------------------
# Ensemble.align / Ensemble.average
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def converged():
    obj, support, intensity = make_data(m=10, seed=3)
    ensemble = qp.reconstruct(intensity, support=support, recipe=RECIPE, starts=6)
    return obj, ensemble


def test_align_makes_every_run_agree(converged):
    _, ensemble = converged
    aligned = ensemble.align()
    reference = aligned.best.object

    assert isinstance(aligned, qp.Ensemble)
    assert len(aligned) == len(ensemble)
    for run in aligned:
        assert complex_similarity(run.object, reference) > 0.99


def test_average_recovers_the_object(converged):
    obj, ensemble = converged
    averaged = ensemble.average()

    assert isinstance(averaged, qp.Reconstruction)
    assert complex_similarity(averaged.object, obj) > 0.95
    assert averaged.error < 1e-2


def test_averaging_without_alignment_cancels_the_runs(converged):
    """The reason align() exists.

    Each run carries its own global phase, so a raw mean interferes
    destructively and bleeds away most of the amplitude -- here about two
    thirds of it.  The direction survives, because a sum of phase-rotated
    copies is still a multiple of the object, which is exactly why a
    similarity score alone would not catch this.
    """
    _, ensemble = converged
    scale = np.mean([np.linalg.norm(np.asarray(run.object)) for run in ensemble])

    naive = sum(np.asarray(run.object) for run in ensemble) / len(ensemble)
    aligned = np.asarray(ensemble.average().object)

    assert np.linalg.norm(naive) / scale < 0.7
    assert np.linalg.norm(aligned) / scale > 0.95


def test_average_without_a_problem_reports_no_error():
    """Hand-built ensembles have no problem attached, so there is no metric."""
    runs = tuple(
        qp.Reconstruction(
            object=jnp.ones((4, 4), complex),
            support=jnp.ones((4, 4), bool),
            metrics=jnp.asarray([0.5]),
        )
        for _ in range(2)
    )
    averaged = qp.Ensemble(runs).average()
    assert averaged.metrics.size == 0
    assert np.isnan(averaged.error)
