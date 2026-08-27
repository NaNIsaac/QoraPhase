"""Ensemble collection protocol (spec 15)."""

import jax.numpy as jnp
import pytest

from qoraphase import Ensemble, Reconstruction


def fake(error):
    return Reconstruction(
        object=jnp.zeros((2, 2), complex),
        support=jnp.ones((2, 2), bool),
        metrics=jnp.asarray([1.0, error]),
    )


def ensemble(*errors):
    return Ensemble(tuple(fake(e) for e in errors))


def test_len_and_iteration():
    e = ensemble(0.3, 0.1, 0.2)
    assert len(e) == 3
    assert [r.error for r in e] == pytest.approx([0.3, 0.1, 0.2], rel=1e-6)


def test_indexing_returns_a_reconstruction():
    assert isinstance(ensemble(0.3, 0.1)[0], Reconstruction)


def test_slicing_returns_an_ensemble():
    sliced = ensemble(0.3, 0.1, 0.2)[:2]
    assert isinstance(sliced, Ensemble)
    assert len(sliced) == 2


def test_ranking_is_best_first():
    assert ensemble(0.3, 0.1, 0.2).ranking == (1, 2, 0)


def test_best_has_the_lowest_final_error():
    assert ensemble(0.3, 0.1, 0.2).best.error == pytest.approx(0.1, rel=1e-6)
