"""Public projection handles and plotting."""

import sys

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_data

import qoraphase as qp
from qoraphase.operators.modulus import modulus_project
from qoraphase.operators.support import support_project


def test_modulus_projection_handle_matches_the_function(data, problem):
    obj, _, _ = data
    start = jnp.asarray(obj) * 0.3 + 0.2
    handle_obj, handle_err = qp.ModulusProjection()(start, problem)
    plain_obj, plain_err = modulus_project(start, problem)
    assert np.array_equal(np.asarray(handle_obj), np.asarray(plain_obj))
    assert float(handle_err) == float(plain_err)


def test_support_projection_handle_matches_the_function(data):
    obj, support, _ = data
    start, support = jnp.asarray(obj) + 1.0, jnp.asarray(support)
    assert np.array_equal(
        np.asarray(qp.SupportProjection()(start, support)),
        np.asarray(support_project(start, support)),
    )


def test_every_advertised_symbol_is_importable():
    """The public surface is what ``import qoraphase as qp`` gives you."""
    for name in qp.__all__:
        assert getattr(qp, name) is not None


def test_core_runs_without_matplotlib_and_plot_names_extra(data, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    _, support, intensity = data
    result = qp.reconstruct(intensity, support=support, recipe=qp.Recipe(qp.ER(5)))
    assert result.amplitude.shape == intensity.shape

    with pytest.raises(ImportError, match=r"qoraphase\[plot\]"):
        result.plot()


def test_plot_returns_two_panels_in_2d(data):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    _, support, intensity = data
    result = qp.reconstruct(intensity, support=support, recipe=qp.Recipe(qp.ER(5)))
    axes = result.plot()
    assert len(axes) == 2
    assert [ax.get_title() for ax in axes] == ["amplitude", "phase"]


def test_plot_returns_central_slices_in_3d():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    _, support, intensity = make_data(shape=(12, 12, 12), m=5, seed=2)
    result = qp.reconstruct(intensity, support=support, recipe=qp.Recipe(qp.ER(5)))
    axes = result.plot()
    assert len(axes) == 6
    assert all(ax.images[0].get_array().shape == (12, 12) for ax in axes)
