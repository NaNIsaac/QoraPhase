"""3D and non-cubic support, end to end."""

import numpy as np
import pytest
from conftest import make_data, similarity

import qoraphase as qp

RECIPE = qp.Recipe(qp.RAAR(150, beta=0.9), qp.ER(100))


def test_reconstructs_a_3d_object():
    obj, support, intensity = make_data(shape=(16, 16, 16), m=6, seed=2)
    ensemble = qp.reconstruct(intensity, support=support, recipe=RECIPE, starts=4)

    assert ensemble.best.object.shape == (16, 16, 16)
    assert ensemble.best.error < 1e-3
    assert similarity(ensemble.best.object, obj) > 0.95


@pytest.mark.parametrize("shape", [(24, 32), (16, 12, 20)])
def test_accepts_non_cubic_shapes(shape):
    obj, support, intensity = make_data(shape=shape, m=6, seed=2)
    result = qp.reconstruct(intensity, support=support, recipe=RECIPE)

    assert result.object.shape == shape
    assert np.isfinite(result.error)


def test_shrinkwrap_and_autosupport_work_in_3d():
    _, _, intensity = make_data(shape=(16, 16, 16), m=6, seed=2)
    recipe = qp.Recipe(qp.repeat(3, qp.RAAR(20), qp.Shrinkwrap()), qp.ER(20))
    result = qp.reconstruct(intensity, recipe=recipe)

    assert result.support.shape == (16, 16, 16)
    assert 0 < int(np.asarray(result.support).sum()) < 16**3
