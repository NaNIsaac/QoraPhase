"""The input contract in docs/input-contract.md, checked on synthetic data.

Each test pins one promise a caller (CXD, or any detector pipeline) relies on.
"""

import numpy as np
import pytest
from conftest import make_data, similarity

import qoraphase as qp

RECIPE = qp.Recipe(qp.RAAR(150, beta=0.9), qp.ER(100))


@pytest.fixture
def data3d():
    return make_data(shape=(16, 16, 16), m=6, seed=2)


def test_masked_values_do_not_change_the_result(data3d):
    """Unconstrained means unread: what sits under the mask has no effect."""
    _, support, intensity = data3d
    mask = np.zeros(intensity.shape, bool)
    mask[:, 7, :] = True  # a dead detector row through the peak
    zeros = np.where(mask, 0.0, intensity)
    junk = np.where(mask, 1e6 * np.random.default_rng(0).random(intensity.shape), intensity)

    a = qp.reconstruct(zeros, support=support, mask=mask, recipe=RECIPE, seed=1)
    b = qp.reconstruct(junk, support=support, mask=mask, recipe=RECIPE, seed=1)
    np.testing.assert_allclose(np.asarray(a.object), np.asarray(b.object), rtol=0, atol=1e-6)


def test_axis_order_carries_no_meaning(data3d):
    """Rocking-first (CXD) or rocking-last (QoraGeom): the object comes back in the same order."""
    obj, support, intensity = data3d
    order = (2, 0, 1)
    ensemble = qp.reconstruct(
        intensity.transpose(order), support=support.transpose(order), recipe=RECIPE, starts=4
    )
    assert ensemble.best.error < 1e-3
    assert similarity(ensemble.best.object, obj.transpose(order)) > 0.95


def test_a_detector_stack_round_trips_through_the_documented_steps(data3d):
    """(frame, y, x) counts with a gap -> crop to n//2, mask, zero-fill -> reconstruction."""
    obj, support, intensity = data3d
    stack = np.zeros((20, 24, 24))
    stack[1:17, 5:21, 3:19] = intensity
    gap = np.zeros(stack.shape, bool)
    gap[:, 12, :] = True
    stack[gap] = np.nan

    peak = np.unravel_index(np.nanargmax(stack), stack.shape)
    crop = tuple(slice(p - 8, p + 8) for p in peak)
    raw, mask = stack[crop], gap[crop]
    with pytest.raises(ValueError):
        qp.CDI(raw, mask=mask, support=support)

    measured = np.where(mask, 0.0, raw)
    report = qp.diagnose(measured, mask=mask, support=support)
    assert report.peak_offset == (0, 0, 0) and report.friedel_offset == (0, 0, 0)

    ensemble = qp.reconstruct(measured, support=support, mask=mask, recipe=RECIPE, starts=4)
    assert ensemble.best.error < 1e-3
    assert similarity(ensemble.best.object, obj) > 0.95
