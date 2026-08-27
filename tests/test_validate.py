"""Input validation (spec 18)."""

import numpy as np
import pytest

import qoraphase as qp
from conftest import make_data

FAST = qp.Recipe(qp.ER(2))


def test_rejects_1d_intensity():
    with pytest.raises(ValueError, match="2D or 3D"):
        qp.CDI(np.ones(8))


def test_rejects_mismatched_support(data):
    _, _, intensity = data
    with pytest.raises(ValueError, match="support.shape"):
        qp.CDI(intensity, support=np.ones((4, 4), bool))


def test_rejects_mismatched_mask(data):
    _, _, intensity = data
    with pytest.raises(ValueError, match="mask.shape"):
        qp.CDI(intensity, mask=np.ones((4, 4), bool))


def test_rejects_non_finite_intensity(data):
    _, support, intensity = data
    broken = intensity.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        qp.CDI(broken, support=support)


def test_rejects_a_fully_masked_measurement(data):
    _, support, intensity = data
    with pytest.raises(ValueError, match="every pixel"):
        qp.CDI(intensity, support=support, mask=np.ones_like(intensity, bool))


def test_warns_about_many_negative_pixels(data):
    _, support, intensity = data
    shifted = intensity - 0.5 * intensity.mean()
    with pytest.warns(UserWarning, match="negative"):
        qp.CDI(shifted, support=support)


def test_rejects_zero_starts(problem):
    with pytest.raises(ValueError, match="starts"):
        problem.reconstruct(FAST, starts=0)


def test_rejects_a_batch_that_does_not_divide_starts(problem):
    with pytest.raises(ValueError, match="divisible"):
        problem.reconstruct(FAST, starts=4, batch=3)


def test_rejects_a_batch_larger_than_starts(problem):
    with pytest.raises(ValueError, match="1 <= batch <= starts"):
        problem.reconstruct(FAST, starts=2, batch=4)
