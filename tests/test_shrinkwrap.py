"""Shrinkwrap and its Gaussian blur."""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import make_data

from qoraphase import Shrinkwrap
from qoraphase.core.state import ReconState
from qoraphase.operators.shrinkwrap import gaussian_blur


def test_blur_preserves_sum():
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.uniform(size=(24, 24)))
    assert np.isclose(float(gaussian_blur(a, 3.0).sum()), float(a.sum()), rtol=1e-5)


def test_blur_with_zero_sigma_is_identity():
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.uniform(size=(16, 16)))
    assert np.allclose(gaussian_blur(a, 0.0), a, atol=1e-5)


def test_blur_spreads_a_delta():
    n = 32
    delta = jnp.zeros((n, n)).at[n // 2, n // 2].set(1.0)
    blurred = np.asarray(gaussian_blur(delta, 2.0))
    assert blurred.argmax() == delta.argmax()  # stays centred
    assert blurred.max() < 0.5  # and spreads out
    assert (blurred > 1e-6).sum() > 20


def test_blur_works_in_3d():
    a = jnp.ones((8, 8, 8))
    assert np.allclose(gaussian_blur(a, 1.5), a, atol=1e-5)


def test_shrinkwrap_tightens_a_loose_support(data):
    """Starting from a support twice too big, shrinkwrap pulls it in."""
    obj, support, intensity = data
    loose = np.ones_like(support)
    state = ReconState(
        obj=jnp.asarray(obj),
        support=jnp.asarray(loose),
        iteration=jnp.int32(0),
        key=None,
    )
    new_state, metrics = Shrinkwrap(threshold=0.2, sigma=1.0).run(state, None)
    new_support = np.asarray(new_state.support)

    assert metrics.size == 0  # shrinkwrap reports no error metric
    assert new_support.sum() < loose.sum()
    assert new_support.sum() >= support.sum() * 0.5
    assert new_support[support].mean() > 0.9  # still covers the object


@pytest.mark.parametrize("spike", [10.0, 50.0, 200.0])
def test_rms_threshold_is_less_sensitive_to_a_bright_outlier(spike):
    """Why 'rms' exists: 'max' keys off a single pixel, 'rms' off the bulk.

    A hot spot drags the 'max' reference up and starves the support.  It is not
    that rms ignores it -- the spike sits inside the support, so it lifts the
    rms too -- but it is diluted over every pixel instead of setting the
    reference single-handedly.  At a 50x spike the support keeps 84% of its
    pixels under 'rms' against 57% under 'max'.
    """
    obj, support, _ = make_data()
    spiked = np.asarray(obj).copy()
    spiked[16, 16] = spike

    def sizes(method, threshold):
        out = []
        for array in (obj, spiked):
            state = ReconState(
                obj=jnp.asarray(array),
                support=jnp.asarray(support),
                iteration=jnp.int32(0),
                key=None,
            )
            updated, _ = Shrinkwrap(threshold, 2.0, method).run(state, None)
            out.append(int(np.asarray(updated.support).sum()))
        return out

    clean_max, spiked_max = sizes("max", 0.2)
    clean_rms, spiked_rms = sizes("rms", 0.28)

    assert spiked_rms / clean_rms > spiked_max / clean_max


def test_rms_threshold_produces_a_usable_support():
    obj, support, _ = make_data()
    state = ReconState(
        obj=jnp.asarray(obj),
        support=jnp.asarray(support),
        iteration=jnp.int32(0),
        key=None,
    )
    updated, _ = Shrinkwrap(0.28, 1.0, "rms").run(state, None)
    updated = np.asarray(updated.support)
    assert 0 < updated.sum() < updated.size
    assert updated[support].mean() > 0.9


def test_unknown_threshold_method_is_rejected():
    with pytest.raises(ValueError, match="max.*rms"):
        Shrinkwrap(method="mean")
