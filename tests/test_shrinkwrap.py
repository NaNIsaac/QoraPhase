"""Shrinkwrap and its Gaussian blur (spec 12)."""

import jax.numpy as jnp
import numpy as np

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
