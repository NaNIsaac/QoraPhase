"""Fourier error definition (spec 10.3)."""

import jax.numpy as jnp
import numpy as np

from qoraphase.metrics.errors import fourier_error


def test_zero_for_an_exact_match():
    amp = jnp.asarray(np.random.default_rng(0).uniform(0.1, 1.0, (8, 8)))
    mask = jnp.zeros((8, 8), bool)
    assert float(fourier_error(amp, amp, mask)) < 1e-6


def test_grows_with_the_discrepancy():
    amp = jnp.ones((8, 8))
    mask = jnp.zeros((8, 8), bool)
    small = float(fourier_error(amp * 1.1, amp, mask))
    large = float(fourier_error(amp * 1.5, amp, mask))
    assert 0 < small < large


def test_ignores_masked_pixels():
    amp = jnp.ones((8, 8))
    mask = jnp.zeros((8, 8), bool).at[0, :].set(True)
    computed = jnp.ones((8, 8)).at[0, :].set(1e6)
    assert float(fourier_error(computed, amp, mask)) < 1e-6


def test_normalisation_is_relative():
    """Scaling both magnitudes leaves the relative error unchanged."""
    amp = jnp.asarray(np.random.default_rng(1).uniform(0.1, 1.0, (8, 8)))
    mask = jnp.zeros((8, 8), bool)
    a = float(fourier_error(amp * 1.2, amp, mask))
    b = float(fourier_error(amp * 1.2 * 7, amp * 7, mask))
    assert np.isclose(a, b, rtol=1e-5)
