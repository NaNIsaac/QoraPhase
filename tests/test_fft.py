"""Forward model contract (spec 4.1, 9)."""

import jax.numpy as jnp
import numpy as np
import pytest

from conftest import centered_fft, make_object
from qoraphase import FFT


@pytest.fixture
def model():
    return FFT()


def test_adjoint_inverts_forward(model):
    obj = jnp.asarray(make_object())
    assert np.allclose(model.adjoint(model.forward(obj)), obj, atol=1e-5)


def test_unitary_preserves_norm(model):
    obj = jnp.asarray(make_object())
    assert np.isclose(
        np.linalg.norm(model.forward(obj)), np.linalg.norm(obj), rtol=1e-5
    )


def test_matches_reference_transform(model):
    obj = make_object()
    assert np.allclose(model.forward(jnp.asarray(obj)), centered_fft(obj), atol=1e-5)


def test_center_origin_convention(model):
    """A delta at the array centre must transform to a flat, real, positive wave.

    This is the test that catches a missing or doubled ``ifftshift``: with the
    wrong convention the phase ramps across the detector instead of being flat.
    """
    n = 16
    obj = jnp.zeros((n, n), complex).at[n // 2, n // 2].set(1.0)
    wave = model.forward(obj)
    assert np.allclose(np.abs(wave), 1.0 / n, atol=1e-6)
    assert np.allclose(np.angle(wave), 0.0, atol=1e-6)


def test_works_in_3d(model):
    rng = np.random.default_rng(1)
    obj = jnp.asarray(rng.normal(size=(8, 8, 8)) + 1j * rng.normal(size=(8, 8, 8)))
    assert np.allclose(model.adjoint(model.forward(obj)), obj, atol=1e-5)
