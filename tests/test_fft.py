"""Forward model contract."""

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
    assert np.isclose(np.linalg.norm(model.forward(obj)), np.linalg.norm(obj), rtol=1e-5)


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


def centered_ifft(wave):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(wave), norm="ortho"))


SHAPES = [(8,), (7,), (8, 8), (7, 9), (6, 5), (6, 5, 4), (7, 5, 3), (16, 16, 16)]


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_shifted_reference_for_even_and_odd_lengths(model, shape):
    rng = np.random.default_rng(2)
    x = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    tol = 1e-5 * np.linalg.norm(x)
    assert np.abs(model.forward(jnp.asarray(x)) - centered_fft(x)).max() < tol
    assert np.abs(model.adjoint(jnp.asarray(x)) - centered_ifft(x)).max() < tol
    assert np.allclose(model.adjoint(model.forward(jnp.asarray(x))), x, atol=1e-5)


@pytest.mark.parametrize("shape", [(8, 8), (7, 9), (6, 5, 4)])
def test_real_input_matches_shifted_reference(model, shape):
    x = np.random.default_rng(3).normal(size=shape)
    assert np.allclose(model.forward(jnp.asarray(x)), centered_fft(x), atol=1e-5)


def test_reconstruction_agrees_with_the_shifted_transform():
    """A short recipe run with the shift-free model and with the rolled one gives the same
    result to float32 rounding (the two cannot be bitwise equal: the data order differs)."""
    import qoraphase as qp

    class Shifted(FFT):
        def forward(self, obj):
            return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(obj), norm="ortho"))

        def adjoint(self, wave):
            return jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(wave), norm="ortho"))

    intensity = np.abs(centered_fft(make_object())) ** 2
    recipe = qp.Recipe(qp.RAAR(30), qp.ER(10))
    new = qp.reconstruct(intensity, recipe=recipe, seed=3)
    old = qp.reconstruct(intensity, recipe=recipe, seed=3, model=Shifted())
    assert np.allclose(new.metrics, old.metrics, rtol=1e-3, atol=1e-6)
    assert np.abs(new.object - old.object).max() < 1e-3 * np.abs(old.object).max()
