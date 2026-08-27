"""Projection contracts (spec 4.3, 10.1, 10.2)."""

import jax.numpy as jnp
import numpy as np

import qoraphase as qp
from conftest import flip_about_origin, make_data
from qoraphase.operators.modulus import modulus_project
from qoraphase.operators.support import autocorrelation_support, support_project


def test_support_projection_zeros_outside(data):
    obj, support, _ = data
    projected = support_project(jnp.asarray(obj) + 1.0, jnp.asarray(support))
    assert np.all(np.asarray(projected)[~support] == 0)


def test_support_projection_is_idempotent(data):
    obj, support, _ = data
    once = support_project(jnp.asarray(obj) + 1.0, jnp.asarray(support))
    assert np.allclose(support_project(once, jnp.asarray(support)), once)


def test_modulus_projection_fixes_the_true_solution(data, problem):
    """The true object already satisfies the modulus constraint."""
    obj, _, _ = data
    projected, err = modulus_project(jnp.asarray(obj), problem)
    assert np.allclose(projected, obj, atol=1e-5)
    assert float(err) < 1e-5


def test_modulus_projection_enforces_measured_magnitude(data, problem):
    """After projection the Fourier magnitude equals the measurement."""
    obj, support, _ = data
    perturbed = jnp.asarray(obj) * 0.3 + 0.2
    projected, _ = modulus_project(perturbed, problem)
    magnitude = np.abs(np.asarray(problem.model.forward(projected)))
    assert np.allclose(magnitude, np.asarray(problem.amplitude), atol=1e-5)


def test_masked_pixels_do_not_constrain_the_object(data):
    """Whatever garbage sits under the mask must not change the result.

    This is the spec 4.3 rule: masked pixels keep their computed magnitude
    instead of being forced to the measured (or worse, to zero).
    """
    obj, support, intensity = data
    mask = np.zeros_like(intensity, bool)
    mask[:4, :4] = True

    start = jnp.asarray(obj) * 0.3 + 0.2
    results = []
    for garbage in (0.0, 1e6):
        corrupted = intensity.copy()
        corrupted[mask] = garbage
        problem = qp.CDI(corrupted, support=support, mask=mask)
        projected, err = modulus_project(start, problem)
        results.append((np.asarray(projected), float(err)))

    assert np.allclose(results[0][0], results[1][0], atol=1e-6)
    assert np.isclose(results[0][1], results[1][1], rtol=1e-6)


def test_autocorrelation_support_fits_the_object(data):
    """The default support must admit the object somewhere, and be loose.

    It cannot be expected to cover the object *in place*: the autocorrelation is
    ``S (+) -S``, always centred at the origin, so what it has to contain is a
    translate of the object support -- which is all that matters, since the
    solution is only defined up to a translation anyway (spec 14.1).
    """
    obj, support, intensity = data
    auto = np.asarray(autocorrelation_support(jnp.asarray(intensity), 0.1))

    # Number of support pixels falling outside `auto`, for every shift at once.
    spill = np.fft.ifftn(
        np.fft.fftn(~auto) * np.conj(np.fft.fftn(support))
    ).real.round()
    assert spill.min() == 0  # some translate of the object fits entirely inside

    assert auto.sum() > support.sum()  # loose: roughly twice the object
    assert np.array_equal(auto, flip_about_origin(auto))  # autocorrelations are centrosymmetric
