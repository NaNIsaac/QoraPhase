"""Projection contracts."""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import flip_about_origin, make_data

import qoraphase as qp
from qoraphase.operators.modulus import modulus_project
from qoraphase.operators.support import (
    autocorrelation_support,
    fill_along_rays,
    masked_autocorrelation_support,
    support_project,
)


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

    Masked pixels keep their computed magnitude
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
    solution is only defined up to a translation anyway.
    """
    obj, support, intensity = data
    auto = np.asarray(autocorrelation_support(jnp.asarray(intensity), 0.1))

    # Number of support pixels falling outside `auto`, for every shift at once.
    spill = np.fft.ifftn(np.fft.fftn(~auto) * np.conj(np.fft.fftn(support))).real.round()
    assert spill.min() == 0  # some translate of the object fits entirely inside

    assert auto.sum() > support.sum()  # loose: roughly twice the object
    assert np.array_equal(auto, flip_about_origin(auto))  # autocorrelations are centrosymmetric


def centre_hole(shape, radius):
    grid = np.indices(shape) - np.array([n // 2 for n in shape]).reshape(-1, *[1] * len(shape))
    return np.sqrt((grid**2).sum(0)) <= radius


def fits_somewhere(region, support):
    spill = np.fft.ifftn(np.fft.fftn(~region) * np.conj(np.fft.fftn(support))).real.round()
    return spill.min() == 0


@pytest.mark.parametrize("sigma", [None, 1.0, 2.0, 4.0, 8.0])
def test_masked_autocorrelation_support_survives_a_centre_hole(sigma):
    """With the centre masked the autocorrelation loses its low frequencies, and a fraction of
    its maximum no longer covers the object (#38). The background-relative estimate does, over a
    range of blur widths."""
    _, support, intensity = make_data(shape=(128, 128), m=20)
    mask = centre_hole(intensity.shape, 6)
    holed = np.where(mask, 0.0, intensity)

    old = np.asarray(autocorrelation_support(jnp.asarray(holed), 0.1))
    assert not fits_somewhere(old, support)

    new = np.asarray(
        masked_autocorrelation_support(jnp.asarray(holed), jnp.asarray(mask), sigma=sigma)
    )
    assert fits_somewhere(new, support)
    assert support.sum() < new.sum() < 0.5 * new.size


@pytest.mark.parametrize(
    ("shape", "m", "radius", "photons"),
    [
        ((128, 128), 40, 4, None),  # the ladder benchmark geometry
        ((128, 128), 40, 2, None),
        ((128, 128), 40, 6, None),
        ((128, 128), 40, 4, 1e6),
        ((128, 128), 40, 16, 1e4),
        ((64, 64, 64), 24, 4, None),
    ],
)
def test_masked_autocorrelation_support_with_a_small_hole_under_a_large_object(
    shape, m, radius, photons
):
    """A hole small against the object leaves a near-zero ring inside the autocorrelation; the
    median cut alone split the inside off and came out smaller than the object (#38)."""
    _, support, intensity = make_data(shape=shape, m=m)
    if photons:
        rng = np.random.default_rng(0)
        intensity = rng.poisson(intensity / intensity.max() * photons).astype(float)
    mask = centre_hole(intensity.shape, radius)
    holed = np.where(mask, 0.0, intensity)
    new = np.asarray(masked_autocorrelation_support(jnp.asarray(holed), jnp.asarray(mask)))
    assert fits_somewhere(new, support)
    assert new.mean() < 0.5


def test_fill_along_rays_adds_what_lies_between_the_centre_and_the_region():
    ring = centre_hole((65, 65), 20) & ~centre_hole((65, 65), 14)
    filled = np.asarray(fill_along_rays(jnp.asarray(ring)))
    assert np.array_equal(filled, centre_hole((65, 65), 20))
    spot = np.zeros((65, 65), bool)
    spot[32, 52] = True
    line = np.asarray(fill_along_rays(jnp.asarray(spot)))
    assert line[32, 32:53].all() and line.sum() == 21


def test_masked_autocorrelation_support_refuses_an_empty_support():
    _, _, intensity = make_data(shape=(128, 128), m=20)
    mask = centre_hole(intensity.shape, 6)
    holed = jnp.asarray(np.where(mask, 0.0, intensity))
    with pytest.raises(ValueError, match="empty support"):
        masked_autocorrelation_support(holed, jnp.asarray(mask), sigma=32.0)


def test_an_explicit_support_threshold_wins_over_the_masked_rule():
    _, _, intensity = make_data(shape=(128, 128), m=20)
    hole = centre_hole(intensity.shape, 6)
    holed = np.where(hole, 0.0, intensity)
    problem = qp.CDI(holed, mask=hole, support_threshold=0.2)
    expected = autocorrelation_support(jnp.asarray(holed), 0.2)
    assert np.array_equal(np.asarray(problem.support), np.asarray(expected))


def test_cdi_picks_the_start_support_by_whether_the_centre_is_masked():
    _, _, intensity = make_data(shape=(128, 128), m=20)
    hole = centre_hole(intensity.shape, 6)
    gap = np.zeros(intensity.shape, bool)
    gap[:, 10] = True

    holed = qp.CDI(np.where(hole, 0.0, intensity), mask=hole)
    expected = masked_autocorrelation_support(
        jnp.asarray(np.where(hole, 0.0, intensity)), jnp.asarray(hole)
    )
    assert np.array_equal(np.asarray(holed.support), np.asarray(expected))

    gapped = qp.CDI(intensity, mask=gap)
    assert np.array_equal(
        np.asarray(gapped.support), np.asarray(autocorrelation_support(jnp.asarray(intensity), 0.1))
    )
