import jax.numpy as jnp
import numpy as np
import pytest
from conftest import centered_fft, make_object, similarity

import qoraphase as qp
from qoraphase.operators.coherence import convolve, lucy_update

COHERENT = qp.Recipe(qp.RAAR(300, beta=0.9), qp.ER(200))


def gaussian_kernel(size, sigma):
    r = np.arange(size) - size // 2
    g = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma**2))
    return g / g.sum()


def blurred_data(sigma=1.0, size=7, shape=(48, 48), seed=0):
    obj = make_object(shape, m=12, seed=seed)
    kernel = gaussian_kernel(size, sigma)
    coherent = np.abs(centered_fft(obj)) ** 2
    measured = np.asarray(convolve(jnp.asarray(coherent), jnp.asarray(kernel)))
    return obj, np.abs(obj) > 0, coherent, measured, kernel


def test_convolving_with_a_delta_changes_nothing():
    _, _, coherent, _, _ = blurred_data()
    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    assert np.allclose(convolve(jnp.asarray(coherent), jnp.asarray(delta)), coherent, atol=1e-6)


def test_convolution_matches_a_direct_sum():
    rng = np.random.default_rng(0)
    image, kernel = rng.random((9, 8)), rng.random((3, 3))
    expected = np.zeros_like(image)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            expected += kernel[di + 1, dj + 1] * np.roll(image, (di, dj), axis=(0, 1))
    assert np.allclose(convolve(jnp.asarray(image), jnp.asarray(kernel)), expected, atol=1e-6)


def test_lucy_recovers_the_kernel_from_the_true_coherent_intensity():
    _, _, coherent, measured, kernel = blurred_data(sigma=1.2)
    flat = jnp.full(kernel.shape, 1.0 / kernel.size)
    no_mask = jnp.zeros(coherent.shape, bool)

    def distance(iterations):
        estimate = lucy_update(
            flat, jnp.asarray(coherent), jnp.asarray(measured), no_mask, iterations
        )
        assert np.isclose(float(estimate.sum()), 1.0)
        return np.abs(np.asarray(estimate) - kernel).sum()

    # Richardson-Lucy closes in slowly but steadily (0.09, 0.05, 0.02 here).
    errors = [distance(n) for n in (50, 200, 1000)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.03


def test_masked_pixels_do_not_vote():
    _, _, coherent, measured, kernel = blurred_data(sigma=1.2)
    mask = np.zeros(coherent.shape, bool)
    mask[:4] = True
    corrupted = measured.copy()
    corrupted[mask] = 1e6
    flat = jnp.full(kernel.shape, 1.0 / kernel.size)
    estimate = lucy_update(
        flat, jnp.asarray(coherent), jnp.asarray(corrupted), jnp.asarray(mask), 1000
    )
    assert np.abs(np.asarray(estimate) - kernel).sum() < 0.03


def test_partial_coherence_fits_blurred_data_that_a_coherent_model_cannot():
    obj, support, _, measured, _ = blurred_data(sigma=1.0)
    problem = qp.CDI(measured, support=support)
    pcdi = qp.Recipe(
        qp.RAAR(200, beta=0.9), qp.partial_coherence(qp.RAAR(100, beta=0.9), qp.ER(200))
    )

    coherent = problem.reconstruct(COHERENT, starts=4, seed=0).best
    partial = problem.reconstruct(pcdi, starts=4, seed=0).best

    # 4.4x lower on the Windows machine that produced the README numbers, 3.0x on Linux CI.
    assert partial.error < coherent.error / 2
    assert similarity(np.asarray(partial.object), obj) > similarity(
        np.asarray(coherent.object), obj
    )
    assert partial.coherence.shape == (7, 7)
    assert coherent.coherence is None


def test_the_kernel_is_updated_before_each_algorithm_stage():
    recipe = qp.partial_coherence(qp.repeat(3, qp.RAAR(10), qp.Shrinkwrap()), qp.ER(5))
    body = recipe.body.stages
    repeat_body = body[0].body.stages[0].stages
    assert isinstance(repeat_body[0], qp.UpdateCoherence)
    assert isinstance(repeat_body[1], qp.RAAR)
    assert isinstance(body[1].stages[0], qp.UpdateCoherence)


def test_an_even_or_oversized_kernel_is_rejected(problem):
    with pytest.raises(ValueError, match="odd"):
        problem.reconstruct(qp.partial_coherence(qp.ER(2), kernel=4))
    with pytest.raises(ValueError, match="odd"):
        problem.reconstruct(qp.partial_coherence(qp.ER(2), kernel=99))
