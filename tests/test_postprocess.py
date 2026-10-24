"""Resolving the CDI ambiguities after the fact."""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import complex_similarity, flip_about_origin, make_data, make_object

import qoraphase as qp

RECIPE = qp.Recipe(qp.RAAR(400, beta=0.9), qp.ER(200))


def test_alignment_leaves_an_already_aligned_object_alone():
    obj = jnp.asarray(make_object())
    assert np.allclose(qp.align_to(obj, obj), obj, atol=1e-5)


def test_alignment_undoes_a_translation():
    obj = jnp.asarray(make_object())
    shifted = jnp.roll(obj, (3, -2), axis=(0, 1))
    assert np.allclose(qp.align_to(shifted, obj), obj, atol=1e-5)


def test_alignment_undoes_a_global_phase():
    obj = jnp.asarray(make_object())
    rotated = obj * jnp.exp(1.1j)
    assert np.allclose(qp.align_to(rotated, obj), obj, atol=1e-5)


def test_alignment_undoes_a_twin():
    obj = jnp.asarray(make_object())
    twin = jnp.conj(jnp.asarray(flip_about_origin(np.asarray(obj))))
    assert np.allclose(qp.align_to(twin, obj), obj, atol=1e-5)


def test_alignment_undoes_all_three_at_once():
    obj = jnp.asarray(make_object())
    twin = jnp.conj(jnp.asarray(flip_about_origin(np.asarray(obj))))
    mangled = jnp.roll(twin, (-4, 5), axis=(0, 1)) * jnp.exp(-2.3j)
    assert np.allclose(qp.align_to(mangled, obj), obj, atol=1e-5)


def test_alignment_reconciles_two_independent_runs():
    """Two starts that both converged disagree only by the ambiguities."""
    obj, support, intensity = make_data(m=10, seed=3)
    ensemble = qp.reconstruct(intensity, support=support, recipe=RECIPE, starts=4)
    reference = ensemble[0].object

    for run in ensemble[1:]:
        assert complex_similarity(qp.align_to(run.object, reference), reference) > 0.99


def fourier_shift(x, shift):
    """Reference sub-pixel displacement, in plain numpy."""
    out = np.fft.fftn(x)
    for axis, delta in enumerate(shift):
        shape = [1] * x.ndim
        shape[axis] = -1
        out = out * np.exp(-2j * np.pi * delta * np.fft.fftfreq(x.shape[axis]).reshape(shape))
    return np.fft.ifftn(out)


@pytest.mark.parametrize(
    "shape, m, shift",
    [
        ((32, 32), 10, (1.37, -2.63)),
        ((32, 32), 10, (-0.5, 0.21)),
        ((24, 24, 24), 6, (1.3, -0.7, 2.45)),
    ],
)
def test_register_recovers_a_fractional_shift(shape, m, shift):
    obj = make_object(shape, m=m, seed=2)
    moved = fourier_shift(obj, shift) * np.exp(0.7j)

    found, _ = qp.register(jnp.asarray(moved), jnp.asarray(obj), upsample=100)

    assert np.abs(np.array(found) - np.array(shift)).max() < 0.05


def test_whole_pixel_registration_cannot_see_a_fraction():
    obj = make_object(seed=2)
    found, _ = qp.register(jnp.asarray(fourier_shift(obj, (0.4, -0.4))), jnp.asarray(obj))
    assert found == (0.0, 0.0)


@pytest.mark.parametrize("shape, m", [((32, 32), 10), ((24, 24, 24), 6)])
def test_subpixel_alignment_beats_whole_pixel(shape, m):
    obj = jnp.asarray(make_object(shape, m=m, seed=2))
    shift = (0.4,) * (len(shape) - 1) + (-0.35,)
    moved = jnp.asarray(fourier_shift(np.asarray(obj), shift))

    coarse = np.linalg.norm(qp.align_to(moved, obj) - obj)
    fine = np.linalg.norm(qp.align_to(moved, obj, upsample=100) - obj)

    assert fine < 0.05 * np.linalg.norm(obj)
    assert fine < 0.3 * coarse


def test_subpixel_alignment_still_resolves_phase_and_twin():
    obj = jnp.asarray(make_object(seed=2))
    twin = jnp.conj(jnp.asarray(flip_about_origin(np.asarray(obj))))
    mangled = jnp.asarray(fourier_shift(np.asarray(twin), (2.3, -1.6))) * jnp.exp(-2.3j)

    aligned = qp.align_to(mangled, obj, upsample=100)

    assert np.linalg.norm(aligned - obj) < 0.05 * np.linalg.norm(obj)


def with_ramp(obj, slopes):
    """Multiply by ``exp(2 pi i slopes . r)``, zero phase at the array centre."""
    phase = 0
    for axis, (n, slope) in enumerate(zip(obj.shape, slopes, strict=False)):
        shape = [1] * obj.ndim
        shape[axis] = -1
        phase = phase + slope * (np.arange(n) - n // 2).reshape(shape)
    return obj * np.exp(2j * np.pi * phase)


def real_object(shape, m):
    """Non-negative, asymmetric, so its spectrum is centrosymmetric but not trivial."""
    return np.abs(make_object(shape, m=m, seed=4)) * (make_object(shape, m=m, seed=4) != 0)


@pytest.mark.parametrize(
    "shape, m, slopes",
    [
        ((32, 32), 10, (0.071, -0.043)),
        ((32, 32), 10, (-0.0123, 0.0)),
        ((24, 24, 24), 6, (0.05, -0.03, 0.021)),
    ],
)
def test_remove_ramp_cancels_a_synthetic_ramp(shape, m, slopes):
    obj = real_object(shape, m)
    ramped = with_ramp(obj, slopes)

    cleaned = np.asarray(qp.remove_ramp(jnp.asarray(ramped)))

    inside = obj > 0
    residual = np.angle(cleaned[inside] * np.conj(obj[inside]))
    assert np.abs(residual).max() < 0.05


def test_remove_ramp_leaves_a_flat_object_alone():
    obj = real_object((32, 32), 10)
    assert np.allclose(qp.remove_ramp(jnp.asarray(obj)), obj, atol=1e-3)


def test_remove_ramp_keeps_the_amplitude():
    obj = real_object((32, 32), 10)
    cleaned = qp.remove_ramp(jnp.asarray(with_ramp(obj, (0.06, 0.02))))
    assert np.allclose(np.abs(cleaned), obj, atol=1e-5)


def test_remove_ramp_zero_phase_sets_the_mean_phase_in_the_support():
    obj = real_object((32, 32), 10)
    support = obj > 0
    ramped = with_ramp(obj, (0.05, -0.02)) * np.exp(1.3j)

    cleaned = np.asarray(qp.remove_ramp(jnp.asarray(ramped), support, zero_phase=True))

    assert abs(np.angle(cleaned[support].sum())) < 1e-4
    assert np.allclose(cleaned[support], obj[support], atol=0.05)


def test_remove_ramp_runs_under_jit():
    import jax

    obj = jnp.asarray(with_ramp(real_object((32, 32), 10), (0.05, -0.02)))
    plain = qp.remove_ramp(obj)
    compiled = jax.jit(qp.remove_ramp, static_argnames=("pad", "iterations", "zero_phase"))(obj)
    assert np.allclose(plain, compiled, atol=1e-5)


@pytest.fixture(scope="module")
def converged():
    obj, support, intensity = make_data(m=10, seed=3)
    ensemble = qp.reconstruct(intensity, support=support, recipe=RECIPE, starts=6)
    return obj, ensemble


def test_align_makes_every_run_agree(converged):
    _, ensemble = converged
    aligned = ensemble.align()
    reference = aligned.best.object

    assert isinstance(aligned, qp.Ensemble)
    assert len(aligned) == len(ensemble)
    for run in aligned:
        assert complex_similarity(run.object, reference) > 0.99


def test_average_recovers_the_object(converged):
    obj, ensemble = converged
    averaged = ensemble.average()

    assert isinstance(averaged, qp.Reconstruction)
    assert complex_similarity(averaged.object, obj) > 0.95
    assert averaged.error < 1e-2


def test_averaging_without_alignment_cancels_the_runs(converged):
    """The reason align() exists.

    Each run carries its own global phase, so a raw mean interferes
    destructively and bleeds away most of the amplitude -- here about two
    thirds of it.  The direction survives, because a sum of phase-rotated
    copies is still a multiple of the object, which is exactly why a
    similarity score alone would not catch this.
    """
    _, ensemble = converged
    scale = np.mean([np.linalg.norm(np.asarray(run.object)) for run in ensemble])

    naive = sum(np.asarray(run.object) for run in ensemble) / len(ensemble)
    aligned = np.asarray(ensemble.average().object)

    assert np.linalg.norm(naive) / scale < 0.7
    assert np.linalg.norm(aligned) / scale > 0.95


def test_average_without_a_problem_reports_no_error():
    """Hand-built ensembles have no problem attached, so there is no metric."""
    runs = tuple(
        qp.Reconstruction(
            object=jnp.ones((4, 4), complex),
            support=jnp.ones((4, 4), bool),
            metrics=jnp.asarray([0.5]),
        )
        for _ in range(2)
    )
    averaged = qp.Ensemble(runs).average()
    assert averaged.metrics.size == 0
    assert np.isnan(averaged.error)


def _run(obj, error):
    return qp.Reconstruction(
        object=jnp.asarray(obj), support=jnp.abs(jnp.asarray(obj)) > 0, metrics=jnp.asarray([error])
    )


def test_consensus_is_the_majority_not_the_lowest_error():
    """Three equivalent copies of one object and a different one that fits slightly better."""
    obj = make_object(seed=1)
    other = make_object(seed=2)
    rng = np.random.default_rng(0)
    noise = 1e-3 * (rng.standard_normal(obj.shape) + 1j * rng.standard_normal(obj.shape))
    copies = [
        obj + noise,
        np.roll(obj, (2, -1), axis=(0, 1)) * np.exp(0.4j),
        np.conj(flip_about_origin(obj)) * np.exp(-1.0j),
    ]
    ensemble = qp.Ensemble(
        (_run(copies[0], 0.02), _run(other, 0.01), *(_run(c, 0.03) for c in copies[1:]))
    )

    assert ensemble.best.error == pytest.approx(0.01)
    assert ensemble.consensus is not ensemble[1]  # any of the three copies will do
    assert np.argmax(ensemble.spread) == 1


def test_consensus_needs_three_runs():
    obj = make_object(seed=1)
    pair = qp.Ensemble((_run(obj, 0.02), _run(make_object(seed=2), 0.01)))
    assert pair.consensus is pair.best


def test_average_is_built_on_the_consensus_run():
    obj = make_object(seed=1)
    other = make_object(seed=2)
    copies = [obj, np.roll(obj, (1, 1), axis=(0, 1)), obj * np.exp(2.0j)]
    ensemble = qp.Ensemble((_run(other, 0.01), *(_run(c, 0.02) for c in copies)))
    averaged = np.asarray(ensemble.average().object)
    on_best = np.asarray(sum(r.object for r in ensemble.align()) / len(ensemble))
    assert complex_similarity(averaged, obj) > 0.7
    assert complex_similarity(averaged, obj) > complex_similarity(on_best, obj)
