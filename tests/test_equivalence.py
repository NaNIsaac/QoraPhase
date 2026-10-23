"""The JAX stages against the NumPy reference in :mod:`qoraphase.testing`, from shared starts.

Every comparison is measured against :func:`qoraphase.testing.floor`, the distance between
the complex64 and complex128 runs of the reference for the same case, rather than against a
fixed rtol. A backend is accepted when it is within ``FACTOR`` times that floor of the
complex128 result. Backends other than JAX on the default device are added to ``BACKENDS``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import qoraphase as qp
from qoraphase import testing as ref
from qoraphase.core.state import initial_state

FACTOR = 10.0
ABSOLUTE = 1e-6  # where the floor itself is zero or below float32 resolution


def run_jax(recipe, intensity, support, init, mask=None):
    problem = qp.CDI(np.asarray(intensity, np.float32), support=support, mask=mask)
    state = initial_state(jax.random.key(0), problem, jnp.asarray(init, jnp.complex64))
    state, errors = jax.jit(lambda s, p: recipe.run(s, p))(state, problem)
    return np.asarray(state.obj), np.asarray(state.support), np.asarray(errors)


BACKENDS = {"jax": run_jax}


def case(shape, seed=0, masked=0.0):
    """Object inside a box support, its intensity, a random-phase start and a mask."""
    rng = np.random.default_rng(seed)
    grids = np.meshgrid(*[np.arange(n) - n // 2 for n in shape], indexing="ij")
    inside = sum((g / (n / 4)) ** 2 for g, n in zip(grids, shape, strict=True)) < 1
    obj = np.where(inside, (1 + 0.3 * rng.random(shape)) * np.exp(1j * rng.random(shape)), 0)
    intensity = np.abs(ref.forward(obj)) ** 2
    support = np.zeros(shape, bool)
    support[tuple(slice(n // 4 - 1, 3 * n // 4 + 1) for n in shape)] = True
    init = ref.adjoint(np.sqrt(intensity) * np.exp(2j * np.pi * rng.random(shape)))
    mask = rng.random(shape) < masked
    return intensity, support, init, mask


def assert_within_floor(recipe, intensity, support, init, mask=None, backend="jax"):
    exact_obj, exact_support, exact_errors = ref.run(recipe, intensity, support, init, mask=mask)
    floor = ref.floor(recipe, intensity, support, init, mask=mask)
    obj, sup, errors = BACKENDS[backend](recipe, intensity, support, init, mask)
    distance = ref.relative_distance(obj, exact_obj)
    assert distance <= FACTOR * floor + ABSOLUTE, f"{distance:.2e} against floor {floor:.2e}"
    np.testing.assert_array_equal(sup, exact_support)
    np.testing.assert_allclose(errors, exact_errors, atol=FACTOR * floor + ABSOLUTE)
    return floor


SHAPES = [(32, 32), (33, 31), (16, 16, 16), (15, 16, 17)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("stage", [qp.ER(10), qp.HIO(10, beta=0.7), qp.RAAR(10, beta=0.85)])
def test_short_runs_agree_to_the_rounding_floor(backend, shape, stage):
    intensity, support, init, _ = case(shape)
    assert_within_floor(qp.Recipe(stage), intensity, support, init, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_masked_pixels_follow_the_reference(backend):
    intensity, support, init, mask = case((32, 32), seed=1, masked=0.2)
    recipe = qp.Recipe(qp.HIO(5), qp.ER(5))
    assert_within_floor(recipe, intensity, support, init, mask, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_long_er_stays_on_the_reference(backend):
    """ER is not chaotic: 200 iterations stay within the floor."""
    intensity, support, init, _ = case((33, 31), seed=2)
    assert_within_floor(qp.Recipe(qp.ER(200)), intensity, support, init, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("method", ["max", "rms"])
def test_support_updates_and_detwin_agree(backend, method):
    intensity, support, init, _ = case((32, 32), seed=3)
    recipe = qp.Recipe(
        qp.ER(3),
        qp.detwin(qp.HIO(3), axis=1),
        qp.repeat(2, qp.RAAR(2), qp.Shrinkwrap(threshold=0.2, method=method)),
        qp.ER(3),
    )
    assert_within_floor(recipe, intensity, support, init, backend=backend)


def test_hio_amplifies_rounding_so_long_runs_cannot_be_compared_step_by_step():
    """Why the comparisons above stop at a few iterations for HIO and RAAR."""
    intensity, support, init, _ = case((32, 32))
    short = ref.floor(qp.Recipe(qp.HIO(1)), intensity, support, init)
    long = ref.floor(qp.Recipe(qp.HIO(50)), intensity, support, init)
    assert short < 1e-6 and long > 1e3 * short


@pytest.mark.parametrize("backend", BACKENDS)
def test_global_phase_carries_through(backend):
    intensity, support, init, _ = case((32, 32), seed=4)
    recipe = qp.Recipe(qp.RAAR(5), qp.ER(5))
    phase = np.exp(0.7j)
    obj, _, errors = BACKENDS[backend](recipe, intensity, support, init)
    turned, _, turned_errors = BACKENDS[backend](recipe, intensity, support, phase * init)
    assert ref.relative_distance(turned, phase * obj) < 1e-5
    np.testing.assert_allclose(turned_errors, errors, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twin_start_gives_the_twin(backend):
    """``conj(x(-r))`` has the same intensity; with the mirrored support the run mirrors."""

    def twin(a):
        flipped = a[(slice(None, None, -1),) * a.ndim]
        return np.roll(flipped, 1, axis=tuple(range(a.ndim)))

    intensity, support, init, _ = case((32, 32), seed=5)
    support[10, 9] = False  # break the symmetry of the box so the mirror is a different support
    recipe = qp.Recipe(qp.HIO(5), qp.ER(5))
    obj, _, errors = BACKENDS[backend](recipe, intensity, support, init)
    mirrored, _, mirrored_errors = BACKENDS[backend](
        recipe, intensity, twin(support), np.conj(twin(init))
    )
    assert ref.relative_distance(mirrored, np.conj(twin(obj))) < 1e-5
    np.testing.assert_allclose(mirrored_errors, errors, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_er_error_never_increases(backend):
    intensity, support, init, _ = case((16, 16, 16), seed=6)
    _, _, errors = BACKENDS[backend](qp.Recipe(qp.ER(50)), intensity, support, init)
    assert np.all(np.diff(errors) <= 1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_degenerate_inputs_stay_finite(backend):
    shape = (16, 16)
    support = np.zeros(shape, bool)
    support[8, 8] = True
    recipe = qp.Recipe(qp.HIO(3), qp.RAAR(3), qp.ER(3))
    rng = np.random.default_rng(7)
    init = rng.random(shape) + 1j * rng.random(shape)

    obj, _, errors = BACKENDS[backend](recipe, np.zeros(shape), support, init)
    assert np.all(np.isfinite(obj)) and np.all(np.isfinite(errors))

    # a delta at the origin index n // 2 has a flat spectrum, so it is a fixed point of ER
    delta = np.zeros(shape, complex)
    delta[8, 8] = 1.0
    flat = np.full(shape, 1.0 / np.prod(shape))
    obj, _, errors = BACKENDS[backend](qp.Recipe(qp.ER(3)), flat, support, delta)
    np.testing.assert_allclose(obj, delta, atol=1e-6)
    np.testing.assert_allclose(errors, 0, atol=1e-6)
