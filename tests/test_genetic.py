"""Breeding and the generation loop."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import complex_similarity, make_data

import qoraphase as qp
from qoraphase.core.state import initial_state

# Sums to a real positive number, so breeding does not rotate it.
ALPHA = jnp.array([4 + 0j, 1 + 1j, 1 - 1j])
BETA = jnp.array([1 + 0j, 1j, -1j])
R2 = 2 ** 0.25  # sqrt(|alpha|) where |alpha| = sqrt 2 and |beta| = 1


def test_avg_ab_is_the_plain_mean():
    assert np.allclose(qp.breed("avg_ab", ALPHA, BETA), [2.5, 0.5 + 1j, 0.5 - 1j])


def test_sqrt_ab_is_geometric_in_amplitude_and_arithmetic_in_phase():
    expected = np.array(
        [2.0, R2 * np.exp(3j * np.pi / 8), R2 * np.exp(-3j * np.pi / 8)]
    )
    assert np.allclose(qp.breed("sqrt_ab", ALPHA, BETA), expected)


def test_pa_modes_keep_the_phase_of_alpha():
    phase = np.exp(1j * np.array([0.0, np.pi / 4, -np.pi / 4]))
    assert np.allclose(qp.breed("sqrt_ab_pa", ALPHA, BETA), [2.0, R2, R2] * phase)
    assert np.allclose(qp.breed("avg_ab_pa", ALPHA, BETA), [2.5, (np.sqrt(2) + 1) / 2, (np.sqrt(2) + 1) / 2] * phase)
    assert np.allclose(qp.breed("b_pa", ALPHA, BETA), [1.0, 1.0, 1.0] * phase)


def test_max_ab_takes_the_larger_amplitude():
    phase = np.exp(1j * np.array([0.0, 3 * np.pi / 8, -3 * np.pi / 8]))
    assert np.allclose(qp.breed("max_ab", ALPHA, BETA), [4.0, np.sqrt(2), np.sqrt(2)] * phase)


@pytest.mark.parametrize("mode", sorted(qp.BREEDERS))
def test_breeding_ignores_a_common_global_phase(mode):
    plain = qp.breed(mode, ALPHA, BETA)
    turned = qp.breed(mode, ALPHA * np.exp(1.1j), BETA * np.exp(1.1j))
    assert np.allclose(plain, turned, atol=1e-6)


@pytest.mark.parametrize("mode", ["sqrt_ab", "avg_ab", "sqrt_ab_pa", "avg_ab_pa", "max_ab"])
def test_breeding_a_run_with_itself_changes_nothing(mode):
    assert np.allclose(qp.breed(mode, ALPHA, ALPHA), ALPHA, atol=1e-6)


def test_unknown_breed_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown breed mode"):
        qp.breed("nope", ALPHA, BETA)
    problem = qp.CDI(make_data()[2])
    with pytest.raises(ValueError, match="unknown breed mode"):
        qp.evolve(problem, qp.ER(1), breed="nope")


# --------------------------------------------------------------------------
# starting from an object
# --------------------------------------------------------------------------


def test_initial_state_can_start_from_an_object():
    obj, support, intensity = make_data()
    problem = qp.CDI(intensity, support=support)
    start = jnp.asarray(obj) * 2 + 0.5

    state = initial_state(jax.random.key(0), problem, start)

    assert np.allclose(state.obj, np.where(support, start, 0))


# --------------------------------------------------------------------------
# generations
# --------------------------------------------------------------------------

QUICK = qp.Recipe(qp.RAAR(20, beta=0.9), qp.ER(10))


@pytest.fixture(scope="module")
def easy():
    _, support, intensity = make_data(m=10, seed=3)
    return qp.CDI(intensity, support=support)


def sizes(ensemble):
    return [len(g) for g in (*ensemble.history, ensemble)]


def test_population_sizes_follow_cull(easy):
    assert sizes(qp.evolve(easy, QUICK, starts=8, generations=3, cull=[2, 3])) == [8, 6, 3]
    assert sizes(qp.evolve(easy, QUICK, starts=8, generations=3, cull=2)) == [8, 6, 4]
    assert sizes(qp.evolve(easy, QUICK, starts=8, generations=4)) == [8, 8, 8, 8]
    assert sizes(qp.evolve(easy, QUICK, starts=8, generations=3, cull=[1])) == [8, 7, 7]


def test_cull_cannot_empty_the_population(easy):
    with pytest.raises(ValueError, match="cull leaves no runs"):
        qp.evolve(easy, QUICK, starts=4, generations=3, cull=2)


def test_one_generation_is_a_plain_multistart(easy):
    ga = qp.evolve(easy, QUICK, starts=4, generations=1, seed=2)
    plain = easy.reconstruct(QUICK, starts=4, seed=2)
    assert ga.history == ()
    assert np.allclose([r.error for r in ga], [r.error for r in plain], atol=1e-6)


def test_evolve_is_reproducible_and_batching_does_not_change_it(easy):
    a = qp.evolve(easy, QUICK, starts=8, generations=3, cull=[4, 0], seed=5)
    b = qp.evolve(easy, QUICK, starts=8, generations=3, cull=[4, 0], seed=5)
    c = qp.evolve(easy, QUICK, starts=8, generations=3, cull=[4, 0], seed=5, batch=4)

    assert np.array_equal([r.error for r in a], [r.error for r in b])
    assert np.allclose([r.error for r in a], [r.error for r in c], atol=1e-4)


def test_a_recipe_per_generation(easy):
    ensemble = qp.evolve(easy, [qp.ER(5), qp.ER(5), qp.RAAR(5)], starts=2)
    assert len(ensemble.history) == 2
    with pytest.raises(ValueError, match="recipes for"):
        qp.evolve(easy, [qp.ER(5)], starts=2, generations=3)


# --------------------------------------------------------------------------
# the point of it
# --------------------------------------------------------------------------


def test_breeding_beats_the_same_budget_without_it_on_a_symmetric_support():
    """A box support admits the object and its twin; random starts stagnate in a
    mixture of both.  Aligning every run to the best before breeding puts the
    population in one branch, which the plain multi-start has no way to do.
    """
    obj, support, intensity = make_data(centrosymmetric=True)
    problem = qp.CDI(intensity, support=support)
    per_generation = qp.Recipe(qp.RAAR(100, beta=0.9), qp.ER(50))
    same_budget = qp.Recipe(qp.RAAR(300, beta=0.9), qp.ER(150))

    def converged(ensemble):
        return sum(run.error < 1e-3 for run in ensemble)

    bred = plain = 0
    for seed in range(6):
        evolved = qp.evolve(problem, per_generation, starts=8, generations=3, seed=seed)
        bred += converged(evolved)
        plain += converged(problem.reconstruct(same_budget, starts=8, seed=seed))
        for run in evolved:
            if run.error < 1e-3:
                assert complex_similarity(run.object, obj) > 0.99

    assert bred >= plain
    assert bred > 0
