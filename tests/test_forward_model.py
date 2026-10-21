"""Swapping the forward model without touching the algorithms.

This is the load-bearing claim of the architecture: ER/HIO/RAAR never name an
FFT, so a different forward model should reconstruct a different physics with
the same recipe.  A toy quadratic-phase (Fresnel-like) model is enough to prove
it, and to prove the choice actually matters.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from conftest import complex_similarity, make_data, similarity

import qoraphase as qp

RECIPE = qp.Recipe(qp.RAAR(400, beta=0.9), qp.ER(200))


@dataclass(frozen=True)
class QuadraticPhase:
    """``FFT(chirp * obj)``: a unitary stand-in for a near-field propagator.

     The chirp is applied in *object* space, so unlike a detector-space phase it
     changes the measured magnitudes -- which is what makes this a genuinely
     different inverse problem rather than a relabelling.

     Holds no arrays, so it stays hashable and can live in a meta field
    .
    """

    # 0.1 wraps the phase several times across the object -- enough to separate
    # chirp * obj from obj -- while staying well sampled.  Past ~0.2 the chirp
    # aliases on the grid and even the correct model stops converging.
    strength: float = 0.1

    def _chirp(self, shape):
        grids = jnp.meshgrid(*[jnp.arange(n) - n // 2 for n in shape], indexing="ij")
        r2 = sum(g.astype(jnp.float32) ** 2 for g in grids)
        return jnp.exp(1j * self.strength * r2)

    def forward(self, obj):
        chirped = obj * self._chirp(obj.shape)
        return jnp.fft.fftshift(jnp.fft.fftn(jnp.fft.ifftshift(chirped), norm="ortho"))

    def adjoint(self, wave):
        obj = jnp.fft.fftshift(jnp.fft.ifftn(jnp.fft.ifftshift(wave), norm="ortho"))
        return obj * jnp.conj(self._chirp(obj.shape))


def test_toy_model_is_unitary():
    """The adjoint contract; everything below depends on it."""
    model = QuadraticPhase()
    obj = jnp.asarray(make_data()[0])
    assert np.allclose(model.adjoint(model.forward(obj)), obj, atol=1e-5)
    assert np.isclose(np.linalg.norm(model.forward(obj)), np.linalg.norm(obj), rtol=1e-5)


def test_the_model_changes_the_measurement():
    """Guards against a vacuous test: the two models must differ observably."""
    obj = jnp.asarray(make_data()[0])
    plain = np.abs(np.asarray(qp.FFT().forward(obj)))
    chirped = np.abs(np.asarray(QuadraticPhase().forward(obj)))
    assert not np.allclose(plain, chirped, atol=1e-3)


def test_algorithms_reconstruct_through_a_custom_model():
    """Same recipe, different physics, no algorithm change."""
    model = QuadraticPhase()
    obj, support, _ = make_data(m=10, seed=3)
    intensity = np.abs(np.asarray(model.forward(jnp.asarray(obj)))) ** 2

    ensemble = qp.reconstruct(intensity, support=support, model=model, recipe=RECIPE, starts=8)
    assert ensemble.best.error < 1e-3
    assert similarity(ensemble.best.object, obj) > 0.95


def test_the_wrong_model_converges_to_the_wrong_object():
    """A wrong forward model fails *silently*, which is the dangerous case.

    The chirp has unit modulus, so ``chirp * obj`` occupies the same support as
    ``obj``.  A plain FFT therefore has a perfectly valid solution for chirped
    data -- it just is not the object.  The error metric drops to zero and the
    magnitude looks right; only the phase gives it away.
    """
    model = QuadraticPhase()
    obj, support, _ = make_data(m=10, seed=3)
    intensity = np.abs(np.asarray(model.forward(jnp.asarray(obj)))) ** 2
    chirped = np.asarray(jnp.asarray(obj) * model._chirp(obj.shape))

    ensemble = qp.reconstruct(intensity, support=support, recipe=RECIPE, starts=8)
    recovered = ensemble.best.object

    assert ensemble.best.error < 1e-3  # fits the data perfectly
    assert similarity(recovered, obj) > 0.95  # and the magnitude matches
    assert complex_similarity(recovered, chirped) > 0.95  # but it is chirp * obj
    assert complex_similarity(recovered, obj) < 0.7  # not the object


def test_the_right_model_recovers_the_phase():
    """The counterpart: with the correct model the phase comes back too."""
    model = QuadraticPhase()
    obj, support, _ = make_data(m=10, seed=3)
    intensity = np.abs(np.asarray(model.forward(jnp.asarray(obj)))) ** 2

    ensemble = qp.reconstruct(intensity, support=support, model=model, recipe=RECIPE, starts=8)
    assert complex_similarity(ensemble.best.object, obj) > 0.95


def test_model_is_a_static_pytree_field():
    """The model must survive jit as metadata, not become a traced leaf."""
    import jax

    obj, support, intensity = make_data()
    problem = qp.CDI(intensity, support=support, model=QuadraticPhase())
    leaves, treedef = jax.tree.flatten(problem)
    assert len(leaves) == 3
    assert jax.tree.unflatten(treedef, leaves).model == QuadraticPhase()
