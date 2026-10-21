"""Shared synthetic data and ambiguity-aware comparison helpers."""

import numpy as np
import pytest

import qoraphase as qp


def centered_fft(obj):
    """Reference forward transform, written with plain numpy on purpose.

    The library must agree with this; if both were the same code the FFT test
    would prove nothing.
    """
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))


def flip_about_origin(a):
    """Reflect ``r -> -r`` for a center-origin array of any ndim.

    Not the same as reversing the axes: with the origin at index ``n // 2`` the
    reflection maps ``i`` to ``(n - i) % n``, which is the reversal rolled by one.
    """
    reversed_ = a[(slice(None, None, -1),) * a.ndim]
    return np.roll(reversed_, 1, axis=tuple(range(a.ndim)))


def make_object(shape=(32, 32), m=10, seed=0, centrosymmetric=False):
    """A compact complex object centred in an array of *shape*.

    The support is a simplex (triangle in 2D, tetrahedron in 3D), not a box, on
    purpose: a centrosymmetric support admits both the object and its twin, and
    most starts then stagnate in a mixture of the two.
    ``centrosymmetric=True`` keeps the full box, to reproduce that.
    """
    rng = np.random.default_rng(seed)
    ndim = len(shape)
    block = rng.uniform(0.5, 1.0, (m,) * ndim) * np.exp(1j * rng.uniform(-0.5, 0.5, (m,) * ndim))
    if not centrosymmetric:
        block = block * (np.indices((m,) * ndim).sum(0) < m)

    obj = np.zeros(shape, complex)
    obj[tuple(slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n in shape)] = block
    return obj


def make_data(shape=(32, 32), m=10, seed=0, centrosymmetric=False):
    """``(object, support, intensity)`` for a noiseless, well-oversampled case."""
    obj = make_object(shape, m, seed, centrosymmetric)
    support = np.abs(obj) > 0
    intensity = np.abs(centered_fft(obj)) ** 2
    return obj, support, intensity


def _max_correlation(A, B):
    """Peak normalised cross-correlation over all shifts.

    Taking the magnitude of the correlation peak also absorbs a global phase,
    so this is invariant to two of the three CDI ambiguities; the caller adds
    the conjugate flip.  1.0 means identical up to those ambiguities.
    """
    peak = np.abs(np.fft.ifftn(np.fft.fftn(A) * np.conj(np.fft.fftn(B)))).max()
    return peak / (np.linalg.norm(A) * np.linalg.norm(B))


def similarity(a, b):
    """Agreement in *magnitude* only -- blind to any phase difference."""
    A, B = np.asarray(np.abs(a)), np.asarray(np.abs(b))
    return max(_max_correlation(A, B), _max_correlation(flip_about_origin(A), B))


def oriented_similarity(a, b):
    """Agreement in magnitude and phase, *without* accepting the twin.

    Use this to ask which of the two conjugate-flipped branches a result landed
    in -- a question :func:`complex_similarity` deliberately cannot answer.
    """
    return _max_correlation(np.asarray(a), np.asarray(b))


def complex_similarity(a, b):
    """Agreement in magnitude *and* phase, up to the CDI ambiguities.

    Needed whenever two candidates share an amplitude but differ in phase --
    for instance an object and the same object under a propagation chirp.
    """
    A, B = np.asarray(a), np.asarray(b)
    twin = np.conj(flip_about_origin(A))
    return max(_max_correlation(A, B), _max_correlation(twin, B))


@pytest.fixture
def data():
    return make_data()


@pytest.fixture
def problem(data):
    _, support, intensity = data
    return qp.CDI(intensity, support=support)
