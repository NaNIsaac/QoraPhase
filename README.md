# QoraPhase

[![PyPI](https://img.shields.io/pypi/v/qoraphase)](https://pypi.org/project/qoraphase/) [![Python](https://img.shields.io/pypi/pyversions/qoraphase)](https://pypi.org/project/qoraphase/) [![Tests](https://github.com/NaNIsaac/QoraPhase/actions/workflows/tests.yml/badge.svg)](https://github.com/NaNIsaac/QoraPhase/actions/workflows/tests.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A JAX-native, composable phase-retrieval framework for coherent diffraction imaging.

**Array in → reconstruction out.** QoraPhase does not read files. You bring a
`numpy` or `jax` array; it gives you back an object.

## Install

```bash
pip install qoraphase
```

## Quick start

```python
import qoraphase as qp

result = qp.reconstruct(intensity, support=support)
result.plot()
```

## Array conventions (read this first)

These three are the source of nearly every phase-retrieval bug, so they are part
of the API contract, not an implementation detail.

| | |
|---|---|
| **Origin** | Every public array is **center-origin**: index `n // 2` is q=0. `ifftshift` is internal; never call it yourself. |
| **`intensity`** | A measured **intensity**, not an amplitude. Internally `sqrt(clip(intensity, 0))`. |
| **`mask`** | `True` marks an **unusable** pixel (beamstop, hot, saturated). Masked pixels are left unconstrained — never forced to zero. |
| **`support`** | `True` is inside the object. Omit it and an autocorrelation support is derived (loose; pass a real one when you have it). |

## Composing a reconstruction

Code order is execution order. There is no operator-overloading DSL.

```python
problem = qp.CDI(intensity=intensity, support=support, mask=mask)

recipe = qp.Recipe(
    qp.repeat(10, qp.RAAR(50, beta=0.9), qp.Shrinkwrap()),
    qp.ER(200),
)

result = problem.reconstruct(recipe)
```

The first positional argument of an algorithm is always its iteration count.

`qp.detwin(...)` runs its body against a support cut in half at the object's
centre of mass, then restores the full support. That cut is not centrosymmetric,
so it fits only one of the two conjugate solutions and biases the run into a
branch. The default recipe places it a quarter of the way through the RAAR
cycles, where PyNX puts it. It is measurably useful when the support is
symmetric and the run would otherwise stall, and harmless when it is not — but
a support with the right shape is worth far more than any detwinning.

`qp.Shrinkwrap(threshold, sigma, method)` takes `method="max"` (classic, relative
to the peak amplitude) or `method="rms"` (relative to the rms inside the current
support, less easily thrown off by a hot pixel). The thresholds are on different
scales: the classic `"max"` value is 0.2, PyNX uses 0.28 for `"rms"`.

The defaults are `0.3, 1.0` rather than the classic `0.2, 2.0`. Starting from an
autocorrelation support — the case that actually needs shrinkwrap — 0.2 / 2.0
never converged in testing and 0.3 / 1.0 did, on every object tried. That is a
measurement on synthetic data, not a law. **These two numbers are the first thing
to re-tune on a real dataset.**

## Multi-start

Multi-start is a core feature, not an add-on — for a support with awkward
symmetry it is the difference between converging and not.

```python
ensemble = problem.reconstruct(recipe, starts=32, batch=4)
best = ensemble.best
```

`vmap` holds every start in memory at once. `batch` chunks that: lower it if a
large 3D volume runs out of GPU memory.

## Swapping the forward model

Algorithms never name an FFT, so the physics is replaceable without touching
them:

```python
qp.CDI(intensity, support=support, model=MyPropagator())
```

A model needs `forward` / `adjoint` and should be unitary. A *wrong* model fails
silently — it can fit the data perfectly and return the wrong object — so this
is worth getting right. See `tests/test_forward_model.py`.

## Before you trust a reconstruction

Check the measurement against the assumptions the library makes about it:

```python
print(qp.diagnose(intensity, mask=mask, support=support))
```

It reports the centring offset two ways (brightest pixel, and Friedel symmetry,
which resolves half a pixel and works on patterns with no dominant peak), how
much of the array the autocorrelation fills, the negative and masked fractions,
and warns about each. An off-centre pattern is the failure that costs the most
and shows the least.

`examples/diagnose.py` does the same from the command line, then reconstructs
and reports whether independent starts agree:

```
python examples/diagnose.py scan.npz --key intensity --support sup.npy --starts 8
```

A low Fourier error says a run fits the data. Agreement between starts, after
alignment, is what says the answer is determined — they are different claims and
only the second one is worth much.

## Combining runs

A reconstruction is determined only up to a global phase, a translation, and a
conjugate flip. No iteration can remove those — they are exact symmetries of the
constraints — so they are resolved afterwards:

```python
aligned = ensemble.align()      # every run brought onto the best one
mean = ensemble.average()       # coherent mean of the aligned runs

```

**Never average runs without aligning them.** Each carries its own global phase,
so a raw mean interferes destructively — in practice it throws away about two
thirds of the amplitude. The direction survives, so a similarity score will not
warn you; only the amplitude shows it.

`align()` picks a *convention*, not the truth. It adopts whichever conjugate
branch the best run happens to be in — enough to combine runs, and no claim
about which branch is real. Nothing in the data distinguishes them: `conj(obj(-r))`
reproduces the measured intensity exactly. What breaks the tie is the support
you supply. Hand the solver a reflected support and every start converges just
as cleanly on the mirrored object, with a flawless error metric and no warning.
The handedness of your support is an assumption, not a result.

Alignment works to whole pixels.

`result.phase` still carries whatever linear phase the measurement geometry put
there, and QoraPhase does not remove it. Doing that properly needs the
diffraction geometry, which arrives with BCDI in v0.3; short of that, any
"correction" is a guess.

If the twin branch matters to you, break the symmetry during the run rather than
after it — see `qp.detwin` below.

## What this does not do

- No phase-ramp removal, no sub-pixel alignment, no clustering, no PRTF.
- No positivity/reality constraint — the object is complex here.
- No file I/O, no BCDI geometry, no NUFFT, no strain, no autodiff optimisers.

## Development

```bash
uv sync --all-groups
uv run pytest
```
