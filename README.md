# QoraPhase

[![PyPI](https://img.shields.io/pypi/v/qoraphase)](https://pypi.org/project/qoraphase/) [![Tests](https://github.com/NaNIsaac/QoraPhase/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/QoraPhase/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/NaNIsaac/QoraPhase/blob/main/LICENSE)

**Phase retrieval for coherent diffraction imaging in JAX: pass a diffraction intensity array, get the complex object back.**

In coherent diffraction imaging (CDI) the detector records only the intensity of
the far-field diffraction pattern; the phase is lost. QoraPhase recovers the
complex object (amplitude and phase) from a measured 2D or 3D pattern with the
standard iterative algorithms (ER, HIO, RAAR, shrinkwrap), on CPU or GPU. It is
for people who already have a centred diffraction pattern as an array and want
a reconstruction they can check.

![Simulated object, its diffraction intensity, and the reconstruction](https://raw.githubusercontent.com/NaNIsaac/QoraPhase/main/docs/reconstruction.png)

A 64×64 simulated object, its noiseless intensity, and the best of 8 random
starts from `qp.reconstruct(intensity, starts=8)` with no support given (it is
derived from the autocorrelation and shrinkwrapped). Relative error against the
true object: 7.9e-7. Script: `examples/readme_figure.py`.

## Quick start

Requires Python 3.12+.

```bash
pip install qoraphase
```

```python
import numpy as np
import qoraphase as qp

intensity = np.load("pattern.npy")   # centred 2D or 3D diffraction intensity
result = qp.reconstruct(intensity)   # pass support=... if you have one
result.plot()                        # result.amplitude, result.phase
```

QoraPhase does not read files: you bring a `numpy` or `jax` array. Iterative
algorithms (ER, HIO, RAAR), shrinkwrap, detwinning and multi-start are
composable pieces of a recipe, and multi-start runs are vectorised with `vmap`.

## Array conventions (read this first)

| | |
|---|---|
| **Origin** | Every public array is **center-origin**: index `n // 2` is q=0. `ifftshift` is internal; never call it yourself. |
| **`intensity`** | A measured **intensity**, not an amplitude. Internally `sqrt(clip(intensity, 0))`. |
| **`mask`** | `True` marks an **unusable** pixel (beamstop, hot, saturated). Masked pixels are left unconstrained — never forced to zero. |
| **`support`** | `True` is inside the object. Omit it and an autocorrelation support is derived (loose; pass a real one when you have it). |

## Composing a reconstruction

Code order is execution order.

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
branch. The default recipe runs one detwin pass before shrinkwrap starts. It
helps when the support is symmetric and the run would otherwise stall; a support
with the right shape helps more.

`qp.Shrinkwrap(threshold, sigma, method)` takes `method="max"` (classic, relative
to the peak amplitude) or `method="rms"` (relative to the rms inside the current
support, less easily thrown off by a hot pixel). The thresholds are on different
scales: the classic `"max"` value is 0.2, PyNX uses 0.28 for `"rms"`.

The defaults are `0.3, 1.0` rather than the classic `0.2, 2.0`, because only
those converged from an autocorrelation support on the synthetic test objects.
**These two numbers are the first thing to re-tune on a real dataset.**

## Multi-start

For a support with awkward symmetry, several random starts are often the
difference between converging and not.

```python
ensemble = problem.reconstruct(recipe, starts=32, batch=4)
best = ensemble.best
```

`vmap` holds every start in memory at once. `batch` chunks that: lower it if a
large 3D volume runs out of GPU memory.

Measured on the README object above (64×64, default recipe, no support), CPU
only. "Converged" counts runs whose final Fourier error is below 1e-3; on this
object the final errors fall either below 1e-5 or above 6e-3.

```
CPU: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel | JAX 0.11.1 (cpu) | Python 3.14.7
```

| starts | first call (incl. compile) | vmap, warm | loop of starts=1, warm | converged (vmap) | converged (loop) |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.34 s | 0.05 s | 0.05 s | 1/1 | 1/1 |
| 8 | 2.19 s | 0.36 s | 0.50 s | 3/8 | 2/8 |
| 32 | 3.71 s | 1.82 s | 2.18 s | 12/32 | 9/32 |

Roughly a third of the starts converge on this object (21 of 64 across the two
32-start columns), so one start fails more often than not, while 8 starts
contained at least one success in both columns. On CPU, `vmap` beat a Python
loop of single starts at 8 and 32 starts, by a margin (17–28% here) that moves
between runs. Each new `starts` value costs a first call of 2–4 s, mostly
compilation.

Regenerate with `python examples/bench_multistart.py`.

## Swapping the forward model

Algorithms never name an FFT, so the physics is replaceable without touching
them:

```python
qp.CDI(intensity, support=support, model=MyPropagator())
```

A model needs `forward` / `adjoint` and should be unitary. A *wrong* model fails
silently: it can fit the data perfectly and return the wrong object. See
`tests/test_forward_model.py`.

## Before you trust a reconstruction

Check the measurement against the assumptions the library makes about it:

```python
print(qp.diagnose(intensity, mask=mask, support=support))
```

It reports the centring offset two ways (brightest pixel, and Friedel symmetry,
which resolves half a pixel and works on patterns with no dominant peak), how
much of the array the autocorrelation fills, the negative and masked fractions,
and warns about each. An off-centre pattern is the most costly and least visible
failure.

`examples/diagnose.py` does the same from the command line, then reconstructs
and reports whether independent starts agree:

```
python examples/diagnose.py scan.npz --key intensity --support sup.npy --starts 8
```

A low Fourier error says a run fits the data. Agreement between starts, after
alignment, is what says the answer is determined.

## Combining runs

A reconstruction is determined only up to a global phase, a translation, and a
conjugate flip. No iteration can remove those — they are exact symmetries of the
constraints — so they are resolved afterwards:

```python
aligned = ensemble.align()      # every run brought onto the best one
mean = ensemble.average()       # coherent mean of the aligned runs
```

**Never average runs without aligning them** (`average()` aligns for you). Each
run carries its own global phase, so a raw mean interferes destructively and
loses amplitude.

`align()` adopts whichever conjugate branch the best run is in; that is a
convention, not a statement about which branch is real. `conj(obj(-r))`
reproduces the measured intensity exactly, so only the support you supply breaks
the tie: a reflected support converges just as cleanly on the mirrored object.
If the twin branch matters, break the symmetry during the run with `qp.detwin`.

Alignment works to whole pixels by default. `result.phase` keeps any linear phase
ramp from the measurement geometry until you call `qp.remove_ramp` (below).

### Sub-pixel alignment

`upsample=N` refines the whole-pixel correlation peak to about `1/N` pixel with
the matrix-multiply DFT of Guizar-Sicairos et al. (Opt. Lett. 33, 156, 2008) and
applies the fractional remainder as a Fourier shift. It is available on
`align_to`, `Ensemble.align` and `Ensemble.average`; `qp.register` returns the
shift itself. The Fourier shift treats the array as periodic, so the object
should not touch the array edge.

```python
import numpy as np
import qoraphase as qp

rng = np.random.default_rng(0)
obj = np.zeros((32, 32), complex)
obj[11:21, 11:21] = rng.uniform(0.5, 1, (10, 10)) * np.exp(1j * rng.uniform(-0.5, 0.5, (10, 10)))

# displace by (1.37, -2.63) pixels through the Fourier shift theorem
freq = [np.fft.fftfreq(n) for n in obj.shape]
ramp = np.exp(-2j * np.pi * (1.37 * freq[0][:, None] - 2.63 * freq[1][None, :]))
moved = np.fft.ifftn(np.fft.fftn(obj) * ramp)

shift, _ = qp.register(moved, obj, upsample=100)
print(shift)
for upsample in (1, 100):
    aligned = qp.align_to(moved, obj, upsample=upsample)
    print(upsample, np.linalg.norm(aligned - obj) / np.linalg.norm(obj))
```

```
(1.3699999999999999, -2.63)
1 0.41172676077039544
100 1.5936516413343817e-07
```

### Phase-ramp removal

A linear phase across the object means its Fourier transform sits off the array
centre. `qp.remove_ramp(obj)` finds that offset as the centre of mass of `|F|^2`
of the zero-padded object, to a fraction of a pixel, and cancels it. Pass the
support to keep noise outside the object out of the measurement, and
`zero_phase=True` to also set the mean phase inside the support to zero.

It removes whatever linear phase is present. A ramp that carries physical
information (a displaced Bragg peak, uniform strain) goes with it, so keep the
original if you need it. The centre of mass is exact for a real-valued object; for
a complex one it is an estimate. It traces under `jax.jit` (`pad`, `iterations`
and `zero_phase` static).

```python
import numpy as np
import qoraphase as qp

rng = np.random.default_rng(0)
obj = np.zeros((32, 32))
obj[11:21, 11:21] = rng.uniform(0.5, 1, (10, 10))
support = obj > 0

x = np.arange(32) - 16
ramped = obj * np.exp(2j * np.pi * (0.071 * x[:, None] - 0.043 * x[None, :]))

cleaned = np.asarray(qp.remove_ramp(ramped, support, zero_phase=True))
print(np.abs(np.angle(ramped[support])).max())
print(np.abs(np.angle(cleaned[support])).max())
```

```
3.135309468282613
0.000930922
```

### PRTF

The phase retrieval transfer function measures where independent reconstructions
stop agreeing. `qp.prtf(ensemble, intensity, mask)` aligns the runs (to `1/100`
pixel by default, `upsample=` changes that), averages their transforms and
divides the radially summed amplitude of the mean by the measured one. It returns
a `PRTF` with `frequency` (shell centres, cycles per object pixel), `value`,
`crossing` (where `value` first falls below 1/e, interpolated, nan if it never
does) and `resolution` (`1 / crossing`, a full period in object pixels). A
sequence of already-aligned objects works too with `align=False`, and
`qp.prtf_curve` is the traceable core. Two or more runs are needed; one run has a
PRTF of 1 by construction.

```python
import numpy as np
import qoraphase as qp

rng = np.random.default_rng(1)
obj = np.zeros((48, 48), complex)
block = rng.uniform(0.5, 1, (14, 14)) * np.exp(1j * rng.uniform(-0.5, 0.5, (14, 14)))
obj[17:31, 17:31] = block * (np.indices((14, 14)).sum(0) < 14)
support = np.abs(obj) > 0

clean = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))) ** 2
intensity = rng.poisson(clean * (100 / clean.max())) * (clean.max() / 100)

ensemble = qp.reconstruct(intensity, support=support, starts=12, seed=0)
result = qp.prtf(ensemble, intensity)
print(np.round(np.asarray(result.value)[::4], 3))
print(result.crossing, result.resolution)
```

```
[0.979 0.852 0.676 0.499 0.468 0.396 0.278]
0.437100887298584 2.2878013498904184
```

On noiseless data a converged ensemble has a PRTF of 1 at every frequency and no
crossing. Here the intensity has Poisson noise (peak of 100 counts), and the
values printed are every fourth shell, from zero frequency to Nyquist (0.5).

## What this does not do

- No clustering of runs.
- No positivity/reality constraint — the object is complex here.
- No file I/O, no BCDI geometry, no NUFFT, no strain, no autodiff optimisers.

## Development

```bash
uv sync --all-groups
uv run pytest
```
