# QoraPhase

QoraPhase recovers a complex object (amplitude and phase) from a measured coherent diffraction
intensity, in 2D or 3D, on CPU or GPU through JAX. You pass a centred intensity array; it returns
the reconstruction. Reading detector files, BCDI geometry and strain are left to other tools
([CXD](https://github.com/NaNIsaac/CXD), [QoraGeom](https://github.com/NaNIsaac/QoraGeom)).

The algorithms are the standard iterative ones (ER, HIO, RAAR, shrinkwrap). What QoraPhase adds is
that they are small pieces you compose into a recipe, run from many random starts at once with
`vmap`, and check: `qp.diagnose` looks at the measurement before you reconstruct, and
`Ensemble.align`, `qp.prtf` and agreement between starts look at the result after.

## Install

Python 3.12 or later.

```sh
uv venv
uv pip install qoraphase
```

`pip install qoraphase` works as well.

## First run

This needs no data file: it builds a small object, computes the intensity a detector would
record, and reconstructs from that intensity alone, without being given a support.

```python
import numpy as np
import qoraphase as qp

# A 64x64 array holding a triangular object with a phase bump
n, m = 64, 24
inside = np.indices((m, m)).sum(0) < m
y, x = np.indices((m, m)) - (m - 1) / 2
block = inside * np.exp(1j * 1.2 * np.exp(-((x + 4) ** 2 + (y + 4) ** 2) / 30))
obj = np.zeros((n, n), complex)
obj[n // 2 - m // 2 : n // 2 + m // 2, n // 2 - m // 2 : n // 2 + m // 2] = block

# Its far-field intensity, centred: this is what a detector records
intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj), norm="ortho"))) ** 2

print(qp.diagnose(intensity))
ensemble = qp.reconstruct(intensity, starts=8, seed=0)
best = ensemble.best
aligned = np.asarray(qp.align_to(best.object, obj))
print(f"Fourier error {best.error:.1e}")
print(f"error against the true object {np.linalg.norm(aligned - obj) / np.linalg.norm(obj):.1e}")
```

The last lines of the output, from qoraphase 0.2.0 on CPU (Windows, Python 3.13):

```
no warnings
Fourier error 2.9e-07
error against the true object 3.0e-06
```

`qp.align_to` is needed before comparing with the true object, because a reconstruction is only
determined up to a global phase, a shift and a conjugate flip. With your own data, replace the
synthetic `intensity` with your array and read the [array conventions](../README.md#array-conventions-read-this-first)
first: the array is center-origin, it is an intensity and not an amplitude, and `mask=True` marks
an unusable pixel.

## Where to go next

The README is the full user guide. Its sections, in the order you are likely to need them:

- [Composing a reconstruction](../README.md#composing-a-reconstruction): recipes, `detwin`, shrinkwrap settings.
- [Multi-start](../README.md#multi-start) and [Combining runs](../README.md#combining-runs): many starts, alignment, averaging, PRTF.
- [Before you trust a reconstruction](../README.md#before-you-trust-a-reconstruction): `qp.diagnose` and agreement between starts.
- [Partial coherence](../README.md#partial-coherence), [Phase constraint](../README.md#phase-constraint), [Genetic algorithm](../README.md#genetic-algorithm), [Low-pass resolution progression](../README.md#low-pass-resolution-progression): optional pieces, each with what it did on test data.
- [Validation](../README.md#validation) and [What this does not do](../README.md#what-this-does-not-do).

On this site:

- [Input contract](input-contract.md): what a detector stack must satisfy before it is passed in, and what stays with the caller.
- [GPU environment of the benchmark machine](hardware.md): an RTX 5070 under Windows, why JAX runs only on its CPU there, and the options for the GPU.
- [Default recipe on simulated BCDI data](bcdi-support.md): why the default recipe stops where it does on simulated BCDI data, and what changes that.
- The Python API page lists every public function and class with its docstring.
