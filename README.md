# QoraPhase

A JAX-native, composable phase-retrieval framework for coherent diffraction imaging.

**Array in → reconstruction out.** QoraPhase does not read files. You bring a
`numpy` or `jax` array; it gives you back an object.

```python
import qoraphase as qp

result = qp.reconstruct(intensity, support=support)
result.plot()
```

Design details live in `QoraPhase 설계 명세서 v0.2.md`; section numbers below
refer to it.

## Array conventions (read this first)

These three are the source of nearly every phase-retrieval bug, so they are part
of the API contract, not an implementation detail (spec 4).

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

## Multi-start

Multi-start is a core feature, not an add-on — for a support with awkward
symmetry it is the difference between converging and not.

```python
ensemble = problem.reconstruct(recipe, starts=32, batch=4)
best = ensemble.best
```

`vmap` holds every start in memory at once. `batch` chunks that: lower it if a
large 3D volume runs out of GPU memory (spec 13.2).

## Swapping the forward model

Algorithms never name an FFT, so the physics is replaceable without touching
them (spec 9):

```python
qp.CDI(intensity, support=support, model=MyPropagator())
```

A model needs `forward` / `adjoint` and should be unitary. A *wrong* model fails
silently — it can fit the data perfectly and return the wrong object — so this
is worth getting right. See `tests/test_forward_model.py`.

## What v0.1 does not do

- No twin-image, translation, or global-phase removal. Results are defined only
  up to those three transforms, so `result.phase` is **not** an absolute phase
  and runs must not be averaged without alignment (spec 14.1).
- No file I/O, no BCDI geometry, no NUFFT, no strain, no autodiff optimisers.

## Development

```bash
uv sync --all-groups
uv run pytest
```
