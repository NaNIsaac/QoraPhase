# Recipes and multiple starts

The Python snippets use `intensity`, `support` and `mask` from your own data. They are session fragments, not standalone shell commands. Read the [array conventions](arrays.md) first. The multi-start timing table is a historical measurement, not a CI benchmark.


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
