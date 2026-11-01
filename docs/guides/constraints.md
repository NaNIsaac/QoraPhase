# Partial coherence and phase constraints

These session fragments continue a `problem` defined in the [recipe guide](recipes.md). The tables retain the original synthetic measurements; this documentation change does not rerun them.

## Partial coherence


A beam that is not fully coherent blurs the measured pattern: the detector
records the coherent intensity convolved with a coherence kernel. A coherent
model cannot fit that, and puts spurious structure into the object instead.
`qp.partial_coherence` estimates the kernel during the run (Richardson-Lucy, as
in Clark et al. 2012 and cohere's `pcdi`) and fits the blurred model:

```python
recipe = qp.Recipe(
    qp.RAAR(200, beta=0.9),  # coherent first: the first kernel
    qp.partial_coherence(  # estimate comes from this object
        qp.RAAR(100, beta=0.9),
        qp.ER(200),
        kernel=7,  # odd size in pixels, or one per axis
    ),
)
result = problem.reconstruct(recipe)
result.coherence  # the estimated kernel, sums to 1
```

The kernel is re-estimated before every algorithm stage inside the wrapper, so
the update interval is the length of those stages
(`qp.repeat(8, qp.RAAR(50), qp.Shrinkwrap())` updates every 50 iterations).

On a 48×48 test object whose pattern was blurred by a Gaussian kernel of
σ = 1 px (best of 4 starts, `tests/test_coherence.py`):

| model | final Fourier error | similarity to the true object |
|---|---:|---:|
| coherent, `RAAR(300) + ER(200)` | 0.0645 | 0.893 |
| `partial_coherence`, as above | 0.0145 | 0.997 |

Measured on Windows (CPU). The iteration is sensitive to floating-point
rounding, so the numbers move between platforms: the same test on Linux CI
gives 0.0644 and 0.0217.

## Phase constraint


`qp.phase_limit(low, high, *stages)` runs every algorithm inside it with the
support narrowed, iteration by iteration, to the pixels whose phase lies in
`[low, high]` radians; the stored support is not changed. It is the `phc`
option of cohere and the "phase constraint" of older BCDI codes, used early in
a run on weakly strained crystals:

```python
recipe = qp.Recipe(
    qp.phase_limit(-0.6, 0.6, qp.HIO(150)),
    qp.HIO(150),
    qp.ER(100),
)
```

It is not part of the default recipe, because on the synthetic objects it did
not help. Successes out of 16 starts, summed over 3 objects, whose phases lie
in ±0.5 rad, with `phase_limit(-0.6, 0.6, ...)` on the first half of the run:

| support | RAAR | RAAR, limited | HIO | HIO, limited |
|---|---:|---:|---:|---:|
| simplex (well determined) | 48 / 48 | 48 / 48 | 48 / 48 | 48 / 48 |
| centrosymmetric box | 19 / 48 | 12 / 48 | 48 / 48 | 46 / 48 |

Try it on real data where unconstrained runs end in phase wraps or vortices,
and compare against runs without it.
