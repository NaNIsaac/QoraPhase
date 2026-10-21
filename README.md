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

intensity = np.load("pattern.npy")  # centred 2D or 3D diffraction intensity
result = qp.reconstruct(intensity)  # pass support=... if you have one
result.plot()  # result.amplitude, result.phase
```

QoraPhase does not read files: you bring a `numpy` or `jax` array. Iterative
algorithms (ER, HIO, RAAR), shrinkwrap, detwinning and multi-start are
composable pieces of a recipe, and multi-start runs are vectorised with `vmap`.

## Array conventions (read this first)

| | |
|---|---|
| **Origin** | Every public array is **center-origin**: index `n // 2` is q=0. `ifftshift` is internal; never call it yourself. |
| **`intensity`** | A measured **intensity**, not an amplitude. Internally `sqrt(clip(intensity, 0))`. |
| **`mask`** | `True` marks an **unusable** pixel (beamstop, hot, saturated). Masked pixels are left unconstrained — never forced to zero. NaN or Inf is refused even under the mask: set those pixels to 0. `diagnose` ignores masked pixels for the peak, negative and NaN checks. |
| **`support`** | `True` is inside the object. Omit it and an autocorrelation support is derived (loose; pass a real one when you have it). |

Coming from a detector stack (for example [CXD](https://github.com/NaNIsaac/CXD)'s
`to_qoraphase`): [docs/input-contract.md](docs/input-contract.md) lists what the arrays must
satisfy, which coordinates stay with the caller, and the effects of cropping, binning and
interpolation.

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
aligned = ensemble.align()  # every run brought onto the best one
mean = ensemble.average()  # coherent mean of the aligned runs
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

## Genetic algorithm

`qp.evolve` runs a recipe for several generations. After each one the worst runs
are dropped (`cull`), the rest are aligned to the best (`Ensemble.align`, so the
whole population ends up in one twin branch) and bred with it, and the next
generation restarts from the children instead of from random phases. The names
of the breeding modes are cohere's: `sqrt_ab`, `avg_ab`, `sqrt_ab_pa`,
`avg_ab_pa`, `max_ab`, `b_pa` (`qp.BREEDERS`; `qp.breed` applies one to two
aligned arrays).

```python
ensemble = qp.evolve(problem, recipe, starts=8, generations=3, breed="sqrt_ab", cull=[2, 2])
len(ensemble)  # 4: the last generation
ensemble.history  # the earlier generations, oldest first
```

`recipe` may be a sequence with one recipe per generation. `cull` is an int or a
list indexed by generation, so `[2, 2]` takes 8 runs to 6 and then 4. Children
restart on the problem's support rather than the shrinkwrapped one, and a
partial-coherence kernel starts afresh in each generation.

`qp.BREEDERS` also has `clone`, which is not cohere's: every child is the aligned
best run. It exists as a control, to separate what breeding adds from what
aligning to and continuing the best run adds.

### What it did here

Two square-support objects (a square admits the object and its twin, so random
starts tend to stagnate in a mixture of the two). 8 starts per seed, 10 seeds,
150 iterations per generation over 3 generations; the independent starts run the
same 450 iterations in one go. A run counts as converged below a Fourier error of
1e-3, and each seed has 8 runs.

```python
import numpy as np
import qoraphase as qp


def box_object(m, seed):
    rng = np.random.default_rng(seed)
    obj = np.zeros((32, 32), complex)
    lo = 16 - m // 2
    obj[lo : lo + m, lo : lo + m] = rng.uniform(0.5, 1, (m, m)) * np.exp(
        1j * rng.uniform(-0.5, 0.5, (m, m))
    )
    return obj


per_generation = qp.Recipe(qp.RAAR(100, beta=0.9), qp.ER(50))
same_budget = qp.Recipe(qp.RAAR(300, beta=0.9), qp.ER(150))


def converged(ensemble):
    return sum(run.error < 1e-3 for run in ensemble)


for m, object_seed in ((10, 0), (8, 5)):
    obj = box_object(m, object_seed)
    intensity = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))) ** 2
    problem = qp.CDI(intensity, support=np.abs(obj) > 0)  # a square: object and twin both fit
    print(f"{m}x{m} box, object seed {object_seed}")

    counts = {"independent starts": [], "clone": [], "sqrt_ab": [], "avg_ab": []}
    for seed in range(10):
        counts["independent starts"].append(
            converged(problem.reconstruct(same_budget, starts=8, seed=seed))
        )
        for mode in ("clone", "sqrt_ab", "avg_ab"):
            bred = qp.evolve(
                problem, per_generation, starts=8, generations=3, breed=mode, seed=seed
            )
            counts[mode].append(converged(bred))
    for name, per_seed in counts.items():
        print(f"  {name:18s} {sum(per_seed):2d}/80  {per_seed}")
```

```
10x10 box, object seed 0
  independent starts  0/80  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  clone               0/80  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  sqrt_ab            45/80  [0, 8, 0, 0, 8, 6, 8, 8, 6, 1]
  avg_ab             49/80  [1, 8, 0, 0, 7, 8, 7, 7, 8, 3]
8x8 box, object seed 5
  independent starts 10/80  [2, 0, 1, 1, 1, 1, 2, 2, 0, 0]
  clone              56/80  [8, 0, 8, 8, 8, 8, 8, 8, 0, 0]
  sqrt_ab            80/80  [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
  avg_ab             80/80  [8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
```

On these two objects breeding converged more runs than the clone control, and the
clone converged more than independent starts or the same as them. On the first
object the clone converged nothing, so there the gain is from breeding. On the
second, aligning to the best run and continuing it already recovered most of the
gain, and breeding added the rest. A clone population is eight copies of one run,
so its count per seed is 0 or 8.

Breeding is also not reliable: on the first object some seeds converged every run
and others none, since the best run of the first generation sets what the rest are
pulled towards. On a 20x20x20 object with a 6x6x6 box (object seed 2), all four
methods converged 80 of 80 over the same 10 seeds, so that case does not separate
them. Two objects and one iteration budget are a small sample; do not read these
counts as a general rate.

`qp.evolve` is provided for compatibility with cohere's genetic-algorithm
workflow. These runs support using it on a support with twin symmetry; they do not
show it improves reconstructions in general.

## Low-pass resolution progression

`qp.low_pass(*stages, sigma=(2.0, 0.5))` runs its stages against a Gaussian-blurred
copy of the measured amplitude, with the blur width ramping linearly from
`sigma[0]` to `sigma[1]` detector pixels over the stages' iterations, then goes
back to the data. A last sigma of 0 ends on the data as measured. It follows
cohere's `lowpass_filter`: the filter is applied to the data, not to the computed
wave, so the modulus constraint just projects onto a smoother target and the
iterate is never filtered. Masked pixels are left out of the blur, and the error
metrics inside the stage are against the blurred data.

```python
recipe = qp.Recipe(
    qp.low_pass(qp.RAAR(50), qp.repeat(8, qp.RAAR(50), qp.Shrinkwrap()), sigma=(1.0, 0.0)),
    qp.ER(200),
)
```

The schedule is carried in the iteration state, so the stages inside can be
anything, including `repeat`, `detwin`, `phase_limit` and `partial_coherence`
(either nesting order). For cohere's per-generation setting, give one generation
of `qp.evolve` its own recipe.

### Does it help

Not consistently, on the cases below. Each cell is the number of converged runs out
of 512 (16 seeds of 32 starts). Noiseless cases count a run converged below a
Fourier error of 1e-3; the noisy one counts a run whose agreement with the true
object (peak normalised cross-correlation, up to twin) exceeds 0.95. The low-pass
columns cover every iteration before the final `ER(200)`; the square-support case
uses `RAAR(300)` then `ER(150)`.

```python
import numpy as np
import qoraphase as qp


def make_object(n, m, seed, box=False):
    rng = np.random.default_rng(seed)
    block = rng.uniform(0.5, 1, (m, m)) * np.exp(1j * rng.uniform(-0.5, 0.5, (m, m)))
    if not box:  # a triangle, so the support is not centrosymmetric
        block = block * (np.indices((m, m)).sum(0) < m)
    obj = np.zeros((n, n), complex)
    lo = n // 2 - m // 2
    obj[lo : lo + m, lo : lo + m] = block
    return obj


def pattern(obj):
    return np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))) ** 2


def agreement(a, b):
    """Peak normalised cross-correlation, taking the better of a run and its twin."""
    twin = np.conj(np.roll(a[::-1, ::-1], 1, axis=(0, 1)))
    best = 0.0
    for x in (a, twin):
        peak = np.abs(np.fft.ifft2(np.fft.fft2(x) * np.conj(np.fft.fft2(b)))).max()
        best = max(best, peak / (np.linalg.norm(x) * np.linalg.norm(b)))
    return best


def default_like(low_pass):
    head = [qp.RAAR(50), qp.detwin(qp.HIO(20)), qp.RAAR(50)]
    loop = qp.repeat(8, qp.RAAR(50), qp.Shrinkwrap())
    if low_pass is None:
        return qp.Recipe(*head, loop, qp.ER(200))
    return qp.Recipe(qp.low_pass(*head, loop, sigma=low_pass), qp.ER(200))


def box_like(low_pass):
    if low_pass is None:
        return qp.Recipe(qp.RAAR(300), qp.ER(150))
    return qp.Recipe(qp.low_pass(qp.RAAR(300), sigma=low_pass), qp.ER(150))


def poisson(intensity, peak):
    scale = peak / intensity.max()
    return np.random.default_rng(0).poisson(intensity * scale) / scale


# name, object, problem, recipe builder, whether success is judged by agreement
cases = []
for seed in (1, 2, 3):
    obj = make_object(48, 14, seed)
    cases.append(
        (f"autocorrelation support, object {seed}", obj, qp.CDI(pattern(obj)), default_like, False)
    )
obj = make_object(32, 10, 0, box=True)
cases.append(
    ("square support", obj, qp.CDI(pattern(obj), support=np.abs(obj) > 0), box_like, False)
)
obj = make_object(48, 14, 1)
cases.append(
    (
        "Poisson noise, 5000 counts at peak",
        obj,
        qp.CDI(poisson(pattern(obj), 5000)),
        default_like,
        True,
    )
)

schedules = [None, (1.0, 0.0), (2.0, 0.0)]
print(f"{'':40s}" + "".join(f"{str(s):>14s}" for s in schedules))
for name, obj, problem, build, by_agreement in cases:
    row = []
    for sigma in schedules:
        good = 0
        for seed in range(16):
            for run in problem.reconstruct(build(sigma), starts=32, seed=seed):
                good += (
                    agreement(np.asarray(run.object), obj) > 0.95
                    if by_agreement
                    else run.error < 1e-3
                )
        row.append(good)
    print(f"{name:40s}" + "".join(f"{g:>14d}" for g in row))
```

```
                                                  None    (1.0, 0.0)    (2.0, 0.0)
autocorrelation support, object 1                   23            55            39
autocorrelation support, object 2                  125           113           128
autocorrelation support, object 3                  249           197           178
square support                                       6             5             0
Poisson noise, 5000 counts at peak                 258           221           245
```

It converged more runs on one autocorrelation-support object (23 to 55 at sigma
1 to 0), made no clear difference on another, and converged fewer on a third (249
to 197 and 178). Summed over the three, no blur setting beat the unblurred recipe.
The square support (6 runs without it) and the noisy pattern showed no gain. A
sigma tied to the fringe spacing of each pattern, or a different span of
iterations, might do better; these runs do not show that. Run it on your own data
and compare before relying on it.

## What this does not do

- No clustering of runs.
- No positivity/reality constraint — the object is complex here.
- No file I/O, no NUFFT, no autodiff optimisers.
- No BCDI geometry and no strain: [QoraGeom](https://github.com/NaNIsaac/QoraGeom)
  reads the SPEC scan, builds the detector-to-lab basis, resamples a
  reconstruction to the lab frame and computes strain from its phase.

## Development

```bash
uv sync --all-groups
uv run pytest
```
