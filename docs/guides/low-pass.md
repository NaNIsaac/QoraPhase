# Low-pass resolution progression

Not in the 0.2.0 release on PyPI. Until the next release, install from GitHub: `uv pip install "git+https://github.com/NaNIsaac/QoraPhase"`.

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

Not run in CI. Published numbers first appear in [commit `e7471c6`](https://github.com/NaNIsaac/QoraPhase/commit/e7471c6d43a8bb588f93ae29adb9fda2e15191b6); use that commit to reproduce them. Planning runtime estimate: several to tens of minutes on a desktop CPU, not a measured duration. The original table did not record its exact CPU or elapsed time. The surrounding historical examples record CPU JAX 0.11.1, Python 3.14.7 and Intel64 Family 6 Model 158; that is context, not a verified per-table hardware record. No benchmark was rerun for this move.

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
