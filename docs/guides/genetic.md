# Genetic algorithm

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

Not run in CI. Published numbers first appear in [commit `9eb8b7c`](https://github.com/NaNIsaac/QoraPhase/commit/9eb8b7c10a4223ea547b3eee2f4ea805ad2bf783); use that commit to reproduce them. Planning runtime estimate: a few minutes on a desktop CPU, not a measured duration. The original table did not record its exact CPU or elapsed time. The surrounding historical examples record CPU JAX 0.11.1, Python 3.14.7 and Intel64 Family 6 Model 158; that is context, not a verified per-table hardware record. No benchmark was rerun for this move.

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
