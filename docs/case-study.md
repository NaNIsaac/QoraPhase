# Case study: making phase retrieval faster without changing its answers

Draft. Sections marked **TODO (GPU)** wait for a GPU environment: JAX has no CUDA build for
native Windows, and the WSL2 distribution that would provide one is NaNIsaac/DotFiles#23. Every
number below was measured on CPU, except one cuFFT probe in section 8, which is not a
performance claim. Where a number is a ratio from a loaded machine it says so. Nothing here claims
a GPU result.

## 1. The physical problem

Coherent diffraction imaging measures the far-field intensity `I(q) = |F[rho](q)|^2` of a
compact object `rho` and recovers `rho`, phase included, by iterating between two constraints:
the measured Fourier magnitude and a real-space support. In Bragg CDI (BCDI) the object is a
crystal grain, the data are a 3D rocking-curve stack, and the recovered phase is `Q·u(r)`:
`|Q|` times the component of the lattice displacement `u` along the scattering vector, with the
sign set by the FFT convention. The strain component along `Q` is `(1/|Q|) dphi/dx_Q`, which is
what makes strain visible.

QoraPhase implements the iterations (ER, HIO, RAAR, shrinkwrap, detwinning, multi-start under
`vmap`) in JAX. A typical run is 720 iterations of the default recipe per start, several starts,
on arrays from 64^2 to 256^3. Each iteration is one forward and one inverse FFT plus
elementwise work, so the cost is FFT-shaped. The goal of this study is to find where the time
goes and reduce it while showing that the reconstructions do not change beyond rounding.

## 2. Baseline (F2, QoraPhase#6)

`benchmarks/perf.py`, CPU only: Ryzen 5 7500F, 6 cores, Windows 11, Python 3.14.5, JAX 0.11.1,
complex64. Full tables are in `docs/perf.md` (#6).

| shape | default recipe, warm | iterations/s | RAAR step | forward+adjoint | raw fftn+ifftn |
|---|---:|---:|---:|---:|---:|
| 256^2 | 0.39 s | 1860 | 0.87 ms | 0.79 ms | 0.49 ms |
| 1024^2 | 4.39 s | 164 | 6.89 ms | 5.29 ms | 3.33 ms |
| 64^3 | 0.87 s | 825 | 1.77 ms | 1.60 ms | 0.98 ms |
| 128^3 | 14.52 s | 50 | 14.47 ms | 14.06 ms | 6.80 ms |

What the baseline showed:
- Time is iteration count times the cost of one modulus projection. A step costs about one
  `modulus_project`.
- `forward + adjoint` takes 77-97% of a RAAR step at 256^2 and up. It costs 1.5x to 2.4x the
  bare `fftn + ifftn` because of the `ifftshift`/`fftshift` around each transform.
- Compilation is a constant 0.5-1.7 s per new shape and matters only for small problems.
- Batching starts halves the per-start cost at 8 starts.

Caveat: at 128^3 the end-to-end run averages 20.2 ms per iteration, against 14.5 ms for an
isolated step measured in another part of the same run. A rerun under load showed no
per-iteration overhead inside `lax.scan`, so the gap is most likely load variation. This is
noted in `docs/perf.md` and was not isolated further.

## 3. Profiler evidence

**TODO (GPU).** Nsight Systems and Nsight Compute traces of the compiled program (BACKLOG H6):
kernel list per iteration, whether XLA fuses the elementwise work around cuFFT, DRAM traffic,
achieved bandwidth against the card's peak, and launch overhead. This needs JAX on CUDA
(DotFiles#23).

What exists instead is timing evidence on CPU (section 2) and the compiled CPU program.
`benchmarks/hlo.py` (#6) reads the top-level instructions of the optimized XLA program for
`adjoint(forward(x))`; each one writes its result, so they count passes over the data outside
the two FFTs (JAX 0.11.1, CPU, complex64; no timing, so machine load does not enter):

| shape | shifted (#6) | phase ramp (#15) | bare fftn+ifftn |
|---|---:|---:|---:|
| 256^2, 1024^2 | 8 | 3 | 2 |
| 64^3, 128^3, 65x64x63 | 12 | 3 | 2 |

XLA does not fuse the shifts into the FFT: each is a roll per axis, compiled to a
slice-and-concatenate that writes the whole array. The phase-ramp form leaves one elementwise
multiply before, between and after the transforms (the middle one merges the forward output
ramp, the scaling and the adjoint input ramp). This confirms the mechanism section 2 inferred;
how long the passes take is still inferred from timing, not profiled, and the GPU program is
compiled by a different backend.

## 4. Measurement method (H2, QoraPhase#13)

`benchmarks/harness.py` times RAAR, HIO and ER per shape. Compile time is measured separately,
followed by warm-up calls and repeated calls that each end in `block_until_ready`. It reports
the median, quartiles and min-max, peak device memory where the backend reports it (the CPU
backend does not), and `--compare before after`.

The first use of the harness changed how every later number was taken. Two runs of identical
code differed by up to 2.5x on this shared machine, and in 4 of 15 cases the min-max ranges did
not overlap. From then on:
- before/after comparisons alternate in rounds (A B B A) and report the spread of the A runs
  next to the A/B ratio;
- absolute times from a loaded machine are not quoted as results;
- every column in a comparison uses the same JAX, NumPy and Python versions. Moving from JAX
  0.11.1 to 0.12 for GPU support changes the CPU baseline, and that change is reported as its
  own row (internal note: NaNIsaac FORUM.md, topic T1).

## 5. Change 1: centered FFT without moving data (F14, QoraPhase#15)

`fftshift(fftn(ifftshift(x)))` with `c = n // 2` per axis is

    sum_j x_j exp(-2 pi i (j - c)(k - c) / n)
      = exp(2 pi i c k / n) exp(-2 pi i c^2 / n) fftn(x_j exp(2 pi i c j / n))_k

so the shifts become a separable phase ramp on input and output. For even `n` the ramps are +-1.
The public convention (center origin, `norm="ortho"`) is unchanged.

Result: alternating A/B rounds on a loaded machine (`docs/perf-fft-shift.md`). The new form
was faster in 39 of 40 per-round comparisons:

| case | base / new |
|---|---:|
| forward+adjoint, 256^2 / 1024^2 / 64^3 / 128^3 | 2.14 / 1.53 / 1.90 / 1.91 |
| RAAR step, same sizes | 1.95 / 1.37 / 1.29 / 1.86 |
| default recipe end to end, 256^2 / 64^3 | 1.23 / 1.31 |

These are ratios from a loaded machine (the spread of identical base runs was 3-99% per case),
not absolute speeds. A quiet-machine rerun with `harness.py --compare` is still to do. Whether
the speedup carries over to a GPU, where cuFFT and XLA fusion behave differently, is **TODO (GPU)**.

## 6. Correctness: the same answers

Bitwise identity is not expected: the change reorders floating-point operations. Three layers
of evidence replace it.

1. **Against the shifted transform** (#15): `forward`/`adjoint` match the shifted NumPy
   transform on even and odd shapes in 1D to 3D, and a full reconstruction through the new
   `FFT` matches the same run through the old one (metrics to rtol 1e-3, object to 1e-3 of its
   maximum).
2. **Against a double-precision reference** (QoraPhase#17, `qoraphase.testing`): a plain NumPy
   implementation of ER, HIO, RAAR, shrinkwrap and detwin runs from the same start. The
   tolerance for each case is measured rather than chosen: it is the distance between the
   complex64 and complex128 runs of the reference itself, and a backend passes within 10x that
   floor. The same measurement showed why tolerances have to be measured. HIO and RAAR amplified
   a 1e-7 rounding difference to 1e-2-1e-1 within 50 iterations for every pair measured, which
   is expected since on non-convex constraint sets they are not contractions. Step-by-step
   comparison is therefore only meaningful over a few iterations. ER did not amplify it over
   200 iterations.
3. **Across implementations** (QoraPhaseCU#1, #2): QoraPhaseCU, the C++/CUDA port, now
   accepts a starting object. Its CPU backend follows QoraPhase iteration by iteration from
   shared starts on 2D and 3D, even and odd shapes, with masks, shrinkwrap and detwin (18 tests,
   relative distance < 1e-4). Its CUDA backend joins the same tests when a device is present:
   **TODO (GPU)**.

## 7. A correctness finding along the way: support and selection (F13d/F13e)

A BCDI benchmark (rocking_xrd_sim#6) suggested that the default recipe "stagnates" on clean
data. Testing that claim turned up two things that matter more than speed
(`docs/bcdi-support.md`, QoraPhase#21):

- **No stagnation.** Started from the true object, the default recipe ends at the same Fourier
  error (0.051) as random starts, and 2000 more ER iterations change nothing. The floor is set
  by the final shrinkwrap support; larger supports reach 0.029-0.033.
- **Selecting by Fourier error is blind to phase.** At strain 1.2 rad the per-start phase error
  splits into two groups (about 0.08 and 0.2 rad) whose Fourier errors overlap within 1e-3.
  Averaging all runs aligned onto the run the others agree with (QoraPhase#20) never did worse
  than the old average in 10 case/seed pairs, and removed all three bad picks (0.18 to
  0.07-0.08 rad).
- **Shrinkwrap threshold.** A looser threshold (0.2) cuts the BCDI phase error to about
  0.3-0.7 of the default's at strain 0-1.2 rad with 1e5 photons or more. At 1e4 photons the
  gain is not consistent (0.8 for the lowest-error pick, 1.2 for the median over starts), and
  at 3.0 rad there is none. On QoraPhase's own synthetic objects, which are exactly compact on the grid,
  it converges in 0 of 32 starts, against 9 and 14 of 32 at 0.3. A schedule (six rounds at
  0.3, then two at 0.2) keeps 9 and 14 of 32 there and matches 0.2 on three BCDI cases. The counts and the BCDI table are in section 4 of
  `docs/bcdi-support.md` (#21). The survey across 3 shapes x 2 sizes x 3 strains x 3 doses
  (F13e) is running; whether the schedule becomes the default is Isaac's decision after it.

## 8. Speed and memory against size

**TODO (GPU).** The table this report is for: wall time and peak VRAM against size, 64^2 to
256^3, for JAX CPU, JAX GPU (WSL2) and QoraPhaseCU CUDA (Windows and WSL2). It will use
`harness.py`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` so that peak memory means use rather than
reservation, the same JAX version in every column, and A/A spread next to every ratio. The
comparison measures the WSL2 cost by building QoraPhaseCU on both sides (internal note: NaNIsaac
FORUM.md, topic T1).

Hardware for that table (QoraPhase#10, `docs/hardware.md`): RTX 5070 (Blackwell, compute
capability 12.0, 12 GB), driver 610.88 / CUDA 13.3. The `jax-cuda13` 0.12 wheels contain sm_120
targets, but no JAX program has run on this card. CuPy on native Windows ran one cuFFT probe
(256^3 complex64 `fftn` in 1.6 ms), which is not a benchmark and not part of any comparison.

## 9. Limits

- CPU numbers come from one machine shared with other jobs. Ratios were taken in alternating
  rounds; absolute times are indicative only.
- No profiler evidence yet. The bottleneck attribution in section 2 is from timing; the
  compiled CPU program (section 3) shows which passes exist, not what they cost.
- The equivalence bounds hold for short HIO/RAAR runs. Long runs are compared by where they
  converge, not step by step.
- The BCDI findings come from one simulator and, until F13e finishes, one particle shape.
- Object size. The simulated particle spans about 9 px and the synthetic test objects 10-12 px.
  At that size a 1-px support rim is a large part of the object (about half the voxels of a
  9-px sphere are in its outer pixel), so the shrinkwrap findings may be much weaker for
  particles 30-60 px across (inference). Photon counts below 1e4 are not covered.
- GPU: nothing measured.

## 10. Reproduce

```console
# baseline (F2)
uv run python benchmarks/perf.py > perf-results.md
# compiled program around the FFTs (section 3); the script is in #6, the #15 column
# is the same script run with #15 checked out
uv run python benchmarks/hlo.py
# harness, before/after (H2)
uv run python benchmarks/harness.py --json before.jsonl   # on the base commit
uv run python benchmarks/harness.py --json after.jsonl    # on the change
uv run python benchmarks/harness.py --compare before.jsonl after.jsonl
# equivalence (H3)
uv run pytest tests/test_equivalence.py tests/test_fft.py
# QoraPhaseCU against QoraPhase
cd ../QoraPhaseCU && uv sync --group qoraphase && uv run pytest python/tests/test_equivalence.py
```

The F13d/F13e probes ran from scratch scripts. Internal note: the scripts, their raw outputs, the
F13e grid, the command and the commit hashes are kept outside the repository in
`$DEV/NaNIsaac/_artifacts/f13e-shrinkwrap-20261022/` (README.txt, `probes/`) on the lab machine.
