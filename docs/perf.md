# Where reconstructions spend their time

Measured on CPU with `benchmarks/perf.py`, on synthetic data, using the public API
(`qp.reconstruct` with the default recipe) and the stage classes directly. This is
Phase A of issue #2 for the CPU only. No GPU was run, so there are no GPU numbers here,
and no custom kernel was tried.

Reproduce (about 3 minutes on the machine below; `--quick` takes under a minute and uses
small sizes):

```console
uv run python benchmarks/perf.py > perf-results.md
uv run python benchmarks/hlo.py    # compiled program around the FFTs, seconds, no timing
```

## Setup

- CPU: AMD Ryzen 5 7500F (6 cores, 12 logical), 31.6 GB RAM
- OS: Windows 11 (10.0.26300), Python 3.14.5
- JAX 0.11.1, NumPy 2.5.2, JAX device `CpuDevice(id=0)`, float32 / complex64 (x64 off)
- The machine has an NVIDIA GeForce RTX 5070, but the JAX installed here is the CPU
  build, so nothing ran on it.
- Data: noiseless intensity of a smooth complex object inside a centred ellipsoid that
  spans half of each axis (`synthetic()` in the script), deterministic, no mask, support
  from the autocorrelation (the default). 2D: 64² to 1024². 3D: 32³ to 128³ end to end,
  up to 256³ for single-iteration timings.
- Default recipe: 720 iterations (RAAR 50, detwin HIO 20, RAAR 50, 8 x [RAAR 50 +
  Shrinkwrap], ER 200), one start, seed 0.
- Every timing ends in `block_until_ready` or a host read. "Warm" is the median of 4
  repeats (2 for arrays of 2^20 voxels and up) after one untimed call; the min-max is
  shown. One run of the whole script produced the numbers below; they were not
  repeated across separate sessions.

## End to end

| shape | voxels | cold s | warm median s | warm min-max s | cold - warm s | iterations/s | final error |
|---|---:|---:|---:|---:|---:|---:|---:|
| 64x64 | 4,096 | 0.78 | 0.03 | 0.02-0.03 | 0.75 | 27315.6 | 1.9e-03 |
| 128x128 | 16,384 | 0.71 | 0.09 | 0.09-0.09 | 0.62 | 7941.2 | 1.9e-03 |
| 256x256 | 65,536 | 1.04 | 0.39 | 0.38-0.44 | 0.65 | 1859.7 | 2.7e-03 |
| 512x512 | 262,144 | 1.85 | 1.27 | 1.25-1.31 | 0.58 | 566.8 | 2.2e-02 |
| 1024x1024 | 1,048,576 | 5.06 | 4.39 | 4.31-4.47 | 0.67 | 163.9 | 1.5e-02 |
| 32x32x32 | 32,768 | 1.78 | 0.24 | 0.23-0.35 | 1.53 | 2939.6 | 6.8e-03 |
| 64x64x64 | 262,144 | 2.58 | 0.87 | 0.86-0.91 | 1.71 | 825.1 | 6.7e-03 |
| 128x128x128 | 2,097,152 | 15.05 | 14.52 | 14.42-14.62 | 0.53 | 49.6 | 9.4e-03 |

"iterations/s" is 720 divided by the warm median. "Final error" is the Fourier error of
the single run; it says the runs did something sensible, not that the recipe is tuned for
these sizes.

## Cost of the primitives inside one iteration

Each row is one jitted call timed on its own (median of 10). At 64² the call is short
enough that Python dispatch is a visible part of it; the scan-based per-iteration numbers
in the stage table below are the better estimate there.

Selected sizes (the full table is the script output):

| shape | raw fftn+ifftn ms | forward+adjoint ms | modulus_project ms | RAAR step ms | support_project ms | Shrinkwrap ms |
|---|---:|---:|---:|---:|---:|---:|
| 256x256 | 0.49 | 0.79 | 0.88 | 0.87 | 0.07 | 0.71 |
| 1024x1024 | 3.33 | 5.29 | 6.28 | 6.89 | 0.69 | 4.51 |
| 64x64x64 | 0.98 | 1.60 | 1.64 | 1.77 | 0.23 | 1.19 |
| 128x128x128 | 6.80 | 14.06 | 15.35 | 14.47 | 1.14 | 9.14 |
| 256x256x256 | 73.3 | 173.6 | 202.5 | 202.3 | 14.3 | 109.6 |

`raw fftn+ifftn` is `jnp.fft.ifftn(jnp.fft.fftn(a))` with no shifts. `forward+adjoint`
is `FFT.adjoint(FFT.forward(a))`, i.e. the same transforms inside the
`ifftshift`/`fftshift` sandwich the model uses to keep arrays center-origin.

## Stages of the default recipe

Each stage jitted and timed separately; "sum of stages" against "whole recipe, one jit"
shows what fusing them into one program changes (little).

| shape | RAAR(50) s | detwin(HIO(20)) s | RAAR(50) s | 8 x [RAAR(50)+Shrinkwrap] s | ER(200) s | sum s | whole recipe s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128x128 | 0.005 | 0.002 | 0.005 | 0.047 | 0.022 | 0.083 | 0.085 |
| 256x256 | 0.030 | 0.012 | 0.030 | 0.225 | 0.100 | 0.396 | 0.417 |
| 512x512 | 0.068 | 0.027 | 0.064 | 0.482 | 0.223 | 0.864 | 0.781 |
| 32x32x32 | 0.013 | 0.005 | 0.014 | 0.112 | 0.054 | 0.198 | 0.197 |
| 64x64x64 | 0.056 | 0.022 | 0.051 | 0.414 | 0.229 | 0.772 | 0.798 |

Shares of the sum are the same at every size, within a few points: 7-8% / 3% / 7% /
54-59% / 24-30%, which is just the iteration count of each stage (50/20/50/400/200 of 720).

## NumPy reference against JAX, RAAR only (20 iterations)

A plain NumPy RAAR with the same equations (`numpy_raar` in the script), started from the
same initial object.

| shape | NumPy complex128 it/s | JAX complex64 it/s | JAX / NumPy | relative diff |
|---|---:|---:|---:|---:|
| 64x64 | 5481 | 30363 | 5.5 | 1.1e-05 |
| 128x128 | 1724 | 8150 | 4.7 | 6.7e-05 |
| 256x256 | 168 | 1758 | 10.5 | 4.5e-04 |
| 512x512 | 36 | 771 | 21.3 | 1.5e-04 |
| 32x32x32 | 668 | 3770 | 5.6 | 1.3e-03 |
| 64x64x64 | 37 | 976 | 26.5 | 5.2e-03 |

This is not an equal-work comparison: NumPy runs in complex128 and JAX in complex64, and
the speed ratio therefore mixes precision, threading and compilation. The relative
difference after 20 iterations (1e-5 to 5e-3) is the distance between the two results;
how much of it is float32 rounding was not isolated.

## Multi-start under vmap

| shape | starts | warm median s | s per start |
|---|---:|---:|---:|
| 128x128 | 1 | 0.10 | 0.10 |
| 128x128 | 4 | 0.32 | 0.08 |
| 128x128 | 8 | 0.44 | 0.05 |
| 32x32x32 | 1 | 0.21 | 0.21 |
| 32x32x32 | 4 | 0.55 | 0.14 |
| 32x32x32 | 8 | 0.92 | 0.11 |

## Compiled program around the FFT

`benchmarks/hlo.py` compiles `FFT.adjoint(FFT.forward(x))` and the bare transforms and
reads the top-level instructions of the optimized program (JAX 0.11.1, CPU, complex64).
Every top-level instruction writes its result, so the last column counts passes over the
data outside the transforms, in units of the input array. This does not depend on machine
load and was not timed.

| shape | case | FFTs | other instructions | arrays written outside the FFTs |
|---|---|---:|---:|---:|
| 256x256, 1024x1024 | forward+adjoint | 2 | 12 | 8 |
| 64x64x64, 65x64x63 | forward+adjoint | 2 | 12 | 12 |
| 128x128x128 | forward+adjoint | 2 | 16 | 12 |
| all of the above | raw fftn+ifftn | 2 | 2 | 2 |

In 2D each of the four shifts (`ifftshift` and `fftshift` around each transform) compiles to
two half-array slices and a concatenate; in 3D to one slice-and-concatenate per axis. The
`ortho` scaling is fused into the shifts next to it, which is the only fusion across them.

## Findings

Each follows from the tables above; the explanations marked *hypothesis* were not tested.

1. **Time is the iteration count times the cost of one modulus projection.** A RAAR, HIO or
   ER step costs about the same as `modulus_project` alone (e.g. 6.89 vs 6.28 ms at 1024²,
   14.5 vs 15.4 ms at 128³); the support projection is 7-15% of a step at 256² and up. Per-stage time
   tracks iteration count, so 400 of the 720 iterations (the shrinkwrap loop) take 54-59%
   of a run and ER(200) takes about a quarter. Shrinkwrap itself costs 54-82% of one step at 256² and up
   and runs 8 times, i.e. under 1% of the recipe.
2. **The FFT work, not the elementwise work, is the largest part of a step, and the
   `fftshift`/`ifftshift` sandwich costs about as much as the FFTs.** At 256² and up the raw
   `fftn + ifftn` is 36-57% of a RAAR step, while `forward + adjoint` is 77-97% of it. So
   the sandwiched transform takes 1.5x to 2.4x the time of the raw one (1.6x at 256²,
   1.6x at 1024², 1.6x at 64³, 2.1x at 128³, 2.4x at 256³). The compiled program shows
   why ("Compiled program around the FFT" above): the shifts are not fused into the FFT. Each shift is a roll per
   axis, compiled to a slice-and-concatenate that writes the whole array, so one
   `forward + adjoint` writes 8 arrays (2D) or 12 (3D) outside the two transforms, against
   2 for the bare transforms, where only the `ortho` scaling remains. How much time those
   passes take was inferred from the timings above, not profiled.
3. **Compilation is a roughly constant 0.5-1.7 s per new shape and recipe**, shown by
   cold minus warm (0.53-1.71 s across all sizes), and there is no recompilation on the
   repeated calls (warm times are stable, min-max within a few percent). It dominates
   small problems (64²: 0.78 s cold against 0.03 s warm) and is irrelevant at 1024² and
   128³ (5.1 vs 4.4 s, 15.1 vs 14.5 s).
4. **Launch overhead is not the limit inside a run.** With `lax.scan` the 64² case runs at
   about 0.03 ms per iteration, 27,000 iterations/s, against 0.07 ms for a single jitted
   step called from Python. Fusing all stages into one jit changes the total by about 10% or
   less in either direction (table above).
5. **Throughput falls faster than the voxel count grows at the top end:**
   1024² (1.05 M voxels) runs at 164 iterations/s, 128³ (2.10 M voxels) at 50, so twice
   the voxels cost 3.3x the time per iteration (cache behaviour of the 3D transform is a
   *hypothesis*, not measured). At 128³ one
   iteration of the recipe averages 20.2 ms (14.52 s / 720, all stages and Shrinkwrap
   included), while a single isolated RAAR step measured 14.5 ms in the primitives table;
   the two came from different parts of the same run and differ by 1.4x. A second run of
   the 128³ stages on a machine busy with other jobs gave 33-38 ms per iteration in
   every stage (RAAR, HIO, ER, Shrinkwrap loop) against 39 ms for an isolated RAAR step in
   the same process, so the per-iteration cost inside `lax.scan` is not higher than the
   isolated step there; the 1.4x gap in the first run is most likely load variation (this
   machine showed up to 2.5x between identical runs), not a per-stage overhead, but this
   was not isolated further. Use 20 ms (end to end) for run-time estimates and 14.5 ms
   only as the best isolated step. A 256³ step measured 202 ms, which would put the
   default recipe near 2.4 minutes per start (an extrapolation of 720 x 0.202 s, not a run).
6. **Batching starts helps but is not free:** per-start cost falls to about half at 8
   starts (128²: 0.10 to 0.05 s; 32³: 0.21 to 0.11 s) and total time still grows with
   the number of starts.

## Bottlenecks, ranked

1. FFT plus the shift sandwich inside `FFT.forward` / `FFT.adjoint`, 77-97% of every
   step at 256² and up (finding 2).
2. The sheer iteration count of the shrinkwrap loop and ER (findings 1, 5).
3. First-call compilation, only for small problems (finding 3).

Not bottlenecks on this machine: dispatch/launch overhead inside a run, the support
projection, Shrinkwrap.

## Not measured

- Any GPU: JAX GPU, peak VRAM/RAM, host-device transfer, PCIe. The numbers above say
  nothing about where a GPU run would be limited.
- 1024² / 256³ end to end with several repeats; 256³ was timed per step only.
- The memory traffic and time of each instruction in a step. Which instructions XLA
  emits around the FFT is now read from the compiled program (above); what they cost needs
  a profiler.
- Whether the CPU run is multi-threaded and how that scales.
- The effect of x64, of `batch=` chunking, a partial-coherence model, masks and low-pass.
- Phase B (a custom CUDA kernel). Nothing here shows it is warranted; the CPU profile
  points at the FFT and shift data movement, which the issue says to leave to cuFFT.

## Candidate changes, untested

- Keep the working arrays in FFT order internally so that the shifts leave the loop
  (finding 2). This changes layout handling in `FFT` and `CDI`; it needs a test that the
  output is unchanged.
- Report the above again on a GPU machine before drawing any conclusion about kernels.
