# Centered FFT without moving data

`docs/perf.md` found that `FFT.forward` / `FFT.adjoint` take 1.5x to 2.4x the time of a bare
`fftn` / `ifftn`, and *hypothesised* that the `ifftshift` / `fftshift` calls are materialised
copies. This change replaces them and measures the effect. Issue #2, CPU only.

## Change

`fftshift(fftn(ifftshift(x)))` with `c = n // 2` per axis equals

    sum_j x_j exp(-2 pi i (j - c)(k - c) / n)
      = exp(2 pi i c k / n) exp(-2 pi i c^2 / n) * fftn(x_j exp(2 pi i c j / n))_k

so the shifts become a separable phase ramp on the input and another on the output. For even
`n` the ramps are +-1 (they are rounded to 14 places so that they come back real), which turns
the multiplications into sign flips that can fuse with the neighbouring elementwise operations.
The adjoint uses the conjugate ramps. Odd lengths use complex ramps and are covered by the
tests. The public convention is unchanged: arrays are center-origin, index `n // 2` is the
origin, `norm="ortho"`.

The compiled CPU program confirms it (`benchmarks/hlo.py` in #6, JAX 0.11.1): outside the two
FFTs, `forward` + `adjoint` writes 12 arrays in 3D and 8 in 2D with the shifts, and 3 with the
ramps, against 2 for the bare transforms.

## Output equality

`tests/test_fft.py` compares `forward` and `adjoint` with the shifted NumPy transform for
even and odd shapes in 1D, 2D and 3D, checks real input, and checks that a full
`reconstruct` (RAAR(30) + ER(10)) with the new `FFT` agrees with the same run through a
subclass that uses the old `fftshift` form: `metrics` to rtol 1e-3 and the object to 1e-3 of
its maximum. The results are not bitwise identical, because the floating-point operations
are ordered differently; the test asserts agreement to a tolerance, not equality.

## Timing

Alternating rounds, order base, new, new, base, 4 rounds, same machine as `docs/perf.md`
(Ryzen 5 7500F, JAX 0.11.1 CPU, complex64) but **with other jobs running on it**: the same
code varied by up to 2x between runs, and the absolute times below are 2-3x the quiet-machine
numbers of `docs/perf.md`. Only the ratio of the interleaved runs carries information.
`base` is the parent commit. Each cell is the median of 15 calls per run (3 runs for
`reconstruct`), then the median over the 8 runs of each variant. "base spread" is
(max - min) / median of the 8 base runs, i.e. the noise between runs of identical code.

| case | base ms | new ms | base / new | per-round base / new | base spread |
|---|---:|---:|---:|---|---:|
| 256x256 forward+adjoint | 0.92 | 0.43 | 2.14 | 2.44 2.49 2.27 1.98 | 61% |
| 256x256 RAAR step | 1.16 | 0.59 | 1.95 | 1.46 2.21 3.04 1.60 | 99% |
| 256x256 `reconstruct` | 611 | 496 | 1.23 | 1.38 1.31 1.24 1.17 | 34% |
| 1024x1024 forward+adjoint | 13.3 | 8.7 | 1.53 | 1.44 1.33 1.26 1.79 | 16% |
| 1024x1024 RAAR step | 15.4 | 11.2 | 1.37 | 1.40 1.34 1.39 1.29 | 30% |
| 64^3 forward+adjoint | 4.03 | 2.12 | 1.90 | 2.29 1.93 1.95 1.56 | 66% |
| 64^3 RAAR step | 3.76 | 2.91 | 1.29 | 1.44 0.98 1.40 1.44 | 64% |
| 64^3 `reconstruct` | 1478 | 1130 | 1.31 | 1.28 1.34 1.35 1.30 | 3% |
| 128^3 forward+adjoint | 36.7 | 19.2 | 1.91 | 2.06 1.76 1.89 2.13 | 15% |
| 128^3 RAAR step | 41.8 | 22.5 | 1.86 | 1.73 1.89 1.71 1.78 | 21% |

`reconstruct` is the default 720-iteration recipe, one start, seed 0; 2D 1024^2 and 3D 128^3
end to end were not repeated here because they are too slow under this load.

Reading: the new form is faster in 39 of the 40 per-round comparisons (the exception is the
64^3 RAAR step in round 2, 0.98). The gain is 1.2x to 2.1x depending on the case; on
`forward+adjoint` alone it is 1.5x to 2.1x, which matches the 1.5x to 2.4x that `perf.md`
attributed to the shifts. A whole step gains less than the transform because the
projections are unchanged, and a whole `reconstruct` gains 1.2x to 1.3x at the two sizes
that were run. The 256^2 numbers are the least certain: at 0.4 to 1.2 ms per call, dispatch
and scheduling noise is the same size as the effect (base spread 61-99%), and the 2.0x and
3.0x step ratios there should be read as "clearly faster", not as a measured factor.

## Quiet-machine rerun

2026-10-23, `benchmarks/perf.py` (#6) run four times in the order base, new, new, base, with no
other job on the machine (load snapshots before each run showed nothing but System and Defender
above 0.6 core). Same JAX 0.11.1, NumPy 2.5.2, Python 3.14.5 for both. Each cell is the mean of
the two runs of a variant; "base spread" is the difference between the two base runs over their
mean.

| case | base | new | base / new | base spread |
|---|---:|---:|---:|---:|
| 256x256 forward+adjoint, ms | 0.96 | 0.64 | 1.49 | 11% |
| 256x256 RAAR step, ms | 1.01 | 0.75 | 1.34 | 9% |
| 1024x1024 forward+adjoint, ms | 3.99 | 3.24 | 1.23 | 2% |
| 1024x1024 RAAR step, ms | 5.47 | 4.67 | 1.17 | 9% |
| 64^3 forward+adjoint, ms | 1.46 | 0.95 | 1.53 | 5% |
| 64^3 RAAR step, ms | 1.53 | 1.19 | 1.29 | 4% |
| 128^3 forward+adjoint, ms | 12.8 | 6.2 | 2.08 | 29% |
| 128^3 RAAR step, ms | 13.7 | 10.3 | 1.33 | 10% |
| 256^3 forward+adjoint, ms | 172 | 85 | 2.02 | 9% |
| 256^3 RAAR step, ms | 192 | 105 | 1.83 | 6% |
| `reconstruct` 512x512, s | 1.19 | 1.01 | 1.17 | 1% |
| `reconstruct` 1024x1024, s | 3.63 | 2.47 | 1.47 | 18% |
| `reconstruct` 64^3, s | 0.76 | 0.58 | 1.32 | 8% |
| `reconstruct` 128^3, s | 10.80 | 5.44 | 1.99 | 8% |

On the quiet machine the new form is faster in every case where the gain exceeds the base
spread, by 1.2x to 2.1x, and the gain grows with size in 3D: a whole 128^3 reconstruction takes
half the time. The bare `fftn + ifftn` did not change (base / new 0.90 to 1.12), so the gain is
the removed copies. At 256^2 end to end the two runs of the new form disagreed (0.29 and 0.45 s)
and nothing is concluded there; at 128^2 and below the two forms are equal within the noise.

## Not measured

- A rerun with `benchmarks/harness.py --compare` (#13), which reports quartiles per case.
- Anything on a GPU. Whether XLA on CUDA fuses the ramps into the cuFFT call or around it is
  not known.
- Memory use.
