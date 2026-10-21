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

Whether XLA really removes the copies was not inspected in the compiled program; what is
measured below is only the wall time.

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

## Not measured

- A quiet-machine rerun. `benchmarks/harness.py` (#13) can repeat this with `--compare` once
  it is merged.
- Anything on a GPU. Whether XLA on CUDA fuses the ramps into the cuFFT call or around it is
  not known.
- Memory use.
