# Default recipe on simulated BCDI data: support, not stagnation

The BCDI benchmark in rocking_xrd_sim#6 reported that `qp.reconstruct` with the default recipe
stops at a Fourier error of 0.05 on noiseless data, while the true object fits the frames to
2e-4, and that at strain 1.2 rad the noiseless cases had a larger phase error than the noisy
ones. This note tests why. In short:

1. The 0.05 is not stagnation. It is the floor imposed by the support the recipe ends with.
2. The "noiseless is worse" result is most likely selection luck. The phase error of single runs
   splits into two groups that the Fourier error cannot tell apart, and the lowest-error run comes
   from either. Picking the run the others agree with fixes this (QoraPhase#20).
3. A lower shrinkwrap threshold (0.2 instead of 0.3) halves the phase error at strain 0 to 1.2 rad
   with 1e5 photons or more. It does not help at 1e4 photons or at 3.0 rad. That is not enough
   to change the default; see the end of this note.

## Data and method

The quick grid of rocking_xrd_sim#6 (`python -m bench.make_cases --grid quick`) was regenerated
locally: an asymmetric 150 nm particle, 61 frames x 128 x 128 pixels at 5.55 keV, strain
amplitude 0, 1.2 and 3.0 rad, noiseless and 1e5 / 1e4 / 1e3 photons at the peak. In the frame
conjugate to the detector indices the particle spans about 9 pixels across and 8 frames.
The reference is that PR's `truth_det.npy`, the inverse DFT of the scattering amplitude at
every detector pixel. Scores come from its `bench.detframe`: the phase RMSE inside the particle
after removing translation, twin, global phase and phase ramp; the amplitude NRMSE; and the
support IoU at 30% of the maximum. "Fourier error" is QoraPhase's own error, unscaled, unless
marked as scaled.

All runs used QoraPhase at #17 (the phase-ramp FFT of #15), JAX 0.11.1 on CPU, default recipe
unless stated: RAAR 50, detwin(HIO 20), RAAR 50, 8 x [RAAR 50 + Shrinkwrap(0.3, sigma 1)],
ER 200. The support starts from the autocorrelation at 10%. The probes were scratch scripts and
are not part of this change. Each number below is from one run on a machine shared with other
jobs; wall time is not reported.

## 1. The floor is the support

Noiseless, strain 1.2 rad, seed 0. At strain 0 the Fourier errors of the same five runs are
0.051, 0.050, 0.051, 0.033 and 0.036:

| run | Fourier error | phase rad | amplitude NRMSE | final support voxels |
|---|---:|---:|---:|---:|
| default, random start (2 seeds) | 0.052 / 0.051 | 0.080 / 0.180 | 0.22 / 0.20 | 450 / 454 |
| default, then 2000 more ER | 0.052 | 0.079 | 0.22 | 450 |
| default, started from the truth | 0.051 | 0.103 | 0.21 | 456 |
| ER 200 from the truth, autocorrelation support | 0.033 | 0.023 | 0.04 | 1083 |
| RAAR 600 + ER 200 from random, true support (> 2% of max) | 0.029 | 0.019 | 0.04 | 1016 |

Starting from the exact answer, the default recipe walks away from it to the same error as a
random start. More iterations change nothing. There is no stagnation basin to escape: the
recipe converges to the best object its final support allows.

That support is too small for this object. The truth is not compact: the q window is finite, so
the object is band-limited and has tails over the whole array. Cutting the truth to a support
and computing its scaled Fourier error gives the best any support-constrained reconstruction
can reach:

| truth cut to | voxels | energy outside | Fourier error (scaled) |
|---|---:|---:|---:|
| nothing | 999,424 | 0 | 0.0002 |
| > 0.1% of max | 17,361 | 2.4e-5 | 0.0034 |
| > 1% of max | 1,924 | 9.4e-4 | 0.025 |
| > 2% of max | 1,016 | 1.8e-3 | 0.034 |
| > 5% of max | 540 | 3.9e-3 | 0.052 |
| shrinkwrap(0.3, sigma 1) of its own amplitude | 433 | | 0.075 |
| shrinkwrap(0.2, sigma 1) of its own amplitude | 582 | | 0.053 |
| shrinkwrap(0.1, sigma 1) of its own amplitude | 760 | | 0.045 |

So 2e-4 is reachable only without a support constraint, and the default recipe's 0.05 matches
the truth cut at 5%. rocking_xrd_sim#6 now reports these support-limited floors next to 2e-4.

## 2. The lowest Fourier error does not pick the best phase

With 8 starts per seed (default recipe), the phase error of single runs at strain 1.2 falls into
two groups, about 0.075-0.09 rad and 0.18-0.26 rad. The Fourier errors of the two groups overlap
within 1e-3 (0.050-0.053 noiseless, 0.060-0.063 at 1e5). The rank correlation between Fourier
error and phase error within one ensemble ranged from -0.14 to 0.69 over the ten ensembles
below. The lowest-error run therefore came from the bad group in 3 of 6 strained ensembles:
twice at 1e5, once noiseless.

That explains the benchmark's "noiseless worse than noisy at 1.2 rad" without any difference
between the data. With best-of-4 by Fourier error in 3 seeds, the noiseless picks happened to
land in the bad group more often. This is an inference from these runs, not a proof.

A selector that needs no truth does better: take the run with the smallest mean distance to the
other runs after alignment (shift, global phase, twin), and average all runs aligned onto it.
Phase RMSE in rad:

| case | seed | median single run | best single run | lowest error | consensus run | average on lowest error | average on consensus |
|---|---|---:|---:|---:|---:|---:|---:|
| strain 1.2, noiseless | 0 | 0.180 | 0.082 | 0.242 | 0.082 | 0.181 | 0.079 |
| | 1 | 0.086 | 0.074 | 0.077 | 0.086 | 0.072 | 0.072 |
| | 2 | 0.095 | 0.073 | 0.078 | 0.079 | 0.075 | 0.075 |
| strain 1.2, 1e5 | 0 | 0.184 | 0.077 | 0.242 | 0.079 | 0.178 | 0.071 |
| | 1 | 0.080 | 0.076 | 0.077 | 0.077 | 0.075 | 0.075 |
| | 2 | 0.082 | 0.076 | 0.243 | 0.078 | 0.178 | 0.070 |
| strain 0, noiseless | 0 | 0.056 | 0.016 | 0.028 | 0.025 | 0.010 | 0.010 |
| | 1 | 0.034 | 0.011 | 0.032 | 0.030 | 0.015 | 0.015 |
| | 2 | 0.043 | 0.004 | 0.046 | 0.014 | 0.013 | 0.013 |
| strain 3.0, noiseless | 0 | 0.188 | 0.125 | 0.214 | 0.214 | 0.209 | 0.209 |

The average on the consensus run is never worse than the average on the lowest-error run. It
removes all three bad picks, and at strain 0 it beats every single run. At 3.0 rad nothing helps
(one seed). QoraPhase#20 makes this the behaviour of `Ensemble.average()` and adds
`Ensemble.consensus`.

## 3. Shrinkwrap threshold

The default recipe with the shrinkwrap threshold changed, 3 seeds x 4 starts per cell. Phase
RMSE in rad, mean over seeds of the lowest-error run, with the mean of the per-seed median
over starts in brackets:

| case | 0.3 / sigma 1 (default) | 0.2 / sigma 1 |
|---|---|---|
| strain 0, noiseless | 0.055 (0.046) | 0.017 (0.020) |
| strain 0, 1e5 | 0.048 (0.044) | 0.028 (0.030) |
| strain 1.2, noiseless | 0.127 (0.140) | 0.080 (0.082) |
| strain 1.2, 1e5 | 0.133 (0.115) | 0.076 (0.079) |
| strain 1.2, 1e4 | 0.116 (0.096) | 0.093 (0.118) |
| strain 3.0, noiseless | 0.212 (0.252) | 0.223 (0.262) |

With 0.2 the final support grows from about 455 to 590 voxels, and the Fourier error falls in
every cell (0.051 to 0.042 noiseless, 0.061 to 0.054 at 1e5, 0.107 to 0.103 at 1e4). The
amplitude NRMSE (about 0.2) and IoU (about 0.87) do not move: they are set by the edge of a
9-pixel particle, not by the threshold.

Other settings tried at strain 1.2 were worse on phase. With rms 0.28 (noiseless 3 seeds, 1e5 2
seeds), the lowest-error runs were at 0.11-0.23 rad. With 0.1 / sigma 2 (noiseless, 3 seeds),
they were at 0.09-0.15 rad, with medians of 0.20-0.25. 0.3 / sigma 2 gave mixed results.

The default stays at 0.3 / sigma 1 for now. The evidence above is one particle shape, one size
and one 128-pixel grid, and the gain is absent at 1e4 photons and at 3.0 rad. The current
default was chosen on QoraPhase's own synthetic set (see `Shrinkwrap`). Changing it should wait
for the full benchmark grid (three shapes, two sizes) and a rerun of QoraPhase's shrinkwrap tests
at 0.2 / sigma 1.

## Not covered

- 1e3 photons, the full grid, GPU runs.
- Why the iterate falls into two phase groups at strain 1.2. Possibly a phase vortex or a partial
  twin; not examined.
- Whether the consensus selector keeps working with fewer than 8 starts. The tests in #20 cover
  only the construction.
