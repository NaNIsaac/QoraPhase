# Default recipe on simulated BCDI data: support, not stagnation

[Back to all reports](reports/README.md)

> **Question.** Why does the default recipe stop at a Fourier error of 0.05 on noiseless simulated BCDI data, and why did noiseless cases show a larger phase error than noisy ones at strain 1.2 rad?
>
> **Result.** The 0.05 is not stagnation but is set by the support the recipe ends with; the noiseless-is-worse result is most likely selection luck; a lower shrinkwrap threshold helps only in part, so the default stays at 0.3.
>
> **Limits.** Small objects (about 9-15 px), no 1e3-photon or GPU runs, one run per number on a shared machine.

The BCDI benchmark in rocking_xrd_sim#6 reported that `qp.reconstruct` with the default recipe
stops at a Fourier error of 0.05 on noiseless data, while the true object fits the frames to
2e-4, and that at strain 1.2 rad the noiseless cases had a larger phase error than the noisy
ones. This note tests why. In short:

1. The 0.05 is not stagnation. It is set by the support the recipe ends with.
2. The "noiseless is worse" result is most likely selection luck. The phase error of single runs
   splits into two groups that the Fourier error cannot tell apart, and the lowest-error run comes
   from either. Picking the run the others agree with fixes this (QoraPhase#20).
3. A lower shrinkwrap threshold (0.2 instead of 0.3) lowers the phase error to about 0.3-0.7 of
   the default's at strain 0 to 1.2 rad with 1e5 photons or more. It does not help consistently
   at 1e4 photons or at all at 3.0 rad. On QoraPhase's own
   sharp test objects it stops converging altogether (section 4). A schedule that loosens only
   in the last two rounds keeps both there, but over the benchmark's three shapes and two sizes
   it is worse than the default in 19 of 54 cells (section 5). The default stays at 0.3; the
   schedule is at most a recipe option.

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
and computing its scaled Fourier error bounds the floor for that support from above, since the
cut truth is one object inside it. The reconstructions land below the bound at similar support
sizes: 0.051 at about 455 voxels against 0.075 for the truth's own 0.3 shrinkwrap, and 0.042 at
about 590 voxels (section 3) against 0.053 for 0.2. Those two reconstruction errors are
unscaled, and scaling can only lower them.

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

So 2e-4 is reachable only with a support far larger than shrinkwrap produces. That the 5% cut
(540 voxels) gives 0.052, close to the recipe's 0.051 at about 455 voxels, is a coincidence of
numbers, not a match. The conclusion rests on the first table instead: started from the truth
the recipe ends at 0.051, 2000 more ER iterations change nothing, and larger supports reach
0.029-0.033. A direct estimate of the floor for a given support S would be ER started from the
truth cut to S with S held fixed, as in row 4 for the autocorrelation support; it was not run
for the shrinkwrap supports. rocking_xrd_sim#6 reports these truth-cut errors next to 2e-4.

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
at 0.2 / sigma 1. Both were done: the tests rule out plain 0.2 (section 4), and the
grid rules out the schedule as a default (section 5).

## 4. The threshold on QoraPhase's own data, and a schedule

The rerun of QoraPhase's tests rules out 0.2 as a plain default. With the `Shrinkwrap` default
set to 0.2, 2 of 206 tests fail.
`test_shrinkwrap_settings_matter_more_than_detwin_placement` only pins the default value.
`test_the_default_recipe_finds_the_object_without_a_support` fails for real. Counted directly on
its data (`make_data` from `tests/conftest.py`, default recipe with the shrinkwrap replaced,
autocorrelation support, 32 starts, seed 0; "converged" means Fourier error < 1e-4):

| shrinkwrap in the 8 rounds | 32^2, m 10, data seed 3 | 64^2, m 12, data seed 1 |
|---|---:|---:|
| 0.3 / sigma 1 (default) | 9 of 32 | 14 of 32 |
| 0.2 / sigma 1 | 0 of 32 (best error 2.2e-2) | 0 of 32 (best error 2.4e-2) |
| 0.25 / sigma 1 | 0 of 32 | 5 of 32 |
| 6 rounds at 0.3, then 2 at 0.2 | 9 of 32 | 14 of 32 |

What separates the two object sets is whether some finite support is exact, not how sharp the
edges are; physically both particles are faceted. `make_object` is exactly compact on the grid,
zero outside a 10-12 px simplex, so a reconstruction needs a support tight enough to pin it
down. The BCDI truth is band-limited and rings across the array, so no finite support is exact,
and a looser one mainly gives back rim energy.

Why 0.2 fails on the test objects was not measured. A likely mechanism, from an estimate that
was not run: 1 px outside a straight edge of the test object, the sigma-1 blurred amplitude is
about 0.25 of its maximum. Then 0.2 keeps a 1-px rim, 0.3 drops it, and 0.25 is borderline,
which fits 0 and 5 of 32. On a 55-px triangle that rim adds about 30 px and rounds the corners,
and the triangles are there because a centrosymmetric support lets starts stagnate in
object/twin mixtures (see `make_object`). Two checks on the 0.2 failures would separate twin
stagnation from plain stagnation and are not done: the final support area against the true
55 / 78 px, and `oriented_similarity` to the object and to its twin.

The schedule in the last row does the shrinking at 0.3 and loosens only after six rounds. That
its counts equal those of 0.3 is what a fixed point predicts: for the same object a lower
threshold gives a superset of the support, so a start that converged to the exact object under
0.3 still satisfies both constraints after loosening and stays there. The schedule can only lose
starts that would have converged in rounds 7-8. Only the counts were compared, not whether the
same starts converge.

On the BCDI quick grid the schedule matches 0.2. Phase RMSE in rad, mean over 3 seeds of the
lowest-error run of 4 starts, with the mean of the per-seed median over starts in brackets:

| case | 0.3 / sigma 1 | 0.2 / sigma 1 | schedule |
|---|---|---|---|
| strain 0, noiseless | 0.055 (0.046) | 0.017 (0.020) | 0.016 (0.019) |
| strain 1.2, noiseless | 0.127 (0.140) | 0.080 (0.082) | 0.076 (0.076) |
| strain 1.2, 1e5 | 0.133 (0.115) | 0.076 (0.079) | 0.072 (0.076) |

The lowest-error pick is blind to phase (section 2), and the differences between 0.2 and the
schedule are well inside the gap between the two phase groups, so this shows a match, not an
improvement; the medians, which do not depend on that pick, say the same. Its final support
(about 585 voxels) and Fourier error (0.042 noiseless) are those of 0.2. These three cases are
all from the one particle of the quick grid; section 5 tests the other shapes and sizes.

## 5. The schedule across the benchmark grid (F13e)

The benchmark's survey grid: asymmetric, sphere and truncated octahedron particles, 150 and
250 nm (about 9 and 15 px), strain 0, 1.2 and 3.0 rad, noiseless, 1e5 and 1e4 photons; 54 cells,
3 seeds x 4 starts per cell and recipe. QoraPhase at #20 (91cf87a), rocking_xrd_sim at #6
(3d135c8), JAX 0.11.1 on CPU.

The scoring rule was fixed before any phase value of the survey was read. Per seed, take the
median phase RMSE over the 4 starts. A cell is "worse" for the schedule if the mean over seeds of
(schedule - default) exceeds the standard deviation of the default over seeds. The schedule would
become the default only if no cell is worse and QoraPhase's tests pass with it. As a check that
does not depend on the spread estimate, a second column counts cells where the schedule is worse
in all three seeds. The scoring was repeated independently from the rule alone, with no
disagreement over 54 cells.

Result: the schedule is worse in 19 of 54 cells (16 by the all-three-seeds count; the two rules
disagree on 7 cells). The test-suite condition could not change that and was not run. The default
stays at 0.3 / sigma 1. Cells by group, by median phase error (the last column uses the mirrored
rule, which was not part of the registered scoring):

| group | cells | worse | worse in all 3 seeds | better (unregistered) |
|---|---:|---:|---:|---:|
| strain 0 | 18 | 10 | 9 | 1 |
| strain 1.2 | 18 | 4 | 2 | 3 |
| strain 3.0 | 18 | 5 | 5 | 3 |
| noiseless | 18 | 6 | 5 | 3 |
| 1e5 photons | 18 | 4 | 4 | 4 |
| 1e4 photons | 18 | 9 | 7 | 0 |
| asymmetric, 150 nm | 9 | 1 | 1 | 2 |
| asymmetric, 250 nm | 9 | 3 | 3 | 2 |
| sphere, 150 nm | 9 | 4 | 2 | 1 |
| sphere, 250 nm | 9 | 5 | 4 | 0 |
| truncated octahedron, 150 nm | 9 | 2 | 2 | 1 |
| truncated octahedron, 250 nm | 9 | 4 | 4 | 1 |

The asymmetric 150 nm particle of sections 1-4 is the case where the schedule helps, and the
survey reproduces that (strain 0 noiseless, median 0.065 to 0.018 rad; strain 1.2 noiseless,
0.119 to 0.082). Elsewhere it mostly hurts, most clearly at strain 0, at 250 nm and at 1e4
photons: truncated octahedron 250 nm strain 0 goes from 0.043 to 0.118 rad noiseless and from
0.027 to 0.093 at 1e5, and the 250 nm sphere at strain 0 goes from about 0.02 to about 0.06 at
every dose. So the gain in sections 3-4 belongs to that particle, not to the threshold.

Why was not measured. A guess that fits the pattern: the 0.2 rounds help where the 0.3 support
clips part of the particle, as for the 9-px asymmetric one, and add a rim that blurs the phase
where the 0.3 support already fits. Comparing the final supports with the truth's would test it;
the survey recorded support sizes but they were not scored.

## Not covered

- 1e3 photons, GPU runs. The survey (section 5) covers the other shapes and sizes only for
  default against schedule; plain 0.2 was run on the asymmetric 150 nm particle only.
- Where the phase gain sits. The benchmark's phase RMSE is unweighted over truth > 30% of max,
  and for a 9-px particle about half of that region is in the outer pixel (1 - (7/9)^3 for a
  sphere). The 0.3 support clips exactly that rim. An interior-only score (truth > 60%) or an
  amplitude-weighted one would show whether 0.2 and the schedule improve the bulk strain field
  or mainly the surface.
- Object size. Both object sets are small: the BCDI particle spans about 9 px (150 nm; 250 nm
  in the survey is about 15 px) and the test objects 10-12 px, so a 1-px rim is a large part of
  the object. For particles 30-60 px across the threshold probably matters much less
  (inference).
- Why the iterate falls into two phase groups at strain 1.2. Possibly a phase vortex or a partial
  twin; not examined.
- Whether the consensus selector keeps working with fewer than 8 starts. The tests in #20 cover
  only the construction.

[Back to all reports](reports/README.md)
