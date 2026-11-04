# Default recipe on simulated BCDI data: support, not stagnation

[Back to all reports](reports/README.md)

> **Question.** Why does the default recipe stop at a Fourier error of 0.05 on noiseless simulated BCDI data, and why did noiseless cases show a larger phase error than noisy ones at strain 1.2 rad?
>
> **Result.** The 0.05 is not stagnation but is set by the support the recipe ends with. The noiseless-is-worse result, and the two phase groups behind it, came from scoring with whole-voxel registration; registered to 0.01 voxel the runs form one group. A shrinkwrap schedule that loosens to 0.2 at the end is worse than the default in 24 of 54 benchmark cells, so the default stays at 0.3. At 3 rad, sphere and truncated-octahedron runs often end above 0.15 rad without a higher Fourier error.
>
> **Limits.** Small objects (about 9-15 px), no 1e3-photon or GPU runs. Sections 1 and 3 and the plain-0.2 runs are scored to whole voxels and were not rescored.

The BCDI benchmark in rocking_xrd_sim#6 reported that `qp.reconstruct` with the default recipe
stops at a Fourier error of 0.05 on noiseless data, while the true object fits the frames to
2e-4, and that at strain 1.2 rad the noiseless cases had a larger phase error than the noisy
ones. This note tests why. In short:

1. The 0.05 is not stagnation. It is set by the support the recipe ends with.
2. The "noiseless is worse" result does not survive sub-voxel registration. Scored to whole
   voxels, single runs split into two phase groups that the Fourier error could not tell apart.
   Registered to 0.01 voxel, the default's runs at strain 0 and 1.2 rad all end below 0.13 rad,
   and the lowest-error run is within 0.004 rad of the best one in the median ensemble.
3. A lower shrinkwrap threshold (0.2 instead of 0.3) stops converging on QoraPhase's own sharp
   test objects (section 4). A schedule that loosens only in the last two rounds keeps those, but
   over the benchmark's three shapes and two sizes it is worse than the default in 24 of 54 cells
   (19 when scored to whole voxels; section 5). The default stays at 0.3; the schedule is at most
   a recipe option.
4. At 3 rad, 109 of 288 runs on the sphere and the truncated octahedron end above 0.15 rad, with
   Fourier errors in the same range as the rest (section 6). Why is not known.

## Rescoring

This report was first scored with whole-voxel registration (rocking_xrd_sim at 3d135c8): the
reconstruction was shifted onto the truth by the integer peak of the amplitude
cross-correlation, then the phase ramp was fitted. A translation by a fraction of a voxel is an
exact ambiguity of the reconstruction. Left uncorrected it adds phase error that the ramp fit
removes only in part, and it can make the wrong twin overlap the truth better.
rocking_xrd_sim#15 registers to 0.01 voxel and chooses the twin after the fine shift. For one noiseless run at strain 1.2 rad the four starts
score 0.186, 0.080, 0.184 and 0.185 rad to whole voxels and 0.057, 0.052, 0.051 and 0.053 rad
registered.

All 324 runs of the survey in section 5 were rerun at the code and environment of that survey
(QoraPhase at #20, JAX 0.11.1 on CPU, Python 3.14). Every run reproduced its Fourier error per
start exactly, so the objects scored are those of the original survey; each was scored with
both registrations. Sections 2, 4 (BCDI table), 5 and 6 use these. The probe runs behind
sections 1 and 3 were not rerun: their phase numbers are whole-voxel and are kept only for the
record. Their Fourier errors and support sizes do not depend on the scorer.

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

The probes of sections 1 and 3 used QoraPhase at #17 (the phase-ramp FFT of #15), JAX 0.11.1 on
CPU, default recipe unless stated: RAAR 50, detwin(HIO 20), RAAR 50, 8 x [RAAR 50 +
Shrinkwrap(0.3, sigma 1)], ER 200. The support starts from the autocorrelation at 10%. The
probes were scratch scripts and are not part of this change. Each number is from one run on a
machine shared with other jobs; wall time is not reported.

## 1. The floor is the support

Noiseless, strain 1.2 rad, seed 0. At strain 0 the Fourier errors of the same five runs are
0.051, 0.050, 0.051, 0.033 and 0.036. Phase scored to whole voxels (see Rescoring):

| run | Fourier error | phase rad, whole-voxel | amplitude NRMSE | final support voxels |
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

## 2. The lowest Fourier error and the phase

Scored to whole voxels, single runs at strain 1.2 rad fell into two groups, about 0.075-0.09 rad
and 0.18-0.26 rad, with Fourier errors overlapping within 1e-3, and the lowest-error run came
from the upper group in 3 of 6 strained ensembles. That was the explanation offered for
"noiseless worse than noisy", and it motivated the consensus selector of QoraPhase#20. The upper
group is a registration artefact. The same runs of the survey (default recipe, asymmetric 150 nm
particle, 3 seeds x 4 starts per case), single starts in rad:

| case | whole-voxel | registered |
|---|---|---|
| strain 1.2, noiseless | 0.075-0.186 | 0.042-0.057 |
| strain 1.2, 1e5 | 0.071-0.248 | 0.041-0.057 |
| strain 1.2, 1e4 | 0.063-0.266 | 0.042-0.101 |

Registered, per seed, phase RMSE in rad (`average` is `Ensemble.average()`, aligned on the
consensus run):

| case | seed | median single run | best single run | lowest error | consensus run | average |
|---|---|---:|---:|---:|---:|---:|
| strain 1.2, noiseless | 0 | 0.053 | 0.051 | 0.051 | 0.052 | 0.049 |
| | 1 | 0.052 | 0.048 | 0.048 | 0.053 | 0.048 |
| | 2 | 0.046 | 0.042 | 0.046 | 0.050 | 0.042 |
| strain 1.2, 1e5 | 0 | 0.051 | 0.043 | 0.052 | 0.052 | 0.043 |
| | 1 | 0.049 | 0.041 | 0.052 | 0.052 | 0.043 |
| | 2 | 0.053 | 0.049 | 0.049 | 0.052 | 0.046 |
| strain 0, noiseless | 0 | 0.033 | 0.026 | 0.039 | 0.026 | 0.021 |
| | 1 | 0.029 | 0.007 | 0.031 | 0.027 | 0.021 |
| | 2 | 0.025 | 0.013 | 0.033 | 0.019 | 0.016 |
| strain 3.0, noiseless | 0 | 0.058 | 0.055 | 0.056 | 0.056 | 0.050 |
| | 1 | 0.062 | 0.053 | 0.059 | 0.065 | 0.073 |
| | 2 | 0.063 | 0.054 | 0.054 | 0.068 | 0.041 |

Over the whole survey (108 ensembles per strain) the picture is the same at strain 0 and 1.2:
no single start above 0.13 rad, and the lowest-error start is a median 0.003-0.004 rad above the
best one. The average is at or below the lowest-error run in 89 of 108 ensembles at strain 0 and
68 of 108 at 1.2; the consensus run alone is not better than the lowest-error run (median 0.069
against 0.067 rad at 1.2). At 3 rad the order reverses (section 6). `Ensemble.average()` and
`Ensemble.consensus` are not changed here.

## 3. Shrinkwrap threshold

Whole-voxel scores, not rescored (see Rescoring). The default recipe with the shrinkwrap
threshold changed, 3 seeds x 4 starts per cell. Phase RMSE in rad, mean over seeds of the
lowest-error run, with the mean of the per-seed median over starts in brackets:

| case | 0.3 / sigma 1 (default) | 0.2 / sigma 1 |
|---|---|---|
| strain 0, noiseless | 0.055 (0.046) | 0.017 (0.020) |
| strain 0, 1e5 | 0.048 (0.044) | 0.028 (0.030) |
| strain 1.2, noiseless | 0.127 (0.140) | 0.080 (0.082) |
| strain 1.2, 1e5 | 0.133 (0.115) | 0.076 (0.079) |
| strain 1.2, 1e4 | 0.116 (0.096) | 0.093 (0.118) |
| strain 3.0, noiseless | 0.212 (0.252) | 0.223 (0.262) |

In section 4, registration lowers the default's numbers much more than the schedule's, which
ends on the same support as 0.2; if 0.2 behaves the same, this table overstates its gain. What
does not depend on the scorer: with 0.2 the final support grows from about 455 to 590 voxels,
and the Fourier error falls in every cell (0.051 to 0.042 noiseless, 0.061 to 0.054 at 1e5,
0.107 to 0.103 at 1e4).

Other settings tried at strain 1.2, also whole-voxel: rms 0.28 (noiseless 3 seeds, 1e5 2
seeds), 0.1 / sigma 2 (noiseless, 3 seeds) and 0.3 / sigma 2. None was rescored.

The default was chosen on QoraPhase's own synthetic set (see `Shrinkwrap`). Changing it waited
for a rerun of QoraPhase's shrinkwrap tests at 0.2 / sigma 1 and for the full benchmark grid
(three shapes, two sizes). The tests rule out plain 0.2 (section 4), and the grid rules out
the schedule as a default (section 5).

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

On the asymmetric 150 nm particle the schedule lowers the phase error at strain 0 and 1.2 rad
with 1e5 photons or more, and not at 1e4 photons or at 3 rad. Phase RMSE in rad from the survey
of section 5, mean over 3 seeds of the lowest-error run of 4 starts, with the mean of the
per-seed median over starts in brackets:

| case | default, whole-voxel | default, registered | schedule, whole-voxel | schedule, registered |
|---|---|---|---|---|
| strain 0, noiseless | 0.064 (0.065) | 0.034 (0.029) | 0.018 (0.018) | 0.014 (0.014) |
| strain 0, 1e5 | 0.048 (0.051) | 0.039 (0.038) | 0.021 (0.028) | 0.017 (0.020) |
| strain 1.2, noiseless | 0.148 (0.119) | 0.048 (0.050) | 0.077 (0.082) | 0.040 (0.045) |
| strain 1.2, 1e5 | 0.133 (0.106) | 0.051 (0.051) | 0.095 (0.082) | 0.043 (0.043) |
| strain 1.2, 1e4 | 0.110 (0.117) | 0.053 (0.062) | 0.083 (0.101) | 0.064 (0.068) |
| strain 3.0, noiseless | 0.256 (0.258) | 0.056 (0.061) | 0.147 (0.193) | 0.062 (0.065) |
| strain 3.0, 1e5 | 0.277 (0.309) | 0.067 (0.102) | 0.197 (0.219) | 0.068 (0.091) |
| strain 3.0, 1e4 | 0.221 (0.227) | 0.076 (0.086) | 0.147 (0.173) | 0.104 (0.106) |

Scored to whole voxels the schedule looked better at 3 rad as well (0.256 to 0.147 noiseless);
registered, it is not. The gain at strain 0 and 1.2 shrinks but stays. The schedule's final
support (about 585 voxels) and Fourier error (0.042 noiseless) are those of plain 0.2; whether
its phase also matches 0.2 when registered was not checked, since the 0.2 runs were not
rescored.

## 5. The schedule across the benchmark grid

The benchmark's survey grid: asymmetric, sphere and truncated octahedron particles, 150 and
250 nm (about 9 and 15 px), strain 0, 1.2 and 3.0 rad, noiseless, 1e5 and 1e4 photons; 54 cells,
3 seeds x 4 starts per cell and recipe. QoraPhase at #20 (91cf87a), JAX 0.11.1 on CPU; scored
with rocking_xrd_sim at #6 (3d135c8, whole-voxel) and at #15 (registered).

The scoring rule was fixed before any phase value of the survey was read. Per seed, take the
median phase RMSE over the 4 starts. A cell is "worse" for the schedule if the mean over seeds of
(schedule - default) exceeds the standard deviation of the default over seeds. The schedule would
become the default only if no cell is worse and QoraPhase's tests pass with it. As a check that
does not depend on the spread estimate, a second column counts cells where the schedule is worse
in all three seeds. The scoring was repeated independently from the rule alone, with no
disagreement over 54 cells, and the rescoring script reproduces the whole-voxel counts below.

Result: the schedule is worse in 24 of 54 cells registered, 22 by the all-three-seeds count
(whole-voxel: 19 and 16). The test-suite condition could not change that and was not run. The
default stays at 0.3 / sigma 1. Cells by group, by median phase error, whole-voxel / registered
(the last column uses the mirrored rule, against the same spread of the default, which was not
part of the rule fixed in advance):

| group | cells | worse | worse in all 3 seeds | better (not fixed in advance) |
|---|---:|---:|---:|---:|
| strain 0 | 18 | 10 / 14 | 9 / 12 | 1 / 3 |
| strain 1.2 | 18 | 4 / 6 | 2 / 5 | 3 / 7 |
| strain 3.0 | 18 | 5 / 4 | 5 / 5 | 3 / 2 |
| noiseless | 18 | 6 / 7 | 5 / 5 | 3 / 6 |
| 1e5 photons | 18 | 4 / 5 | 4 / 6 | 4 / 4 |
| 1e4 photons | 18 | 9 / 12 | 7 / 11 | 0 / 2 |
| asymmetric, 150 nm | 9 | 1 / 3 | 1 / 4 | 2 / 4 |
| asymmetric, 250 nm | 9 | 3 / 6 | 3 / 5 | 2 / 1 |
| sphere, 150 nm | 9 | 4 / 1 | 2 / 1 | 1 / 3 |
| sphere, 250 nm | 9 | 5 / 4 | 4 / 3 | 0 / 2 |
| truncated octahedron, 150 nm | 9 | 2 / 4 | 2 / 1 | 1 / 2 |
| truncated octahedron, 250 nm | 9 | 4 / 6 | 4 / 8 | 1 / 0 |

No particle is one where the schedule simply helps: on the asymmetric 150 nm particle of sections
1-4 it is worse in 3 cells and better in 4. Where it hurts it does so most clearly at strain 0,
at 250 nm and at 1e4 photons, and registration does not change those cases much: truncated
octahedron 250 nm at strain 0 goes from 0.025 to 0.103 rad noiseless and from 0.021 to 0.084 at
1e5 (mean of the per-seed medians; 0.043 to 0.118 and 0.027 to 0.093 whole-voxel), and the
250 nm sphere at strain 0 from about 0.02 to 0.04-0.06 at every dose.

Why was not measured. A guess that fits the pattern: the 0.2 rounds help where the 0.3 support
clips part of the particle, as for the 9-px asymmetric one, and add a rim that blurs the phase
where the 0.3 support already fits. Comparing the final supports with the truth's would test it;
the survey recorded support sizes but they were not scored.

## 6. Strain 3 rad on the symmetric particles

Registered, every start in the survey above 0.15 rad is at strain 3 rad: 116 of 432 there,
against 303 of 432 scored to whole voxels, and none of 864 at strain 0 and 1.2 (175 whole-voxel).
Almost all are on the sphere and the truncated octahedron. Default and schedule pooled, 72 starts
and 18 ensembles per row:

| particle | starts > 0.15 rad | median start, rad | lowest-error run > 0.15 | consensus run > 0.15 | median support IoU |
|---|---:|---:|---:|---:|---:|
| asymmetric, 150 nm | 6 | 0.074 | 0 | 0 | 0.96 |
| asymmetric, 250 nm | 1 | 0.060 | 0 | 0 | 0.98 |
| sphere, 150 nm | 23 | 0.126 | 2 | 6 | 0.96 |
| sphere, 250 nm | 36 | 0.150 | 2 | 8 | 0.97 |
| truncated octahedron, 150 nm | 23 | 0.115 | 2 | 2 | 0.95 |
| truncated octahedron, 250 nm | 27 | 0.121 | 0 | 4 | 0.97 |

The support is found (IoU 0.95-0.98); the phase inside it is not. The Fourier error does not
mark these runs across cases: its median is 0.071 for starts above 0.15 rad and 0.068 for the
rest. Within one ensemble it does better than the agreement between runs. The lowest-error run
is above 0.15 rad in 6 of 108 ensembles at 3 rad, the consensus run in 20 and
`Ensemble.average()` in 17, so at 3 rad the consensus selector picks worse than the lowest error.

A hypothesis, not tested: at 3 rad the phase wraps inside the particle, and for a
centrosymmetric shape the twin of a partly wrapped solution is close to another solution, so
runs stall between them. Twin-check statistics and a count of phase wraps inside the support on
these runs would test it.

## Not covered

- 1e3 photons, GPU runs. The survey (section 5) covers the other shapes and sizes only for
  default against schedule; plain 0.2 was run on the asymmetric 150 nm particle only, and only
  scored to whole voxels.
- Where the phase gain sits. The benchmark's phase RMSE is unweighted over truth > 30% of max,
  and for a 9-px particle about half of that region is in the outer pixel (1 - (7/9)^3 for a
  sphere). The 0.3 support clips exactly that rim. An interior-only score (truth > 60%) or an
  amplitude-weighted one would show whether the schedule improves the bulk strain field or
  mainly the surface.
- Object size. Both object sets are small: the BCDI particle spans about 9 px (150 nm; 250 nm
  in the survey is about 15 px) and the test objects 10-12 px, so a 1-px rim is a large part of
  the object. For particles 30-60 px across the threshold probably matters much less
  (inference).
- Why runs at 3 rad on the symmetric particles end above 0.15 rad (section 6).
- Whether the consensus selector should give way to the lowest error at large strain. The
  survey has 4 starts per ensemble; the tests in #20 cover only the construction.

[Back to all reports](reports/README.md)
