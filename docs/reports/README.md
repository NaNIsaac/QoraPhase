# Reports

Benchmarks and investigations of QoraPhase, newest first. Status: **checked** is a finished
report; **draft** is not final; **superseded** points to what replaced it. Reports marked
*generated* are written by the named script and rewritten on every run, so they carry no summary
box; the question, result and limits are given here instead.

## Start here

- **Key result:** [Measured data](../benchmarks/measured.md). On one measured 3D dataset, Fourier residual
  0.310 for the best of 10 starts against 0.315 for the lab MATLAB reconstruction; mutual error 0.235.
  Physics check needed.
- **Method:** [Benchmark ladder](../benchmarks/ladder.md). Success rate and error over 20 starts as synthetic
  data gets worse (noise, support, beamstop, gaps, strong phase).
- **Verification and limits:** [Public measured data (CXIDB)](../benchmarks/cxidb.md). Neither published
  reconstruction (CXIDB 3, CXIDB 28) is reproduced yet
  ([#38](https://github.com/NaNIsaac/QoraPhase/issues/38)); the measured comparison is one dataset,
  summary numbers only.

## All reports

| Date | Report | Result | Status |
|---|---|---|---|
| 2026-11-04 | [Lab ER/HIO recipe on simulated BCDI data](../benchmarks/bcdi-lab.md) | On simulated 3D data of one particle, registered to the truth to 0.01 voxel, median phase RMSE over 20 starts 0.055 rad (10⁵ photons) and 0.048 rad (noiseless) for the default recipe, against 0.17–0.26 and 0.11–0.16 rad for a re-implemented lab MATLAB recipe; at 0 and 3 rad peak phase 0.042 and 0.061 rad against 0.21–0.30 and 0.19–0.29 rad. A second pass on a consensus of QoraPhase's own supports gives no consistent improvement on the default. Physics check done, two independent passes. | checked |
| 2026-10-27 | [Public measured data (CXIDB)](../benchmarks/cxidb.md) | Neither published reconstruction (CXIDB 3, CXIDB 28) is reproduced yet ([#38](https://github.com/NaNIsaac/QoraPhase/issues/38)). Physics check needed. | checked, generated (`benchmarks/cxidb.py`) |
| 2026-10-27 | [Measured data](../benchmarks/measured.md) | On one measured 3D dataset, Fourier residual 0.310 for the best of 10 starts against 0.315 for the lab MATLAB reconstruction; mutual error 0.235. Physics check needed. | checked, generated (`benchmarks/measured.py`) |
| 2026-10-27 | [Benchmark ladder](../benchmarks/ladder.md) | With the exact support every stage succeeds in 20 of 20 starts; with a dilated or autocorrelation support, 0 to 10 of 20. | checked, generated (`benchmarks/ladder.py`) |
| 2026-10-25 | [Where reconstructions spend their time](../perf.md) | Time is the iteration count times one modulus projection; the FFT and its shift sandwich take 77-97 % of a step at 256² and up. | checked |
| 2026-10-23 | [Default recipe on simulated BCDI data: support, not stagnation](../bcdi-support.md) | The 0.05 Fourier error floor is set by the final support, not stagnation; rescored with sub-voxel registration, the two phase groups were an artefact; the default shrinkwrap threshold stays at 0.3. At 3 rad, symmetric particles often end above 0.15 rad; the level of the Fourier error does not flag them across cases, but within one ensemble it does. Physics check needed. | draft |

Details for the generated reports:

- **CXIDB.** Question: does QoraPhase reproduce the published reconstructions of two public
  measured 2D patterns (CXIDB 3, CXIDB 28)? Result: not reproduced; the paper schedule and the
  default recipe end on a cross or a blob, and HIO with the start support kept fits the data but
  its starts do not agree with each other. Limits: the numbers are reported without
  interpretation; physics check needed.
- **Measured data.** Question: how does QoraPhase compare with one saved reconstruction from the
  lab MATLAB code on one measured 3D dataset? Result: Fourier residual 0.315 for the reference
  cut to its support, 0.310 for the best of 10 QoraPhase starts; mutual error 0.235 against a
  median of 0.099 among QoraPhase's own starts. Limits: one reference run, the lab code's own
  spread is not measured (needs MATLAB); summary numbers only; physics check needed.
- **Ladder.** Question: how do success rate and error over 20 starts change as the synthetic
  data gets worse (noise, support, beamstop, gaps, strong phase)? Result: the exact support
  succeeds in 20 of 20 at every stage; the dilated and autocorrelation supports reach 0 to 10 of
  20. Limits: synthetic data only; the success threshold (real-space error < 0.1) is arbitrary.
