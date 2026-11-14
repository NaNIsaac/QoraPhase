# Validation and limits

All test data is **synthetic**; no measured data ships with the repository.

- `tests/test_end_to_end.py` reconstructs a known 24³ object from Poisson-noisy counts (peak
  1e6 photons) through the default path: `default_recipe`, autocorrelation support,
  shrinkwrap and 4 starts. The metrics are pinned in `tests/golden/end_to_end.json`.
  Physical checks:
  - the Fourier error must sit at the noise floor, 0.9 to 1.05 times the error of the true
    object against the same counts;
  - the floor itself must be close to sqrt(N / 4 / ΣI), from var(sqrt(Poisson)) ≈ 1/4;
  - the shrinkwrap support must be no smaller than the object.
- The other tests reconstruct synthetic objects in 2D and 3D against an independent numpy
  FFT (`tests/conftest.py`).
- [benchmark ladder](../benchmarks/ladder.md): success rate and error over 20
  starts as the data gets worse (noiseless, then Poisson noise at three count levels, each
  with the exact, a dilated and an autocorrelation support), after shift, phase and twin
  are resolved, then with a beamstop, detector gaps and a stronger phase.
  `uv run --python 3.14+gil python benchmarks/ladder.py` regenerates it in about a minute on CPU.
- [public measured data](../benchmarks/cxidb.md): two public measured 2D patterns, downloaded
  at run time and checked by SHA-256, not stored here (CC0). Neither published reconstruction is
  reproduced yet ([#38](https://github.com/NaNIsaac/QoraPhase/issues/38)). Data from CXIDB 3:
  H. N. Chapman et al., Nat. Phys. 2, 839 (2006), doi:10.1038/nphys461, data doi:10.11577/1096905;
  CXIDB 28: S. Marchesini et al., Phys. Rev. B 68, 140101(R) (2003),
  doi:10.1103/PhysRevB.68.140101, data doi:10.11577/1172615.
- Known limits:
  - The noise-floor check needs the true object, so it applies to synthetic data only.
  - The fixture has Poisson noise and nothing else: no detector gaps, no partial coherence,
    no background.
  - No measured BCDI scan has been validated yet
    ([#1](https://github.com/NaNIsaac/QoraPhase/issues/1)).

## What this does not do


- No clustering of runs.
- No positivity/reality constraint — the object is complex here.
- No file I/O, no NUFFT, no autodiff optimisers.
- No BCDI geometry and no strain: [QoraGeom](https://github.com/NaNIsaac/QoraGeom)
  reads the SPEC scan, builds the detector-to-lab basis, resamples a
  reconstruction to the lab frame and computes strain from its phase.
