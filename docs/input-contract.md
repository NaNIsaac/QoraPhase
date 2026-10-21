# Input contract

What QoraPhase expects from a measured 3D (or 2D) diffraction pattern, and what it
leaves to the caller. [CXD](https://github.com/NaNIsaac/CXD) implements the CXD side as
`cxd.core.handoff.to_qoraphase`; `tests/test_input_contract.py` here checks each rule
below on synthetic data.

## Arrays

| | QoraPhase expects | Checked by |
|---|---|---|
| Shape | 2D or 3D, any size, odd or even per axis | `check_inputs` |
| Axis order | none: axis *k* of the result is axis *k* of the input | `test_axis_order_carries_no_meaning` |
| Origin | the q = 0 / Bragg peak voxel at index `n // 2` of every axis | `diagnose` (`peak_offset`, `friedel_offset`) |
| Values | intensity (counts, any linear scale), not amplitude; internally `sqrt(clip(I, 0))` | |
| `mask` | bool, same shape, `True` = not a measurement (gap, beamstop, hot, saturated, unmeasured) | `check_inputs` |
| Masked voxels | not read by the modulus constraint, the error metric, the low-pass target or the random start. Exception: without a `support`, the autocorrelation support (and `diagnose`'s autocorrelation fill) is computed from the array as stored, so zero-fill masked voxels | `test_masked_values_do_not_change_the_result` |
| NaN / Inf | refused anywhere, also under the mask: zero-fill masked voxels | `check_inputs` |
| Negative values | clipped to 0 with a warning above 1 %; mask them instead when they are a background-subtraction artefact | `check_inputs`, `diagnose` |
| Everything masked | refused | `check_inputs` |

A detector stack therefore goes in as: crop around the peak so it lands on `n // 2`,
build the mask (`True` = bad), set masked voxels to 0, pass both.
`test_a_detector_stack_round_trips_through_the_documented_steps` does exactly this from a
`(frame, y, x)` stack with a NaN gap.

## Coordinates are the caller's

QoraPhase has no units and no geometry. It reconstructs on the grid it is given, so the
physical meaning of a result voxel comes from the reciprocal-space sampling the caller
used:

- If the voxel steps in reciprocal space are the columns of a 3x3 matrix **B** (one column
  per array axis, q including the factor 2π, so q = 4π sin θ / λ), the real-space voxel
  steps are the columns of **R** = 2π (**B**⁻¹)ᵀ diag(1/N₁, 1/N₂, 1/N₃), N being the
  array size per axis. Without the 2π in q (q in 1/length), drop the 2π.
- For a rocking scan phased in detector frame, **B** is not orthogonal (the rocking axis is
  sheared against the detector axes), and so neither is **R**. Keep **B** with the result;
  [QoraGeom](https://github.com/NaNIsaac/QoraGeom) builds it from the SPEC geometry and
  resamples a reconstruction to lab axes.
- QoraPhase's FFT is unitary and centre-origin. The sign convention of the forward
  transform decides which of the two twin objects (r, φ) and (−r, −φ) is "the" object; a
  phase-retrieval result is only defined up to that twin, a global phase and a
  translation in any case.

## Effects that are not corrected for you

These change the physics of the input. QoraPhase does not detect them, so they belong to
the caller and should be recorded with the result.

- **Uniform grid assumption.** The FFT assumes the voxels are uniformly spaced along each
  array axis. A detector-frame rocking stack is uniform in (rocking angle, pixel), which
  is linear in q only to first order in the rocking range; the curvature grows with the
  range and with the detector angle. Phasing in detector frame and transforming the object
  afterwards is the usual way to avoid interpolation.
- **Interpolation to a q grid before phasing** smooths the pattern, mixes masked and
  measured voxels and correlates noise between voxels. If it is done anyway, the mask has
  to be interpolated conservatively (a voxel is valid only if all its contributors were).
- **Cropping** in reciprocal space keeps the field of view (set by the step) and lowers the
  real-space resolution (set by the extent). **Binning** does the opposite: it shrinks the
  field of view and the oversampling, which `diagnose` reports.
- **Unmeasured regions** (missing angles, detector gaps, beyond the detector edge inside
  the array) are masked, not zero: zero is a measurement of no intensity.
- **Background subtraction** that leaves negative values should mask them rather than let
  them be clipped; clipping biases the weak, high-q intensities upwards.

## What a caller should store with a reconstruction

The crop offset in the original stack, the mask, **B** (or the geometry that produces it),
the recipe and seed, and whether the input was detector-frame or interpolated.
