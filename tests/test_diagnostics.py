"""Pre-flight checks.

A broken centring check is worse than no centring check, so the checks
themselves get tested against data with a known defect injected.
"""

import numpy as np
import pytest
from conftest import make_data

import qoraphase as qp
from qoraphase.diagnostics import friedel_offset, oversampling, peak_offset


def shifted(intensity, offset):
    return np.roll(intensity, offset, axis=tuple(range(intensity.ndim)))


def test_centred_data_reports_no_offset(data):
    _, _, intensity = data
    assert peak_offset(intensity) == (0, 0)
    assert friedel_offset(intensity) == (0, 0)


@pytest.mark.parametrize("offset", [(1, 0), (0, -3), (2, 4)])
def test_both_centring_checks_recover_an_injected_shift(data, offset):
    _, _, intensity = data
    moved = shifted(intensity, offset)
    assert peak_offset(moved) == offset
    assert friedel_offset(moved) == offset


def test_friedel_check_works_without_a_dominant_peak(data):
    """The peak test needs a bright pixel; the Friedel test does not."""
    _, _, intensity = data
    flattened = np.sqrt(np.sqrt(intensity))  # squash the dynamic range
    moved = shifted(flattened, (3, -2))
    assert friedel_offset(moved) == (3, -2)


def test_off_centre_data_is_flagged(data):
    _, support, intensity = data
    report = qp.diagnose(shifted(intensity, (2, 0)), support=support)
    assert any("off centre" in w for w in report.warnings)


def test_oversampling_tracks_the_object_size():
    """A smaller object in the same array is more oversampled."""
    small = oversampling(make_data(shape=(64, 64), m=6)[2])
    large = oversampling(make_data(shape=(64, 64), m=24)[2])
    assert min(small) > min(large)
    assert min(small) > 2.0


def test_oversampling_is_reported_per_axis():
    ratios = oversampling(make_data(shape=(24, 64), m=8)[2])
    assert len(ratios) == 2
    assert ratios[1] > ratios[0]  # the long axis oversamples more


def test_the_ratio_cannot_report_undersampling_on_its_own():
    """Why the fill is reported alongside the ratio.

    The autocorrelation cannot extend past the array, so ``2 / fill`` bottoms
    out at exactly 2 however badly undersampled the data is.  Here the true
    autocorrelation would span 27 pixels in a 16-pixel array; the ratio still
    says 2.0.  Only the fill hitting 1 gives it away.
    """
    _, support, intensity = make_data(shape=(16, 16), m=14)
    report = qp.diagnose(intensity, support=support)

    assert report.oversampling == (2.0, 2.0)  # looks fine, is not
    assert report.autocorrelation_fill == (1.0, 1.0)
    assert any("upper bound" in w for w in report.warnings)


def test_a_well_sampled_measurement_leaves_room():
    _, support, intensity = make_data(shape=(64, 64), m=8)
    report = qp.diagnose(intensity, support=support)
    assert max(report.autocorrelation_fill) < 0.5
    assert report.warnings == ()


def test_clean_data_produces_no_warnings(data):
    _, support, intensity = data
    assert qp.diagnose(intensity, support=support).warnings == ()


def test_negative_background_is_flagged(data):
    _, support, intensity = data
    report = qp.diagnose(intensity - 0.5 * intensity.mean(), support=support)
    assert any("negative" in w for w in report.warnings)


def test_non_finite_pixels_are_flagged(data):
    _, support, intensity = data
    broken = intensity.copy()
    broken[0, 0] = np.nan
    assert any("NaN" in w for w in qp.diagnose(broken, support=support).warnings)


def test_masked_pixels_do_not_set_the_peak(data):
    _, support, intensity = data
    hot = intensity.copy()
    hot[3, 5] = 100 * intensity.max()
    mask = np.zeros_like(intensity, bool)
    mask[3, 5] = True
    assert qp.diagnose(hot, support=support).peak_offset != (0, 0)
    report = qp.diagnose(hot, mask=mask, support=support)
    assert report.peak_offset == (0, 0)
    assert report.warnings == ()


def test_masked_negative_pixels_are_not_flagged(data):
    _, support, intensity = data
    shifted_background = intensity - 0.5 * intensity.mean()
    mask = shifted_background < 0
    report = qp.diagnose(shifted_background, mask=mask, support=support)
    assert report.negative_fraction == 0
    assert not any("negative" in w for w in report.warnings)


def test_nan_under_the_mask_is_explained(data):
    _, support, intensity = data
    gap = intensity.copy()
    gap[0] = np.nan
    mask = np.zeros_like(intensity, bool)
    mask[0] = True
    warnings = qp.diagnose(gap, mask=mask, support=support).warnings
    assert any("refuses these even under the mask" in w for w in warnings)
    with pytest.raises(ValueError):
        qp.CDI(gap, mask=mask, support=support)
    filled = np.where(mask, 0.0, gap)
    assert qp.diagnose(filled, mask=mask, support=support).warnings == ()
    qp.CDI(filled, mask=mask, support=support)


def test_rejects_a_mask_of_the_wrong_shape(data):
    _, _, intensity = data
    with pytest.raises(ValueError, match="mask has shape"):
        qp.diagnose(intensity, mask=np.zeros((3, 3), bool))


def test_a_loose_support_is_flagged(data):
    _, _, intensity = data
    report = qp.diagnose(intensity, support=np.ones_like(intensity, bool))
    assert any("barely" in w for w in report.warnings)


def test_mask_fraction_is_reported(data):
    _, support, intensity = data
    mask = np.zeros_like(intensity, bool)
    mask[:8] = True
    report = qp.diagnose(intensity, mask=mask, support=support)
    assert report.masked_fraction == pytest.approx(0.25)


def test_report_prints_every_field(data):
    _, support, intensity = data
    text = str(qp.diagnose(intensity, support=support))
    for label in ("shape", "peak offset", "Friedel", "oversampling", "support fills"):
        assert label in text


def test_rejects_1d_input():
    with pytest.raises(ValueError, match="2D or 3D"):
        qp.diagnose(np.ones(8))


def test_works_in_3d():
    _, support, intensity = make_data(shape=(16, 16, 16), m=6)
    report = qp.diagnose(intensity, support=support)
    assert len(report.oversampling) == 3
    assert report.friedel_offset == (0, 0, 0)
