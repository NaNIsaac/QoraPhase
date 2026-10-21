"""Inputs built the wrong way must be rejected or flagged, not reconstructed quietly.

Covers the mistakes the README warns about: a mask or support with its axes swapped, a mask with
inverted polarity (True must mean unusable), and a pattern with the origin in the corner instead
of at index n // 2. Cases the code does not catch yet are strict xfails tied to QoraPhase#16, so a
fix turns them into failures that prompt removing the marker.
"""

import warnings

import numpy as np
import pytest
from conftest import make_data

import qoraphase as qp

SHAPE = (32, 48)  # not square, so a swapped axis order changes the shape


@pytest.fixture
def pattern():
    _, support, intensity = make_data(SHAPE)
    beamstop = np.zeros(SHAPE, bool)
    beamstop[14:19, 22:27] = True  # around the centre (16, 24)
    return support, np.where(beamstop, 0.0, intensity), beamstop


def test_a_transposed_mask_is_rejected(pattern):
    _, intensity, beamstop = pattern
    with pytest.raises(ValueError, match="mask.shape"):
        qp.CDI(intensity, mask=beamstop.T)
    with pytest.raises(ValueError, match="mask has shape"):
        qp.diagnose(intensity, mask=beamstop.T)


def test_a_transposed_support_is_rejected(pattern):
    support, intensity, beamstop = pattern
    with pytest.raises(ValueError, match="support.shape"):
        qp.CDI(intensity, support=support.T, mask=beamstop)


def test_a_transposed_pattern_is_rejected_against_its_mask(pattern):
    _, intensity, beamstop = pattern
    with pytest.raises(ValueError, match="mask.shape"):
        qp.CDI(intensity.T, mask=beamstop)


def test_diagnose_flags_a_corner_origin_pattern(pattern):
    _, intensity, beamstop = pattern
    corner = np.fft.ifftshift(intensity)
    assert any("off centre" in w for w in qp.diagnose(corner).warnings)


@pytest.mark.xfail(strict=True, reason="QoraPhase#16: no warning about an inverted mask")
def test_diagnose_names_an_inverted_mask(pattern):
    _, intensity, beamstop = pattern
    report = qp.diagnose(intensity, mask=~beamstop)
    assert report.masked_fraction > 0.9
    assert any("mask" in w for w in report.warnings)


@pytest.mark.xfail(strict=True, reason="QoraPhase#16: CDI accepts an inverted mask silently")
def test_cdi_warns_about_an_inverted_mask(pattern):
    _, intensity, beamstop = pattern
    with pytest.warns(UserWarning, match="mask"):
        qp.CDI(intensity, mask=~beamstop)


@pytest.mark.xfail(strict=True, reason="QoraPhase#16: CDI accepts a corner-origin pattern silently")
def test_cdi_warns_about_a_corner_origin_pattern(pattern):
    _, intensity, beamstop = pattern
    with pytest.warns(UserWarning, match="centre"):
        qp.CDI(np.fft.ifftshift(intensity), mask=np.fft.ifftshift(beamstop))


def test_the_correct_mask_raises_nothing(pattern):
    support, intensity, beamstop = pattern
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qp.CDI(intensity, support=support, mask=beamstop)
