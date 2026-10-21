"""Pre-flight checks on a measurement.

Everything here takes arrays and returns numbers -- no file handling, no
reconstruction.  The point is to catch the mistakes that otherwise surface as a
plausible-looking but wrong result: a pattern that is not centred, an object
that is not oversampled enough to be unique, a background subtraction that went
negative.

None of these are fatal on their own.  They are the questions worth answering
before believing a reconstruction.
"""

from dataclasses import dataclass, field

import numpy as np

#: Autocorrelation extent, as a fraction of the array, above which the
#: oversampling estimate stops meaning anything.
MAX_AUTOCORRELATION_FILL = 0.9

#: Fraction of negative pixels above which clipping is likely to matter.
MAX_NEGATIVE_FRACTION = 0.01

#: An autocorrelation support this large is barely a constraint.
LOOSE_SUPPORT_FRACTION = 0.5


def flip_about_origin(a: np.ndarray) -> np.ndarray:
    """Reflect ``r -> -r`` for a center-origin array."""
    return np.roll(a[(slice(None, None, -1),) * a.ndim], 1, axis=tuple(range(a.ndim)))


def peak_offset(intensity: np.ndarray) -> tuple:
    """Offset of the brightest pixel from the array centre, in pixels."""
    peak = np.unravel_index(np.argmax(intensity), intensity.shape)
    return tuple(int(p - n // 2) for p, n in zip(peak, intensity.shape, strict=False))


def friedel_offset(intensity: np.ndarray) -> tuple:
    """Centring error inferred from Friedel symmetry, in pixels.

    A correctly centred pattern satisfies ``I(q) == I(-q)``.  If instead the
    pattern sits at ``c``, correlating it with its own reflection peaks at
    ``2c`` -- so this resolves a half-pixel error that the brightest pixel
    cannot, and it works on broad patterns with no single dominant peak.

    Returns half the correlation peak position, rounded, so an odd result means
    the true offset is a half pixel.
    """
    data = np.nan_to_num(np.asarray(intensity, float))
    spectrum = np.fft.fftn(data)
    correlation = np.fft.ifftn(spectrum * np.conj(np.fft.fftn(flip_about_origin(data))))
    doubled = np.unravel_index(np.argmax(np.abs(correlation)), data.shape)
    return tuple(
        int(round((((p + n // 2) % n) - n // 2) / 2))
        for p, n in zip(doubled, data.shape, strict=False)
    )


def autocorrelation_fill(intensity: np.ndarray, threshold: float = 0.1) -> tuple:
    """Autocorrelation extent per axis, as a fraction of the array.

    This is the quantity actually measured, and the honest one to read: a
    well-oversampled measurement puts the autocorrelation well inside the array,
    and a value approaching 1 means it no longer fits.

    An estimate, not a measurement -- it inherits the arbitrariness of
    *threshold*, and a noisy autocorrelation floor inflates the extent.
    """
    autocorrelation = np.abs(
        np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(np.nan_to_num(intensity))))
    )
    above = autocorrelation > threshold * autocorrelation.max()
    fills = []
    for axis, n in enumerate(intensity.shape):
        others = tuple(i for i in range(intensity.ndim) if i != axis)
        occupied = np.flatnonzero(above.any(axis=others))
        extent = max(occupied.max() - occupied.min() + 1, 1) if occupied.size else 1
        fills.append(min(extent / n, 1.0))
    return tuple(float(f) for f in fills)


def oversampling(intensity: np.ndarray, threshold: float = 0.1) -> tuple:
    """Linear oversampling ratio per axis, from :func:`autocorrelation_fill`.

    The autocorrelation spans twice the object, so ``ratio = 2 / fill``.

    Read this together with the fill, not on its own.  The autocorrelation
    cannot extend past the array, so ``fill <= 1`` and this ratio is **always at
    least 2** -- including on data that is badly undersampled, where the
    autocorrelation has simply been clipped.  A ratio of 2.0 is not a pass; it
    means the estimate ran out of room.
    """
    return tuple(2.0 / f for f in autocorrelation_fill(intensity, threshold))


@dataclass(frozen=True)
class Diagnosis:
    """What the checks found.  ``print()`` it."""

    shape: tuple
    dtype: str
    peak_offset: tuple
    friedel_offset: tuple
    autocorrelation_fill: tuple
    oversampling: tuple
    total_oversampling: float
    negative_fraction: float
    masked_fraction: float
    support_fraction: float
    dynamic_range: float
    warnings: tuple = field(default_factory=tuple)

    def __str__(self) -> str:
        lines = [
            f"shape                {self.shape}  {self.dtype}",
            f"peak offset          {self.peak_offset}  (pixels from array centre)",
            f"Friedel offset       {self.friedel_offset}",
            "autocorrelation fill "
            + " x ".join(f"{f:.2f}" for f in self.autocorrelation_fill)
            + "  (of the array; must stay below 1)",
            "oversampling         "
            + " x ".join(f"{r:.1f}" for r in self.oversampling)
            + f"  (total {self.total_oversampling:.1f}; >=2 by construction)",
            f"negative pixels      {self.negative_fraction:.2%}",
            f"masked pixels        {self.masked_fraction:.2%}",
            f"support fills        {self.support_fraction:.1%} of the array",
            f"dynamic range        {self.dynamic_range:.3g}",
        ]
        if self.warnings:
            lines.append("")
            lines += [f"!! {w}" for w in self.warnings]
        else:
            lines.append("")
            lines.append("no warnings")
        return "\n".join(lines)


def diagnose(intensity, mask=None, support=None, threshold: float = 0.1) -> Diagnosis:
    """Check a measurement against the assumptions QoraPhase makes about it."""
    intensity = np.asarray(intensity)
    if intensity.ndim not in (2, 3):
        raise ValueError(f"intensity must be 2D or 3D, got ndim={intensity.ndim}")

    bad = np.zeros(intensity.shape, bool) if mask is None else np.asarray(mask, bool)
    if bad.shape != intensity.shape:
        raise ValueError(f"mask has shape {bad.shape}, intensity {intensity.shape}")
    usable = intensity[~bad]

    # masked pixels carry no measurement: a hot pixel under the mask must not
    # set the peak, and a masked negative or NaN is not a problem with the data
    finite = np.where(bad, 0.0, np.nan_to_num(intensity.astype(float)))
    peak = peak_offset(finite)
    friedel = friedel_offset(finite)
    fills = autocorrelation_fill(finite, threshold)
    ratios = tuple(2.0 / f for f in fills)
    total = float(np.prod(ratios))

    if support is None:
        autocorrelation = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(finite))))
        support = autocorrelation > threshold * autocorrelation.max()
    support = np.asarray(support, bool)

    positive = usable[usable > 0]
    negative_fraction = float(np.mean(usable < 0)) if usable.size else 0.0
    warnings = []
    if any(peak) or any(friedel):
        warnings.append(
            f"pattern is off centre (peak {peak}, Friedel {friedel}); QoraPhase "
            "assumes the origin sits at index n//2"
        )
    if max(fills) > MAX_AUTOCORRELATION_FILL:
        crowded = [i for i, f in enumerate(fills) if f > MAX_AUTOCORRELATION_FILL]
        warnings.append(
            f"the autocorrelation fills axis {crowded} to "
            f"{max(fills):.0%} of the array; it no longer fits, so the "
            "oversampling figure is an upper bound and the measurement may be "
            "undersampled"
        )
    if negative_fraction > MAX_NEGATIVE_FRACTION:
        warnings.append(
            f"{negative_fraction:.1%} of unmasked pixels are negative and will be "
            "clipped to zero; consider masking them instead"
        )
    if not np.all(np.isfinite(usable)):
        warnings.append(
            "intensity contains NaN or Inf; mask those pixels and set them to a "
            "finite value such as 0"
        )
    if not np.all(np.isfinite(intensity[bad])):
        warnings.append(
            "masked pixels contain NaN or Inf; CDI refuses these even under the "
            "mask, so set them to a finite value such as 0"
        )
    if support.mean() > LOOSE_SUPPORT_FRACTION:
        warnings.append(
            f"support fills {support.mean():.0%} of the array and barely "
            "constrains anything; supply a measured support if you have one"
        )

    return Diagnosis(
        shape=tuple(intensity.shape),
        dtype=str(intensity.dtype),
        peak_offset=peak,
        friedel_offset=friedel,
        autocorrelation_fill=fills,
        oversampling=ratios,
        total_oversampling=total,
        negative_fraction=negative_fraction,
        masked_fraction=float(bad.mean()),
        support_fraction=float(support.mean()),
        dynamic_range=float(positive.max() / positive.min()) if positive.size else 0.0,
        warnings=tuple(warnings),
    )
