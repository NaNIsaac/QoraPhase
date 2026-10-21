"""Host-side validation.

Runs on concrete numpy arrays before anything is traced, so failures surface as
plain Python errors with useful messages instead of XLA shape errors.
"""

import warnings

import numpy as np

from ..diagnostics import inverted_mask_message, looks_inverted, robust_peak_offset

NEGATIVE_WARN_FRACTION = 0.01


def check_inputs(intensity, support, mask) -> None:
    if intensity.ndim not in (2, 3):
        raise ValueError(f"intensity must be 2D or 3D, got ndim={intensity.ndim}")
    if not np.all(np.isfinite(intensity)):
        raise ValueError("intensity contains NaN or Inf")

    for name, arr in (("support", support), ("mask", mask)):
        if arr is not None and arr.shape != intensity.shape:
            raise ValueError(f"{name}.shape {arr.shape} != intensity.shape {intensity.shape}")

    if mask is not None and bool(np.all(mask)):
        raise ValueError("mask marks every pixel as bad; nothing left to fit")

    if mask is not None and looks_inverted(intensity, np.asarray(mask, bool)):
        warnings.warn(inverted_mask_message(float(np.mean(mask))), stacklevel=3)
    else:
        # Only a gross offset: a corner origin (ifftshifted pattern) puts the peak n/2 away.
        usable = intensity if mask is None else np.where(mask, 0.0, intensity)
        offset = robust_peak_offset(usable)
        if any(abs(o) > n // 4 for o, n in zip(offset, intensity.shape, strict=True)):
            warnings.warn(
                f"the brightest region is {offset} pixels from the array centre; QoraPhase "
                "expects the origin at index n//2, so do not ifftshift the pattern "
                "(qp.diagnose gives details)",
                stacklevel=3,
            )

    negative = float(np.mean(intensity < 0))
    if negative > NEGATIVE_WARN_FRACTION:
        warnings.warn(
            f"{negative:.1%} of intensity pixels are negative and will be clipped to 0; "
            "consider masking them instead",
            stacklevel=3,
        )


def check_run(recipe, starts: int, batch: int | None) -> None:
    if len(recipe) == 0:
        raise ValueError("recipe is empty")
    if starts < 1:
        raise ValueError(f"starts must be >= 1, got {starts}")
    if batch is not None:
        if not 1 <= batch <= starts:
            raise ValueError(f"batch must satisfy 1 <= batch <= starts, got {batch}")
        if starts % batch:
            raise ValueError(f"starts ({starts}) must be divisible by batch ({batch})")
