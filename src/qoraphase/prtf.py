"""Phase retrieval transfer function.

Independent reconstructions agree on the Fourier components the data pins down
and disagree elsewhere.  Averaging their transforms, after removing the
ambiguities, keeps the first and washes out the second, so the ratio of the
averaged amplitude to the measured one falls from 1 towards 0 as the runs stop
agreeing (Chapman et al., J. Opt. Soc. Am. A 23, 1179, 2006).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .models.fft import FFT
from .postprocess import align_to

DROP = float(np.exp(-1.0))


@dataclass(frozen=True)
class PRTF:
    """Radially averaged PRTF.

    ``frequency`` holds the shell centres in cycles per object pixel and
    ``value`` the PRTF in each shell (nan for a shell with no usable pixel).
    ``crossing`` is the frequency at which ``value`` first falls below 1/e,
    linearly interpolated, or nan if it never does; ``resolution`` is its
    reciprocal, a full period in object pixels.
    """

    frequency: jnp.ndarray
    value: jnp.ndarray
    crossing: float

    @property
    def resolution(self) -> float:
        return 1.0 / self.crossing


def prtf_curve(objects, intensity, mask=None):
    """PRTF of already-aligned *objects* as ``(frequency, value, crossing)``.

    ``value`` is the ratio, per shell, of the summed amplitude of the mean
    transform to the summed measured amplitude, over unmasked pixels.  Summing
    before dividing keeps near-empty pixels from dominating a shell.  Pure
    ``jnp``: traceable under ``jit``.
    """
    objects = jnp.asarray(objects)
    amplitude = jnp.sqrt(jnp.clip(jnp.asarray(intensity), 0))
    shape = amplitude.shape
    usable = jnp.ones(shape, bool) if mask is None else ~jnp.asarray(mask)

    mean = jnp.mean(jnp.stack([FFT().forward(o) for o in objects]), axis=0)

    grid = jnp.meshgrid(*[jnp.arange(n) - n // 2 for n in shape], indexing="ij")
    longest = max(shape)
    radius = jnp.sqrt(sum((g / n) ** 2 for g, n in zip(grid, shape, strict=True)))
    shell = jnp.round(radius * longest).astype(int)
    count = longest // 2 + 1
    usable = usable & (shell < count)
    shell = jnp.where(usable, shell, 0)

    def total(weights):
        return jnp.bincount(shell.ravel(), (weights * usable).ravel(), length=count)

    measured = total(amplitude)
    value = jnp.where(measured > 0, total(jnp.abs(mean)) / jnp.where(measured > 0, measured, 1), jnp.nan)
    frequency = jnp.arange(count) / longest

    below = value < DROP
    first = jnp.argmax(below)
    before = jnp.maximum(first - 1, 0)
    v0, v1 = value[before], value[first]
    f0, f1 = frequency[before], frequency[first]
    interpolated = jnp.where(
        jnp.isnan(v0) | (v1 == v0), f1, f0 + (DROP - v0) * (f1 - f0) / (v1 - v0)
    )
    crossing = jnp.where(below.any(), interpolated, jnp.nan)
    return frequency, value, crossing


def prtf(objects, intensity, mask=None, *, align: bool = True, upsample: int = 100) -> PRTF:
    """PRTF of an :class:`~qoraphase.Ensemble` or a sequence of objects.

    With ``align=True`` every run is first brought onto the reference (the best
    run of an ensemble, the first of a sequence) in phase, position and twin
    branch, to ``1 / upsample`` pixel.  Sub-pixel alignment matters here: a
    residual shift of a fraction of a pixel is a phase error that grows with
    frequency and would read as lost resolution.  Pass ``align=False`` for
    objects that are already aligned.

    *intensity* and *mask* are the measurement (``mask=True`` marks an unusable
    pixel, which is left out).  The curve is meaningful only for two or more
    runs; a single run has a PRTF of 1 by construction.
    """
    if hasattr(objects, "runs"):
        runs = objects.align(upsample).runs if align else objects.runs
        stack = [run.object for run in runs]
    else:
        stack = [jnp.asarray(o) for o in objects]
        if align:
            stack = stack[:1] + [align_to(o, stack[0], upsample) for o in stack[1:]]
    frequency, value, crossing = prtf_curve(jnp.stack(stack), intensity, mask)
    return PRTF(frequency, value, float(crossing))
