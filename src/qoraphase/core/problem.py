"""The CDI problem: fixed reconstruction inputs (spec 8.1)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..models.fft import FFT
from ..operators.support import autocorrelation_support
from .validate import check_inputs


@dataclass(frozen=True, init=False)
class CDI:
    """Immutable reconstruction inputs.  Holds no iteration state.

    Constructed from *intensity* (spec 4.2); the stored field is the amplitude
    ``sqrt(clip(intensity, 0))`` so the square root is not recomputed inside the
    iteration loop.

    Registered as a pytree (spec 8.3) so it can cross ``jit`` / ``scan`` /
    ``vmap`` boundaries.  ``model`` is a meta field: it is static and hashable.
    """

    amplitude: jnp.ndarray
    mask: jnp.ndarray
    support: jnp.ndarray
    model: object

    def __init__(
        self,
        intensity=None,
        *,
        support=None,
        mask=None,
        model=None,
        support_threshold: float = 0.1,
        amplitude=None,
    ):
        if amplitude is not None:
            # Pytree unflattening path: values may be tracers, so no validation
            # and no host-side work here.
            self._set(amplitude, mask, support, model)
            return

        if intensity is None:
            raise TypeError("CDI() requires intensity=")

        intensity = np.asarray(intensity)
        support_np = None if support is None else np.asarray(support)
        mask_np = None if mask is None else np.asarray(mask)
        check_inputs(intensity, support_np, mask_np)

        intensity = jnp.asarray(intensity)
        amplitude = jnp.sqrt(jnp.clip(intensity, 0.0, None))
        mask = (
            jnp.zeros(intensity.shape, bool)
            if mask_np is None
            else jnp.asarray(mask_np, bool)
        )
        support = (
            autocorrelation_support(intensity, support_threshold)
            if support_np is None
            else jnp.asarray(support_np, bool)
        )
        self._set(amplitude, mask, support, model if model is not None else FFT())

    def _set(self, amplitude, mask, support, model) -> None:
        for name, value in (
            ("amplitude", amplitude),
            ("mask", mask),
            ("support", support),
            ("model", model),
        ):
            object.__setattr__(self, name, value)

    @property
    def intensity(self) -> jnp.ndarray:
        return self.amplitude**2

    @property
    def shape(self) -> tuple:
        return self.amplitude.shape

    def reconstruct(self, recipe=None, *, starts=1, batch=None, seed=None):
        from ..api import run  # local import: api depends on this module

        return run(self, recipe, starts=starts, batch=batch, seed=seed)


jax.tree_util.register_dataclass(
    CDI, data_fields=["amplitude", "mask", "support"], meta_fields=["model"]
)
