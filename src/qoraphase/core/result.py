"""Single reconstruction result."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Reconstruction:
    """One reconstructed object plus its error history.

    The solution is only defined up to a global phase, a translation, and a
    conjugate flip (twin image).  ``phase`` is therefore *not*
    an absolute phase; remove the ramp before interpreting it physically.
    """

    object: jnp.ndarray
    support: jnp.ndarray
    metrics: jnp.ndarray
    # Estimated coherence kernel when the recipe used qp.partial_coherence.
    coherence: jnp.ndarray | None = None

    @property
    def amplitude(self) -> jnp.ndarray:
        return jnp.abs(self.object)

    @property
    def phase(self) -> jnp.ndarray:
        return jnp.angle(self.object)

    @property
    def error(self) -> float:
        """Final Fourier error; the ranking key."""
        return float(self.metrics[-1]) if self.metrics.size else float("nan")

    def plot(self, axes=None):
        """Amplitude / phase panels.  3D shows the central slice of each axis."""
        import matplotlib.pyplot as plt

        amp, phase = np.asarray(self.amplitude), np.asarray(self.phase)
        if amp.ndim == 2:
            panels = [("amplitude", amp), ("phase", phase)]
        else:
            panels = [
                (f"{name} axis {axis}", arr.take(arr.shape[axis] // 2, axis=axis))
                for name, arr in (("amplitude", amp), ("phase", phase))
                for axis in range(3)
            ]
        if axes is None:
            _, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        axes = np.atleast_1d(axes)
        for ax, (title, data) in zip(axes, panels):
            ax.imshow(data, cmap="gray" if "amplitude" in title else "twilight")
            ax.set_title(title)
            ax.axis("off")
        return axes
