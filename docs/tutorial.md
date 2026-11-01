# Reconstruct a synthetic object

This bounded example creates a noiseless intensity from a known 64×64 object and reconstructs
from the intensity without a supplied support. It writes a Python script into the current directory.
Run it from a repository checkout after `uv sync --python 3.12`, using PowerShell 7.3 or later (also on Ubuntu).
The object definition is unchanged from `examples/readme_figure.py`.

<!-- test -->
```powershell
@'
import numpy as np
import qoraphase as qp

n, m = 64, 24
inside = np.indices((m, m)).sum(0) < m
y, x = np.indices((m, m)) - (m - 1) / 2
block = (1 - 0.5 * np.exp(-(x**2 + y**2) / 20)) * np.exp(
    1j * 1.2 * np.exp(-((x + 4) ** 2 + (y + 4) ** 2) / 30)
)
obj = np.zeros((n, n), complex)
obj[n // 2 - m // 2 : n // 2 + m // 2, n // 2 - m // 2 : n // 2 + m // 2] = block * inside

intensity = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj), norm="ortho"))) ** 2

print(qp.diagnose(intensity))
ensemble = qp.reconstruct(intensity, starts=8, seed=0)
best = ensemble.best
aligned = np.asarray(qp.align_to(best.object, obj))
print(f"Fourier error {best.error:.1e}")
print(f"error against the true object {np.linalg.norm(aligned - obj) / np.linalg.norm(obj):.1e}")
'@ | Set-Content -Encoding utf8 qoraphase-tutorial.py
uv run --python 3.12 -q python qoraphase-tutorial.py
```

The output reports diagnostics, the Fourier residual and aligned error against the known object.
The amplitude dip breaks the plain object's uniform-amplitude pattern. With seed 0 and eight starts,
the plain object converged on hosted Linux but not Windows; the README object converged on both
([investigation #47](https://github.com/NaNIsaac/QoraPhase/issues/47)).
With a fixed seed, reproducibility applies to one platform and setup; results are not bit-identical across platforms.

Last lines from a local Windows run with Python 3.12.13, JAX 0.11.1 and NumPy 2.5.2:

```text
no warnings
Fourier error 1.5e-07
error against the true object 7.9e-07
```

Floating-point results depend on the JAX version and platform; this example is not a published
benchmark or evidence of measured-data accuracy. Alignment resolves the global phase, translation
and conjugate-flip ambiguity before comparison. Read the [array conventions](guides/arrays.md)
before substituting your own intensity.

