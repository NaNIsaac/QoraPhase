# Alignment, averaging and PRTF

The short alignment and PRTF examples below write temporary Python scripts in the repository checkout and are CI-marked. Their floating-point output is an illustrative recorded run, not an exact cross-platform assertion.


A reconstruction is determined only up to a global phase, a translation, and a
conjugate flip. No iteration can remove those — they are exact symmetries of the
constraints — so they are resolved afterwards:

```python
aligned = ensemble.align()  # every run brought onto the best one
mean = ensemble.average()  # coherent mean of the aligned runs
```

**Never average runs without aligning them** (`average()` aligns for you). Each
run carries its own global phase, so a raw mean interferes destructively and
loses amplitude.

`align()` adopts whichever conjugate branch the best run is in; that is a
convention, not a statement about which branch is real. `conj(obj(-r))`
reproduces the measured intensity exactly, so only the support you supply breaks
the tie: a reflected support converges just as cleanly on the mirrored object.
If the twin branch matters, break the symmetry during the run with `qp.detwin`.

Alignment works to whole pixels by default. `result.phase` keeps any linear phase
ramp from the measurement geometry until you call `qp.remove_ramp` (below).

### Sub-pixel alignment

`upsample=N` refines the whole-pixel correlation peak to about `1/N` pixel with
the matrix-multiply DFT of Guizar-Sicairos et al. (Opt. Lett. 33, 156, 2008) and
applies the fractional remainder as a Fourier shift. It is available on
`align_to`, `Ensemble.align` and `Ensemble.average`; `qp.register` returns the
shift itself. The Fourier shift treats the array as periodic, so the object
should not touch the array edge.

<!-- test -->
```powershell
@'
import numpy as np
import qoraphase as qp

rng = np.random.default_rng(0)
obj = np.zeros((32, 32), complex)
obj[11:21, 11:21] = rng.uniform(0.5, 1, (10, 10)) * np.exp(1j * rng.uniform(-0.5, 0.5, (10, 10)))

# displace by (1.37, -2.63) pixels through the Fourier shift theorem
freq = [np.fft.fftfreq(n) for n in obj.shape]
ramp = np.exp(-2j * np.pi * (1.37 * freq[0][:, None] - 2.63 * freq[1][None, :]))
moved = np.fft.ifftn(np.fft.fftn(obj) * ramp)

shift, _ = qp.register(moved, obj, upsample=100)
print(shift)
for upsample in (1, 100):
    aligned = qp.align_to(moved, obj, upsample=upsample)
    print(upsample, np.linalg.norm(aligned - obj) / np.linalg.norm(obj))
'@ | Set-Content -Encoding utf8 qoraphase-alignment-2.py
uv run --python 3.12 -q python qoraphase-alignment-2.py
```

```
(1.3699999999999999, -2.63)
1 0.41172676077039544
100 1.5936516413343817e-07
```

### Phase-ramp removal

A linear phase across the object means its Fourier transform sits off the array
centre. `qp.remove_ramp(obj)` finds that offset as the centre of mass of `|F|^2`
of the zero-padded object, to a fraction of a pixel, and cancels it. Pass the
support to keep noise outside the object out of the measurement, and
`zero_phase=True` to also set the mean phase inside the support to zero.

It removes whatever linear phase is present. A ramp that carries physical
information (a displaced Bragg peak, uniform strain) goes with it, so keep the
original if you need it. The centre of mass is exact for a real-valued object; for
a complex one it is an estimate. It traces under `jax.jit` (`pad`, `iterations`
and `zero_phase` static).

<!-- test -->
```powershell
@'
import numpy as np
import qoraphase as qp

rng = np.random.default_rng(0)
obj = np.zeros((32, 32))
obj[11:21, 11:21] = rng.uniform(0.5, 1, (10, 10))
support = obj > 0

x = np.arange(32) - 16
ramped = obj * np.exp(2j * np.pi * (0.071 * x[:, None] - 0.043 * x[None, :]))

cleaned = np.asarray(qp.remove_ramp(ramped, support, zero_phase=True))
print(np.abs(np.angle(ramped[support])).max())
print(np.abs(np.angle(cleaned[support])).max())
'@ | Set-Content -Encoding utf8 qoraphase-alignment-3.py
uv run --python 3.12 -q python qoraphase-alignment-3.py
```

```
3.135309468282613
0.000930922
```

### PRTF

The phase retrieval transfer function measures where independent reconstructions
stop agreeing. `qp.prtf(ensemble, intensity, mask)` aligns the runs (to `1/100`
pixel by default, `upsample=` changes that), averages their transforms and
divides the radially summed amplitude of the mean by the measured one. It returns
a `PRTF` with `frequency` (shell centres, cycles per object pixel), `value`,
`crossing` (where `value` first falls below 1/e, interpolated, nan if it never
does) and `resolution` (`1 / crossing`, a full period in object pixels). A
sequence of already-aligned objects works too with `align=False`, and
`qp.prtf_curve` is the traceable core. Two or more runs are needed; one run has a
PRTF of 1 by construction.

<!-- test -->
```powershell
@'
import numpy as np
import qoraphase as qp

rng = np.random.default_rng(1)
obj = np.zeros((48, 48), complex)
block = rng.uniform(0.5, 1, (14, 14)) * np.exp(1j * rng.uniform(-0.5, 0.5, (14, 14)))
obj[17:31, 17:31] = block * (np.indices((14, 14)).sum(0) < 14)
support = np.abs(obj) > 0

clean = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj), norm="ortho"))) ** 2
intensity = rng.poisson(clean * (100 / clean.max())) * (clean.max() / 100)

ensemble = qp.reconstruct(intensity, support=support, starts=12, seed=0)
result = qp.prtf(ensemble, intensity)
print(np.round(np.asarray(result.value)[::4], 3))
print(result.crossing, result.resolution)
'@ | Set-Content -Encoding utf8 qoraphase-alignment-4.py
uv run --python 3.12 -q python qoraphase-alignment-4.py
```

```
[0.979 0.852 0.676 0.499 0.468 0.396 0.278]
0.437100887298584 2.2878013498904184
```

On noiseless data a converged ensemble has a PRTF of 1 at every frequency and no
crossing. Here the intensity has Poisson noise (peak of 100 counts), and the
values printed are every fourth shell, from zero frequency to Nyquist (0.5).

