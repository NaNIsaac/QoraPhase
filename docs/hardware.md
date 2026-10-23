# GPU environment of the benchmark machine

State of the machine used for `docs/perf.md`, recorded on 2026-10-22, and what it takes to
run QoraPhase on its GPU. Everything under "Measured" was read from this machine; the rest is
marked as such.

## Measured

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| compute capability | 12.0 (`nvidia-smi --query-gpu=compute_cap`) |
| VRAM | 12,227 MiB total (11.6 GB free of 12.8 GB reported by CuPy `mem_info` with the desktop running) |
| driver | 610.88, WDDM driver model, reports CUDA 13.3 |
| CPU / RAM | AMD Ryzen 5 7500F (6 cores, 12 threads), 31.6 GB |
| OS | Windows 11 Home 10.0.26300 |
| Python / uv | 3.14.5 / 0.11.32 |
| CUDA toolkit | not installed (`nvcc` not found) |
| WSL | installed, but no distribution |
| JAX in the project env | 0.11.1 (jaxlib 0.11.1), CPU backend only, `jax.devices()` is `[CpuDevice(id=0)]` |

## Can JAX use this GPU?

Not on native Windows. `uv pip install --dry-run jax-cuda13-plugin` and `jax-cuda12-plugin`
find no installable wheel; the resolver lists only `manylinux_2_27_x86_64` and
`manylinux_2_27_aarch64` builds (v0.12.0.1). So the options for JAX on the GPU are:

1. **WSL2 with a Linux distribution** and `jax[cuda13]` inside it. Not tried: it needs a
   distribution installed (`wsl --install`, which may require an administrator prompt and a
   reboot) and the Windows NVIDIA driver, which is present. Whether the RTX 5070 (compute
   capability 12.0) works with the JAX CUDA wheels under WSL2 has not been tested.
2. A Linux machine with an NVIDIA GPU.

### Blackwell (sm_120) support in the JAX CUDA wheels

The RTX 5070 is compute capability 12.0 (Blackwell). Inspected without running anything: the
Linux wheels `jax-cuda13-plugin` 0.12.0.1 and `jax-cuda13-pjrt` 0.12.0.1 were downloaded
(`pip download --platform manylinux_2_27_x86_64`) and their binaries searched for target
strings. `jax_plugins/xla_cuda13/xla_cuda_plugin.so` (396 MB) contains `sm_120` (12 times),
`sm_121` (9), `sm_100` (19) and `compute_100` (31), and no `compute_120`; the plugin
extension modules contain none. That shows the XLA backend knows these architectures; it does
not show that a program runs on this card, and the CUDA 13 line of wheels (as opposed to
CUDA 12) is the one that targets Blackwell. The `with-cuda` extra pulls `nvidia-cuda-nvcc`
13.x (for `ptxas`). Whether JAX 0.12 runs on the RTX 5070 under WSL2 is untested here; the
project env is JAX 0.11.1 and would have to be upgraded to 0.12 to use these wheels.

For any VRAM measurement set `XLA_PYTHON_CLIENT_PREALLOCATE=false`; otherwise JAX reserves
75% of the device memory at start-up and the peak-memory figure reflects the reservation, not
the use.

## What does work today: CuPy on native Windows

A throwaway environment outside the repository (`uv pip install "cupy-cuda13x[ctk]"`,
CuPy 14.2.0, CUDA runtime 13.2 from wheels, no toolkit install) sees the GPU and runs cuFFT:

- device name `NVIDIA GeForce RTX 5070`, compute capability 120, runtime 13020, driver 13030;
- `cp.fft.fftn` of a 256 x 256 x 256 complex64 array took 1.6 ms per call (mean of 10 calls
  after one warm-up, synchronized), and `ifftn(fftn(a))` returned `a` to a maximum absolute
  error of 7.8e-7.

This is one probe, not a benchmark: one operation, one size, no comparison with the CPU run
under identical conditions. For scale only, the CPU run in `docs/perf.md` measured 73 ms for
`fftn + ifftn` at 256^3 on JAX (about 36 ms per transform if the two cost the same).

CuPy is not a dependency of QoraPhase and was not added to the project.

## Consequences for the GPU track

- GPU numbers for the existing JAX code require option 1 or 2 above. Until then, no GPU
  figure for QoraPhase exists; `docs/perf.md` stays CPU-only.
- A CuPy port of the RAAR/ER loop would run natively on this machine, but it would be a
  second implementation. It is only a fair comparison against the JAX code if the two are
  shown to produce the same results to a stated tolerance, and the issue asks for equal
  algorithms and work.
- A custom CUDA kernel (`QoraPhaseCU`) additionally needs the CUDA toolkit (`nvcc`) and a
  compiler supported by it; neither is installed. `cupy-cuda13x[ctk]` ships the runtime and
  libraries but not `nvcc`.
