# A lab ER/HIO shrinkwrap recipe against the QoraPhase default recipe on simulated 3D Bragg CDI data

QoraPhase benchmark note, 2026-11-04. Code: `benchmarks/bcdi_lab.py` (runs) and
`benchmarks/bcdi_lab_figures.py` (figures).

## Abstract

We asked whether QoraPhase's default phase-retrieval recipe recovers the strain phase of a
Bragg coherent diffraction imaging (BCDI) data set as well as a two-pass ER/HIO shrinkwrap
recipe written in MATLAB and used in our laboratory. The lab recipe was re-implemented in JAX
from reading its source. Both recipes were applied to simulated 3D data of one particle
(asymmetric, 250 nm, 61 × 128 × 128 detector voxels) and scored against the known object: at a
peak phase of 1.2 rad noiseless and with Poisson noise at 10⁵ photons in the brightest voxel,
and at peak phases of 0 and 3 rad with Poisson noise. At 1.2 rad and 10⁵ photons the default
recipe gave a median phase RMSE of 0.076 rad over 20 starts (0.070 rad when run for the lab's
1000 iterations), against 0.27–0.35 rad for the second-pass outputs of the lab recipe. Without
noise it gave 0.083 rad against 0.16–0.18 rad (0.12–0.22 rad with a twin check added to the
lab's support averaging). At 0 and 3 rad the default gave 0.052 and 0.115 rad against
0.21–0.42 and 0.24–0.37 rad. A twin check in the lab's support averaging helped at 0 and 3 rad
and without noise, but not at 1.2 rad and 10⁵ photons. Repeating a run with a different jax
release moved single runs by up to 0.1 rad at fixed seeds, so only ensemble statistics are
compared. Restarting QoraPhase from a twin-checked consensus of its own 20 final supports, with
that support held fixed, narrowed the ensemble at 1.2 rad (median 0.070 rad at 10⁵ photons and
0.062 rad noiseless, interquartile range about 0.01 rad); restarting from it with shrinkwrap made
the ensemble worse. With one particle shape and one noise realisation per case, these results
indicate a direction, not a general ranking.

## 1 Introduction

In BCDI a nanocrystal is illuminated by a coherent X-ray beam and the 3D intensity around one
Bragg peak is recorded by rocking the crystal through the peak [4]. The detector measures only
the intensity $I(\mathbf q) = |\mathcal F\rho(\mathbf q)|^2$ of the complex object
$\rho(\mathbf r) = |\rho(\mathbf r)|\,e^{i\phi(\mathbf r)}$. The phase
$\phi = \mathbf Q\cdot\mathbf u$ encodes the lattice displacement $\mathbf u$ along the
scattering vector $\mathbf Q$, which is why its recovery is the goal. Phase retrieval restores
the missing Fourier phase by iterating between two constraints: the *modulus constraint* (the
Fourier magnitude must equal $\sqrt I$; projection $P_M$) and the *support constraint* (the
object is zero outside a region $S$, the *support*; projection $P_S$).

The algorithms used here are:

- **ER** (error reduction), $x_{n+1} = P_S P_M x_n$ [1].
- **HIO** (hybrid input–output) [1]: $x_{n+1} = P_M x_n$ inside $S$ and
  $x_n - \beta P_M x_n$ outside, with feedback $\beta$.
- **RAAR** (relaxed averaged alternating reflections) [3]:
  $x_{n+1} = \tfrac{\beta}{2}(R_S R_M + I)x_n + (1-\beta)P_M x_n$, $R = 2P - I$.
- **Shrinkwrap** [2]: the support is periodically replaced by the region where the
  Gaussian-blurred object amplitude exceeds a threshold, so a loose initial support (here
  derived from the autocorrelation, the inverse transform of $I$) contracts onto the object.
- **Twin**: $\rho(\mathbf r)$ and $\rho^*(-\mathbf r)$ have the same $|\mathcal F\rho|$, so a run
  can converge to either; the twin's support is the point reflection of the true one, up to a
  shift.

The lab recipe is in use on experimental data. Before QoraPhase replaces it, we need to know
whether the default recipe is at least as accurate on data for which the true object is known,
and which parts of the lab recipe, if any, should be carried over.

## 2 Methods

### 2.1 Simulated data

Data were generated with rocking_xrd_sim (`bench.make_cases`, commit `65ac18d`). The object is a
faceted asymmetric particle 250 nm across (voxel 156.25 Å, 16 voxels across on the simulation
grid, about 15 px in the reconstruction frame) with $\rho = \chi_\Omega\,e^{i\phi}$. The phase is
a sum of linear, bilinear and two off-centre Gaussian terms, mean-free over the particle and
scaled so that $\max|\phi| = 1.2$ rad; two further cases use 0 rad (no strain) and 3 rad.
Diffraction geometry: 5.55 keV ($\lambda$ = 2.234 Å), $2\theta$ = 69°, rocking angle
34.2°–34.8° in 0.01° steps (61 frames), detector at 0.511 m with 128 × 128 pixels of 55 µm. Two
doses were used: noiseless, and intensities scaled so that the brightest voxel holds 10⁵
photons, followed by one Poisson draw (NumPy generator, seed 0). The 1.2 rad case was run at
both doses, the 0 and 3 rad cases at 10⁵ photons only.
Readout noise, background, beamstop, partial coherence and detector point spread were not
modelled. The reference object is the inverse DFT of the complex scattering amplitude sampled
at each detector voxel, that is, the object in the frame the reconstructions live in. Its
voxels are not cubic, so the slices in Figure 2 are distorted.

### 2.2 Lab recipe (re-implemented, not run in MATLAB)

The input amplitude is $F = \sqrt{I}$, with $F \le 1$ set to 0 in pass 1.

*Pass 1 (support estimate).* The start support is $|\mathcal F(F^2)|$, blurred with a Gaussian
($\sigma$ = 5 px), thresholded at min + 0.05 (max − min). We ran 20 random starts, each with
1000 iterations in 11 blocks alternating ER and HIO (20, 180, 20, 180, …, 20; HIO
$\beta$ = 0.9, complex object, no positivity). During blocks 1–6 (600 iterations):

- shrinkwrap runs every 20 iterations: the amplitude after the modulus step is blurred
  ($\sigma$ = 2 px, finite kernel, replicated edges) and thresholded at min + $t$(max − min),
  with $t$ = 0.10, 0.11, …, 0.15 per block, then intersected with the start support;
- the data are multiplied by a Gaussian window in $q$, $W$, whose width grows linearly from
  0.8 to 2.0 times the half array size.

The 20 final supports are shifted onto the first (cross-correlation peak to 0.1 px, clamped to
±5 px), clipped to [0, 1], summed, and thresholded at 20 % of the maximum.

*Pass 2 (phasing).* The same schedule, with the raw $F$, the summed support held fixed and no
shrinkwrap, over 3 seeds. The seed with the lowest best-so-far error is kept, and its last
iterate is the result.

*Deviations from the lab code:*

- the random start phase varies in all three dimensions (the lab's is constant along the third
  axis, a defect);
- the cross-correlation shift and the clipping replace the lab's L1 fit and de-ringing;
- the first support is counted once (the lab counts it twice);
- the support sum is run twice, as written and with a twin check, in which the point-reflected
  support is also tried and the closer one kept.

*Seed choice.* The error is

$$E_{\mathrm{lab}} = \frac{\sum_{\mathbf q}\big|F W - |\mathcal F(P_S x)|\big|}{\sum_{\mathbf q} F W}.$$

The *lab rule* uses $W$ as above, so errors from the windowed blocks are on a different scale.
The *unwindowed rule* sets $W = 1$ at every iteration. Only iterations after the 5th are
eligible.

### 2.3 QoraPhase recipes

The default recipe is `default_recipe()` of QoraPhase 0.2.0:

RAAR(50) → detwin[HIO(20)] → RAAR(50) → 8 × [RAAR(50) + Shrinkwrap] → ER(200)

That is 720 iterations, with RAAR and HIO $\beta$ = 0.9. Detwin runs HIO on the support cut in
half along axis 0 and then restores it. Shrinkwrap thresholds the blurred amplitude
($\sigma$ = 1 px, periodic Fourier blur) at 0.3 × max. The start support is
$|\mathcal F^{-1} I| > 0.1\max$. QoraPhase's own error, reported for scale only, is

$$E_F = \Big(\sum_{\mathbf q}(|\mathcal F x| - F)^2 / \sum_{\mathbf q}F^2\Big)^{1/2}.$$

Two variants were also run, each with 20 starts:

- **1000 iterations** (10⁵ photons only): the eight shrinkwrap rounds run RAAR(85) instead of
  RAAR(50), so the recipe has the lab's 1000 iterations and still updates the support eight
  times.
- **Own consensus** (both doses): the 20 final supports of the default run are combined like the
  lab's pass 1 supports, with the twin check, and the result is the start support of a second
  set of 20 starts, once with the default recipe and once with its shrinkwrap rounds replaced by
  the same 400 RAAR iterations, so the support stays fixed (the analogue of the lab's pass 2).

Each 20-start set yields two single outputs: `best`, the lowest final $E_F$, and `consensus`,
the run with the smallest mean distance to the other runs after alignment.

### 2.4 Accuracy metrics

Each reconstruction $\hat\rho$ is aligned to the truth $\rho$: we choose $\hat\rho$ or its
twin, shift it by the integer cross-correlation peak, and remove the global phase and a fitted
linear phase ramp. The amplitude is then rescaled by
$s = \langle|\hat\rho|,|\rho|\rangle/\||\hat\rho|\|^2$. With supports
$S_t = \{|\rho| > 0.3\max|\rho|\}$ and $S_r$ (the same for $s|\hat\rho|$), and $N_t = |S_t|$:

$$\mathrm{IoU} = \frac{|S_r\cap S_t|}{|S_r\cup S_t|}$$

$$\mathrm{NRMSE}_A = \frac{\|s|\hat\rho| - |\rho|\|_{S_r\cup S_t}}{\||\rho|\|_{S_t}}$$

$$\mathrm{RMSE}_\phi = \Big(\frac{1}{N_t}\sum_{S_t}\arg^2(\hat\rho\,\rho^*)\Big)^{1/2}$$

$$\mathrm{NRMSE}_c = \frac{\|c\hat\rho - \rho\|}{\|\rho\|},\quad c = \frac{\langle\hat\rho,\rho\rangle}{\|\hat\rho\|^2}$$

$\arg$ is wrapped to $(-\pi,\pi]$. The phase retrieval transfer function (PRTF) [5] over each
set of runs was also computed (`qp.prtf`, 1/e crossing).

### 2.5 Software, hardware and reproducibility

All runs were single CPU processes in float32 (x64 off) on an AMD Ryzen 5 7500F (6 cores,
12 threads, 31.6 GB RAM) under Windows 11 (10.0.26300), on 2026-11-04. Software, from the
project lock file: Python 3.14.5 (GIL build), jax and jaxlib 0.11.1, NumPy 2.5.2, SciPy 1.18.1.
JAX random keys were split from seed 0 (pass 1 and QoraPhase) and seed 1 (pass 2). Wall times
include compilation.

*Determinism.* Two runs of the 10⁵ case in this environment, 15 minutes apart and with other
jobs on the machine during the second, gave identical scores for every start of every method.
At fixed seeds the results are deterministic within one environment.

*Version sensitivity.* A first run of the same script on 2026-11-02 used jax and jaxlib
0.12.0.1 on Python 3.13 (outside the lock). With the same seeds, single runs differed between
the two environments by up to 0.1 rad in phase RMSE, and medians by up to 0.03 rad. The two
environments therefore produce different random draws or different floating-point paths from
the same seed, and a single run is not a reproducible quantity across versions. Only ensemble
statistics are compared below; the numbers are those of the locked environment.

## 3 Results

Tables 1 and 2 give the accuracy at 1.2 rad for the two doses, Table 3 at 0 and 3 rad. For 20-run sets we report the median, the
interquartile range [Q1–Q3] and, for the phase, the min–max range over starts. For pass 2 of
the lab recipe we report the seed chosen by each rule, with all three seeds given for the phase.
The lab recipe's output is pass 2; pass 1 is shown for reference. Phase is in rad; the other
metrics are dimensionless. Lower is better, except for IoU.

**Table 1.** Peak phase 1.2 rad, 10⁵ photons.

| Method | Phase RMSE (rad) | Amplitude NRMSE | Support IoU | Complex NRMSE |
|---|---|---|---|---|
| Lab pass 1, 20 runs | 0.217 [0.201–0.285] (0.137–0.362) | 0.258 [0.239–0.267] | 0.880 [0.842–0.894] | 0.354 [0.332–0.381] |
| Lab pass 2, support as lab (4911 vx), lab rule | 0.330 (seeds 0.274–0.344) | 0.304 | 0.813 | 0.451 |
| ditto, unwindowed rule | 0.274 | 0.314 | 0.809 | 0.452 |
| Lab pass 2, twin-checked support (4568 vx), lab rule | 0.282 (seeds 0.275–0.351) | 0.289 | 0.820 | 0.445 |
| ditto, unwindowed rule | 0.351 | 0.274 | 0.835 | 0.394 |
| QoraPhase default, 720 it, 20 runs | 0.076 [0.055–0.112] (0.045–0.206) | 0.170 [0.124–0.204] | 0.931 [0.912–0.955] | 0.211 [0.162–0.245] |
| QoraPhase default, 720 it, `best` / `consensus` | 0.052 / 0.072 | 0.128 / 0.166 | 0.936 / 0.934 | 0.166 / 0.206 |
| QoraPhase default, 1000 it, 20 runs | 0.070 [0.065–0.084] (0.047–0.159) | 0.171 [0.155–0.192] | 0.918 [0.906–0.937] | 0.211 [0.196–0.233] |
| QoraPhase default, 1000 it, `best` / `consensus` | 0.071 / 0.056 | 0.186 / 0.165 | 0.901 / 0.918 | 0.224 / 0.202 |
| Own consensus (2140 vx) → default recipe, 20 runs | 0.104 [0.073–0.147] (0.055–0.217) | 0.204 [0.183–0.219] | 0.912 [0.886–0.925] | 0.244 [0.220–0.258] |
| ditto, `best` / `consensus` | 0.086 / 0.156 | 0.197 / 0.214 | 0.917 / 0.908 | 0.237 / 0.256 |
| Own consensus (2140 vx) → fixed support, 20 runs | 0.070 [0.063–0.076] (0.055–0.102) | 0.144 [0.139–0.151] | 0.950 [0.946–0.953] | 0.187 [0.183–0.194] |
| ditto, `best` / `consensus` | 0.055 / 0.076 | 0.141 / 0.142 | 0.950 / 0.953 | 0.180 / 0.187 |

**Table 2.** Peak phase 1.2 rad, noiseless.

| Method | Phase RMSE (rad) | Amplitude NRMSE | Support IoU | Complex NRMSE |
|---|---|---|---|---|
| Lab pass 1, 20 runs | 0.171 [0.146–0.188] (0.094–1.755) | 0.179 [0.158–0.211] | 0.920 [0.902–0.935] | 0.279 [0.267–0.300] |
| Lab pass 2, support as lab (3525 vx), lab rule | 0.165 (seeds 0.155–0.182) | 0.244 | 0.875 | 0.334 |
| ditto, unwindowed rule | 0.182 | 0.218 | 0.911 | 0.329 |
| Lab pass 2, twin-checked support (3460 vx), lab rule | 0.117 (seeds 0.117–0.215) | 0.112 | 0.966 | 0.217 |
| ditto, unwindowed rule | 0.150 | 0.181 | 0.918 | 0.262 |
| QoraPhase default, 720 it, 20 runs | 0.083 [0.062–0.148] (0.049–0.204) | 0.175 [0.155–0.220] | 0.912 [0.904–0.927] | 0.215 [0.193–0.256] |
| QoraPhase default, 720 it, `best` / `consensus` | 0.049 / 0.147 | 0.145 / 0.181 | 0.929 / 0.909 | 0.183 / 0.218 |
| Own consensus (2380 vx) → default recipe, 20 runs | 0.131 [0.072–0.169] (0.051–0.247) | 0.214 [0.183–0.225] | 0.908 [0.898–0.917] | 0.251 [0.221–0.262] |
| ditto, `best` / `consensus` | 0.078 / 0.186 | 0.206 / 0.220 | 0.889 / 0.878 | 0.240 / 0.259 |
| Own consensus (2380 vx) → fixed support, 20 runs | 0.062 [0.056–0.066] (0.050–0.353) | 0.162 [0.156–0.175] | 0.924 [0.921–0.926] | 0.199 [0.192–0.209] |
| ditto, `best` / `consensus` | 0.063 / 0.056 | 0.175 / 0.152 | 0.924 / 0.927 | 0.209 / 0.189 |

**Table 3.** Peak phase 0 and 3 rad, 10⁵ photons.

| Method | Phase RMSE (rad) | Amplitude NRMSE | Support IoU | Complex NRMSE |
|---|---|---|---|---|
| *0 rad* | | | | |
| Lab pass 1, 20 runs | 0.216 [0.151–0.274] (0.116–0.349) | 0.234 [0.206–0.279] | 0.887 [0.860–0.903] | 0.323 [0.289–0.371] |
| Lab pass 2, support as lab (4650 vx), both rules | 0.415 (seeds 0.207–0.415) | 0.332 | 0.766 | 0.493 |
| Lab pass 2, twin-checked support (3705 vx), both rules | 0.262 (seeds 0.229–0.274) | 0.279 | 0.821 | 0.383 |
| QoraPhase default, 720 it, 20 runs | 0.052 [0.043–0.064] (0.029–0.205) | 0.156 [0.129–0.194] | 0.933 [0.913–0.955] | 0.189 [0.167–0.229] |
| QoraPhase default, 720 it, `best` / `consensus` | 0.033 / 0.032 | 0.104 / 0.180 | 0.955 / 0.915 | 0.138 / 0.209 |
| *3 rad* | | | | |
| Lab pass 1, 20 runs | 0.262 [0.225–0.328] (0.203–0.435) | 0.266 [0.238–0.289] | 0.866 [0.840–0.882] | 0.398 [0.357–0.447] |
| Lab pass 2, support as lab (5019 vx), lab rule | 0.320 (seeds 0.320–0.360) | 0.285 | 0.844 | 0.449 |
| ditto, unwindowed rule | 0.326 | 0.297 | 0.834 | 0.466 |
| Lab pass 2, twin-checked support (3845 vx), lab rule | 0.237 (seeds 0.237–0.365) | 0.252 | 0.899 | 0.370 |
| ditto, unwindowed rule | 0.246 | 0.240 | 0.882 | 0.376 |
| QoraPhase default, 720 it, 20 runs | 0.115 [0.078–0.134] (0.059–0.233) | 0.171 [0.117–0.202] | 0.925 [0.917–0.961] | 0.230 [0.162–0.258] |
| QoraPhase default, 720 it, `best` / `consensus` | 0.115 / 0.131 | 0.202 / 0.165 | 0.920 / 0.935 | 0.258 / 0.222 |

![Phase RMSE of every start, per method and dose](bcdi-lab-spread.png)

**Figure 1.** Phase RMSE of every start. Dots are single runs (jittered vertically), bars are
medians. Values above 0.45 rad are drawn at the edge and labelled.

![Central slice of the truth and of single reconstructions at 10⁵ photons](bcdi-lab-slices.png)

**Figure 2.** One detector-frame slice through the centre of the particle at 10⁵ photons, after
alignment to the truth. Top: amplitude scaled to each object's maximum. Bottom: phase where the
amplitude exceeds 0.3. The lab outputs are the pass-2 seeds chosen by the lab rule.

**Twin convergence.** In the lab's support sum, the pass-1 supports that matched the first one
better after point reflection were 8 of 20 (1.2 rad, 10⁵), 4 of 20 (1.2 rad, noiseless) and 11
of 20 (0 and 3 rad). The twin-checked support was smaller by 7 %, 2 %, 20 % and 23 %. Where many
supports were flipped it helped clearly: at 0 rad the chosen seed went from 0.415 to 0.262 rad,
at 3 rad from 0.320 to 0.237 rad (lab rule). Without noise it improved the seed chosen by either
rule (0.165 → 0.117 rad, 0.182 → 0.150 rad). At 1.2 rad and 10⁵ photons it did not: the three
seeds span 0.274–0.344 rad with the lab's support and 0.275–0.351 rad with the twin-checked one.
In the 2026-11-02 run the twin check had improved every pass-2 score in that case too; that gain
did not survive the change of environment, and with three seeds it is within the seed spread.

**Seed choice.** Under the lab rule the best-so-far error fell inside the windowed phase in
all 80 pass-1 and 24 pass-2 runs, mostly at iterations 219–220. Under the unwindowed rule it
fell outside it in 77 of 80 pass-1 runs and in all 24 pass-2 runs. The two rules chose
different pass-2 seeds in 6 of 8 cases. Neither rule was consistently better.

**Equal iterations.** At 10⁵ photons, running the default recipe for 1000 iterations instead of
720 moved the median phase RMSE from 0.076 to 0.070 rad and narrowed the spread (IQR 0.057 →
0.019 rad); the gap to the lab recipe is not an effect of the iteration count.

**Own consensus.** When QoraPhase's own 20 final supports were combined, 8 of 20 matched the
first one better after point reflection at both doses: QoraPhase's runs also end in either twin
orientation. The scores do not see this, because the alignment picks the twin. The consensus
support had 2140 (10⁵) and 2380 (noiseless) voxels, against 1970–2045 for single final supports.

- *Support fixed.* The spread nearly vanished: IQR 0.063–0.076 rad (10⁵) and 0.056–0.066 rad
  (noiseless), against 0.055–0.112 and 0.062–0.148 rad for the default from the autocorrelation
  support. Without noise, 16 of 20 runs reached 0.050–0.068 rad, all with $E_F$ 0.0422–0.0425;
  the other four (0.12–0.13 and 0.35 rad) ended at $E_F$ 0.049 and 0.054, so `best` picks a good
  run. At 10⁵ photons $E_F$ (0.0765–0.0774) does not separate the runs: the third-lowest is the
  worst (0.102 rad).
- *Shrinkwrap on.* The ensemble was worse than the plain default (median 0.104 and 0.131 rad).
  Shrinkwrap tightened the support back to 1970–2043 voxels, and the start from a better support
  did not carry over.

**Supports.**

| Support | Voxels |
|---|---|
| Lab start support | 19 297 (1.2 rad), 19 267 (0 rad), 19 467 (3 rad) |
| Lab pass-1 final | 3129–3290 (1.2 rad, 10⁵), 3163–3289 (1.2 rad, noiseless), 3127–3264 (0 rad), 3190–3376 (3 rad) |
| QoraPhase final (default) | 1984–2041 (1.2 rad, 10⁵), 1980–2031 (1.2 rad, noiseless), 1960–2051 (0 rad), 1950–2035 (3 rad) |
| Consensus of QoraPhase's final supports | 2140 (1.2 rad, 10⁵), 2380 (1.2 rad, noiseless) |

**Fourier error.** QoraPhase's $E_F$ ended at 0.079–0.080 (1.2 rad, 10⁵), 0.049–0.051
(noiseless), 0.079–0.081 (0 rad) and 0.073–0.075 (3 rad).

**PRTF.** The PRTF stayed above 1/e to the array edge for every set except lab pass 1 at 10⁵
photons (crossing 0.43–0.48 cycles/px). It does not discriminate between the methods here.

**Run time** (CPU, 20 starts in batches of 5, including compilation; 1.2 rad, 10⁵ photons):

| Run | Wall time | Per start-iteration |
|---|---|---|
| Lab pass 1 (20 × 1000 iterations) | 415 s | 21 ms |
| Lab pass 2 (3 × 1000 iterations) | 59–62 s | 20 ms |
| QoraPhase default (20 × 720 iterations) | 166 s | 11.5 ms |

The first noiseless timings (pass 1 471 s, default 189 s) were taken while another process
used the CPU and are not used. In the own-consensus run the default recipe took 161 s
(noiseless) and 178 s (10⁵, with a short plotting job alongside), the consensus 1 s and each
restart 159–164 s. In the 0 and 3 rad runs pass 1 took 361–370 s, pass 2 49 s and the default
162–163 s. The repeat of the 10⁵ case for the 1000-iteration variant overlapped other
jobs (default 193 s; 1000 iterations 239 s, 12 ms per start-iteration).

## 4 Discussion

In every case the default recipe's typical run has between an eighth and a half of the phase
error of the lab output as written (lab support, lab rule), and better amplitude and support
scores. The ratio is smallest without
strain (0.052 rad against 0.26–0.42 rad) and largest without noise. Running it for
the lab's 1000 iterations changes little, so the difference is not one of compute. The
noiseless default ensemble is again bimodal (IQR 0.062–0.148 rad), as reported in
[Default recipe on simulated BCDI data](../bcdi-support.md); here `best` landed in the good
group and `consensus` in the poor one.

Two properties of the lab recipe plausibly account for part of its deficit (inference):

- its support sum adds twin-converged supports unflipped, which blurs the fixed support;
- its seed choice compares errors computed against differently windowed data, so in practice
  the choice is made at a low-resolution iteration.

The first matters when many pass-1 runs end as twins (11 of 20 at 0 and 3 rad) and without
noise. At 1.2 rad and 10⁵ photons the lab's fixed support (4568–4911 voxels, against about 2000
for QoraPhase's final supports) is loose whether or not the twins are flipped, and the pass-2
result is limited by that. Even twin-checked, the lab's fixed supports (3460–4568 voxels) are
much larger than QoraPhase's.

The own-consensus runs suggest that the lab's two-pass structure is worth keeping, with
QoraPhase's runs as pass 1 (inference). Averaging 20 shrinkwrapped supports with the twin check
gives a support slightly larger than any single one, and phasing on it with the support fixed
removes most of the bimodal spread. The lab's pass 2 does the same on a support more than twice
as large, from a looser pass 1, which is consistent with its larger error. Letting shrinkwrap run
again in the second pass gives the gain away.

Limitations:

1. **Simulation only.** No experimental data; the noise model is Poisson only.
2. **One particle.** One shape and size, three strain amplitudes, one Poisson realisation per
   case. The equal-iteration and own-consensus variants were run at 1.2 rad only.
   [Default recipe on simulated BCDI data](../bcdi-support.md) shows that results on this
   simulator change with shape and size.
3. **Few seeds.** Pass 2 has three seeds, so differences within that range are not significant.
   No statistical test was applied.
4. **Not the lab code itself.** The lab recipe is a re-implementation checked by reading the
   code, not against a MATLAB run, and two of its steps (alignment, de-ringing) were
   substituted.
5. **Environment.** One Windows CPU in float32. Single runs are not reproducible across jax
   versions (Section 2.5); the medians moved by up to 0.03 rad.
6. **Unequal iterations without noise.** The noiseless comparison is 1000 (lab) against 720
   (QoraPhase) iterations; only the 10⁵ case was repeated at equal iterations.

## 5 Conclusion

On simulated BCDI data of one particle, QoraPhase's default recipe recovered the strain phase
more accurately than a re-implementation of the lab's ER/HIO shrinkwrap recipe in all four
cases: at 1.2 rad a median of 0.076 rad (0.070 rad at equal iterations) against 0.27–0.35 rad at
10⁵ photons and 0.083 rad against 0.12–0.18 rad without noise; at 0 and 3 rad 0.052 and 0.115
rad against 0.21–0.42 and 0.24–0.37 rad. The lab's support averaging needs a twin check. A
second pass with the support fixed at a twin-checked consensus of QoraPhase's own final supports
narrowed the ensemble to an interquartile range of about 0.01 rad at both doses; it is a
candidate for a two-pass recipe, not yet a default. More shapes and noise draws are needed
before the result can be generalised.

## Reproduce

```
ROCKING_XRD_SIM=<checkout> uv run --python 3.14+gil --with scipy \
    python benchmarks/bcdi_lab.py CASE_DIR [CASE_DIR ...] --out DIR [--keep] [--equal-iterations]
ROCKING_XRD_SIM=<checkout> uv run --python 3.14+gil --with scipy \
    python benchmarks/bcdi_lab.py CASE_DIR [CASE_DIR ...] --out DIR --own-consensus --keep
ROCKING_XRD_SIM=<checkout> uv run --python 3.14+gil --with scipy --with matplotlib \
    python benchmarks/bcdi_lab_figures.py CASES_DIR --results DIR [DIR ...] --out docs/benchmarks
```

Cases are made with rocking_xrd_sim `bench.make_cases` (asymmetric particle, 250 nm, box 128;
peak phase 1.2 rad at `ph1e5` and `phinf`, 0 and 3 rad at `ph1e5`). The slice figure uses
`--slices asymmetric_250nm_strain1.2_ph1e5_s0`.

## References

1. J. R. Fienup, "Phase retrieval algorithms: a comparison," *Appl. Opt.* **21**, 2758–2769 (1982).
2. S. Marchesini et al., "X-ray image reconstruction from a diffraction pattern alone," *Phys. Rev. B* **68**, 140101(R) (2003).
3. D. R. Luke, "Relaxed averaged alternating reflections for diffraction imaging," *Inverse Problems* **21**, 37–50 (2005).
4. I. Robinson and R. Harder, "Coherent X-ray diffraction imaging of strain at the nanoscale," *Nat. Mater.* **8**, 291–298 (2009).
5. H. N. Chapman et al., "High-resolution ab initio three-dimensional x-ray diffraction microscopy," *J. Opt. Soc. Am. A* **23**, 1179–1200 (2006).
