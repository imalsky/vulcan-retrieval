# Method

What the retrieval does, with the source of each method. Results and the reasons behind the rules are in `notes.md`.

## Modules

| Module | Job |
|---|---|
| `config_schema` | `Config`, the parameter list, `specs_from_config`, the run banner; no jax |
| `run_setup` | The run specification (`load_spec`), logging, a run's identity files |
| `observations` | The observed spectrum and the exact binning and offset operators |
| `tp_profile` | The Guillot T-P evaluator |
| `retrieval_forward` | theta to the native transit depth, on the engine in `vulcan-forward` |
| `pipeline` | The u-space posterior and the batched likelihood evaluators |
| `provenance` | Code and data identity, the target manifest and digest |
| `run_nautilus` | The sampler driver |
| `nautilus_report` | Certificate and figures of a finished run |
| `validate_env` | Environment validation, run first by every job |
| `forward/config` | This repository's paths; importing it hands the engine its data root |

## Parameters

$$\theta = [\ln Z,\ \Delta\ln({\rm C/O}),\ \ln K_{zz},\ [s_o],\ [\alpha_K],\ \text{T-P},\ \ln R_0,\ [\ln M],\ \text{cloud},\ \text{offsets},\ \text{noise}]$$

$s_o$ is ln of the S/O multiple of the baseline, $\alpha_K$ the Kzz power-law slope, and brackets mark optional entries.
The chemistry and T-P entries re-converge VULCAN (Tsai et al. 2021) and are the expensive part of every likelihood call.
$\ln R_0$, $\ln M$ (Gaussian, RT-only) and the cloud parameters enter only the RT, and the instrument offsets apply after
the spectrum. Planet identity enters through what the spec sets: the observed spectrum, the VULCAN baseline config,
gravity and radii, and the priors.

## T-P profile

exojax's `atmprof_Guillot` (Kawahara et al. 2022) is the irradiated analytic profile of Guillot (2010), with plain
`jnp.exp` rather than the $E_2$ exponential integral, so it is forward-mode clean. The same curve is evaluated on the VULCAN
grid and the ART grid. The chemistry side rebuilds the rates and the T- and composition-dependent structure from it; the
photolysis cross sections are T-independent. The exojax side builds its own hydrostatic transit geometry from the same
T(P) and the chemistry's mean molecular weight, interpolated with `interp_map`.

$f = 1/4$ is the global-average irradiation convention, an analytic-shape choice and not a terminator measurement, so the
retrieved $(T_{\rm irr}, \kappa, \gamma)$ are flexible shape parameters of the limb profile. The deep parameter, a fixed
$T_{\rm int}$, a retrieved $T_{\rm int}$, or with `tp_deep_param="T1bar"` the temperature at `tp_deep_p_bar`, adds
$\frac{3}{4} T_{\rm int}^4 \tau$ and so sets the deep column. With constant $\kappa$ the deep gradient is 0.25, below the
adiabat, so the profile never turns convective on its own, and $T_{\rm int}$ is a deep-temperature knob, not the planet's
measured intrinsic flux. Retrieving T at a deep level puts the prior on the quantity the validity window acts on:
$T_{\rm int}$ follows in closed form, since $T^4$ is linear in $T_{\rm int}^4$, and a draw whose $T_{\rm int}^4$ would be
negative evaluates to NaN and is rejected. `tp_adiabat` replaces the profile below a free boundary pressure by the
adiabat matched there, a sensitivity model: the join is imposed.

## Observations

Product CSVs follow the Carter & May (2024) convention. The binning operator is an exact matrix, so a binned depth's JVP is
$B$ times the native depth's. It applies no instrument line-spread function, so a bin must span at least two model bands of
the R1000 tables; that admits the native 0.015 um NIRCam bins of Welbanks et al. (2024).

## Likelihood

Diagonal Gaussian with an optional noise inflation (Line et al. 2015 style). `make_nuisance_marginal` integrates the offsets
and the inflation(s) out at the same priors: the offset integral is closed form, the inflation a trapezoid in $\log_{10} b$.

## Sampler

Stock nautilus-sampler 1.0.6 (Lange 2023) with its default networks, `N_NETWORKS` ReLU MLPs, $n_{\rm update} = n_{\rm live}/2$,
exploration until `f_live = F_LIVE`, bounds enlarged by `ENLARGE_PER_DIM`, and the exploration phase discarded
(`DISCARD_EXPLORATION`, the unbiased estimator; keeping it biased log Z by about -0.2 on surrogates of the W39b posterior).
Each sampling-phase batch goes to one shell, as in stock nautilus. nautilus samples the unit cube; $u = {\rm logit}(z)$ is the
retrieval's own u-space, so theta follows the box prior exactly. `validate_env` refuses the former fork (SiLU networks,
spread_batch), which carried the same version string.

`n_batch` defaults to `BATCH_PER_LANE` times the lanes, so a lane whose column certifies refills from the same batch.
Padding and out-of-window draws cost no solve: a chunk shorter than `n_batch` is filled with copies of its last row, and the
evaluator masks those and every T-P-invalid row. The RT runs only on the rows whose log L can be kept.

Lanes are `cfg.cold_lanes`, times `FFI_BLOCKS_PER_SM` under the ffi solver, whose one-buffer kernels keep that many lanes
resident per SM.

### Warm starts

Every certified column becomes an anchor keyed by its unit-cube chemistry and T-P coordinates, and each later solve continues
from its nearest anchor; batches before the first certified column run cold. The certified state and which draws are rejected
then depend on the evaluation order at the convergence tolerance. Anchors are written inside the likelihood call, before
nautilus checkpoints the batch, so a job killed in between resumes by redrawing that batch, and the tally counts it twice.
The set is never pruned: 44 KB per certified column, on disk and in memory.

### Startup check

The first chunk of a fresh run (cold, box-prior draws) stops the run, before any anchor or checkpoint is written, if no draw
was kept or if a decade of log10 Kzz at the case's reference pressure holds `STARTUP_MIN_DRAWS` or more in-window draws and
none certified, so a prior region the solver cannot certify is refused in minutes.

### Output files

In `output/<case>/<name>_nautilus`: `nautilus_checkpoint.hdf5`, rewritten every iteration; `anchors/`, one npz of certified
columns per batch; `rejected/`, one npz per chunk of the rejected draws (`z`, class `cls` 0 T-P out, 1 exhausted, 2 stalled,
3 non-finite, `acc`, `warm`); with integrated nuisances `nuisance/`, one npz per chunk of the certified rows (`z`, `log_l`,
`stats`); `batches.jsonl`, each chunk's wall time, start mode and accepted steps; `nautilus_tally.json`, the rejection and
warm-start counts summed over jobs. A finished run adds `nautilus_posterior.npz`, `nautilus_summary.json` and
`nautilus_certificate.json`. The run stops at `cfg.walltime_seconds` and exits 0 unfinished; resubmit with `--resume`.

## Report

`derived_quantities` gives weighted 16/50/84% quantiles of: metallicity as O/H and as total metals (C and S at their own
knobs, with the Lodders 2020 photospheric abundances), C/O, S/O, the Kzz factor and the absolute log10 Kzz at 1 mbar, the
weight near the prior floor, on a case with a ceiling the weight above it and the posterior truncated at it, T at the deep
level and $T_{\rm int}$, and the integrated offsets and inflations. `conditioning_map` shows where the rejected draws sit.

Model spectra (`--spectra N`) cost the pipeline build and compile plus one batched cold two-stage solve of N columns and N RTs,
about one likelihood batch of the run at width N. A draw whose cold solve does not certify is left out of the bands and counted
in the log. Draws and spectra go to `nautilus_spectra.npz`.

## Validation scripts

### e2e_check

Rungs, top to bottom; batch rungs run on the run's GPU, and each fixes every other setting:

| Rung | Change |
|---|---|
| `production` | The spec's runtime and lanes, one batch of every draw through the production cold evaluator |
| `shuffled` | The draws in another order; production against shuffled is the noise floor between two equivalent solves |
| `cmdbuf_off` | Command buffers off, the other XLA flags kept |
| `no_xla_flags` | `XLA_FLAGS` unset |
| `codegen` | The codegen chemistry RHS |
| `fast` | The fast solver (JAX scans) in place of the CUDA kernel |
| `solo_gpu` | Each draw alone, fast and codegen, on the GPU |
| `solo_cpu` | The same on the CPU |
| `reference_cpu` | The reference solver on the CPU: the parity pair behind the VULCAN 2 gates |

The draws are `--draws` seeded in-window prior draws with the first rows replaced by designed corners. `--rows a:b` solves a
slice; `--u-from <npz>` takes the set from that file, so a CPU machine solves the GPU's draws row for row. `--posterior <out_dir>`
draws from a finished run's weighted posterior, keeping each row's run log L.

`compare` reports, over the draws both sides certify, the worst bin shift and the total displacement
$D = \sqrt{\sum (\delta\mu/\sigma)^2}$, $|\delta\log_{10}{\rm VMR}|$ of the six species at $10^{-5}$ to 1 bar, $|\delta\log L|$ on the
draws within 50 nats of the best, and every certificate mismatch. Gates, provisional: worst bin 0.1 sigma, $D$ 0.5, 95th
percentile 0.03 dex and worst 0.1 dex, $|\delta\log L|$ 0.5, no certificate mismatch. An exceedance is a disagreement to explain,
not a verdict by itself.

`warmcold` is the warm-against-cold check on posterior rows solved cold by the production rung: the subsample reweighted by
$\exp(L_{\rm cold} - L_{\rm run})$, a row that does not certify cold given weight 0, and every parameter's median shift in
posterior sigma, low-Kzz rows ($\ln K_{zz} < -0.7$) apart. Gate: every median within 0.1 sigma. A warm run's numbers are quotable
only after it passes.

### posterior_forward_check variants

| Variant | Change |
|---|---|
| `base` | The run's own model |
| `network2025` | `thermo/SNCHO_photo_network_2025.txt` |
| `dayside` | Dayside-average photochemistry geometry: `sl_angle` acos(0.5), `f_diurnal` 0.5 |
| `twolimb` | Two Guillot limbs at Tirr -/+ `TWO_LIMB_DTIRR_K`, depths averaged |
| `mass` | The planet mass at +-1 sigma in the RT, at frozen chemistry |
| `spot` | The transit-light-source factor of a spotted star |

## Glossary

| Symbol | Meaning |
|---|---|
| `lnZ`, `c_o`, `s_o` | ln of the metallicity, C/O and S/O multiples of the baseline |
| `lnKzz`, `kzz_slope` | ln of the eddy-diffusion multiple, the power-law slope |
| `Tirr`, `Tint`, `T1bar` | Irradiation and interior temperature, the temperature at the deep level |
| `lnR0`, `lnM` | ln of the radius and mass multiples, RT-only |
| `P_rcb` | The radiative-convective boundary pressure |
| `B`, `O_*`, `N`, `Nm` | Binning, offset and group matrices |
| `R`, `S`, `G`, `W` | The group statistics of the integrated nuisances, the group count, the group weight |
| `U`, `Y`, `Z`, `C_`, `L`, `Theta`, `AUX`, `CD` | Batched u, columns, unit-cube points, chemistry blocks, log L, theta, profiles, `ConvDiag` |
| `T`, `P`, `M` | Temperature, pressure, number density |

## References

- Carter, A. L., May, E. M., et al. 2024, Nature Astronomy, 8, 1008
- Guillot, T. 2010, A&A, 520, A27
- Kawahara, H., et al. 2022, ApJS, 258, 31
- Lange, J. U. 2023, MNRAS, 525, 3181, doi:10.1093/mnras/stad2441
- Line, M. R., et al. 2015, ApJ, 807, 183, doi:10.1088/0004-637X/807/2/183
- Lodders, K. 2020, Oxford Research Encyclopedia of Planetary Science, doi:10.1093/acrefore/9780190647926.013.145
- Tsai, S.-M., et al. 2021, ApJ, 923, 264
- Welbanks, L., et al. 2024, Nature, 630, 836
