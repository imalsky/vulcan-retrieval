# Method

What the retrieval does, with the source of each method. Results and the reasons behind the rules are in `notes.md`.

## Modules

| Module | Job |
|---|---|
| `constants` | The fixed numbers, each with its unit and reason; imports nothing |
| `config_schema` | `Config` and `validate_config`, the parameter list, `specs_from_config`, the run banner; no jax |
| `run_setup` | The run specification (`load_spec`, `export_runtime`), logging, a run's identity files |
| `observations` | The observed spectrum and the exact binning and offset operators |
| `tp_profile` | The Guillot T-P evaluator |
| `retrieval_forward` | `RetrievalForward`: theta to the native transit depth, on the engine in `vulcan-forward`; the lane route |
| `pipeline` | `Pipeline`: the u-space posterior, the batched likelihood evaluators, the T-P window |
| `nuisance` | The integrated offsets and inflations |
| `provenance` | Code and data identity, the target manifest and digest |
| `run_nautilus` | The sampler driver: anchors, the resume refusal |
| `batch_profiler` | The environment-gated CUDA profiling hook of the likelihood batches |
| `nautilus_report` | Certificate of a finished run, and the command that writes it and the figures |
| `report_figures` | The three figures |
| `validate_env` | Environment validation, run first by every job |
| `forward/config` | This repository's paths; importing it hands the engine its data root |

## Parameters

$$\theta = [\ln Z,\ \Delta\ln({\rm C/O}),\ \ln K_{zz},\ [s_o],\ [\alpha_K],\ \text{T-P},\ \ln R_0,\ [\ln M],\ \text{cloud},\ \text{offsets},\ \text{noise}]$$

$s_o$ is ln of the S/O multiple of the baseline, $\alpha_K$ the Kzz power-law slope, and brackets mark optional entries.
The chemistry and T-P entries re-converge VULCAN (Tsai et al. 2021) and are the expensive part of every likelihood call.
$\ln R_0$, $\ln M$ (Gaussian, RT-only) and the cloud parameters (two for `powerlaw`, one for `deck`, none for `none`)
enter only the RT, and the instrument offsets apply after the spectrum. Planet identity enters through what the spec
sets: the observed spectrum, the VULCAN baseline config, gravity and radii, and the priors.

## T-P profile

exojax's `atmprof_Guillot` (Kawahara et al. 2022) is the irradiated analytic profile of Guillot (2010), with plain
`jnp.exp` rather than the $E_2$ exponential integral, so it is forward-mode clean. The same curve is evaluated on the VULCAN
grid and the ART grid. The chemistry side rebuilds the rates and the T- and composition-dependent structure from it; the
photolysis cross sections are T-independent. The exojax side builds its own hydrostatic transit geometry from the same
T(P) and the chemistry's mean molecular weight, interpolated with `interp_map`.

The profile is never clipped: a draw with a layer outside [220, 2980] K on either grid (the engine's 200-3000 K opacity
window inset by `T_EDGE_INSET_K`) is rejected.

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

Product CSVs follow the Carter & May (2024) convention. A spec that names a data source is fitted on its bins, or
refused if fewer than `MIN_REAL_BINS` lie in the window and the model band, in a synthetic run too; a synthetic run that
names no source runs on the constant-R synthetic grid (`SYNTH_R`, `SYNTH_SIGMA_PPM`), with the noise scaled by
`truth_noise_inflation`. The binning operator is an exact matrix, so a binned depth's JVP is
$B$ times the native depth's. It applies no instrument line-spread function, so a bin must span at least two model bands of
the R1000 tables; that admits the native 0.015 um NIRCam bins of Welbanks et al. (2024).

## Likelihood

Diagonal Gaussian with an optional noise inflation (Line et al. 2015 style). `make_nuisance_marginal` integrates the offsets
and the inflation(s) out at the same priors: the offset integral is closed form, the inflation a trapezoid in $\log_{10} b$.

## Sampler

A draw that does not certify within `count_max` is rejected, with log L = -inf. A column frozen at `dt_min` is different:
it is a solver failure rather than a verdict on the draw. A warm start that froze is re-solved cold, alone, and that
result stands; each chunk's `termination_reason` is then checked and a cold freeze stops the run with `DtMinFrozenError`. `nautilus_report` and the forward checks re-solve finished runs'
draws and stop the same way; `nautilus_report --spectra 0` still writes the certificate and the figures that need no solve.

Stock nautilus-sampler 1.0.6 (Lange 2023) with its default networks, `N_NETWORKS` ReLU MLPs, $n_{\rm update} = n_{\rm live}/2$,
exploration until `f_live = F_LIVE`, bounds enlarged by `ENLARGE_PER_DIM`, and the exploration phase discarded
(`DISCARD_EXPLORATION`, the unbiased estimator; keeping it biased log Z by about -0.2 on surrogates of the W39b posterior).
Each sampling-phase batch goes to one shell, as in stock nautilus. nautilus samples the unit cube; $u = {\rm logit}(z)$ is the
retrieval's own u-space, so theta follows the box prior exactly. `validate_env` refuses a fork (SiLU networks,
spread_batch) that carried the same version string.

`n_batch` defaults to `BATCH_PER_LANE` times the lanes, so a lane whose column certifies refills from the same batch.
Padding and out-of-window draws cost no solve: a chunk shorter than `n_batch` is filled with copies of its last row, and the
evaluator masks those and every T-P-invalid row. The RT runs only on the rows whose log L can be kept.

Lanes are `cfg.cold_lanes`, the number that runs under every solver; the ffi solver's one-buffer kernels keep three per
SM resident, so the GH200 specs state 528. `--dry-run` prints the lanes and the batch width.

### Warm starts

Every certified column becomes an anchor keyed by its unit-cube chemistry and T-P coordinates, and each later solve continues
from its nearest anchor; batches before the first certified column run cold. The certified state and which draws are rejected
then depend on the evaluation order at the convergence tolerance. Anchors are written inside the likelihood call, before
nautilus checkpoints the batch, so a job killed in between resumes by redrawing that batch, and the tally counts it twice.
The set is never pruned: `8 nz ni` bytes per certified column (74 KB at `nz` 100 and 93 species), on disk and in memory.

### Startup check

The first chunk of a fresh run (cold, box-prior draws) stops the run, before any anchor or checkpoint is written, if no draw
was kept or if a decade of log10 Kzz at the case's reference pressure holds `STARTUP_MIN_DRAWS` or more in-window draws and
none certified, so a prior region the solver cannot certify is refused in minutes.

### Output files

In `output/<case>/<name>_nautilus`: the run's identity, written at every start (`config.json`, `observations.npz`,
`target_manifest.json`, `target_digest.txt`, and the spec as `spec.yaml`, its base if any as `spec_base.yaml`); `run.log`;
`nautilus_checkpoint.hdf5`, rewritten every iteration; `anchors/`, one npz of certified
columns per batch; `rejected/`, one npz per chunk of the rejected draws (`z`, class `cls` 0 T-P out, 1 exhausted, 2 stalled,
3 non-finite, `acc`, `warm`); with integrated nuisances `nuisance/`, one npz per chunk of the certified rows (`z`, `log_l`,
`stats`); `batches.jsonl`, each chunk's wall time, start mode and accepted steps; `nautilus_tally.json`, the rejection and
warm-start counts summed over jobs. A finished run adds `nautilus_posterior.npz`, `nautilus_summary.json` and
`nautilus_certificate.json`. The run stops at `cfg.walltime_seconds` and exits 0 unfinished; resubmit with `--resume`.

## Report

`derived_quantities` gives weighted 16/50/84% quantiles of: metallicity as O/H and as total metals (C and S at their own
knobs, with the Lodders 2020 photospheric abundances), C/O (e^c_o times the case's C_H / O_H,
`provenance.baseline_c_to_o`), S/O, the Kzz factor and the absolute log10 Kzz at 1 mbar, the weight near the prior floor,
with the spec's `report_kzz_ceiling_x` the weight above it and the posterior truncated at it, T at the deep level and
$T_{\rm int}$, and the integrated offsets and inflations. `_conditioning_map` shows where the rejected draws sit. The
certificate's caveats name every network species with no opacity table whose mean gas VMR (condensed reservoirs left
out, as in the RT) over 1e-5 to 1e-2 bar exceeds 1e-4 in a re-solved draw (`--spectra N`), the rule of
jwst-transit-authority's `certify.unmodeled_absorbers`; when no re-solved draw certified, or no layer lies in that band,
they say the check was not performed and why.

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
| `dayside` | Dayside-average photochemistry geometry: `sl_angle` acos(0.5), `f_diurnal` 0.5 |
| `twolimb` | Two Guillot limbs at Tirr -/+ `TWO_LIMB_DTIRR_K`, depths averaged |
| `mass` | The planet mass at +-`prior_lnM_sigma` (the run's) in the RT, at frozen chemistry |
| `spot` | The transit-light-source factor of a spotted star: `--spot-t-phot`, `--spot-dt`, `--spot-f`, required |

The GPU validation set takes the case to measure: `elemental_audit.py`, `resolution_ladder.py`, `top_pressure_ladder.py`
and `opacity_leave_one_out.py` each need `--spec runs/<case>/<name>.yaml` and write to `output/<case>/validation/`. The
elemental audit draws the case's whole chemistry block, `s_o` and `kzz_slope` included when the case infers them. The
resolution ladder's rungs are the case's `art_nlayer` times 1, 1.5 and 2.25, rounded up (67, 101, 151 for 67).
`posterior_forward_check.py` writes its solves and their comparison into the run's own `validation/` folder.

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
| `b`, `o_g`, `K` | A noise inflation, group g's offset, the trapezoid node count in $\log_{10} b$ |
| `U`, `Y`, `Z`, `C_`, `L`, `Theta`, `AUX`, `CD` | Batched u, columns, unit-cube points, chemistry blocks, log L, theta, profiles, `ConvDiag` |
| `T`, `P`, `M` | Temperature, pressure, planet mass |

## References

- Carter, A. L., May, E. M., et al. 2024, Nature Astronomy, 8, 1008
- Guillot, T. 2010, A&A, 520, A27
- Kawahara, H., et al. 2022, ApJS, 258, 31
- Lange, J. U. 2023, MNRAS, 525, 3181, doi:10.1093/mnras/stad2441
- Line, M. R., et al. 2015, ApJ, 807, 183, doi:10.1088/0004-637X/807/2/183
- Lodders, K. 2020, Oxford Research Encyclopedia of Planetary Science, doi:10.1093/acrefore/9780190647926.013.145
- Tsai, S.-M., et al. 2021, ApJ, 923, 264
- Welbanks, L., et al. 2024, Nature, 630, 836
