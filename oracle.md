# oracle.md — what vulcan-retrieval must achieve

This file states the goal of vulcan-retrieval (this repo) and what counts as success or failure. It
is the yardstick for judging whether the repo, or a run made with it, is ready to be used. It holds
targets and decisions, not measurements: the evidence lives in notes.md, named here by section
heading. Only the maintainer changes this file. The forward model's own criteria are in
vulcan-forward's `oracle.md`, and the chemistry solver's in vulcan3's.

## Goal: a correct retrieval with photochemistry solved per draw

vulcan-retrieval infers an exoplanet atmosphere's metallicity, C/O, sulfur abundance, vertical
mixing and temperature structure from a transmission spectrum. For every draw of the sampler it
solves the full photochemical kinetics and computes the spectrum from that column; it uses no
chemistry grid, emulator or reduced network. The method is the same for every planet: a planet is a
case file, and nothing in the code belongs to one planet.

Correct means three things. Every likelihood value comes from a converged, physically valid model.
The model contains what the data need, or the run refuses to start. Every reported number carries
the assumptions it rests on.

### Success: the machinery

- **One certificate.** Every likelihood is computed from a column that passes the chemistry
  solver's own convergence test at the case's settings. A draw that does not certify within its
  step cap is rejected, never extended or accepted at the cap. A column frozen at the smallest step
  in a cold solve stops the run; a frozen warm start is re-solved cold, and the re-solves are counted.
- **Rejected, never clipped.** A temperature profile that leaves the opacity window is rejected.
  A non-finite spectrum or likelihood is rejected and counted. Nothing is clamped to make a draw
  usable.
- **The physics is the full model.** Chemistry in 64-bit on a published upstream network named in
  the case; no reduced or self-made network; the elemental map is the only abundance map.
- **Settings are visible.** A number that changes a result is a named setting the user can read and
  change in the configuration, with its unit and its reason. It is not a value buried in the code.
  A physical constant or a numerical guard that is fixed on purpose is a named constant with its
  reason beside it.
- **Runs are reproducible.** A run records its commits, network, data files and settings. A change
  to anything in its target needs a new case, never a resume.
- **The RT grid is converged.** The number of RT layers and the model top are changed only when the
  resolution and top-pressure ladders (`validation/resolution_ladder.py`,
  `validation/top_pressure_ladder.py`) pass their gates.

### Success: the science

- **The case names its planet.** Radius, gravity, stellar radius, reference pressure, molecule
  list, spectral band, network and stellar spectrum are set by the case. A value left unset is an
  error, never a default that belongs to another planet.
- **The model covers the data.** The spectral band of the model contains every data bin that is
  fitted; a bin outside it is refused or counted and reported, never dropped silently. The model
  is computed at a resolution the binning operator can use, and a data bin finer than that is
  refused.
- **The core absorbers are there.** H2O, CO2, CO, CH4 and NH3 are in the radiative transfer
  wherever the band overlaps their absorption, or the case names the one it leaves out; HCN, H2S
  and SO2 left out are reported (vulcan-forward `constants.CORE_MOLECULE_BANDS_UM`). A minor species may be left out, with a
  recorded bound when a run is reported. A species abundant in the observable layers with no
  opacity table is reported with its abundance.
- **The standard opacity terms are on.** H2-H2 and H2-He collision-induced absorption and H2 and He
  Rayleigh scattering are in every retrieval.
- **A cloud is never a default.** The case states its cloud model; nothing turns one on for it. A
  retrieval of real data usually needs one (Rustamkulov et al. 2023 find a grey cloud at high
  significance on one hot Saturn), and that is the case's stated choice.
- **Instrument offsets can be free.** Data from more than one instrument or detector can be fitted
  with an offset between groups, and the case states whether it is (Carter & May et al. 2024 find
  offsets of tens to hundreds of ppm between JWST modes on one planet).

### Success: the repository's tests

These decide whether the code is correct. None takes more than 100 minutes on a CPU, and the aim is
under an hour. None runs a production-size retrieval: the job path is tested on a small synthetic
case.

- **The gates are tested without chemistry.** The certificate, the temperature window on both
  grids, the exact likelihood and the refusal of an uncertified synthetic truth
  (`tests/test_pipeline_gates.py`); every entry point's rejection of an uncertified spectrum
  (`tests/test_cold_reject.py`).
- **One end-to-end forward solve.** A shipped case assembles its real data, binning and offset
  operators and returns one finite likelihood from one draw solved through chemistry and radiative
  transfer (`tests/test_set_observations.py`).
- **The forward map can find its own truth.** Synthetic data made at known parameters score highest
  at those parameters against perturbed ones, with no sampler in the loop.
- **Each refusal has a test** that fails without it.

### Success: a run that is reported

These apply to a finished run, not to the test suite. They need the run's own output and may need
its hardware.

- **Warm starts did not bias it.** Weighted posterior draws re-solved from a cold start, through
  the same chain and tolerances, agree with the warm results within the gate of
  `validation/e2e_check.py`. Evidence claims use cold starts and the spread over at least three
  seeds.
- **Its absorbers were screened.** Every species the case left out has a recorded bound, for this
  planet, below the leave-one-out gate (`validation/opacity_leave_one_out.py`).
- **Its priors are what it sampled.** The window rejection and the certificate reshape the stated
  priors. The accepted prior is reported beside the posterior, with the fraction of draws rejected
  and where in parameter space they fall.
- **Its numbers are the posterior's own.** A reported number is read from the weighted posterior
  points and the certificate's `derived` block, never from the equal-weight samples and never
  re-derived by hand.
- **Its result is a model, not a detection.** A reported abundance or significance names the model
  it was computed in: the molecule list, the cloud model, the network and the tolerances
  (Welbanks & Nixon et al. 2025). Bayes factors are quoted only between runs with the same
  chemistry settings and rejection rule.
- **Its limits are named.** Reported beside the posterior: what notes "Physics assumptions and
  limitations" lists (one limb and one illumination geometry, a closed column, a prescribed
  temperature profile, no condensation, mass and radius acting in the spectrum only), each with its
  measured size where one exists.

### Failure

- A likelihood or spectrum from an uncertified column, or a relaxed gate to reduce rejections.
- A clipped temperature profile, or a clamped opacity.
- A case that inherits another planet's gravity, reference pressure, molecule list or stellar
  spectrum.
- A fitted data bin outside the model band, or dropped without a count.
- An absorber left out without the case naming it, or reported without a bound for that planet.
- Condensation in a retrieval before it is validated on that case.
- A quoted number from a run that cannot be re-solved, or without its limits.
- Bayes factors between runs at mismatched solver settings, or a detection claim without its model
  set.
- A repository test that needs a production-size retrieval, or more than 100 minutes on a CPU.
- A cloud, a molecule list or a planet value that the case did not state.
- A number that changes a result, hidden in the code where the user cannot see or set it.
- A silent fallback or default where the code should refuse.

## Ready to use

The repo is ready when, at the commit to be used:

- the normal tier passes locally and in CI, and the full tier passes when the change touches what it
  covers;
- its pins name released, tested commits of vulcan3 and vulcan-forward;
- no `[owed]` item in todo.md backs a claim being made.

A run is ready to report when its certificate is written, everything under "a run that is reported"
holds, and nothing above under Failure applies.
