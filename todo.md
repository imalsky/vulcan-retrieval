# todo

Findings behind these items are in `notes.md` (this repo and VULCAN3's / vulcan-forward's).

## Next run: baseline v4 at 100 layers (needs the user's launch OK)

- Settings: `runs/w39b_smc_retrieval/overrides/w39b_baseline_v4_nz100.json` (presolve
  chain, uniform 100 layers, 528 lanes, per-instrument noise, log10gamma down to -3) with
  case.py's Tint prior uniform 50-600 K and stock nautilus 1.0.6.
- Wall [E]: ~16 h at 100 layers with the old fork's call count; stock without spreading
  took +62% calls on the surrogate, so expect ~20-26 h. The governor stops at 46 h of a
  48 h PBS wall; RESUME=1 continues.
- From `runs/w39b_smc_retrieval`:
  `qsub -l walltime=48:00:00 -v SMC_RETRIEVAL_OVERRIDES_FILE=overrides/w39b_baseline_v4_nz100.json,SMC_RETRIEVAL_OUT_DIR=data/w39b_baseline_v4_nz100,NAUTILUS_N_EFF=5000,VULCAN3_CHEM_RHS=grouped,PROFILE_BATCH=3 run_nas_w39b.pbs`
- Optional: 660 lanes compiled at ~79 GiB but were never timed (528 is measured).

## After the run

- Tint: push prior-only draws through `tp_valid` and compare the posterior with that
  effective prior; plot Tint against log10kappa. On a Tint ~ kappa^-1/4 ridge, report
  T at 1 bar (or at the quench level) instead of Tint.

## T-P profile

- Decide between fix B (retrieve T at 1 bar instead of Tint, ~1/2-1 day) and C (pressure-
  dependent opacity + adiabat, e.g. Parmentier+2015; several days, plan first). See
  notes.md, "Priors and the T-P profile".

## Confirm (provenance unknown, live on main)

- VULCAN3: the element-conservation projection, `_repair_stage` and the extra
  certificate terms (geom_conv_tol, element_budget_tol) against the rule "keep VULCAN's
  original convergence criteria"; the FastChem abundance file swap.
- vulcan-forward: conv_normal's geometry / element-budget terms and per-theta re-anchoring.
- vulcan-retrieval: the Kzz bounds (ceiling x1000 chosen after seeing the data, floor
  x0.1) and the other items listed in notes.md, "Provenance audit".
- VULCAN3 audit batches B-D (328 commits) were never done.

## Deferred ideas (none approved)

- Abort doomed lanes early (dt < 1e4 s and reject fraction > 0.3 at step 801: 64/75
  caught, 1 false positive). A new rejection rule.
- RT overlapped with the retry pass; longest-first ordering; halving the host update.
- AD warm-start polish (-8 to -16% wall [E]).
- nautilus: bootstrap logZ error + per-shell diagnostics; parallel network fits
  (`pool=(None, Pool(4))`); n_live 700-800 / f_live 0.05-0.1 after a surrogate test.
- RT: unrolled searchsorted interp (bitwise, ~4% faster); CUDA fold.
- Non-uniform pressure grid support in VULCAN3 (a 284-layer non-uniform grid matched
  uniform 600 to <= 0.06 sigma at ~55 h).

## Outside NAS (user)

- Archive the GitHub fork imalsky/nautilus (Settings > Archive); its branches are kept
  as tags archive/*-2026-10-19.
- Confirm the 2026-10-17 TestPyPI releases (vulcan3 0.18.0, vulcan-forward 0.29.0,
  vulcan-retrieval 0.27.0, JTA 0.72.0) and their tags, made by another session.
- NAS: the passphrase-less SSH key; the global `qsub` allow in ~/.claude/settings.json.
- The NAS working rules are in /nobackupp17/imalsky/CLAUDE.md (no repo); copy them into
  the daw CLAUDE.md if wanted.
