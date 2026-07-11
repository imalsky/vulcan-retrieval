#!/usr/bin/env python3
"""run_smc.py -- case-directory driver for a differentiable VULCAN-JAX -> ExoJax
transmission retrieval (adaptive-tempered SMC + forward-mode-jvp MALA).

All the science lives in the framework modules (config_schema / observations /
tp_profile / retrieval_forward / pipeline); the planet lives in the case directory's
``case.py`` (PRESETS dict of Config factories). This driver only: resolves the case +
preset into a Config, sets up logging + output dir, builds the pipeline, loads real
(or generates synthetic) observations, optionally calibrates timing, runs SMC, and
writes the .npz bundles plot_smc.py reads. The layout deliberately mirrors the SWAMPE
MY_SWAMP/retrieval driver.

Presets / overrides (env vars)
------------------------------
- ``SMC_RETRIEVAL_PRESET``        : key into the case's PRESETS dict (default: the
                                    case's DEFAULT_PRESET, else "smoke").
- ``SMC_RETRIEVAL_OVERRIDES``     : JSON object of Config field overrides,
                                    e.g. '{"nz": 50, "smc_num_particles": 32}'.
- ``SMC_RETRIEVAL_OVERRIDES_FILE``: JSON file of Config field overrides (relative
                                    paths resolve against the run dir).
- ``SMC_RETRIEVAL_OUT_DIR``       : output directory (default <run_dir>/data/<preset>).

CLI
---
    python -m retrieval_framework.run_smc <run_dir>              # full run for the chosen preset
    python -m retrieval_framework.run_smc <run_dir> --calibrate  # build + time 1 likelihood batch
                                                       # and 1 MALA sweep, project, exit

Examples (from the repo root)
----------------------------------
    SMC_RETRIEVAL_PRESET=smoke python -m retrieval_framework.run_smc runs/w39b_smc_retrieval
    SMC_RETRIEVAL_PRESET=gpu   python -m retrieval_framework.run_smc runs/w39b_smc_retrieval --calibrate
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from retrieval_framework import config_schema as C  # light import (no jax)


def load_case(run_dir: Path):
    """Import <run_dir>/case.py as a module. It must define PRESETS (dict preset-name
    -> zero-arg Config factory) and may define DEFAULT_PRESET."""
    run_dir = Path(run_dir).resolve()
    case_py = run_dir / "case.py"
    if not case_py.is_file():
        raise FileNotFoundError(
            f"{case_py} not found -- a retrieval run dir must contain case.py "
            "(see runs/w39b_smc_retrieval/case.py for the template)")
    spec = importlib.util.spec_from_file_location(f"retrieval_case_{run_dir.name}", case_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PRESETS"):
        raise AttributeError(f"{case_py} defines no PRESETS dict")
    return mod


def resolve_preset(case_mod) -> str:
    preset = os.environ.get("SMC_RETRIEVAL_PRESET", "").strip().lower()
    if not preset:
        preset = str(getattr(case_mod, "DEFAULT_PRESET", "smoke")).lower()
    if preset not in case_mod.PRESETS:
        raise ValueError(f"Unknown SMC_RETRIEVAL_PRESET={preset!r} "
                         f"(this case has: {sorted(case_mod.PRESETS)})")
    return preset


def make_config(run_dir: Path) -> Tuple[C.Config, str]:
    """Resolve (case, preset, env overrides) -> (Config, preset_name). Relative
    override-file paths and out_dir resolve against the RUN DIR, not the cwd."""
    run_dir = Path(run_dir).resolve()
    case_mod = load_case(run_dir)
    preset = resolve_preset(case_mod)
    cfg = case_mod.PRESETS[preset]()

    overrides: Dict[str, Any] = {}
    ov_file = os.environ.get("SMC_RETRIEVAL_OVERRIDES_FILE", "").strip()
    if ov_file:
        ov_path = Path(ov_file)
        if not ov_path.is_absolute():
            ov_path = run_dir / ov_path
        overrides.update(json.loads(ov_path.read_text()))
    ov = os.environ.get("SMC_RETRIEVAL_OVERRIDES", "").strip()
    if ov:
        overrides.update(json.loads(ov))
    overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}

    out_env = os.environ.get("SMC_RETRIEVAL_OUT_DIR", "").strip()
    if out_env:
        overrides["out_dir"] = out_env
    if "out_dir" in overrides:
        od = Path(overrides["out_dir"])
        overrides["out_dir"] = od if od.is_absolute() else (run_dir / od).resolve()
    elif cfg.out_dir is None:
        cfg = replace(cfg, out_dir=run_dir / "data" / preset)

    # tuples in the dataclass may arrive as JSON lists
    for k, v in list(overrides.items()):
        if isinstance(v, list):
            overrides[k] = tuple(v)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg, preset


def write_config_json(cfg: C.Config, pipe, preset: str) -> None:
    d = asdict(cfg)
    d.update(dict(
        preset=preset,
        inferred_param_names=pipe.names,
        inferred_param_labels=pipe.labels,
        inferred_param_kinds=pipe.kinds,
        inferred_param_prior_types=pipe.prior_types,
        inferred_param_prior_lo=pipe.param_prior_lo.tolist(),
        inferred_param_prior_hi=pipe.param_prior_hi.tolist(),
        inferred_param_truth=output_truth(cfg, pipe).tolist(),
        instrument_groups=pipe.groups,
        n_obs_bins=int(pipe.n_bin),
        real_bins=bool(pipe.real_bins),
        n_chem_tp=int(pipe.n_chem_tp),
        gradient_mode=pipe.gradient_mode,
    ))
    (cfg.out_dir / "config.json").write_text(json.dumps(d, indent=2, default=str))


def output_truth(cfg: C.Config, pipe) -> np.ndarray:
    """Truth vector for outputs: real values for synthetic injections, NaN for real
    data (where the cfg truth_* fields are just initialization placeholders)."""
    if cfg.generate_synthetic_data:
        return np.asarray(pipe.param_truth, dtype=np.float64)
    return np.full(pipe.n_dim, np.nan, dtype=np.float64)


def calibrate(cfg: C.Config, pipe, P, jax) -> Dict[str, float]:
    """Time the cold state initialization (one batched two-stage chemistry solve per
    particle -- paid once per run) and one full mutation call (compile and warm
    steady-state separately), then project the SMC cost. Writes timing.json.

    The mutation is benchmarked at the ladder's OWN stage-0 conditions -- ESS-bisected
    first beta from the carried likelihoods, stage-0 systematic resample, cloud-width
    preconditioner, clamped step -- exactly what run_smc_loop's first stage will run,
    so the timing is representative and the health check exercises the real kernel.

    Under the staged architecture a stage costs ~one mutation call: the tempering
    reweight uses the CARRIED likelihood, so no extra likelihood batch is paid."""
    import jax.numpy as jnp
    log = logging.getLogger("retrieval")
    N = int(cfg.smc_num_particles)
    # Derive U exactly as run_smc_loop does, from the run's own seed, so the timing
    # gate exercises the same prior corners the production init will hit (a PRNGKey(0)
    # pilot cloud let job 64073's >16 h worst-corner init slip past calibration).
    key = jax.random.PRNGKey(int(cfg.seed))
    key, sub = jax.random.split(key)
    # oversampled cold-init draw (rejected corners culled back to N healthy in _init_state)
    U = pipe.sample_prior_u(sub, P._init_draw_count(pipe, N))

    t0 = time.perf_counter()
    U, L, G, Y, refs, DY, _init_stats = P._init_state(pipe, U, target_n=N)
    jax.block_until_ready(L)
    t_init = time.perf_counter() - t0
    log.info(f"state init (cold likelihood + move-map gradient): {t_init:.1f}s "
             f"| L range [{float(jnp.min(L)):.1f}, {float(jnp.max(L)):.1f}]")

    mutate = P._make_mutation(pipe, int(cfg.smc_num_mcmc_steps))
    # Stage-0 conditions, not an arbitrary proposal: the drift term of a MALA move is
    # step*scale^2*beta*G, and a prior-like cloud carries |L| (hence |G|) up to ~1e6 --
    # the old hard-coded (beta=0.5, step=mala_step_size, scale=1) benchmark launched
    # proposals so far off the converged map that their tangents went non-finite, and
    # _check_mutation_health aborted the calibration on an "AD pathology" the ladder's
    # tiny adaptive first beta can never produce (NAS job 64961: 8 bad grads/sweep at
    # accept=0.00). Reproduce run_smc_loop's stage 0 instead.
    L_np = np.asarray(jax.device_get(L), np.float64)
    dbeta = P._next_dbeta(L_np, 0.0, float(cfg.smc_target_ess_frac) * N)
    beta = jnp.asarray(dbeta, pipe.dtype)
    w = np.exp(dbeta * (L_np - L_np.max()))
    key, sub = jax.random.split(key)
    idx = P._systematic_resample_idx(sub, jnp.asarray(w / w.sum(), pipe.dtype), N)
    U, Y, refs, L, G = U[idx], Y[idx], refs[idx], L[idx], G[idx]
    if DY is not None:
        DY = DY[idx]
    scale_np = (P._abs_scale_diag(np.asarray(jax.device_get(U)), cap=float(cfg.mcmc_scale_clip))
                if cfg.mcmc_stage_adapt else np.ones(pipe.n_dim))
    scale = jnp.asarray(scale_np, pipe.dtype)
    step_f = min(max(float(cfg.mala_step_size), cfg.mcmc_step_size_min), cfg.mcmc_step_size_max)
    step = jnp.asarray(step_f, pipe.dtype)
    log.info(f"calibration mutation at stage-0 conditions: beta={dbeta:.3e} "
             f"step={step_f:.3g} scale=[{float(scale_np.min()):.3g}, {float(scale_np.max()):.3g}]")
    t0 = time.perf_counter()
    out = mutate(key, U, Y, refs, L, G, DY, beta, step, scale)
    jax.block_until_ready(out[0]); t_mut_compile = time.perf_counter() - t0
    P._check_mutation_health(out[7], "calibration mutation (compile pass)")
    U2, Y2, refs2, L2, G2, DY2 = out[:6]
    t0 = time.perf_counter()
    out = mutate(key, U2, Y2, refs2, L2, G2, DY2, beta, step, scale)
    jax.block_until_ready(out[0]); t_mut = time.perf_counter() - t0
    P._check_mutation_health(out[7], "calibration mutation (steady-state pass)")

    per_stage = t_mut
    proj = {
        "n_particles": N, "n_mcmc_steps": int(cfg.smc_num_mcmc_steps), "n_dim": int(pipe.n_dim),
        "n_chem_tp": int(pipe.n_chem_tp), "gradient_mode": pipe.gradient_mode,
        "smc_chem_mode": pipe.chem_mode,
        "smc_rt_chunk": int(cfg.smc_rt_chunk), "smc_rt_vjp_chunk": int(cfg.smc_rt_vjp_chunk),
        "calibration_beta_stage0": float(dbeta), "calibration_step": step_f,
        "calibration_scale_min": float(scale_np.min()), "calibration_scale_max": float(scale_np.max()),
        "t_state_init_s": t_init,
        "t_mutation_compile_s": t_mut_compile, "t_mutation_sweep_s": t_mut,
        "mutation_accept_frac": float(jax.device_get(out[6])),
        "t_per_stage_s": per_stage,
        "projected_hours_15_stages": (t_init + t_mut_compile + 15 * per_stage) / 3600.0,
        "projected_hours_25_stages": (t_init + t_mut_compile + 25 * per_stage) / 3600.0,
        "projected_hours_40_stages": (t_init + t_mut_compile + 40 * per_stage) / 3600.0,
        "walltime_budget_hours": float(cfg.walltime_seconds) / 3600.0,
    }
    (cfg.out_dir / "timing.json").write_text(json.dumps(proj, indent=2))
    log.info("=== CALIBRATION ===")
    for k, v in proj.items():
        log.info(f"  {k:32s} {v}")
    budget = float(cfg.walltime_seconds)
    if budget > 0 and (15 * per_stage) > budget:
        log.warning("even 15 tempering stages exceed the walltime budget -- reduce "
                    "smc_num_mcmc_steps / smc_num_particles / nz, or raise yconv_cri")
    else:
        log.info("projection fits the walltime budget (the in-run governor still guards it).")
    return proj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=".",
                    help="retrieval case directory containing case.py (default: cwd)")
    ap.add_argument("--calibrate", action="store_true",
                    help="build + time one likelihood batch and one mutation sweep, then exit")
    args = ap.parse_args()

    cfg, preset = make_config(Path(args.run_dir))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(cfg.out_dir / "run.log", mode="w" if cfg.overwrite else "a")],
        force=True,
    )
    log = logging.getLogger("retrieval")
    log.info(f"run_dir={Path(args.run_dir).resolve()} preset={preset} out_dir={cfg.out_dir}")
    # loud, up-front dump of the RESOLVED config so nothing (nu_pts, count_max, priors,
    # ...) is a surprise; shown BEFORE the ~minutes-long forward build.
    log.info(C.describe_config(cfg, preset))

    # pipeline import triggers VULCAN-JAX env setup + jax x64 (heavy)
    from retrieval_framework import pipeline as P
    import jax
    log.info(f"jax backend={jax.default_backend()} devices={jax.devices()} "
             f"(x64 flips on during the VULCAN-JAX import inside build_pipeline)")

    t0 = time.perf_counter()
    pipe = P.build_pipeline(cfg)
    log.info(f"Built pipeline in {time.perf_counter()-t0:.1f}s | n_dim={pipe.n_dim}: {pipe.names} "
             f"| {pipe.n_bin} bins | groups={pipe.groups} | gradient_mode={pipe.gradient_mode} "
             f"| dtype={pipe.dtype.__name__} two_stage_z={cfg.two_stage_z}")
    write_config_json(cfg, pipe, preset)

    # ---- observations (set exactly once, BEFORE any jitted likelihood call) ----
    obs_path = cfg.out_dir / "observations.npz"
    if cfg.generate_synthetic_data:
        if obs_path.exists() and not cfg.overwrite:
            d = np.load(obs_path)
            pipe.set_observations(d["depth"], d["sigma"])
            pipe.flux_true = d["flux_true"] if "flux_true" in d.files else None
            log.info(f"Loaded existing synthetic observations: {obs_path}")
        else:
            log.info("Generating synthetic observations (injection at truth_*)...")
            o = P.generate_observations(pipe, seed=int(cfg.seed))
            P.save_npz(obs_path, wl=pipe.obs["wl"], wl_lo=pipe.obs["wl_lo"], wl_hi=pipe.obs["wl_hi"],
                       depth=o["depth"], sigma=o["sigma"], flux_true=o["flux_true"],
                       group=np.asarray(pipe.obs["group"], dtype="<U16"),
                       synthetic=np.asarray(1, np.int32),
                       inferred_param_names=np.asarray(pipe.names, dtype="<U64"),
                       inferred_param_truth=np.asarray(pipe.param_truth))
            log.info(f"Saved synthetic observations: {obs_path}")
    else:
        o = P.load_real_into_pipe(pipe)
        P.save_npz(obs_path, wl=pipe.obs["wl"], wl_lo=pipe.obs["wl_lo"], wl_hi=pipe.obs["wl_hi"],
                   depth=o["depth"], sigma=o["sigma"],
                   group=np.asarray(pipe.obs["group"], dtype="<U16"),
                   synthetic=np.asarray(0, np.int32))
        log.info(f"Using REAL observed spectrum: {pipe.n_bin} bins "
                 f"({', '.join(pipe.groups)}); saved copy to {obs_path}")

    dspan = float(np.nanmax(pipe.obs_depth) - np.nanmin(pipe.obs_depth))
    smean = float(np.mean(pipe.obs_sigma))
    log.info(f"Depth span {dspan*1e6:.0f} ppm | mean sigma {smean*1e6:.0f} ppm | span/sigma {dspan/smean:.1f}")

    if args.calibrate:
        calibrate(cfg, pipe, P, jax)
        return

    # ---- inference ----
    samples_path = cfg.out_dir / "posterior_samples.npz"
    extra_path = cfg.out_dir / "smc_extra_fields.npz"
    if cfg.run_inference:
        log.info(f"Running adaptive-tempered SMC (N={cfg.smc_num_particles}, "
                 f"mcmc_steps={cfg.smc_num_mcmc_steps}, kernel=preconditioned fwd-jvp MALA)...")
        ckpt = cfg.out_dir / "smc_checkpoint.npz"
        resume = os.environ.get("SMC_RESUME", "").strip().lower() in ("1", "true", "yes")
        if resume:
            # SMC_RESUME=1 means "continue a killed run". If the checkpoint is missing,
            # fail loud rather than silently restarting a multi-hour job from scratch.
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"SMC_RESUME=1 but no checkpoint at {ckpt}. Refusing to silently "
                    "start a fresh run; unset SMC_RESUME to start over, or point out_dir "
                    "at the killed run's directory.")
            log.info(f"SMC_RESUME=1: continuing the ladder from {ckpt}")
        t0 = time.perf_counter()
        res = P.run_smc_loop(pipe, key=jax.random.PRNGKey(int(cfg.seed)), progress=True,
                             checkpoint_path=ckpt,
                             walltime_seconds=float(cfg.walltime_seconds),
                             resume_from=ckpt if resume else None)
        log.info(f"SMC finished in {(time.perf_counter()-t0)/60:.1f} min | "
                 f"reached_beta1={res['reached_beta1']} | stages={len(res['betas'])-1} | logZ={res['logZ']:.2f}")
        if not res["reached_beta1"]:
            log.warning(f"SMC stopped before beta=1 (final beta={res['final_beta']:.4f}) -- "
                        "posterior is TEMPERED; treat widths as lower bounds and rerun/extend.")

        P.save_npz(samples_path,
                   param_names=np.asarray(pipe.names, dtype="<U64"),
                   param_labels=np.asarray(pipe.labels, dtype="<U64"),
                   samples=res["theta_draws"],
                   u_particles=res["U"], final_beta=np.asarray(res["final_beta"]),
                   # travels WITH the samples so no consumer can miss it: beta<1
                   # draws are tempered, not the posterior
                   reached_beta1=np.asarray(int(res["reached_beta1"]), np.int32))
        P.save_npz(extra_path,
                   inference_method=np.asarray(2, np.int32),
                   smc_kernel=np.asarray("mala+precond", dtype="<U16"),
                   smc_num_particles=np.asarray(int(cfg.smc_num_particles), np.int32),
                   smc_num_mcmc_steps=np.asarray(int(cfg.smc_num_mcmc_steps), np.int32),
                   smc_betas=res["betas"], smc_ess=res["ess"],
                   smc_acceptance_rate=res["acceptance_rate"],
                   smc_logZ_increment=res["logZ_increment"], smc_logZ=np.asarray(res["logZ"]),
                   smc_step_size_history=res["step_size_history"],
                   smc_unique_particles=res["unique_particles"],
                   smc_scale_diag_final=res["scale_diag_final"],
                   smc_warm_capped=res["warm_capped"],
                   # evidence conditioning: smc_logZ is under the OPERATIONAL prior
                   # (T-P window x converged support, renormalized); the measured
                   # support fraction + the box-prior value ride along -- quote them
                   # together (see run_smc_loop docstring)
                   smc_log_support_fraction=np.asarray(res["log_support_fraction"]),
                   smc_log_support_fraction_err=np.asarray(res["log_support_fraction_err"]),
                   smc_logZ_box=np.asarray(res["logZ_box"]),
                   init_stats_keys=np.asarray(list(res["init_stats"].keys()), dtype="<U32"),
                   init_stats_vals=np.asarray(list(res["init_stats"].values()), np.int64),
                   reached_beta1=np.asarray(int(res["reached_beta1"]), np.int32),
                   inferred_param_names=np.asarray(pipe.names, dtype="<U64"),
                   inferred_param_truth=output_truth(cfg, pipe))
        log.info(f"Saved posterior + diagnostics: {samples_path}, {extra_path}")

        theta = res["theta_draws"].reshape(-1, pipe.n_dim)
        truth = output_truth(cfg, pipe)
        tag = "" if res["reached_beta1"] else \
            f" [TEMPERED beta={res['final_beta']:.3f} -- NOT the posterior]"
        log.info(("Posterior (median [5%,95%]):" if not cfg.generate_synthetic_data
                 else "Recovery (median [5%,95%], truth):") + tag)
        for i, name in enumerate(pipe.names):
            q = np.percentile(theta[:, i], [5, 50, 95])
            msg = f"  {name:16s} {q[1]:9.3f} [{q[0]:9.3f},{q[2]:9.3f}]"
            if np.isfinite(truth[i]):
                msg += f"  truth={truth[i]:9.3f}  ({'in' if q[0] <= truth[i] <= q[2] else 'OUT'} 90% CI)"
            log.info(msg)
    else:
        if not samples_path.exists():
            log.info("run_inference=False and no existing samples; stopping after build/obs.")
            return
        log.info("run_inference=False; using existing posterior_samples.npz.")

    # ---- posterior predictive (binned, on the observed grid) ----
    if cfg.do_ppc:
        log.info("Posterior predictive...")
        import jax.numpy as jnp
        s = np.load(samples_path)
        if "reached_beta1" in s.files and not bool(int(s["reached_beta1"])):
            log.warning(f"PPC input samples are TEMPERED (final beta="
                        f"{float(s['final_beta']):.3f} < 1): the predictive band is "
                        "under the tempered measure, not the posterior predictive.")
        theta_all = np.asarray(s["samples"]).reshape(-1, pipe.n_dim)
        rng = np.random.default_rng(cfg.seed + 1)
        n_take = min(int(cfg.ppc_draws), theta_all.shape[0])
        sel = theta_all[rng.choice(theta_all.shape[0], size=n_take, replace=False)]
        preds = []
        for i0 in range(0, n_take, int(cfg.ppc_chunk_size)):
            batch = jnp.asarray(sel[i0:i0 + int(cfg.ppc_chunk_size)], pipe.dtype)
            preds.append(np.asarray(jax.vmap(pipe.observed_depth_model_jit)(batch)))
            log.info(f"  ppc {min(i0+int(cfg.ppc_chunk_size), n_take)}/{n_take}")
        ppc = np.concatenate(preds, axis=0)
        theta_med = np.median(theta_all, axis=0)
        mu_med = np.asarray(pipe.observed_depth_model_jit(jnp.asarray(theta_med, pipe.dtype)))
        P.save_npz(cfg.out_dir / "posterior_predictive.npz",
                   ppc_draws=ppc, theta_sel=sel, wl=pipe.obs["wl"],
                   p05=np.nanquantile(ppc, 0.05, axis=0), p50=np.nanquantile(ppc, 0.50, axis=0),
                   p95=np.nanquantile(ppc, 0.95, axis=0),
                   theta_median=theta_med, mu_at_median=mu_med,
                   obs_depth=pipe.obs_depth, obs_sigma=pipe.obs_sigma)
        chi2 = float(np.sum(((pipe.obs_depth - mu_med) / pipe.obs_sigma) ** 2) / pipe.n_bin)
        log.info(f"Saved posterior predictive | reduced chi2 at posterior median: {chi2:.2f}")

    log.info("DONE.")


if __name__ == "__main__":
    main()
