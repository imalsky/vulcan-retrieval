"""NAS job 65200 post-mortem -- hunt the finite-likelihood/non-finite-gradient class.

Job 65200 died at SMC stage 0: 16/864 warm MALA proposals returned a finite
likelihood but a non-finite gradient, and the accept-count-only usable gate
(``ACC < warm_count_max``) had certified them. Hypothesis: the warm solve exits
via the stall fallback / loose branch on a marginally-stable column -- the primal
certifies while the jvp tangent (which relaxes through the same while_loop with no
stopping criterion of its own) has not settled, and can amplify to Inf/NaN.

This script reproduces the EXACT production map -- warm-capped continuation from a
converged column, MALA-sized theta move, forward-mode jvp -- at prior corners from
the W39b gpu preset (the init-phase-2 "oscillating/stall-fallback" class), and
classifies every solve against the candidate rejection predicates:

  P0  longdy < yconv_min at exit           (provably implied by ALL certified exits
                                            under central diffusion -- expected no-op)
  P1  canonical certification (conv_normal recomputed at exit; stall/budget
      exits read False)                     <- the gate now wired into pipeline
  P2  tight branch only (longdy < yconv_cri AND longdydt < slope_cri)

plus the tangent ground truth: finite? and (with --fd) does it agree with a
re-solved central difference of the SAME warm map?

Modes (--scheme): default = whatever the installed VULCAN-JAX resolves (hybrid
upwind since 2026-07-14; warm continuation then runs the central phase-1 operator
via vulcan_chem's carry seeding) | central = pin use_vm_mol/use_hybrid_vm_mol off
(job 65200 physics) | hybrid = force both on.

--equivalence instead cold-solves the baseline + corners under BOTH schemes and
reports the fixed-point agreement (the re-baseline evidence that following the
new hybrid defaults does not move the posterior's forward map).

Run (vulcan env, from vulcan-retrieval/):
    SMC_RETRIEVAL_PRESET=gpu python validation/diag_warm_stall_tangent.py [--scheme central] [--fd]
    SMC_RETRIEVAL_PRESET=gpu python validation/diag_warm_stall_tangent.py --equivalence
Cost: tangent hunt = 4 corners x (2 anchor + 2 jvp) = 16 solves (+8 with --fd);
equivalence = 2 builds x 5 cold solves.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# import order is load-bearing: vulcan_chem before anything exojax
from retrieval_framework.forward import vulcan_chem
import jax
import jax.numpy as jnp

from retrieval_framework import tp_profile                  # noqa: E402  (exojax)
from retrieval_framework.run_smc import make_config         # noqa: E402

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "w39b_smc_retrieval"
TRACERS = ["SO2", "CO2", "CO", "H2O", "CH4"]


def _mid(lo_hi):
    return 0.5 * (float(lo_hi[0]) + float(lo_hi[1]))


def _corners(cfg):
    """MALA-sized (theta0 -> theta1) pairs at the prior corners the init phase-2
    cull flags (hot Tirr / extreme lnKzz). theta layout = [lnZ, c_o, lnKzz,
    Tirr, log10kappa, log10gamma] (the chem+T-P block of the 10-D gpu preset)."""
    mid = np.array([_mid(cfg.prior_lnZ), _mid(cfg.prior_c_o), _mid(cfg.prior_lnKzz),
                    _mid(cfg.prior_Tirr), _mid(cfg.prior_log10kappa),
                    _mid(cfg.prior_log10gamma)])

    def toward(frac, **edges):
        th = mid.copy()
        names = ["lnZ", "c_o", "lnKzz", "Tirr", "log10kappa", "log10gamma"]
        pri = [cfg.prior_lnZ, cfg.prior_c_o, cfg.prior_lnKzz, cfg.prior_Tirr,
               cfg.prior_log10kappa, cfg.prior_log10gamma]
        for k, hi_side in edges.items():
            i = names.index(k)
            edge = float(pri[i][1] if hi_side else pri[i][0])
            th[i] = mid[i] + frac * (edge - mid[i])
        return th

    out = []
    for label, edges in [
        ("hot+highKzz", dict(Tirr=True, lnKzz=True)),
        ("hot+lowKzz", dict(Tirr=True, lnKzz=False)),
        ("hiZ+hot+highKzz", dict(lnZ=True, Tirr=True, lnKzz=True)),
        ("hiCO+hot", dict(c_o=True, Tirr=True)),
    ]:
        out.append((label, toward(0.90, **edges), toward(0.97, **edges)))
    return out


def _build(cfg, scheme: str):
    prof = cfg.profile()
    ov = dict(prof.get("cfg_overrides") or {})
    if scheme == "central":
        ov.update(use_vm_mol=False, use_hybrid_vm_mol=False)
    elif scheme == "hybrid":
        ov.update(use_vm_mol=True, use_hybrid_vm_mol=True)
    prof["cfg_overrides"] = ov
    tpm = tp_profile.build_tp_model(cfg)
    chem = vulcan_chem.build_chem_model(prof, tp_eval=tpm.eval,
                                        n_tp_params=tpm.n_params)
    return chem


def _pack_cd(cd):
    return jax.lax.stop_gradient(jnp.stack([
        jnp.asarray(cd.accept_count, jnp.float64),
        jnp.asarray(cd.longdy, jnp.float64),
        jnp.asarray(cd.longdydt, jnp.float64),
        jnp.asarray(cd.count_since_new_min, jnp.float64),
        jnp.asarray(cd.conv_normal, jnp.float64)]))


def tangent_hunt(cfg, scheme: str, do_fd: bool) -> int:
    chem = _build(cfg, scheme)
    wcmax = int(chem.warm_count_max)
    yconv_min, yconv_cri, slope_cri = (float(chem.yconv_min),
                                       float(chem.yconv_cri), float(chem.slope_cri))
    print(f"[stall] scheme={scheme} wcmax={wcmax} yconv_min={yconv_min} "
          f"yconv_cri={yconv_cri} slope_cri={slope_cri}", flush=True)

    rows = []
    for label, th0_np, th1_np in _corners(cfg):
        th0 = jnp.asarray(th0_np)
        th1 = jnp.asarray(th1_np)
        t0 = time.time()
        # production-faithful anchor: two-stage cold at theta0 (T-relax then warm)
        th_rel = th0.at[0].set(0.0).at[1].set(0.0)
        y_rel, d_rel = chem.converged_y(th_rel, return_conv_diag=True)
        y0, d0 = chem.converged_y(th0, warm_y=y_rel, lnZ_ref=0.0, c_o_ref=0.0,
                                  return_conv_diag=True)
        anchor_ok = bool(d0.conv_normal) and int(d0.accept_count) < chem.count_max
        print(f"[stall] {label}: anchor two-stage in {time.time()-t0:.0f}s "
              f"(acc={int(d_rel.accept_count)}/{int(d0.accept_count)}, "
              f"conv_normal={bool(d_rel.conv_normal)}/{bool(d0.conv_normal)})",
              flush=True)
        if not anchor_ok:
            print(f"[stall] {label}: ANCHOR itself not certified -- this corner is a "
                  "phase-1-reject in production; recording and skipping the warm probe",
                  flush=True)
            rows.append((label, "anchor-reject", None))
            continue

        lnZ0, co0 = float(th0[0]), float(th0[1])

        def f(th):
            y, cd = chem.converged_y(th, warm_y=y0, lnZ_ref=lnZ0, c_o_ref=co0,
                                     warm_cap=True, return_conv_diag=True)
            return y / jnp.sum(y, axis=1, keepdims=True), _pack_cd(cd)

        for k, name in [(2, "lnKzz"), (3, "Tirr")]:
            t1 = time.time()
            e = jnp.zeros(th1.shape[0], jnp.float64).at[k].set(1.0)
            (ymix, cdv), (dymix, _) = jax.jvp(f, (th1,), (e,))
            ymix = np.asarray(ymix)
            dymix = np.asarray(dymix)
            cdv = np.asarray(cdv)
            acc, longdy, longdydt, csnm, convn = (int(cdv[0]), float(cdv[1]),
                                                  float(cdv[2]), int(cdv[3]),
                                                  bool(cdv[4] > 0.5))
            primal_ok = bool(np.all(np.isfinite(ymix)))
            tan_ok = bool(np.all(np.isfinite(dymix)))
            p0 = longdy < yconv_min
            p1 = convn
            p2 = (longdy < yconv_cri) and (longdydt < slope_cri)
            verdict = ("BAD-TANGENT" if (primal_ok and not tan_ok and acc < wcmax)
                       else "capped" if acc >= wcmax
                       else "clean" if tan_ok else "primal-blown")
            fd_note = ""
            if do_fd and tan_ok and acc < wcmax and k == 2:
                h = 0.05
                yp = np.asarray(f(th1.at[k].add(+h))[0])
                ym = np.asarray(f(th1.at[k].add(-h))[0])
                fd = (yp - ym) / (2 * h)
                mag = np.abs(fd)
                hi = mag >= np.quantile(mag, 0.99)
                rel = (np.abs(dymix[hi] - fd[hi])
                       / np.maximum(np.abs(fd[hi]), 1e-300))
                fd_note = f" fd_med_rel={np.median(rel):.2e}"
            rows.append((label, f"d/d{name}", dict(
                acc=acc, longdy=longdy, longdydt=longdydt, csnm=csnm,
                conv_normal=convn, P0=p0, P1=p1, P2=p2,
                primal_finite=primal_ok, tangent_finite=tan_ok,
                verdict=verdict)))
            print(f"[stall] {label} d/d{name} ({time.time()-t1:.0f}s): acc={acc} "
                  f"longdy={longdy:.3g} longdydt={longdydt:.3g} csnm={csnm} "
                  f"conv_normal={convn} | primal={'ok' if primal_ok else 'BAD'} "
                  f"tangent={'ok' if tan_ok else 'NON-FINITE'} | "
                  f"P0={p0} P1={p1} P2={p2} -> {verdict}{fd_note}", flush=True)

    print("=" * 78, flush=True)
    print("[stall] CLASSIFICATION TABLE (does each predicate separate bad tangents?)",
          flush=True)
    n_bad = n_caught_p1 = n_caught_p0 = n_caught_p2 = n_clean_killed_p1 = 0
    for label, lane, r in rows:
        if r is None or not isinstance(r, dict):
            continue
        if r["verdict"] == "BAD-TANGENT":
            n_bad += 1
            n_caught_p0 += int(not r["P0"])
            n_caught_p1 += int(not r["P1"])
            n_caught_p2 += int(not r["P2"])
        if r["verdict"] == "clean" and not r["P1"]:
            n_clean_killed_p1 += 1
    print(f"[stall] bad tangents found: {n_bad}; caught by P0={n_caught_p0} "
          f"P1={n_caught_p1} P2={n_caught_p2}; clean solves P1 would reject: "
          f"{n_clean_killed_p1}", flush=True)
    if n_bad == 0:
        print("[stall] no non-finite tangent reproduced at these corners/moves -- "
              "the class is stochastic (16/864 in production); rely on the in-run "
              "forensics dump for exact offenders, and read the P1-vs-clean column "
              "above for gate safety.", flush=True)
    return 0


def equivalence(cfg) -> int:
    """Cold fixed points under hybrid vs central diffusion at baseline + corners."""
    labels = ["baseline"] + [c[0] for c in _corners(cfg)]
    thetas = [jnp.asarray(np.array([_mid(cfg.prior_lnZ), _mid(cfg.prior_c_o),
                                    _mid(cfg.prior_lnKzz), _mid(cfg.prior_Tirr),
                                    _mid(cfg.prior_log10kappa),
                                    _mid(cfg.prior_log10gamma)]))]
    thetas += [jnp.asarray(c[1]) for c in _corners(cfg)]

    results = {}
    for scheme in ("central", "hybrid"):
        chem = _build(cfg, scheme)
        sols = []
        for lab, th in zip(labels, thetas):
            t0 = time.time()
            y, cd = chem.converged_y(th, return_conv_diag=True)
            ymix = np.asarray(y / jnp.sum(y, axis=1, keepdims=True))
            sols.append((ymix, int(cd.accept_count), bool(cd.conv_normal)))
            print(f"[equiv] {scheme}: {lab} acc={int(cd.accept_count)} "
                  f"conv_normal={bool(cd.conv_normal)} ({time.time()-t0:.0f}s)",
                  flush=True)
        results[scheme] = (sols, chem.sidx)

    sidx = results["central"][1]
    print("=" * 78, flush=True)
    # RT-relevant mask: transit spectra are insensitive to VMR below ~1e-12; the
    # clip floor (exact zeros -> 1e-300 guard) makes unmasked dex ratios on dead
    # cells read as ~hundreds of dex of pure noise.
    VMR_FLOOR = 1e-12
    worst = 0.0
    for i, lab in enumerate(labels):
        a, acc_a, ok_a = results["central"][0][i]
        b, acc_b, ok_b = results["hybrid"][0][i]
        certified = ok_a and ok_b
        mask = (a > VMR_FLOOR) & (b > VMR_FLOOR)
        dex = np.abs(np.log10(np.where(mask, b, 1.0) / np.where(mask, a, 1.0)))
        mx = float(np.max(dex[mask])) if mask.any() else float("nan")
        if certified:
            # only CERTIFIED points count toward the verdict: an uncertified
            # point compares two non-converged transients (production rejects
            # that draw in phase 1 under either scheme)
            worst = max(worst, mx)

        def _tr(s):
            av, bv = a[:, sidx[s]], b[:, sidx[s]]
            m = (av > VMR_FLOOR) & (bv > VMR_FLOOR)
            if not m.any():
                return f"{s}=n/a"
            return f"{s}={np.max(np.abs(np.log10(bv[m] / av[m]))):.3f}"

        tr = "  ".join(_tr(s) for s in TRACERS if s in sidx)
        cert_note = "" if certified else "  [UNCERTIFIED in >=1 scheme -- excluded from verdict]"
        print(f"[equiv] {lab}: max |dlog10 ymix| = {mx:.3f} dex "
              f"(cells with VMR > {VMR_FLOOR:g} in both); tracers(dex): {tr}"
              f"{cert_note}", flush=True)
    print(f"[equiv] WORST across CERTIFIED points (RT-relevant cells): {worst:.3f} dex "
          f"({'within the central-scheme convergence floor (~0.16 dex)' if worst < 0.2 else 'ABOVE the expected floor -- investigate before trusting hybrid defaults'})",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["default", "central", "hybrid"],
                    default="default")
    ap.add_argument("--fd", action="store_true",
                    help="FD-check finite tangents on the lnKzz lane (2 extra solves/corner)")
    ap.add_argument("--equivalence", action="store_true",
                    help="hybrid-vs-central cold fixed-point comparison instead")
    ap.add_argument("--run-dir", default=str(RUN_DIR))
    args = ap.parse_args()
    cfg, preset = make_config(Path(args.run_dir))
    print(f"[stall] preset={preset} run_dir={args.run_dir}", flush=True)
    t0 = time.time()
    rc = equivalence(cfg) if args.equivalence else tangent_hunt(cfg, args.scheme, args.fd)
    print(f"[stall] DONE in {time.time()-t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
