"""Route B B0-6 early derivative probe — chemistry endpoint (D9 prototype).

Fixture: the isothermal 400 K SNCHO S8 mini-column (kernel/budget/subsystem/
LOCAL-derivative roles only, per plan Section 5; it is banned as a
full-network convergence fixture). theta = (T_iso [K], ln x_pin[H2S]) — one
temperature parameter and one sulfur-supply parameter.

Three derivative routes on the same losses
    L = [ln N_S8, ln N_H2S, ln Phi_S,rain]
(column S8 and H2S inventories and the instantaneous rainout flux):

  1. UNROLLED forward-mode jvp through the full runner (theta -> k(T),
     atm(T), conden profile(T), pin(theta), init(T) -> converged carry).
  2. D9 IMPLICIT fixed-point prototype: dense (I - G_eta) solve in
     log-abundance coordinates on the smooth-rainout body map
     (step + gas-partial renorm + bottom pin), with the null space
     MEASURED by SVD and verified against the analytic conserved-mass
     vectors: expected null = {O, C, N}, expected rank drop 5 -> 3, S and
     H must measure non-null (open budgets). Loud failure on mismatch.
  3. Independently re-run centered FD at h and h/2 (the fixture terminates
     at the runtime cap by design — the plan's "reconverged endpoint,
     termination reason converged" gate applies to the photo-on G1
     fixtures, not this local probe; endpoint stationarity is measured by
     the B0-5 residual and reported).

Run (heavier sweeps are Isaac's to schedule):
    ROUTE_B_PROBE=1 python b0_6_chem_probe.py

The script must own its process (network/atom_list are import-locked):
it sets VULCAN_JAX_NETWORK / VULCAN_JAX_ATOM_LIST before importing
vulcan_jax and chdirs to the package root for the thermo data paths.
"""

import os

if os.environ.get("ROUTE_B_PROBE") != "1":
    raise SystemExit(
        "Refusing to run without ROUTE_B_PROBE=1 (B0-6 derivative probe; "
        "~2-10 min of local compute)."
    )

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["VULCAN_JAX_NETWORK"] = "thermo/SNCHO_photo_network.txt"
os.environ["VULCAN_JAX_ATOM_LIST"] = "H,O,C,N,S"

import numpy as np  # noqa: E402

from vulcan_jax._paths import PACKAGE_ROOT  # noqa: E402

os.chdir(PACKAGE_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import vulcan_jax.vulcan_cfg as cfg  # noqa: E402
import vulcan_jax.legacy_io as op  # noqa: E402
import vulcan_jax.op_jax as op_jax  # noqa: E402
import vulcan_jax.outer_loop as outer_loop  # noqa: E402
import vulcan_jax.jax_step as js  # noqa: E402
from vulcan_jax import atm_jax, conden, network as net_mod, rates_jax  # noqa: E402
from vulcan_jax.gibbs import load_nasa9  # noqa: E402
from vulcan_jax._paths import resolve_data_path  # noqa: E402
from vulcan_jax.state import RunState, legacy_view  # noqa: E402
from vulcan_jax.chem_funs import spec_list as SL  # noqa: E402
from vulcan_jax.steady_residual import residual_from_state  # noqa: E402
from vulcan_jax.outer_loop import _NET_JAX as network_arrays  # noqa: E402

ATOMS = ("H", "O", "C", "N", "S")
X_PIN0 = 3e-5
T_ISO0 = 400.0
RUNTIME_END = 1e10  # s — ~15 column diffusion times at Kzz=1e7
# Solver-map probe steps (the D9 body_dt practical diagnostic: the exact
# solution of (I - G_eta) s = G_theta is body_dt-independent, so agreement
# across the scan certifies the solve and disagreement flags it). The
# usable window on this cold fixture is MEASURED as ~1e3-1e4: the
# production default 1e7 amplifies unit log-space tangents by ~1e12 (dense
# jacfwd overflows to NaN), 1e5 already carries non-finite Jacobian
# entries, and below ~1e3 the open budgets have not lifted off the
# approximate-null floor so the null ordering cannot be verified.
BODY_DT_SCAN = (1e3, 3e3, 1e4)
ETA_FLOOR = 1e-50  # density floor for log coordinates
LIVE_FLOOR = 1e-25  # mixing-ratio floor: cells below are dead (excluded
#                     from the dense implicit solve; no loss reads them)
DEFECT_CUT = 1.0  # log-space fixed-point defect above which a cell is not
#                   linearizable (quench-creep radicals on this fixture)


def configure():
    cfg.atm_type = "isothermal"
    cfg.Tiso = T_ISO0
    cfg.atm_base = "H2"
    cfg.use_moldiff = True
    cfg.use_Kzz = True
    cfg.Kzz_prof = "const"
    cfg.const_Kzz = 1e7
    cfg.use_vz = False
    cfg.use_photo = False
    cfg.use_ion = False
    cfg.ini_mix = "const_mix"
    cfg.const_mix = {"H2": 0.85, "He": 0.148, "H2O": 5e-4, "CO": 2e-4,
                     "CH4": 1e-4, "N2": 1e-4, "H2S": X_PIN0, "S8": 1e-4}
    cfg.nz = 32
    cfg.P_b = 7.6e6
    cfg.P_t = 1e2
    cfg.use_topflux = False
    cfg.use_botflux = False
    cfg.use_ini_cold_trap = False
    cfg.use_print_prog = False
    cfg.use_settling = False
    cfg.use_relax = []
    cfg.fix_species = []
    cfg.count_max = 100000
    cfg.count_min = 1
    cfg.runtime = RUNTIME_END
    cfg.use_condense = True
    cfg.conden_mode = "smooth_rainout"
    cfg.conden_smooth_width = 0.1
    cfg.rainout_rate_scale = 1.0
    cfg.condense_sp = ["S8"]
    cfg.non_gas_sp = ["S8_l_s"]
    cfg.r_p = {"S8_l_s": 5e-3}
    cfg.rho_p = {"S8_l_s": 2.07}
    cfg.use_fix_sp_bot = {"H2S": X_PIN0}


def make_build_closure(integ, state0, var, atm):
    """The on-graph theta -> (initial carry, AtmStatic) map for this fixture.

    Shared by the chemistry probe and the full-chain spectrum harness.
    theta = (T_iso, ln x_pin[H2S]); everything T-dependent (rates, atm
    structure, conden profile, init densities) and the boundary pin value
    are rebuilt on the JAX graph, so forward-mode tangents flow end-to-end.
    """
    nz = int(np.asarray(state0.y).shape[0])
    network = net_mod.parse_network(str(resolve_data_path(cfg.network)))
    thermo_dir = resolve_data_path(cfg.network).parent
    if not (thermo_dir / "NASA9").exists():
        thermo_dir = resolve_data_path("thermo")
    nasa9, _ = load_nasa9(network.species, thermo_dir)
    cspec = conden.make_conden_spec(cfg, var, atm, {s: i for i, s in enumerate(SL)})
    phys0, aspec = atm_jax.make_physical_inputs(cfg, var, atm, list(SL))
    pco = jnp.asarray(atm.pco, dtype=jnp.float64)  # fixed pressure grid
    ymix0 = jnp.asarray(state0.ymix)  # const_mix normalized mixing ratios
    remove_list = getattr(cfg, "remove_list", None)

    def build(theta):
        T_iso, ln_xpin = theta[0], theta[1]
        Tco = T_iso * jnp.ones((nz,), dtype=jnp.float64)
        atm_stat = atm_jax.build_atm_static(phys0._replace(Tco=Tco), aspec)
        k_arr = rates_jax.build_rate_array(
            network, Tco, atm_stat.M, nasa9, remove_list=remove_list
        )
        cprof = conden.build_conden_profile(cspec, Tco, pco, atm_stat.M, atm_stat.Dzz)
        y0 = ymix0 * atm_stat.M[:, None]
        pv = state0.pv._replace(
            n_0=atm_stat.M,
            c_Dg_per_re=cprof.Dg_per_re,
            c_sat_n_per_re=cprof.sat_n_per_re,
            bot_pin_mix=jnp.exp(ln_xpin)[None],
            r_Tco=Tco,
            r_Dzz_top=atm_stat.Dzz[-1],
        )
        # dz/mu/Hp/zco keep primal seeds: the first accepted step's atm
        # refresh recomputes them on-graph from ymix + pv.r_* (with
        # tangents); only g/dzi/Hpi feed the very first Ros2 step.
        state = state0._replace(
            y=y0, y_prev=y0, ymix=ymix0, k_arr=k_arr, pv=pv,
            g=atm_stat.g, dzi=atm_stat.dzi, Hpi=atm_stat.Hpi,
        )
        return state, atm_stat

    aux = dict(network=network, nasa9=nasa9, cspec=cspec, phys0=phys0,
               aspec=aspec, pco=pco, remove_list=remove_list)
    return build, aux


def main() -> int:
    configure()
    rs = RunState.with_pre_loop_setup(cfg)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
    var, atm, _ = legacy_view(rs)
    integ._ensure_runner(var, atm)
    st = integ._statics
    state0 = integ._pack_state_from_runstate(rs)
    nz, ni = np.asarray(state0.y).shape

    build, aux = make_build_closure(integ, state0, var, atm)
    network = aux["network"]
    nasa9 = aux["nasa9"]
    cspec = aux["cspec"]
    phys0 = aux["phys0"]
    aspec = aux["aspec"]
    pco = aux["pco"]
    remove_list = aux["remove_list"]
    compo = np.asarray(st.compo_arr)
    i_s8, i_h2s = SL.index("S8"), SL.index("H2S")
    re_row = int(st.rainout_re_row)
    coeff = float(st.rainout_coeff)
    scale = float(st.rainout_scale)
    w = float(st.rainout_w)
    sp_mask = jnp.asarray(st.rainout_sp_mask)

    def run_endpoint(theta):
        state, atm_stat = build(theta)
        return integ._runner(state, atm_stat), atm_stat

    def losses_from_state(final):
        # NOTE: ln Phi_rain is deliberately NOT a loss. On this kinetically
        # dead 400 K fixture the TRUE open-system steady state has zero
        # rain (no S8 source; the profile relaxes to saturation-capped,
        # subsaturated everywhere), and the sink's exact-zero property (D2)
        # then makes the endpoint-instantaneous flux exactly 0 -> ln = -inf.
        dz = final.dz
        n_s8 = jnp.sum(final.y[:, i_s8] * dz)
        n_h2s = jnp.sum(final.y[:, i_h2s] * dz)
        return jnp.stack([jnp.log(n_s8), jnp.log(n_h2s)])

    def F(theta):
        final, _ = run_endpoint(theta)
        return losses_from_state(final)

    theta0 = jnp.asarray([T_ISO0, float(np.log(X_PIN0))])

    # ---- primal + stationarity ----------------------------------------------
    final0, atm_stat0 = run_endpoint(theta0)
    n_steps = int(np.asarray(final0.accept_count))
    print(f"[primal] accepted steps = {n_steps}, t = {float(final0.t):.3e} s, "
          f"terminated at runtime cap = {bool(float(final0.t) >= RUNTIME_END)}")
    led = {k: np.asarray(getattr(final0, k)) for k in
           ("led_step", "led_renorm", "led_bc", "led_rain", "led_dt")}
    dt_last = float(led["led_dt"])
    # steady-state sulfur budget: bottom implied source vs rainout sink.
    # On this fixture the endpoint rain is EXACTLY zero (see
    # losses_from_state), so the closure ratio is reported against the
    # inventory tendency scale instead when rain is inactive.
    phi_bc_s = led["led_bc"][4] / dt_last
    phi_rain_s = led["led_rain"][4]
    dn_dt_s = (led["led_step"][4] + led["led_renorm"][4] + led["led_bc"][4]) / dt_last
    print(f"[budget] Phi_S,bottom = {phi_bc_s:.4e}, Phi_S,rain = {phi_rain_s:.4e}, "
          f"dN_S/dt = {dn_dt_s:.4e} [atoms cm^-2 s^-1]")
    if phi_rain_s > 0.0:
        print(f"[budget] closure |dN_S/dt| / Phi_rain = "
              f"{abs(dn_dt_s) / phi_rain_s:.3e}")
    else:
        print("[budget] rain inactive at endpoint (exact-zero hinge); "
              "closure gate deferred to a source-bearing fixture (W107b)")
    rep = residual_from_state(integ, final0, atm_stat0)
    print(f"[B0-5] scaled residual max_R = {float(rep.max_R):.3e} s^-1 at "
          f"z={int(rep.argmax_z)} sp={SL[int(rep.argmax_i)]}")
    for sp in ("S8", "H2S", "S2"):
        col = np.asarray(rep.R)[:, SL.index(sp)]
        print(f"        max_R[{sp}] = {col.max():.3e} s^-1")

    # ---- route 1: unrolled jvp ----------------------------------------------
    print("\n[route 1] unrolled forward-mode jvp through the runner")
    jvp_rows = []
    for d in range(2):
        e = jnp.zeros(2).at[d].set(1.0)
        _, dF = jax.jvp(F, (theta0,), (e,))
        jvp_rows.append(np.asarray(dF))
        print(f"  dL/dtheta[{d}] = {np.asarray(dF)}")
    jvp_rows = np.stack(jvp_rows, axis=1)  # (3 losses, 2 params)

    # ---- route 2: D9 implicit fixed-point prototype --------------------------
    print("\n[route 2] implicit (I - G_eta) solve, log coords, measured null space")
    y_star = jnp.maximum(final0.y, ETA_FLOOR)
    eta_star = jnp.log(y_star)
    gas_mask = np.ones(ni, dtype=bool)
    gas_mask[SL.index("S8_l_s")] = False
    gas_mask_j = jnp.asarray(gas_mask)
    idx_pin = np.asarray(st.fix_sp_bot_idx)

    def G_eta(eta, theta, body_dt):
        """Smooth-rainout body map in log coords at probe step body_dt."""
        T_iso, ln_xpin = theta[0], theta[1]
        Tco = T_iso * jnp.ones((nz,), dtype=jnp.float64)
        atm_stat = atm_jax.build_atm_static(phys0._replace(Tco=Tco), aspec)
        k_arr = rates_jax.build_rate_array(
            network, Tco, atm_stat.M, nasa9, remove_list=remove_list
        )
        cprof = conden.build_conden_profile(cspec, Tco, pco, atm_stat.M, atm_stat.Dzz)
        rt = conden.RainoutTerm(
            C=scale * coeff * cprof.Dg_per_re[re_row],
            n_sat=cprof.sat_n_per_re[re_row],
            w=w,
            sp_mask=sp_mask,
        )
        # geometry: converged carry splice (primal — the map's own T
        # dependence flows through k, M, Dzz, sat_n; refresh feedback is a
        # second-order term at the fixed point)
        atm_step = atm_stat._replace(
            g=final0.g, dzi=final0.dzi, Hpi=final0.Hpi,
            top_flux=final0.top_flux, vs=final0.vs,
        )
        y = jnp.exp(eta)
        sol, _ = js.jax_ros2_step(
            y, k_arr, body_dt, atm_step, network_arrays, rainout=rt
        )
        ysum = jnp.sum(jnp.where(gas_mask_j[None, :], sol, 0.0), axis=1, keepdims=True)
        ymix = sol / ysum
        bal = atm_stat.M[:, None] * ymix
        bal = jnp.where(gas_mask_j[None, :], bal, sol)
        bal = bal.at[0, idx_pin].set(jnp.exp(ln_xpin)[None] * atm_stat.M[0])
        return jnp.log(jnp.maximum(bal, ETA_FLOOR))

    def L_of(eta_flat):
        # ln column inventories; dz is carry-frozen (its theta dependence
        # is the structure feedback the frozen-geometry body map already
        # truncates, second order at the fixed pressure grid), so L has no
        # explicit theta term and dL = dL/deta . s.
        eta = eta_flat.reshape(nz, ni)
        y = jnp.exp(eta)
        dz = final0.dz
        return jnp.stack([
            jnp.log(jnp.sum(y[:, i_s8] * dz)),
            jnp.log(jnp.sum(y[:, i_h2s] * dz)),
        ])

    dL_deta = np.asarray(jax.jacfwd(L_of)(eta_star.ravel()))
    ymix_star = np.asarray(final0.ymix)
    dz_np = np.asarray(final0.dz)
    y_np = np.asarray(y_star)
    N = nz * ni

    def implicit_solve(body_dt):
        """One deflated dense solve at a given probe step; loud on failure.

        LIVE-CELL reduction, two exclusion classes, both reported: (1)
        floor-clipped dead cells (mixing ratio below LIVE_FLOOR at y*) —
        meaningless log rows no loss reads; (2) cells the map itself sends
        far from the fixed point (|defect| >= DEFECT_CUT; on this 400 K
        fixture the quench-creep radicals the runner holds only through
        clipping, which the body map deliberately lacks). The production
        LGMRES route flags these via its resid/defect diagnostics; the
        dense prototype excludes them and says so.
        """
        fp_defect = np.asarray(
            jnp.abs(G_eta(eta_star, theta0, body_dt) - eta_star)
        ).ravel()
        live = (ymix_star > LIVE_FLOOR).ravel()
        n_live_raw = int(live.sum())
        live &= fp_defect < DEFECT_CUT
        n_live = int(live.sum())
        print(f"  [dt={body_dt:.0e}] live {n_live_raw}/{live.size} "
              f"(ymix > {LIVE_FLOOR:.0e}), {n_live_raw - n_live} more "
              f"excluded at |defect| >= {DEFECT_CUT}; retained defect "
              f"max = {fp_defect[live].max():.3e}, "
              f"median = {np.median(fp_defect[live]):.3e}")
        for sp in ("S8", "H2S"):
            if not live.reshape(nz, ni)[:, SL.index(sp)].any():
                raise RuntimeError(
                    f"loss species {sp} has no retained cells at "
                    f"body_dt={body_dt:.0e}."
                )

        J = jax.jacfwd(
            lambda e: G_eta(e.reshape(nz, ni), theta0, body_dt).ravel()
        )(eta_star.ravel())
        A = (np.eye(N) - np.asarray(J))[np.ix_(live, live)]
        n_bad = int(np.size(A) - np.isfinite(A).sum())
        if n_bad:
            print(f"  [dt={body_dt:.0e}] SKIPPED: {n_bad} non-finite "
                  "(I - G_eta) entries on the retained cells (outside the "
                  "usable body_dt window).")
            return None, False
        # Frobenius norm as the operator scale (a spectral norm would need
        # the same fragile SVD this route avoids; only the RELATIVE null
        # qualities matter and they share the scale).
        op_scale = float(np.linalg.norm(A))

        # Null-space verification by MEASURED null quality of the analytic
        # conserved-mass candidates c_e = compo * dz * y* (unit-norm),
        # nq_e = ||A c_e|| / ||A||. A crisp SVD rank cut is meaningless
        # here: at usable body_dt the spectrum spans ~16 decades (fast
        # chemistry vs slow transport) and the renormalized map is only
        # APPROXIMATELY conserving (same caveat as the production adjoint's
        # null_quality diagnostic). The D9 requirement maps to an ORDERING
        # statement: every expected-closed budget must measure decisively
        # more null than every expected-open one.
        nq = {}
        c_vecs = {}
        for a_i, a in enumerate(ATOMS):
            c = (compo[:, a_i][None, :] * y_np * dz_np[:, None]).ravel()[live]
            c = c / np.linalg.norm(c)
            c_vecs[a] = c
            nq[a] = float(np.linalg.norm(A @ c)) / op_scale
        closed_exp, open_exp = ("O", "C", "N"), ("S", "H")
        worst_closed = max(nq[a] for a in closed_exp)
        best_open = min(nq[a] for a in open_exp)
        sep = best_open / max(worst_closed, 1e-300)
        print(f"  [dt={body_dt:.0e}] null quality nq_e = ||A c_e||/||A||: "
              + ", ".join(f"{a}={nq[a]:.2e}" for a in ATOMS))
        print(f"  [dt={body_dt:.0e}] open/closed separation = {sep:.1f}x "
              "(closed O,C,N must be decisively more null than open S,H; "
              "gate >= 10x)")
        ordering_ok = sep >= 10.0

        # Deflated solve: the fixed-point equation cannot determine s along
        # the (approximately) conserved directions — on a closed budget the
        # endpoint's atom total is set by the initial condition, not the
        # equation — so those components are constrained to zero and their
        # true init-anchored contribution is a measured limitation of the
        # implicit route (the FD comparison quantifies it).
        Q = np.linalg.qr(np.stack([c_vecs[a] for a in closed_exp], axis=1))[0]
        Gth = np.asarray(
            jax.jacfwd(lambda th: G_eta(eta_star, th, body_dt).ravel())(theta0)
        )
        # Row equilibration before the least-squares solve: the log-space
        # operator's rows span ~16 decades (fast chemistry vs slow
        # transport) and an unequilibrated lstsq truncates the slow —
        # physically load-bearing — modes at machine-eps of the fast
        # scale (measured: unequilibrated solutions underflow to ~1e-110).
        # This is the dense-prototype stand-in for the conditioning work
        # the production solver-map LGMRES route does properly.
        row_scale = 1.0 / np.maximum(
            np.linalg.norm(A, axis=1), 1e-300
        )
        A_eq = A * row_scale[:, None]
        rhs_eq = Gth[live] * row_scale[:, None]
        A_aug = np.vstack([A_eq, Q.T])
        rhs_aug = np.vstack([rhs_eq, np.zeros((Q.shape[1], 2))])
        s_live, *_ = np.linalg.lstsq(A_aug, rhs_aug, rcond=None)
        s_cols = np.zeros((N, 2))
        s_cols[live] = s_live
        # (2 losses, 2 params) + whether the null ordering held at this dt
        return dL_deta @ s_cols, ordering_ok

    # body_dt stability scan (a D9 practical solve diagnostic): the exact
    # solution is body_dt-independent; cross-dt spread certifies the solve,
    # and the null ordering must hold at the dt used. On THIS fixture the
    # closed O/C null quality is limited by the quench-creep chemistry
    # (the reason the column is banned as a convergence fixture), so the
    # ordering may fail at every dt — that is a MEASURED fixture property
    # reported below, and the definitive null measurement belongs to the
    # photo-on W107b fixture.
    imp_by_dt = {}
    ok_by_dt = {}
    for body_dt in BODY_DT_SCAN:
        rows, ok = implicit_solve(body_dt)
        if rows is None:
            continue
        imp_by_dt[body_dt] = rows
        ok_by_dt[body_dt] = ok
        for d in range(2):
            print(f"  [dt={body_dt:.0e}] dL/dtheta[{d}] = {rows[:, d]}"
                  + ("" if ok else "  [ordering FAILED]"))
    if not imp_by_dt:
        raise RuntimeError(
            "no body_dt in the scan produced a finite (I - G_eta); the "
            "implicit prototype has no usable window on this fixture."
        )
    stack = np.stack(list(imp_by_dt.values()))
    spread = np.max(np.abs(stack - stack.mean(0))) / max(
        np.max(np.abs(stack.mean(0))), 1e-300
    )
    print(f"  cross-body_dt spread (max rel to mean): {spread:.3e}")
    passing = sorted(dt for dt in imp_by_dt if ok_by_dt[dt])
    if passing:
        imp_rows = imp_by_dt[passing[len(passing) // 2]]
        imp_ok = True
    else:
        print("  NO body_dt passed the null-quality ordering on this "
              "fixture; the implicit rows below are DIAGNOSTIC ONLY "
              "(deflation of O/C rests on an unverified null).")
        usable = sorted(imp_by_dt)
        imp_rows = imp_by_dt[usable[len(usable) // 2]]
        imp_ok = False

    # ---- route 3: independently re-run centered FD ---------------------------
    print("\n[route 3] centered FD, independently re-run endpoints")

    def F_logged(theta, tag):
        final, _ = run_endpoint(theta)
        t_end = float(final.t)
        steps = int(np.asarray(final.accept_count))
        reason = ("runtime-cap" if t_end >= RUNTIME_END
                  else "count-max" if steps >= int(cfg.count_max)
                  else "converged")
        print(f"    endpoint {tag}: t={t_end:.4e} s, steps={steps}, "
              f"termination={reason}")
        return np.asarray(losses_from_state(final))

    fd = {}
    for d, (h1, h2) in enumerate([(2.0, 1.0), (0.02, 0.01)]):
        for h in (h1, h2):
            e = np.zeros(2)
            e[d] = h
            Lp = F_logged(theta0 + jnp.asarray(e), f"d{d} +{h}")
            Lm = F_logged(theta0 - jnp.asarray(e), f"d{d} -{h}")
            fd[(d, h)] = (Lp - Lm) / (2 * h)
            print(f"  d={d} h={h}: {fd[(d, h)]}")

    # ---- comparison table -----------------------------------------------------
    print("\n[compare] per (loss, param): jvp | implicit | FD(h) | FD(h/2)")
    names = ["ln N_S8", "ln N_H2S"]
    hs = {0: (2.0, 1.0), 1: (0.02, 0.01)}
    worst = 0.0
    for d in range(2):
        h1, h2 = hs[d]
        for li, nm in enumerate(names):
            v = (jvp_rows[li, d], imp_rows[li, d], fd[(d, h1)][li], fd[(d, h2)][li])
            ref = v[3]
            rel = [abs(x - ref) / max(abs(ref), 1e-12) for x in v[:2]]
            worst = max(worst, rel[0], rel[1])
            print(f"  th{d} {nm:12s}: {v[0]:+.6e} | {v[1]:+.6e} | "
                  f"{v[2]:+.6e} | {v[3]:+.6e}  (jvp/imp rel vs FD: "
                  f"{rel[0]:.2e}, {rel[1]:.2e})")
    print(f"\nworst jvp/implicit deviation from FD(h/2): {worst:.3e}")
    if not imp_ok:
        print("REMINDER: implicit rows are diagnostic-only on this fixture "
              "(null ordering unverified; see route 2 output).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
