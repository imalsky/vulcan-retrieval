"""Route B B0-6 full-chain harness — ONE binned-spectrum derivative row.

Scratch-level driver (NOT production wiring, plan Section 5): theta ->
chemistry with smooth rainout (VULCAN-JAX driven directly, same on-graph
build as b0_6_chem_probe.py) -> transit depth (the retrieval's
exojax_rt.build_rt_model) -> one binned bandpass row (flux-weightless plain
cell average over a wavelength window — the tool's full count-space
operator is production wiring, B1-11). Compares the unrolled jvp of the
binned row against independently re-run centered FD.

Fixture: the isothermal SNCHO S8 mini-column (LOCAL mechanics validation
of the theta -> spectrum chain; W107b Guillot + photo is the G1/G6 science
fixture and needs the B1-7 retrieval plumbing — this harness raises loudly
if asked for it). theta = (T_iso, ln x_pin[H2S]); the sulfur-sensitive
molecules on the RT side are H2S and SO2.

Run (heavy: exojax opacity build; Isaac schedules):
    ROUTE_B_SPECTRUM_PROBE=1 python b0_6_spectrum_probe.py

Requires the vulcan-retrieval opacity caches (data/opacity_cache/,
data/exojax_linelists/) and must own its process (import-locked network).
"""

import os

if os.environ.get("ROUTE_B_SPECTRUM_PROBE") != "1":
    raise SystemExit(
        "Refusing to run without ROUTE_B_SPECTRUM_PROBE=1 (B0-6 full-chain "
        "probe; builds exojax opacities, several minutes + linelist data)."
    )
if os.environ.get("ROUTE_B_FIXTURE", "isothermal") != "isothermal":
    raise SystemExit(
        "Only the isothermal mechanics fixture is wired; the W107b Guillot "
        "photo-on fixture needs the B1-7 retrieval plumbing (conden_mode + "
        "on-graph pin in vulcan_chem._prep). Refusing rather than running "
        "an unsupported configuration."
    )

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["VULCAN_JAX_NETWORK"] = "thermo/SNCHO_photo_network.txt"
os.environ["VULCAN_JAX_ATOM_LIST"] = "H,O,C,N,S"

import numpy as np  # noqa: E402

from vulcan_jax._paths import PACKAGE_ROOT  # noqa: E402

os.chdir(PACKAGE_ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

# Chemistry side: reuse the chem probe's fixture/build wholesale.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "b0_6_chem_probe",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "b0_6_chem_probe.py"),
)


def main() -> int:
    # The chem probe module refuses import without its own gate; grant it.
    os.environ["ROUTE_B_PROBE"] = "1"
    probe = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(probe)

    probe.configure()
    from vulcan_jax.state import RunState, legacy_view  # noqa: E402
    import vulcan_jax.legacy_io as op  # noqa: E402
    import vulcan_jax.op_jax as op_jax  # noqa: E402
    import vulcan_jax.outer_loop as outer_loop  # noqa: E402
    import vulcan_jax.vulcan_cfg as cfg  # noqa: E402
    from vulcan_jax.chem_funs import spec_list as SL  # noqa: E402

    rs = RunState.with_pre_loop_setup(cfg)
    integ = outer_loop.OuterLoop(op_jax.Ros2JAX(), op.Output(cfg=cfg), cfg=cfg)
    var, atm, _ = legacy_view(rs)
    integ._ensure_runner(var, atm)

    # RT side: the retrieval's transmission model over an H2S/SO2-sensitive
    # band; import order contract (config -> vulcan_chem -> exojax) does not
    # apply here because vulcan_chem is never imported (scratch harness).
    from retrieval_framework.forward import config as fcfg  # noqa: E402
    from retrieval_framework.forward import exojax_rt, interp_map  # noqa: E402

    profile = dict(
        molecules=["H2O", "CO", "CH4", "H2S", "SO2"],
        nu_min=2400.0, nu_max=2800.0, nu_pts=800,  # ~3.6-4.2 um, R ~ 5000
        art_nlayer=60,
        broadening=str(getattr(fcfg, "BROADENING", "air")),
        use_rayleigh=True,
    )
    rt = exojax_rt.build_rt_model(profile)
    p_art = np.asarray(rt.p_art_bar)
    p_vul = np.asarray(atm.pco, dtype=np.float64) / 1.0e6  # dyne -> bar
    to_art = interp_map.make_to_art(p_vul, p_art)
    mols = list(rt.molecules)
    mol_idx = {m: SL.index(m) for m in mols}
    i_h2 = SL.index("H2")

    # One binned row: plain cell average over a fixed wavelength window
    # (scratch operator; the tool's count-space operator is B1-11).
    wl = np.asarray(rt.wl_um)
    band = jnp.asarray((wl >= 3.90) & (wl <= 4.10), dtype=jnp.float64)
    band = band / jnp.sum(band)

    state0 = integ._pack_state_from_runstate(rs)
    build, _aux = probe.make_build_closure(integ, state0, var, atm)

    def binned_row(theta):
        state, atm_stat = build(theta)
        final = integ._runner(state, atm_stat)
        y = final.y
        n_tot = jnp.sum(y, axis=1)
        T_art = jnp.full((p_art.shape[0],), theta[0])
        vmr = {m: to_art(y[:, mol_idx[m]] / n_tot) for m in mols}
        vmr_h2 = to_art(y[:, i_h2] / n_tot)
        mmw = to_art(final.mu)
        depth = rt.transmission_depth(vmr, vmr_h2, T_art, mmw)
        return jnp.sum(band * depth)

    theta0 = jnp.asarray([probe.T_ISO0, float(np.log(probe.X_PIN0))])

    print("[full-chain] binned-row value at theta0 ...")
    row0 = float(binned_row(theta0))
    print(f"  row(theta0) = {row0:.6e}")

    for d, h in ((0, 1.0), (1, 0.02)):
        e = jnp.zeros(2).at[d].set(1.0)
        _, drow = jax.jvp(binned_row, (theta0,), (e,))
        ep = np.zeros(2)
        ep[d] = h
        rp = float(binned_row(theta0 + jnp.asarray(ep)))
        rm = float(binned_row(theta0 - jnp.asarray(ep)))
        fd = (rp - rm) / (2 * h)
        rel = abs(float(drow) - fd) / max(abs(fd), 1e-300)
        print(f"  d row/d theta[{d}]: jvp = {float(drow):+.6e}, "
              f"FD(h={h}) = {fd:+.6e}, rel = {rel:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
