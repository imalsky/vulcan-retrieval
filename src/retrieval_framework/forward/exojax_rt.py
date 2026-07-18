"""ExoJax side of the demo: a differentiable ``ArtTransPure`` transmission model.

``build_rt_model(profile)`` builds (once) the wavenumber grid, one premodit opacity
per molecule, the H2-H2 collision-induced-absorption table, and an ``ArtTransPure``
radiative-transfer object. It returns a model whose ``transmission_depth(vmr, vmr_h2,
T_art, mmw_art)`` maps per-layer VMR + temperature + mean-molecular-weight profiles
(already interpolated onto the ART grid) to the transit depth ``(R_p(lambda)/R_star)^2``.

Everything inside ``transmission_depth`` is pure JAX, so forward-mode tangents from the
chemistry pass straight through to the spectrum. Opacities are built in float64 (x64 is
globally enabled), matching the chemistry side -- no dtype break.

Adapted from the validated emission pipeline in
``emulator-demo/emulator_tests/smc.py`` (lines ~590-716), swapping ``ArtEmisPure`` for
``ArtTransPure`` and the single CO opacity for a configurable molecule set.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # OpaPremodit refuses 32-bit; safe if already set
import jax.numpy as jnp

from retrieval_framework.forward import config

from exojax.utils.grids import wavenumber_grid
from exojax.database.api import MdbExomol, MdbHitran
from exojax.opacity import OpaPremodit, OpaCIA
from exojax.rt import ArtTransPure, ArtEmisPure
from exojax.atm.atmconvert import vmr_to_mmr
from exojax.atm.simple_clouds import powerlaw_clouds
from exojax.opacity.rayleigh import xsvector_rayleigh_gas
from exojax.atm.polarizability import polarizability as _POLARIZABILITY
from exojax.database.contdb import CdbCIA

_H2_MOLMASS, _HE_MOLMASS = 2.016, 4.0026   # g/mol, for the Rayleigh mmr conversion

# bar -> dyn/cm^2, for per-MASS (cm^2/g of atmosphere) opacities: dtau = kappa*dP_cgs/g.
# (exojax's layer_optical_depth folds bar_cgs/m_u into its opacity_factor because its
# xs are per-molecule cross sections; a per-gram kappa must NOT pick up the 1/m_u.)
_BAR_CGS = 1.0e6


def _blend_h2he_broadening(mdb, key: str) -> None:
    """Overwrite ``mdb.gamma_air``/``n_air`` with an H2/He-weighted Lorentz width.

    HITRAN's default gamma_air/n_air describe TERRESTRIAL air, which is the wrong
    perturber for an H2/He-dominated envelope. With ``nonair_broadening=True`` exojax
    exposes the HITRAN planetary-broadener columns (gamma_h2/n_h2, gamma_he/n_he)
    where the database provides them; this blends them with the fixed number-fraction
    mix ``config.H2HE_BROADENING_MIX`` and writes the result into the gamma_air/n_air
    slots that OpaPremodit (and MDBSnapshot) consume, so the rest of the opacity
    pipeline is untouched. Lines lacking a valid H2/He entry fall back to gamma_air
    FOR THAT PARTNER, and the per-molecule coverage is printed loudly; a molecule
    with NO coverage at all raises (use broadening="air" for it, knowingly).
    The temperature exponent is the gamma-weighted mean (exact at T_ref, first-order
    in the mix elsewhere).
    """
    g_air = np.asarray(mdb.gamma_air, dtype=np.float64)
    n_air = np.asarray(mdb.n_air, dtype=np.float64)
    f_h2, f_he = config.H2HE_BROADENING_MIX
    parts = []
    for partner, frac in (("h2", f_h2), ("he", f_he)):
        g = getattr(mdb, f"gamma_{partner}", None)
        n = getattr(mdb, f"n_{partner}", None)
        if g is None or n is None:
            parts.append((frac, None, None, 0.0))
            continue
        g = np.asarray(g, dtype=np.float64)
        n = np.asarray(n, dtype=np.float64)
        ok = np.isfinite(g) & (g > 0.0)
        n_ok = np.where(np.isfinite(n), n, n_air)
        parts.append((frac, np.where(ok, g, g_air), np.where(ok, n_ok, n_air),
                      float(ok.mean()) if g.size else 0.0))
    if all(p[1] is None for p in parts):
        raise RuntimeError(
            f"broadening='h2he' requested but HITRAN supplies no H2/He broadening "
            f"columns for {key} in this band (or the cache predates "
            "nonair_broadening=True -- h2he mode uses a separate 'h2he/<db>' cache "
            "dir precisely to force a fresh extended download). Either drop the "
            f"molecule, or run it with broadening='air' knowingly.")
    gamma_mix = np.zeros_like(g_air)
    gn_mix = np.zeros_like(g_air)
    for frac, g, n, _cov in parts:
        g_eff = g_air if g is None else g
        n_eff = n_air if n is None else n
        gamma_mix = gamma_mix + frac * g_eff
        gn_mix = gn_mix + frac * g_eff * n_eff
    n_mix = gn_mix / np.where(gamma_mix > 0.0, gamma_mix, 1.0)
    cov = {p: parts[i][3] for i, p in enumerate(("H2", "He"))}
    print(f"[rt]   {key}: H2/He broadening blend f=({f_h2:.2f},{f_he:.2f}); line "
          f"coverage H2 {cov['H2']:.1%}, He {cov['He']:.1%} (uncovered lines keep "
          f"gamma_air for that partner); median gamma_mix/gamma_air = "
          f"{np.median(gamma_mix / np.where(g_air > 0, g_air, np.nan)):.3f}", flush=True)
    mdb.gamma_air = gamma_mix
    mdb.n_air = n_mix


def _build_opa(key: str, spec: dict, nu_grid, broadening: str = "air",
               dit_grid_resolution: float = 1.0):
    """Build one premodit opacity for ``key`` (CO cached; others downloaded).

    ``broadening``: "air" (HITRAN default, terrestrial perturber -- documented
    approximation) or "h2he" (HITRAN planetary H2/He widths where available, blended
    per ``config.H2HE_BROADENING_MIX``; separate ``h2he/<db>`` download cache).
    ExoMol sources already carry their own default broadening and ignore the knob.
    ``dit_grid_resolution``: the PreMODIT broadening-parameter grid spacing;
    1.0 is this pipeline's long-standing value, smaller resolves the
    pressure-broadening grid finer (exojax's own default is 0.2) at a slower
    opacity build. Profile-overridable via ``profile["dit_grid_resolution"]``.
    """
    src = spec["source"]
    if src in ("exomol", "exomol_cached"):
        path = spec["db"] if src == "exomol_cached" else str(config.DEMO_DATABASE / spec["db"])
        mdb = MdbExomol(path, nurange=nu_grid)
    elif src == "hitran":
        # isotope=1 (main isotopologue): isotope=0 pulls minor isotopologues whose
        # TIPS partition functions are missing from hapi (e.g. SO2 (9,3) -> KeyError).
        # HITRAN line intensities include the terrestrial isotopic abundance factor,
        # so applying the main-isotopologue opacity to the TOTAL molecular VMR is the
        # standard (slightly conservative) approximation documented in config.py.
        if broadening == "h2he":
            # separate h2he/<db> cache dir: forces a fresh nonair_broadening
            # download, AND keeps the path stem a valid HITRAN molecule token.
            # (The old "<db>_h2he" naming broke under exojax>=2.x, which derives
            # the molecule id from the stem -- radis raised NotImplementedError
            # before anything downloaded, killing every h2he run.)
            mdb = MdbHitran(str(config.DEMO_DATABASE / "h2he" / spec["db"]),
                            nurange=nu_grid, isotope=1, nonair_broadening=True)
            _blend_h2he_broadening(mdb, key)
        elif broadening == "air":
            mdb = MdbHitran(str(config.DEMO_DATABASE / spec["db"]), nurange=nu_grid,
                            isotope=1)
        else:
            raise ValueError(f"unknown broadening mode {broadening!r} "
                             "(expected 'air' or 'h2he')")
    else:
        raise ValueError(f"unknown opacity source {src!r} for {key}")
    try:
        opa = OpaPremodit.from_snapshot(
            mdb.to_snapshot(), nu_grid,
            auto_trange=(config.T_OPA_MIN_K, config.T_OPA_MAX_K),
            dit_grid_resolution=dit_grid_resolution)
    except AttributeError:
        opa = OpaPremodit(
            mdb, nu_grid,
            auto_trange=(config.T_OPA_MIN_K, config.T_OPA_MAX_K),
            dit_grid_resolution=dit_grid_resolution)
    n_lines = int(np.asarray(getattr(mdb, "nu_lines", np.zeros(0)).shape[0])) if hasattr(mdb, "nu_lines") else -1
    return opa, n_lines


def _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_btm,
                     vmr, vmr_h2, T_art, mmw_art,
                     opacia_he=None, vmr_he=None, cloud=None, rayleigh_xs=None,
                     mie_pack=None, mie=None):
    """Per-layer optical-depth matrix ``dtau`` (nlayer, n_nu) on the ART pressure grid.

    The sum of each molecule's line opacity plus the H2-H2 CIA continuum -- and,
    optionally, H2-He CIA (``opacia_he`` + ``vmr_he``), the ExoJax power-law
    retrieval cloud (``cloud``), and H2/He Rayleigh scattering (``rayleigh_xs``).
    Shared by the transmission and emission models: line + CIA + cloud terms are
    identical; Rayleigh is transmission-only BY DESIGN (it is scattering, not
    absorption -- adding it to the pure-absorption ibased emission solver would
    fake thermal extinction; it is also negligible at the >1 um thermal bands).

    Parameters
    ----------
    art : ArtTransPure | ArtEmisPure   provides the pressure grid + opacity_profile_* ops
    vmr : dict molecule -> (nlayer,) volume mixing ratio
    vmr_h2 : (nlayer,) H2 volume mixing ratio (both H2-H2 CIA collision partners)
    opacia_he, vmr_he : optional H2-He CIA table + He VMR profile (term skipped if
        either is None -- keeps the parent demo callers byte-compatible)
    cloud : optional (2,) array [log10 kappac0 (cm^2/g at config.CLOUD_NUC0), alphac]
        for ``exojax.atm.simple_clouds.powerlaw_clouds`` (alphac=0 -> gray cloud;
        per-gram-of-atmosphere opacity, uniformly mixed: dtau = kappa(nu)*dP_cgs/g)

    Pure JAX throughout, so forward-mode tangents from the chemistry pass straight through.

    Each molecule's line-opacity term (and each CIA term) is wrapped in
    ``jax.checkpoint``: REVERSE-mode differentiation (the SMC retrieval's RT vjp)
    otherwise has to STORE every molecule's PreMODIT intermediates
    (~(nlayer x n_nu x broadening-grid) fp64 tensors) until the backward pass --
    ~30-50 GB per spectrum on the gpu-preset grid, which OOM'd a GH200 at a 6-wide
    particle chunk (2026-07-06, ~300 GiB requested). With checkpoint the backward
    recomputes one molecule at a time and the tape keeps only each term's (nlayer,
    n_nu) output. Exact same values; forward eval and forward-mode jvp (the
    published sensitivity path) are unaffected.
    """
    dtau = jnp.zeros((art.pressure.shape[0], nu_grid.shape[0]))
    for key in mols:
        def _line_term(T_art_, vmr_key_, mmw_art_, _key=key):
            xs = opas[_key].xsmatrix(T_art_, art.pressure)         # (nlayer, n_nu)
            mmr = vmr_to_mmr(vmr_key_, molmass[_key], mmw_art_)
            return art.opacity_profile_xs(xs, mmr, molmass[_key], g_btm)
        dtau = dtau + jax.checkpoint(_line_term)(T_art, vmr[key], mmw_art)
    # opacity_profile_cia divides a (nlayer, n_nu) matrix by mmw, so mmw must broadcast
    # as (nlayer, 1) here -- note art.run separately wants the 1-D (nlayer,) form.

    def _cia_term(opacia_, T_art_, vmr_a, vmr_b, mmw_art_):
        logacia_ = opacia_.logacia_matrix(T_art_)
        return art.opacity_profile_cia(
            logacia_, T_art_, vmr_a, vmr_b, mmw_art_[:, None], g_btm)

    dtau = dtau + jax.checkpoint(
        lambda t, va, vb, m: _cia_term(opacia, t, va, vb, m))(
        T_art, vmr_h2, vmr_h2, mmw_art)
    if opacia_he is not None and vmr_he is not None:
        dtau = dtau + jax.checkpoint(
            lambda t, va, vb, m: _cia_term(opacia_he, t, va, vb, m))(
            T_art, vmr_h2, vmr_he, mmw_art)
    if cloud is not None:
        # ExoJax's shipped retrieval cloud (pRT convention, per gram of atmosphere).
        kappa_c = powerlaw_clouds(nu_grid, kappac0=10.0 ** cloud[0],
                                  nuc0=config.CLOUD_NUC0, alphac=cloud[1])  # (n_nu,)
        dP = jnp.asarray(art.dParr)                                # (nlayer,) bar
        dtau = dtau + kappa_c[None, :] * (dP[:, None] * _BAR_CGS / g_btm)
    if mie is not None:
        # Mie cloud deck (v16, exojax PdbCloud/OpaMie): column-uniform
        # condensate MMR with a single lognormal size distribution.
        # mie = [log10 rg (cm), sigmag, log10 MMR]; mie_pack = (OpaMie,
        # substance density) baked at build time. sigma_extinction
        # (absorption + scattering) is the correct chord attenuation in
        # transmission AND the pure-absorption emission approximation for
        # the extinction term (scattering as removal; the tool documents
        # the forward-scattering caveat). Differentiable: the miegrid
        # interpolation is piecewise-linear in (log rg, sigmag), clamped at
        # the grid edges (callers keep parameters strictly inside).
        if mie_pack is None:
            raise ValueError(
                "mie parameters passed but the RT was built without "
                "mie_condensate: rebuild with profile['mie_condensate'] set "
                "(never silently ignore a requested cloud).")
        opa_mie_, rho_c = mie_pack
        rg = 10.0 ** mie[0]
        sigmag = mie[1]
        mmr = 10.0 ** mie[2]
        sig_ext, _sig_sca, _g_asym = opa_mie_.mieparams_vector(rg, sigmag)
        mmr_prof = jnp.full((art.pressure.shape[0],), 1.0) * mmr
        dtau = dtau + art.opacity_profile_cloud_lognormal(
            sig_ext, rho_c, mmr_prof, rg, sigmag, g_btm)
    if rayleigh_xs is not None:
        # H2 (+He) Rayleigh scattering -- zero-free-parameter known physics that
        # matters short of ~1.5 um; omitting it would bias the retrieved haze slope.
        # rayleigh_xs = (xs_h2, xs_he), each (n_nu,) from exojax xsvector_rayleigh_gas.
        xs_h2, xs_he = rayleigh_xs
        nlayer = art.pressure.shape[0]
        mmr_h2 = vmr_to_mmr(vmr_h2, _H2_MOLMASS, mmw_art)
        dtau = dtau + art.opacity_profile_xs(
            jnp.broadcast_to(xs_h2[None, :], (nlayer, xs_h2.shape[0])),
            mmr_h2, _H2_MOLMASS, g_btm)
        if vmr_he is not None:
            mmr_he = vmr_to_mmr(vmr_he, _HE_MOLMASS, mmw_art)
            dtau = dtau + art.opacity_profile_xs(
                jnp.broadcast_to(xs_he[None, :], (nlayer, xs_he.shape[0])),
                mmr_he, _HE_MOLMASS, g_btm)
    return dtau


def build_rt_model(profile: dict) -> SimpleNamespace:
    """Build the transmission-spectrum model for the molecules named in ``profile``.

    Returns a SimpleNamespace with:
        transmission_depth(vmr, vmr_h2, T_art, mmw_art) -> (n_nu,) transit depth
        nu_grid : (n_nu,) wavenumber grid (cm^-1)
        wl_um   : (n_nu,) wavelength grid (micron), descending->ascending sorted handled by caller
        p_art_bar : (nlayer,) ART pressure grid (bar)
        molecules : list[str]
    """
    t0 = time.time()
    nu_grid, wav, resolution = wavenumber_grid(
        profile["nu_min"], profile["nu_max"], profile["nu_pts"],
        unit="cm-1", xsmode="premodit")
    print(f"[rt] nu_grid {nu_grid.shape[0]} pts, R~{resolution:.0f}, "
          f"lambda[{1e4/profile['nu_max']:.2f},{1e4/profile['nu_min']:.2f}] um", flush=True)

    mols = list(profile["molecules"])
    broadening = str(profile.get("broadening", config.BROADENING))
    print(f"[rt] pressure broadening: {broadening}"
          + (" (terrestrial-air widths -- documented approximation; set "
             "broadening='h2he' for HITRAN planetary H2/He widths)"
             if broadening == "air" else ""), flush=True)
    # Profile-overridable RT knobs (defaults = the historical hard-coded
    # values, so every existing consumer is byte-unchanged). Validated
    # loudly here -- an out-of-range value must never build a wrong model.
    dit_res = float(profile.get("dit_grid_resolution", 1.0))
    if not 0.05 <= dit_res <= 2.0:
        raise ValueError(
            f"dit_grid_resolution={dit_res:g} outside [0.05, 2.0] (PreMODIT "
            "broadening-grid spacing; 1.0 = this pipeline's default, 0.2 = "
            "exojax's own default)")
    ptop = float(profile.get("art_ptop_bar", config.ART_PTOP_BAR))
    pbtm = float(profile.get("art_pbtm_bar", config.ART_PBTM_BAR))
    if not (0.0 < ptop < pbtm):
        raise ValueError(
            f"art_ptop_bar={ptop:g} / art_pbtm_bar={pbtm:g}: need "
            "0 < top < bottom (bar)")
    integration = str(profile.get("rt_integration", "simpson"))
    if integration not in ("simpson", "trapezoid"):
        raise ValueError(
            f"rt_integration={integration!r}: exojax ArtTransPure supports "
            "'simpson' (default) or 'trapezoid'")
    opas, molmass = {}, {}
    for key in mols:
        spec = config.MOLECULES[key]
        tb = time.time()
        opa, n_lines = _build_opa(key, spec, nu_grid, broadening=broadening,
                                  dit_grid_resolution=dit_res)
        opas[key] = opa
        molmass[key] = float(spec["molmass"])
        print(f"[rt]   {key}: {n_lines} lines, opa built in {time.time()-tb:.1f}s", flush=True)

    art = ArtTransPure(
        pressure_top=ptop,
        pressure_btm=pbtm,
        nlayer=int(profile["art_nlayer"]),
        integration=integration)
    art.change_temperature_range(config.T_OPA_MIN_K, config.T_OPA_MAX_K)
    p_art_bar = np.asarray(art.pressure)
    print(f"[rt] ArtTransPure {profile['art_nlayer']} layers, "
          f"P=[{p_art_bar.min():.1e},{p_art_bar.max():.1e}] bar, "
          f"chord integration {integration}, dit_grid_resolution {dit_res:g}",
          flush=True)

    if not config.CIA_H2H2_FILE.exists():
        # exojax auto-fetches it (~24 MB from hitran.org), but its downloader
        # swallows failures -- say up front what is about to happen so an
        # offline failure is attributable (fail-loud rule).
        print(f"[rt] H2-H2 CIA absent at {config.CIA_H2H2_FILE}; exojax will "
              "download ~24 MB from https://hitran.org/data/CIA/main/"
              "H2-H2_2011.cia now (network required)", flush=True)
    cdb = CdbCIA(str(config.CIA_H2H2_FILE), nurange=nu_grid)
    opacia = OpaCIA(cdb, nu_grid=nu_grid)
    # H2-He CIA is required physics (He is ~14% by number). Missing it
    # would silently drop a real continuum term -> a wrong spectrum with no error.
    # Fail loud instead; the file ships in data/opacity_cache/ and the PBS preflight
    # also checks for it.
    if not config.CIA_H2HE_FILE.exists():
        raise FileNotFoundError(
            f"H2-He CIA table missing ({config.CIA_H2HE_FILE}). It ships in the bundle's "
            "data/opacity_cache/; if absent download "
            "https://hitran.org/data/CIA/main/H2-He_2011.cia (~147 MB; note the "
            "/main/ path -- the bare /data/CIA/ URL 404s) to that exact path. "
            "Refusing to build the RT without it (silently skipping the He "
            "continuum would bias the spectrum).")
    opacia_he = OpaCIA(CdbCIA(str(config.CIA_H2HE_FILE), nurange=nu_grid), nu_grid=nu_grid)
    print("[rt] H2-He CIA loaded", flush=True)
    print(f"[rt] CIA + RT built; total {time.time()-t0:.1f}s", flush=True)

    # H2/He Rayleigh cross sections (nu-only, precomputed once; opt-in via profile
    # so the parent demo's published outputs are untouched)
    if profile.get("use_rayleigh", False):
        rayleigh_xs = (
            xsvector_rayleigh_gas(nu_grid, _POLARIZABILITY["H2"]),
            xsvector_rayleigh_gas(nu_grid, _POLARIZABILITY["He"]),
        )
        print("[rt] H2/He Rayleigh scattering enabled", flush=True)
    else:
        rayleigh_xs = None

    # Mie cloud deck (v16 tool work): opt-in via profile["mie_condensate"]
    # + profile["mie_data_dir"] (a pinned ABSOLUTE cache dir -- exojax's own
    # default is CWD-relative, which is banned here). The forward model only
    # LOADS a pre-generated miegrid (pure numpy/jnp, differentiable); a
    # missing grid raises with the generation command. The 4 MB virga
    # refractive-index archive auto-downloads from Zenodo on first use --
    # announced up front so an offline failure is attributable.
    mie_pack = None
    mie_cond = profile.get("mie_condensate") or None
    if mie_cond:
        from pathlib import Path as _Path
        from exojax.database.pardb import PdbCloud
        from exojax.opacity import OpaMie
        mie_dir = profile.get("mie_data_dir")
        if not mie_dir:
            raise ValueError(
                "mie_condensate set but mie_data_dir missing: pass the "
                "absolute Mie cache directory (a CWD-relative default would "
                "scatter caches per launch dir).")
        if not (_Path(mie_dir) / "virga.zip").exists():
            print(f"[rt] virga refractive-index archive absent under "
                  f"{mie_dir}; exojax will download ~4 MB from Zenodo now "
                  "(network required)", flush=True)
        pdb = PdbCloud(str(mie_cond), nurange=nu_grid, path=str(mie_dir))
        if not pdb.miegrid_path.exists():
            raise FileNotFoundError(
                f"Mie grid for {mie_cond} not found at {pdb.miegrid_path}. "
                "Generate it once (vulcan-jwst-tool: python "
                f"tools/generate_miegrid.py {mie_cond}, ~1 h/condensate); "
                "the forward model only loads grids, never a silent "
                "fallback.")
        pdb.load_miegrid()
        opa_mie = OpaMie(pdb, nu_grid)
        mie_pack = (opa_mie, float(pdb.condensate_substance_density))
        print(f"[rt] Mie cloud deck: {mie_cond} (rho = "
              f"{pdb.condensate_substance_density:g} g/cm3, miegrid loaded)",
              flush=True)

    # Planet identity: overridable per profile (retrieval case presets set these);
    # defaults are the shared-lib W39b constants for all legacy consumers.
    Rp_btm = float(profile.get("rp_cm", config.RP_CM))
    g_btm = float(profile.get("gs_cgs", config.GS_CGS))
    depth_norm = (Rp_btm / float(profile.get("rstar_cm", config.RSTAR_CM))) ** 2  # (R_btm/R_star)^2

    def _require_he(vmr_he):
        # He is ~14% by number and its CIA is real continuum physics;
        # an accidental None here used to SILENTLY drop the term (the sensitivity
        # demo shipped that way). Fail loud instead -- standing repo rule.
        if vmr_he is None:
            raise ValueError(
                "vmr_he is required: pass the He VMR profile (chem.sidx['He']) so the "
                "H2-He CIA term is included. There is no supported He-less mode.")

    def transmission_depth(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None,
                           mie=None):
        """Transit depth (R_p(lambda)/R_star)^2 from ART-grid profiles.

        vmr : dict molecule -> (nlayer,) VMR; vmr_h2 : (nlayer,) H2 VMR (for CIA);
        vmr_he : (nlayer,) He VMR (H2-He CIA partner; REQUIRED -- the None default
        exists only so an omission raises the explanatory ValueError, not TypeError).
        Optional: cloud=[log10 kappac0, alphac] (ExoJax powerlaw_clouds);
        mie=[log10 rg (cm), sigmag, log10 MMR] (needs mie_condensate at build).
        """
        _require_he(vmr_he)
        dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_btm,
                                vmr, vmr_h2, T_art, mmw_art,
                                opacia_he=opacia_he, vmr_he=vmr_he, cloud=cloud,
                                rayleigh_xs=rayleigh_xs,
                                mie_pack=mie_pack, mie=mie)
        Rp2 = art.run(dtau, T_art, mmw_art, Rp_btm, g_btm)          # (radius/Rp_btm)^2
        return Rp2 * depth_norm                                     # (radius/R_star)^2

    def transmission_depth_r(vmr, vmr_h2, T_art, mmw_art, lnR0, vmr_he=None,
                             cloud=None, mie=None):
        """transmission_depth with a reference-radius scaling: the radius at the bottom
        pressure P_btm is Rp_btm * e^lnR0 (gravity held fixed -- the standard xR_p
        normalization nuisance, cf. Batalha & Line 2017). lnR0 = 0 reproduces
        transmission_depth exactly; the lnR0 jvp is the exact geometric+hydrostatic
        response, RT-only (chemistry profiles enter frozen). Because gravity is held
        fixed, lnR0 must be read as a pressure-radius normalization, NOT a physical
        planet-radius change at fixed mass."""
        _require_he(vmr_he)
        Rp_r = Rp_btm * jnp.exp(lnR0)
        dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_btm,
                                vmr, vmr_h2, T_art, mmw_art,
                                opacia_he=opacia_he, vmr_he=vmr_he, cloud=cloud,
                                rayleigh_xs=rayleigh_xs,
                                mie_pack=mie_pack, mie=mie)
        Rp2 = art.run(dtau, T_art, mmw_art, Rp_r, g_btm)            # (radius/Rp_r)^2
        return Rp2 * depth_norm * jnp.exp(2.0 * lnR0)               # (radius/R_star)^2

    return SimpleNamespace(
        transmission_depth=transmission_depth,
        transmission_depth_r=transmission_depth_r,
        nu_grid=np.asarray(nu_grid),
        wl_um=1e4 / np.asarray(nu_grid),
        p_art_bar=p_art_bar,
        molecules=mols,
        broadening=broadening,
        # echo of the profile-overridable RT knobs, so downstream consumers
        # (vulcan-jwst-tool) can VERIFY the engine honored them -- an older
        # engine that ignores an unknown profile key must fail loudly there,
        # never silently compute a different model than the cache key claims
        art_ptop_bar=ptop,
        art_pbtm_bar=pbtm,
        rt_integration=integration,
        dit_grid_resolution=dit_res,
        has_cia_h2he=opacia_he is not None,
        # Mie deck echo (v16): "" when no Mie cloud was built -- consumers
        # verify this against what they requested (an engine too old to know
        # mie_condensate would ignore it silently; the echo makes that loud)
        mie_condensate=str(mie_cond or ""),
        # internals reused by build_emis_model (so opacities aren't rebuilt)
        _nu_grid=nu_grid, _opas=opas, _molmass=molmass, _opacia=opacia,
        _opacia_he=opacia_he, _mie_pack=mie_pack,
    )


def build_emis_model(trt, profile: dict) -> SimpleNamespace:
    """Build an ArtEmisPure thermal-emission model that SHARES trt's opacities/grid.

    Returns a model whose ``emission_flux(vmr, vmr_h2, T_art, mmw_art, vmr_he)`` maps
    the same ART-grid profiles to the planet's EMERGENT flux spectrum
    (erg s^-1 cm^-2 / cm^-1). This is the top-of-atmosphere planetary flux, NOT an
    eclipse depth / planet-star contrast -- do not compare it to an observed
    secondary-eclipse spectrum without dividing by the stellar flux and applying
    (Rp/Rstar)^2. Opacity terms match transmission (lines + H2-H2 + H2-He CIA,
    optional cloud); Rayleigh scattering is deliberately excluded here (see
    _accumulate_dtau -- a pure-absorption solver must not count scattering as
    thermal absorption, and it is negligible in the thermal bands).
    """
    nu_grid = trt._nu_grid
    opas, molmass, opacia, mols = trt._opas, trt._molmass, trt._opacia, trt.molecules
    opacia_he = trt._opacia_he
    mie_pack = getattr(trt, "_mie_pack", None)   # shared Mie deck (v16)
    g_btm = float(profile.get("gs_cgs", config.GS_CGS))

    # pressure bounds follow the transmission model's (possibly profile-
    # overridden) grid -- the two share opacities and must share the column
    art = ArtEmisPure(nu_grid=nu_grid, pressure_top=trt.art_ptop_bar,
                      pressure_btm=trt.art_pbtm_bar, nlayer=int(profile["art_nlayer"]),
                      rtsolver="ibased", nstream=8)
    art.change_temperature_range(config.T_OPA_MIN_K, config.T_OPA_MAX_K)
    print(f"[rt] ArtEmisPure {profile['art_nlayer']} layers (shares opacities)", flush=True)

    def emission_flux(vmr, vmr_h2, T_art, mmw_art, vmr_he=None, cloud=None,
                      mie=None):
        """Emergent thermal flux (n_nu,) from ART-grid VMR/T/mmw profiles.

        vmr_he is REQUIRED (H2-He CIA -- same continuum physics as transmission;
        the None default only upgrades the omission error message)."""
        if vmr_he is None:
            raise ValueError(
                "vmr_he is required: pass the He VMR profile so the H2-He CIA term "
                "is included in the emission opacity (parity with transmission).")
        dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_btm,
                                vmr, vmr_h2, T_art, mmw_art,
                                opacia_he=opacia_he, vmr_he=vmr_he, cloud=cloud,
                                mie_pack=mie_pack, mie=mie)
        return art.run(dtau, T_art)

    def tau_bottom(vmr, vmr_h2, T_art, mmw_art, vmr_he, cloud=None, mie=None):
        """Total vertical optical depth at the BOTTOM of the RT column,
        per wavenumber (n_nu,). ArtEmisPure has NO surface/interior source
        term, so its flux is only trustworthy where the column is optically
        thick at the bottom -- a caller must check min(tau_bottom) and
        refuse/flag windows that see through the grid (the flux there is
        silently underestimated). Not on the hot AD path (diagnostic)."""
        dtau = _accumulate_dtau(art, nu_grid, mols, opas, molmass, opacia, g_btm,
                                vmr, vmr_h2, T_art, mmw_art,
                                opacia_he=opacia_he, vmr_he=vmr_he, cloud=cloud,
                                mie_pack=mie_pack, mie=mie)
        return jnp.sum(dtau, axis=0)

    return SimpleNamespace(
        emission_flux=emission_flux,
        tau_bottom=tau_bottom,
        nu_grid=np.asarray(nu_grid),
        wl_um=trt.wl_um,
        p_art_bar=np.asarray(art.pressure),
        art_pbtm_bar=float(trt.art_pbtm_bar),
        molecules=mols,
    )
