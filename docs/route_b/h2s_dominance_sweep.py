"""B0A Stage-1 GAS-PHASE H2S-dominance check (Route B plan, D3b; revised per
the collaborator's B0A review, round 3).

Runs the VULCAN-vendored FastChem binary (exoclime FastChem, (C) 2019
Kitzmann & Stock — the GAS-PHASE code; the source tree contains no
condensation solver, verified by inspection, so every result here is
gas-phase equilibrium) over the supported (T_bottom, Z, C/O) domain at the
engine bottom pressure and measures:

  1. f_H2S_gas = x_H2S / (S-atom-weighted sum over ALL gas species in the
     FastChem output). This is the H2S share of GAS-PHASE sulfur, not of
     total (gas + condensed) sulfur.
  2. The saturation ratio r_c = x_c * P / p_sat,c(T) of every sulfur
     condensable with a VULCAN runtime channel (S2, S4, S8) at each node.
     Where max_c r_c < 1, no sulfur condensate is saturated at that node,
     so the gas-only equilibrium IS the complete equilibrium for sulfur
     there (a condensation-capable solver would leave it unchanged); the
     gas-phase dominance number is then a total-sulfur statement at that
     node. Where max_c r_c >= 1 the gas-only result is an upper bound only.
  3. Accuracy of two candidate on-graph pin formulas vs FastChem x_H2S
     (both PROTOTYPE-grade; the production boundary is the equilibrium
     lookup built by h2s_boundary_table.py):
       A. closed form  x_A = R_S / (0.5 + R_He)
       B. anchor-scale x_B = x_H2S(baseline lnZ=0, same T,P,c_o) * exp(lnZ)

Domain swept: T_bottom in [400, 3000] K (14 points), P_b = 7.6 bar (1 and
76 bar margin), lnZ in [-2.303, +2.303] about the 10x-solar baseline (He
fixed), c_o in [-1.70, +0.50].

Baseline element abundances = vulcan_cfg_W39b (Tsai 2023, 10x solar):
  O_H 5.37e-3, C_H 2.95e-3, N_H 7.08e-4, S_H 1.41e-4, He_H 0.0838 (fixed).

Output JSON carries a full provenance block: binary/input checksums,
platform, grid, definitions, and per-run FastChem nonconvergence counts
parsed from monitor_output.dat.

Run against a PRIVATE copy of VULCAN-JAX's fastchem_vulcan/ tree (binary +
input/ + output/), never the package tree itself: this script overwrites
input/element_abundances_vulcan.dat, input/parameters.dat, and the TP file.
  cp -R VULCAN-JAX/src/vulcan_jax/fastchem_vulcan /some/scratch/fastchem_check
  cp .../input/parameters_wo_ion.dat .../input/parameters.dat
  H2S_CHECK_FASTCHEM_DIR=/some/scratch/fastchem_check python h2s_dominance_sweep.py
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

FC_DIR = Path(
    os.environ.get(
        "H2S_CHECK_FASTCHEM_DIR", Path(__file__).parent / "fastchem_check"
    )
)
OUT_JSON = Path(__file__).parent / "h2s_dominance_results.json"

LOG10E = np.log10(np.e)

# Baseline (10x solar, Tsai 2023 values from vulcan_cfg_W39b)
BASE = {"O": 5.37e-3, "C": 2.95e-3, "N": 7.08e-4, "S": 1.41e-4}
HE_DEX = 10.9232  # solar He, fixed (elemental map never scales He)
# Rocky elements: engine baseline suppresses to -3.0 dex then adds
# log10(fastchem_met_scale=10) -> -2.0; the theta map never rescales them.
ROCKY = ["P", "Si", "Ti", "V", "Cl", "K", "Na", "Mg", "F", "Ca", "Fe"]
ROCKY_DEX = -2.0

LNZ_GRID = [-2.303, -1.151, 0.0, 1.151, 2.303]  # 1x .. 100x solar
CO_GRID = [-1.70, -1.00, -0.50, 0.00, 0.24, 0.50]
T_GRID = [400.0 + 200.0 * i for i in range(14)]  # 400 .. 3000 K
P_GRID = [7.6, 1.0, 76.0]  # bar; 7.6 = engine P_b, others = robustness margin

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

F_DEFINITION = (
    "f_H2S_gas = x_H2S / sum_sp(nS(sp) * x_sp) over every species column of "
    "the FastChem MR output (gas phase only; the vendored FastChem has no "
    "condensation solver). nS(sp) parsed from the column name."
)


def s_atoms(name: str) -> int:
    """Count S atoms in a FastChem species column name (Si/Cl-safe)."""
    n = 0
    for sym, cnt in _TOKEN.findall(name):
        if sym == "S":
            n += int(cnt) if cnt else 1
    return n


def sat_p_dyne(sp: str, T: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (dyne/cm^2) for the runtime S condensables.

    MUST MATCH VULCAN-JAX atm_setup.sat_p_jax exactly (S2/S8 413 K break);
    replicated here so this validation artifact stays numpy-standalone.
    """
    T = np.asarray(T, dtype=np.float64)
    if sp == "S2":
        return np.where(
            T < 413.0,
            np.exp(27.0 - 18500.0 / T) * 1e6,
            np.exp(16.1 - 14000.0 / T) * 1e6,
        )
    if sp == "S4":
        return 10 ** (6.0028 - 6047.5 / T) * 1.01325e6
    if sp == "S8":
        return np.where(
            T < 413.0,
            np.exp(20.0 - 11800.0 / T) * 1e6,
            np.exp(9.6 - 7510.0 / T) * 1e6,
        )
    raise ValueError(f"no saturation curve wired for {sp}")


S_CONDENSABLES = ("S2", "S4", "S8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_abundances(lnZ: float, c_o: float) -> Path:
    # FastChem's abundance reader consumes the first line as a header, so a
    # comment line must lead the file or the H row is silently dropped.
    lines = ["# element abundances (dex, H=12)", "H\t12.0000", f"He\t{HE_DEX:.4f}"]
    for el, base in BASE.items():
        dex = 12.0 + np.log10(base) + lnZ * LOG10E
        if el == "C":
            dex += c_o * LOG10E
        lines.append(f"{el}\t{dex:.4f}")
    for el in ROCKY:
        lines.append(f"{el}\t{ROCKY_DEX:.4f}")
    lines.append("e-\t0")
    p = FC_DIR / "input" / "element_abundances_vulcan.dat"
    p.write_text("\n".join(lines) + "\n")
    return p


def write_tp() -> list:
    rows = [(p, t) for p in P_GRID for t in T_GRID]
    body = "#p (bar)    T (K)\n" + "\n".join(
        f"{p:.3e}\t{t:.1f}" for p, t in rows
    )
    (FC_DIR / "input" / "vulcan_TP" / "vulcan_TP.dat").write_text(body)
    return rows


def run_fastchem() -> tuple:
    subprocess.run(
        ["./fastchem", "input/config.input"],
        cwd=FC_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    raw = (FC_DIR / "output" / "vulcan_EQ.dat").read_text().strip().splitlines()
    header = raw[0].split()
    data = np.array([[float(v) for v in ln.split()] for ln in raw[1:]])
    n_bad = count_nonconverged()
    return header, data, n_bad


def count_nonconverged() -> int:
    """Rows of monitor_output.dat whose convergence flags are not all 'ok'."""
    lines = (
        (FC_DIR / "output" / "monitor_output.dat").read_text().strip().splitlines()
    )
    n_bad = 0
    for ln in lines[1:]:
        fields = ln.split()
        # fields: grid_point, c_iterations, c_convergence, P, T, n_tot, n_g,
        # m, then one flag per element. Everything non-numeric must be 'ok'.
        flags = [fields[2]] + fields[8:]
        if any(f != "ok" for f in flags):
            n_bad += 1
    return n_bad


def provenance(abundance_hashes: dict) -> dict:
    params = FC_DIR / "input" / "parameters.dat"
    # The species logK database actually referenced by parameters.dat.
    plines = [
        ln.strip()
        for ln in params.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    referenced = [ln for ln in plines if ln.endswith(".dat")]
    file_hashes = {
        "fastchem_binary": sha256_file(FC_DIR / "fastchem"),
        "parameters.dat": sha256_file(params),
        "config.input": sha256_file(FC_DIR / "input" / "config.input"),
    }
    for rel in referenced:
        f = FC_DIR / rel
        if f.is_file():
            file_hashes[rel] = sha256_file(f)
    return {
        "fastchem_identity": (
            "exoclime FastChem (https://github.com/exoclime/fastchem), "
            "Copyright (C) 2019 Daniel Kitzmann, Joachim Stock, as vendored in "
            "VULCAN-JAX src/vulcan_jax/fastchem_vulcan (compiled from the "
            "vendored source by ini_abun._ensure_fastchem_binary)"
        ),
        "condensation_enabled": False,
        "condensation_evidence": (
            "gas-phase-only code: zero matches for 'condens' across "
            "fastchem_src/*.cpp,*.h; FastChem Cond (equilibrium condensation) "
            "is a later upstream release not vendored here"
        ),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "sha256": file_hashes,
        "abundance_file_sha256_by_lnZ_c_o": abundance_hashes,
        "grid": {
            "T_K": T_GRID,
            "P_bar": P_GRID,
            "lnZ": LNZ_GRID,
            "c_o": CO_GRID,
            "baseline_X_H": BASE,
            "He_dex_fixed": HE_DEX,
            "rocky_dex": ROCKY_DEX,
        },
        "f_definition": F_DEFINITION,
        "sat_ratio_definition": (
            "r_c = x_c * P[dyne] / p_sat,c(T) for c in (S2, S4, S8), "
            "p_sat replicating VULCAN-JAX atm_setup.sat_p_jax; max_c r_c < 1 "
            "means no sulfur condensate is saturated at the node, so the "
            "gas-only equilibrium is the complete sulfur equilibrium there"
        ),
    }


def main() -> int:
    fc_bin = FC_DIR / "fastchem"
    if not fc_bin.is_file():
        raise RuntimeError(
            f"FastChem binary not found at {fc_bin}. Copy "
            "VULCAN-JAX/src/vulcan_jax/fastchem_vulcan to a private dir, copy "
            "input/parameters_wo_ion.dat to input/parameters.dat there, and "
            "point H2S_CHECK_FASTCHEM_DIR at it (see module docstring)."
        )
    tp_rows = write_tp()
    results = []
    baseline_x = {}  # (P, T, c_o) -> x_H2S at lnZ = 0, for candidate B
    abundance_hashes = {}
    total_bad = 0

    for c_o in CO_GRID:
        for lnZ in LNZ_GRID:
            abun_path = write_abundances(lnZ, c_o)
            abundance_hashes[f"({lnZ:+.3f},{c_o:+.2f})"] = sha256_file(abun_path)
            header, data, n_bad = run_fastchem()
            total_bad += n_bad
            if n_bad:
                print(f"WARNING: {n_bad} nonconverged FastChem nodes at "
                      f"lnZ={lnZ:+.3f}, c_o={c_o:+.2f}")
            # Columns: 0 P(bar), 1 T(K), 2 n_<tot>, 3 n_g, 4 m, 5.. species.
            sp_names = header[5:]
            s_counts = np.array([s_atoms(n) for n in sp_names], dtype=float)
            h2s_col = 5 + sp_names.index("H2S")
            cond_cols = {c: 5 + sp_names.index(c) for c in S_CONDENSABLES}
            for i, (p_bar, t_k) in enumerate(tp_rows):
                x = data[i, 5:]
                s_tot = float(np.dot(x, s_counts))
                x_h2s = float(data[i, h2s_col])
                f = x_h2s / s_tot if s_tot > 0 else 0.0
                sat_ratios = {
                    c: float(
                        data[i, cond_cols[c]] * p_bar * 1e6
                        / sat_p_dyne(c, t_k)
                    )
                    for c in S_CONDENSABLES
                }
                if lnZ == 0.0:
                    baseline_x[(p_bar, t_k, c_o)] = x_h2s
                results.append(
                    dict(lnZ=lnZ, c_o=c_o, P_bar=p_bar, T_K=t_k,
                         x_H2S=x_h2s, S_tot_gas=s_tot, f_H2S_gas=f,
                         sat_ratios=sat_ratios,
                         max_sat_ratio=max(sat_ratios.values()),
                         n_nonconverged_in_run=n_bad)
                )

    # Candidate pin formulas (both prototype-grade; see module docstring).
    r_he = 10.0 ** (HE_DEX - 12.0)
    for r in results:
        r_s = BASE["S"] * np.exp(r["lnZ"])
        r["x_candA"] = r_s / (0.5 + r_he)
        r["errA"] = r["x_candA"] / r["x_H2S"] - 1.0 if r["x_H2S"] > 0 else np.inf
        xb = baseline_x.get((r["P_bar"], r["T_K"], r["c_o"]))
        r["x_candB"] = xb * np.exp(r["lnZ"]) if xb is not None else np.nan
        r["errB"] = r["x_candB"] / r["x_H2S"] - 1.0 if r["x_H2S"] > 0 else np.inf

    OUT_JSON.write_text(json.dumps(
        {"provenance": provenance(abundance_hashes),
         "n_nonconverged_total": total_bad,
         "results": results}, indent=1))

    # ---- report -------------------------------------------------------------
    core = [r for r in results if r["P_bar"] == 7.6]
    print(f"{len(results)} nodes total; {len(core)} in the documented domain "
          f"(P = 7.6 bar); {total_bad} nonconverged FastChem nodes\n")

    print("GAS-PHASE f_H2S (H2S share of gas-phase S atoms) and worst")
    print("S-condensable saturation ratio, P = 7.6 bar, worst over (lnZ, c_o):")
    print(f"{'T (K)':>7} {'min f_gas':>10} {'max r_sat':>12}  "
          f"{'argmax-r cond':>13}")
    for t in T_GRID:
        sub = [r for r in core if r["T_K"] == t]
        worst_f = min(sub, key=lambda r: r["f_H2S_gas"])
        worst_r = max(sub, key=lambda r: r["max_sat_ratio"])
        which = max(worst_r["sat_ratios"], key=worst_r["sat_ratios"].get)
        print(f"{t:7.0f} {worst_f['f_H2S_gas']:10.4f} "
              f"{worst_r['max_sat_ratio']:12.3e}  {which:>13}")

    sat_ok = [t for t in T_GRID
              if max(r["max_sat_ratio"] for r in core if r["T_K"] == t) < 1.0]
    print(f"\nT nodes where NO sulfur condensate is saturated at the bottom "
          f"node\n(gas-only = complete sulfur equilibrium): "
          f"{[int(t) for t in sat_ok]}")

    wc = min(core, key=lambda r: r["f_H2S_gas"])
    print(f"\nWorst gas-phase node in documented domain: f = "
          f"{wc['f_H2S_gas']:.4f} at T={wc['T_K']:.0f} K, "
          f"lnZ={wc['lnZ']:+.3f}, c_o={wc['c_o']:+.2f}")
    marg = [r for r in results if r["P_bar"] != 7.6]
    wm = min(marg, key=lambda r: r["f_H2S_gas"])
    print(f"Worst margin node (P=1/76 bar): f = {wm['f_H2S_gas']:.4f} at "
          f"P={wm['P_bar']} bar, T={wm['T_K']:.0f} K, lnZ={wm['lnZ']:+.3f}, "
          f"c_o={wm['c_o']:+.2f}")

    print("\nPROTOTYPE pin-formula errors vs FastChem x_H2S (P = 7.6 bar, "
          "T <= 1600 K):")
    dom = [r for r in core if r["T_K"] <= 1600.0]
    for tag in ("errA", "errB"):
        errs = np.array([r[tag] for r in dom])
        print(f"  {tag}: max |err| = {np.max(np.abs(errs)):.4f}, "
              f"median = {np.median(errs):+.4f}, "
              f"range [{np.min(errs):+.4f}, {np.max(errs):+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
