"""B0A: differentiable equilibrium lookup for the deep H2S boundary (D3b).

Builds the PRODUCTION boundary-condition table proposed in the revised B0A
record (collaborator review round 3): ln x_H2S on a (T_bottom, lnZ, c_o)
grid at the engine bottom pressure P_b = 7.6 bar, computed with the
VULCAN-vendored gas-phase FastChem (valid as the complete sulfur equilibrium
at the boundary node: the companion sweep measured max S-condensable
saturation ratio ~3e-9 there, so condensation is thermodynamically
irrelevant at the node).

Then prototypes the interpolant the JAX port must reproduce — TRILINEAR
interpolation of ln x_H2S over (T, lnZ, c_o) — and validates, at seeded
off-node points:
  - interpolated VALUES against direct FastChem evaluations, and
  - interpolant PARTIAL DERIVATIVES d ln x / d{T, lnZ, c_o} against centered
    finite differences of FastChem itself (the equilibrium truth, not the
    table).

Outputs h2s_boundary_table.json: grids, ln-x table, validation report, and
full provenance (checksums, platform, grid, FastChem identity/convergence).
The B1 JAX port must interpolate THIS table (checksum-gated) with the same
trilinear rule; d/dT and d/d(c_o) through the boundary are then nonzero by
construction, unlike the anchor-scale exp(lnZ) prototype.

Same private-FastChem-copy requirement as h2s_dominance_sweep.py
(H2S_CHECK_FASTCHEM_DIR).
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

FC_DIR = Path(
    os.environ.get(
        "H2S_CHECK_FASTCHEM_DIR", Path(__file__).parent / "fastchem_check"
    )
)
OUT_JSON = Path(__file__).parent / "h2s_boundary_table.json"

LOG10E = np.log10(np.e)
BASE = {"O": 5.37e-3, "C": 2.95e-3, "N": 7.08e-4, "S": 1.41e-4}
HE_DEX = 10.9232
ROCKY = ["P", "Si", "Ti", "V", "Cl", "K", "Na", "Mg", "F", "Ca", "Fe"]
ROCKY_DEX = -2.0

P_BAR = 7.6  # engine bottom pressure (vulcan_cfg_W39b P_b = 7.6e6 dyne/cm^2)
T_GRID = np.arange(400.0, 2000.0 + 1e-9, 100.0)          # 17
LNZ_GRID = np.linspace(-2.303, 2.303, 9)                  # 9
CO_GRID = np.linspace(-1.70, 0.50, 7)                     # 7

# Validation: off-node points + FD steps against FastChem truth.
N_VAL = 16
VAL_SEED = 20260713
H_T = 25.0
H_LNZ = 0.06
H_CO = 0.06


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_abundances(lnZ: float, c_o: float) -> None:
    # Leading comment line is mandatory: FastChem's reader consumes the first
    # line of this file as a header (H would be silently dropped).
    lines = ["# element abundances (dex, H=12)", "H\t12.0000", f"He\t{HE_DEX:.4f}"]
    for el, base in BASE.items():
        dex = 12.0 + np.log10(base) + lnZ * LOG10E
        if el == "C":
            dex += c_o * LOG10E
        lines.append(f"{el}\t{dex:.4f}")
    for el in ROCKY:
        lines.append(f"{el}\t{ROCKY_DEX:.4f}")
    lines.append("e-\t0")
    (FC_DIR / "input" / "element_abundances_vulcan.dat").write_text(
        "\n".join(lines) + "\n"
    )


def run_h2s(T_values: np.ndarray) -> np.ndarray:
    """Run FastChem at P_BAR over T_values; return x_H2S per T row."""
    body = "#p (bar)    T (K)\n" + "\n".join(
        f"{P_BAR:.3e}\t{t:.3f}" for t in T_values
    )
    (FC_DIR / "input" / "vulcan_TP" / "vulcan_TP.dat").write_text(body)
    subprocess.run(
        ["./fastchem", "input/config.input"],
        cwd=FC_DIR, check=True, capture_output=True, text=True,
    )
    raw = (FC_DIR / "output" / "vulcan_EQ.dat").read_text().strip().splitlines()
    header = raw[0].split()
    h2s_col = 5 + header[5:].index("H2S")
    mon = (FC_DIR / "output" / "monitor_output.dat").read_text().strip().splitlines()
    for ln in mon[1:]:
        f = ln.split()
        if any(fl != "ok" for fl in [f[2]] + f[8:]):
            raise RuntimeError(f"nonconverged FastChem node: {ln[:120]}")
    return np.array([float(ln.split()[h2s_col]) for ln in raw[1:]])


def build_table() -> np.ndarray:
    """ln x_H2S with shape (n_T, n_lnZ, n_co); one FastChem run per comp."""
    tab = np.empty((T_GRID.size, LNZ_GRID.size, CO_GRID.size))
    for j, lnZ in enumerate(LNZ_GRID):
        for k, c_o in enumerate(CO_GRID):
            write_abundances(lnZ, c_o)
            tab[:, j, k] = np.log(run_h2s(T_GRID))
    return tab


def _axis_cell(grid: np.ndarray, v: float) -> tuple:
    i = int(np.clip(np.searchsorted(grid, v) - 1, 0, grid.size - 2))
    w = (v - grid[i]) / (grid[i + 1] - grid[i])
    return i, w


def interp_lnx(tab: np.ndarray, T: float, lnZ: float, c_o: float) -> tuple:
    """Trilinear ln-x interpolation + analytic partials (the rule the JAX
    port must reproduce). Returns (lnx, d/dT, d/dlnZ, d/dc_o)."""
    i, u = _axis_cell(T_GRID, T)
    j, v = _axis_cell(LNZ_GRID, lnZ)
    k, w = _axis_cell(CO_GRID, c_o)
    c = tab[i:i + 2, j:j + 2, k:k + 2]
    wu = np.array([1 - u, u])
    wv = np.array([1 - v, v])
    ww = np.array([1 - w, w])
    lnx = np.einsum("a,b,c,abc->", wu, wv, ww, c)
    du = np.einsum("b,c,bc->", wv, ww, c[1] - c[0]) / (T_GRID[i + 1] - T_GRID[i])
    dv = np.einsum("a,c,ac->", wu, ww, c[:, 1] - c[:, 0]) / (
        LNZ_GRID[j + 1] - LNZ_GRID[j])
    dw = np.einsum("a,b,ab->", wu, wv, c[:, :, 1] - c[:, :, 0]) / (
        CO_GRID[k + 1] - CO_GRID[k])
    return float(lnx), float(du), float(dv), float(dw)


def validate(tab: np.ndarray) -> dict:
    rng = np.random.default_rng(VAL_SEED)
    pts = np.column_stack([
        rng.uniform(500.0, 1900.0, N_VAL),
        rng.uniform(-2.0, 2.0, N_VAL),
        rng.uniform(-1.5, 0.35, N_VAL),
    ])
    rows = []
    for T, lnZ, c_o in pts:
        # Truth: center + T-FD from one run; lnZ/c_o FDs from 4 more runs.
        write_abundances(lnZ, c_o)
        x0 = run_h2s(np.array([T - H_T, T, T + H_T]))
        write_abundances(lnZ - H_LNZ, c_o)
        xzm = run_h2s(np.array([T]))[0]
        write_abundances(lnZ + H_LNZ, c_o)
        xzp = run_h2s(np.array([T]))[0]
        write_abundances(lnZ, c_o - H_CO)
        xcm = run_h2s(np.array([T]))[0]
        write_abundances(lnZ, c_o + H_CO)
        xcp = run_h2s(np.array([T]))[0]
        truth = dict(
            lnx=float(np.log(x0[1])),
            dT=float((np.log(x0[2]) - np.log(x0[0])) / (2 * H_T)),
            dlnZ=float((np.log(xzp) - np.log(xzm)) / (2 * H_LNZ)),
            dco=float((np.log(xcp) - np.log(xcm)) / (2 * H_CO)),
        )
        lnx, dT, dlnZ, dco = interp_lnx(tab, T, lnZ, c_o)
        rows.append(dict(
            T_K=float(T), lnZ=float(lnZ), c_o=float(c_o), truth=truth,
            interp=dict(lnx=lnx, dT=dT, dlnZ=dlnZ, dco=dco),
            err=dict(
                value_rel=float(np.expm1(lnx - truth["lnx"])),
                dT_abs=float(dT - truth["dT"]),
                dT_truth=truth["dT"],
                dlnZ_abs=float(dlnZ - truth["dlnZ"]),
                dlnZ_truth=truth["dlnZ"],
                dco_abs=float(dco - truth["dco"]),
                dco_truth=truth["dco"],
            ),
        ))
    return dict(points=rows, seed=VAL_SEED,
                fd_steps=dict(T=H_T, lnZ=H_LNZ, c_o=H_CO))


def main() -> int:
    if not (FC_DIR / "fastchem").is_file():
        raise RuntimeError(
            f"FastChem binary not found under {FC_DIR}; see module docstring."
        )
    tab = build_table()
    val = validate(tab)

    params = FC_DIR / "input" / "parameters.dat"
    referenced = [ln.strip() for ln in params.read_text().splitlines()
                  if ln.strip().endswith(".dat")]
    sha = {
        "fastchem_binary": sha256_file(FC_DIR / "fastchem"),
        "parameters.dat": sha256_file(params),
        "config.input": sha256_file(FC_DIR / "input" / "config.input"),
    }
    for rel in referenced:
        f = FC_DIR / rel
        if f.is_file():
            sha[rel] = sha256_file(f)

    out = {
        "description": (
            "ln x_H2S(T_bottom, lnZ, c_o) at P_b = 7.6 bar; gas-phase "
            "FastChem equilibrium (complete for sulfur at this node: "
            "companion sweep max S-condensable saturation ratio ~3e-9). "
            "Interpolation rule: trilinear in (T, lnZ, c_o) on ln x. "
            "The B1 JAX port must reproduce this rule and gate on the "
            "sha256 of this file's 'lnx_table' block."
        ),
        "P_bar": P_BAR,
        "T_K": T_GRID.tolist(),
        "lnZ": LNZ_GRID.tolist(),
        "c_o": CO_GRID.tolist(),
        "lnx_table": tab.tolist(),
        "validation": val,
        "provenance": {
            "fastchem_identity": (
                "exoclime FastChem, (C) 2019 Kitzmann & Stock, gas-phase, "
                "vendored in VULCAN-JAX fastchem_vulcan"
            ),
            "condensation_enabled": False,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "sha256": sha,
            "baseline_X_H": BASE,
            "He_dex_fixed": HE_DEX,
            "rocky_dex": ROCKY_DEX,
            "nonconvergence_policy": "hard raise on any non-ok monitor flag",
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))

    # ---- report -------------------------------------------------------------
    e = [r["err"] for r in val["points"]]
    vrel = np.array([x["value_rel"] for x in e])
    print(f"table: {tab.shape} nodes at P = {P_BAR} bar; "
          f"{N_VAL} off-node validation points (seed {VAL_SEED})")
    print(f"value:      max |rel err| = {np.max(np.abs(vrel)):.4f}, "
          f"median = {np.median(vrel):+.5f}")
    for key, hstep, unit in (("dT", H_T, "1/K"), ("dlnZ", H_LNZ, "1/lnZ"),
                             ("dco", H_CO, "1/c_o")):
        a = np.array([x[f"{key}_abs"] for x in e])
        t = np.array([x[f"{key}_truth"] for x in e])
        scale = np.maximum(np.abs(t), np.median(np.abs(t)))
        print(f"d lnx/{key:>5}: truth range [{t.min():+.3e}, {t.max():+.3e}] "
              f"{unit}; max |err| = {np.max(np.abs(a)):.3e}; "
              f"max |err|/scale = {np.max(np.abs(a) / scale):.4f}")
    print(f"\nwrote {OUT_JSON.name} "
          f"(lnx_table sha256 {hashlib.sha256(json.dumps(tab.tolist()).encode()).hexdigest()[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
