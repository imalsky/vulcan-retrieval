"""Assemble the single B0C reproducibility artifact (directive M / plan 9).

Collects the LATEST artifact of every gate from results/, verifies they
share one provenance (same repo SHAs, same lookup-table checksum, same
network hash -- mixed-provenance campaigns are REFUSED, not merged),
extracts per-gate verdicts, and writes one self-contained record:

    docs/route_b/b0c_reproducibility_artifact.json
    docs/route_b/b0c_reproducibility_artifact.txt   (human summary)

Go/no-go logic (directive N): B1 is authorized ONLY if every ESSENTIAL gate
(G1, G2, G3, G4, G5a, G6, spectrum-FD) carries a PASS/MEASURED verdict; any
FAIL, TAINTED, INCOMPLETE, or missing artifact keeps the campaign OPEN,
gates are never weakened, and Fisher with condensation stays disabled.
G5b is characterization (never blocks). The artifact names the exact
validated preset (the fixture config) and the documented thresholds in
force; "no vague config classes" -- what is validated is exactly what ran.

Run any time (cheap, no solves): python docs/route_b/harness/b0c_artifact.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as gc  # noqa: E402

OUT_JSON = gc.HARNESS_DIR.parent / "b0c_reproducibility_artifact.json"
OUT_TXT = gc.HARNESS_DIR.parent / "b0c_reproducibility_artifact.txt"

GATES = {
    "G1": ("w107b_g1_*.json", True),
    "G2": ("w107b_g2_*.json", True),
    "G3": ("w107b_g3_*.json", True),
    "G4": ("w107b_g4_*.json", True),
    "G5": ("w107b_g5_*.json", True),          # G5a essential, G5b characterize
    "G6": ("w107b_g6_*.json", True),
    "spectrum_fd": ("w107b_spectrum_fd_*.json", True),
    "spectrum_agreement_g2": ("w107b_spectrum_w107b_g2_*.json", False),
    "spectrum_agreement_g5": ("w107b_spectrum_w107b_g5_*.json", False),
}

PROVENANCE_IDENTITY = ("vulcan_retrieval_sha", "vulcan_jax_sha",
                       "network_sha256", "h2s_table_sha256")


def latest(pattern: str) -> Path | None:
    hits = sorted(gc.RESULTS_DIR.glob(pattern))
    return hits[-1] if hits else None


def verdict_of(gate: str, payload: dict) -> str:
    if gate == "G5":
        return str(payload.get("verdict_G5a", "MISSING"))
    return str(payload.get("verdict", "MISSING"))


def is_passing(verdict: str) -> bool:
    return verdict.startswith("PASS") or verdict.startswith("MEASURED")


def main():
    collected = {}
    prov_identity = None
    prov_mixed = []
    for gate, (pattern, essential) in GATES.items():
        path = latest(pattern)
        if path is None:
            collected[gate] = {"artifact": None, "verdict": "MISSING",
                               "essential": essential}
            continue
        payload = json.loads(path.read_text())
        prov = payload.get("provenance", {})
        ident = tuple(prov.get(k) for k in PROVENANCE_IDENTITY)
        if prov_identity is None and all(ident):
            prov_identity = ident
        elif all(ident) and ident != prov_identity:
            prov_mixed.append((gate, path.name))
        collected[gate] = {
            "artifact": path.name,
            "verdict": verdict_of(gate, payload),
            "essential": essential,
            "timestamp_utc": prov.get("timestamp_utc"),
        }
        if gate == "G5":
            collected[gate]["verdict_G5b"] = payload.get("verdict_G5b")

    if prov_mixed:
        raise SystemExit(
            f"REFUSED: mixed provenance across gate artifacts {prov_mixed} "
            "(different repo SHAs / table checksums). Re-run the stale gates "
            "on the current tree; a reproducibility artifact must describe "
            "ONE campaign.")

    essential_open = [g for g, v in collected.items()
                      if v["essential"] and not is_passing(v["verdict"])]
    if essential_open:
        go = ("NO-GO: B1 NOT authorized. Open essential gates: "
              f"{essential_open}. Gates are never weakened; Fisher with "
              "condensation stays DISABLED.")
    else:
        go = ("ALL ESSENTIAL GATES PASSING (as documented): B1 may be "
              "proposed to Isaac + collaborator for sign-off. Fisher with "
              "condensation stays DISABLED until the B1 eta_C guard passes "
              "at the enabled preset.")

    prov_now = gc.provenance()
    artifact = {
        "artifact": "Route B B0C reproducibility artifact (directive M)",
        "campaign_provenance_identity": dict(zip(PROVENANCE_IDENTITY,
                                                 prov_identity or ())),
        "assembled": prov_now,
        "gates": collected,
        "go_no_go": go,
        "validated_preset": {
            "fixture": "docs/route_b/harness/w107b_fixture.py "
                       "(build(); the EXACT cfg_overrides dict in each gate "
                       "artifact's fixture block is the validated preset -- "
                       "no other configuration is claimed)",
            "forward_only": "inference refused unconditionally "
                            "(validate_config); unrolled jvp diagnostic-only",
        },
        "standing_constraints": [
            "B1 not authorized until every essential gate passes",
            "Fisher with condensation stays disabled",
            "gates are never weakened to accommodate a failure",
            "any FAIL: stop, report measured numbers, collaborator review",
        ],
    }
    OUT_JSON.write_text(json.dumps(artifact, indent=1))

    lines = ["ROUTE B B0C REPRODUCIBILITY ARTIFACT",
             "=" * 60,
             f"assembled: {prov_now['timestamp_utc']}",
             f"vulcan-retrieval: {prov_now['vulcan_retrieval_sha']}",
             f"VULCAN-JAX:       {prov_now['vulcan_jax_sha']}",
             f"lookup table:     {prov_now['h2s_table_sha256'][:16]}...",
             f"network:          {prov_now['network_sha256'][:16]}...",
             ""]
    for g, v in collected.items():
        tag = "ESSENTIAL" if v["essential"] else "informational"
        lines.append(f"{g:24s} [{tag}] {v['verdict']}")
        if v["artifact"]:
            lines.append(f"{'':24s} artifact: {v['artifact']}")
        if v.get("verdict_G5b"):
            lines.append(f"{'':24s} G5b: {v['verdict_G5b']}")
    lines += ["", go, ""]
    OUT_TXT.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written: {OUT_JSON}\n         {OUT_TXT}")


if __name__ == "__main__":
    main()
