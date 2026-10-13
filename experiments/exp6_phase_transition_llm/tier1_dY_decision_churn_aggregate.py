"""Aggregate the FROZEN per-stratum churn results (tier1_dY_decision_churn.json) into one
regime-level headline. Offline, zero GPU, zero new generation, no new experiment.

WHY THIS EXISTS (2026-10-12, after the churn-audit). `tier1_dY_decision_churn.py` reports
ΔY_cross and dC per stratum, but the manuscript needs one number per regime with uncertainty,
not a "k/n significant" tally over 35 strata. This script does nothing but pool the already-
frozen per-stratum point estimates: it does NOT recompute per-question quantities, does NOT
touch the raw dumps, and reuses `_boot_ci` IMPORTED verbatim from `tier1_dY_decision_churn` —
the exact same bootstrap-of-the-mean procedure already frozen there, applied one level up
(resampling STRATA instead of questions), mirroring the unweighted stratum-bootstrap already
used for H1 in `tier1_dA_regime_test.py`. No alternative estimator, threshold, or significance
rule is introduced.

ΔY_cross is DESCRIPTIVE ONLY here (raw cross-arm churn, inflated by within-arm replicate
noise — see PREREG... no, see the audit note in tier1_dY_decision_churn.py's module
docstring). dC = ΔY_cross − C_within is the INFERENTIAL excess-churn estimand: the only
quantity whose significance is reported as a finding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tier1_dY_decision_churn import N_BOOT, SEED, _boot_ci  # noqa: E402  frozen, imported

ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
SRC = TABLES / "tier1_dY_decision_churn.json"


def aggregate(cells: list[dict], regime: str, tag: str, rng: np.random.Generator) -> dict:
    g = [c[tag] for c in cells if c["regime"] == regime]
    dY = np.array([x["dY"] for x in g], float)         # descriptive only
    dC = np.array([x["dC"] for x in g], float)          # inferential estimand
    Cw = np.array([x["C_within"] for x in g], float)
    dY_lo, dY_hi = _boot_ci(dY, rng)
    dC_lo, dC_hi = _boot_ci(dC, rng)
    return {
        "regime": regime, "tie_convention": tag, "n_strata": len(g),
        "dY_cross_mean": float(dY.mean()), "dY_cross_ci": [dY_lo, dY_hi],
        "C_within_mean": float(Cw.mean()),
        "dC_mean": float(dC.mean()), "dC_ci": [dC_lo, dC_hi],
        "dC_sig": bool(not np.isnan(dC_lo) and (dC_lo > 0 or dC_hi < 0)),
        "n_strata_dC_sig_individually": sum(1 for x in g if x["dC_sig"] and x["dC"] > 0),
    }


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing — run tier1_dY_decision_churn.py first")
    blob = json.loads(SRC.read_text())
    cells = blob["cells"]
    rng = np.random.default_rng(SEED)   # same seed family as the frozen per-cell bootstrap

    rows = []
    print("=== AGGREGATE churn, pooled over strata (stratum-bootstrap, unweighted) ===")
    print("ΔY_cross: descriptive only. dC = ΔY_cross - C_within: the inferential estimand.\n")
    for tag in ("strict", "tiebreak"):
        for regime in ("easy", "hard"):
            r = aggregate(cells, regime, tag, rng)
            rows.append(r)
            tagtxt = "SIGNIFICANT" if r["dC_sig"] else "not significant"
            print(f"[{tag}/{regime}] n_strata={r['n_strata']}  "
                  f"ΔY_cross={r['dY_cross_mean']:.3f} (descriptive)  "
                  f"C_within={r['C_within_mean']:.3f}  "
                  f"dC={r['dC_mean']:+.3f} 95% CI [{r['dC_ci'][0]:+.3f},{r['dC_ci'][1]:+.3f}] "
                  f"{tagtxt}  (individually-sig strata: "
                  f"{r['n_strata_dC_sig_individually']}/{r['n_strata']}, supporting detail only)")

    out = {"source": str(SRC.relative_to(ROOT)), "n_boot": N_BOOT, "seed": SEED,
          "note": "dY_cross is descriptive; dC is the inferential excess-churn estimand",
          "rows": rows}
    dst = TABLES / "tier1_dY_decision_churn_aggregate.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
