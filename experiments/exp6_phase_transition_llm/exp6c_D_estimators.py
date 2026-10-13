"""AUDIT ONLY — legacy per-question D vs the canonical pooled ratio, side by side.

This script holds NO estimator of its own. Every number below comes from
`dependence_estimators` (one canonical implementation, imported by both this and the primary
analysis). It makes NO estimability decision: the `n_eff` gate lives in
`tier1_split_sample_strata.py` and only there.

WHAT IT IS FOR. It is the reproducible audit trail for an estimator change that moved a
conclusion, so it is kept rather than folded away:

  * The frozen per-question estimator `exp6c_shared_context.pooled_pair_corr` returns
    `num_q/den_q` and NaN when `den_q <= 1e-12`. On a saturated stratum that discards most
    questions -- and the surviving subset is selected on error variance, which is the very
    quantity D measures.
  * Worse, and silently: when only two agents vary, `num_q` and `den_q` are both nearly zero
    and the ratio collapses onto ±1 exactly. Across the 17 completed probe-era cells, ALL 27
    question-arms with `den_q < 0.3` returned `|D_q| = 1.000`. They sit mostly at high margin,
    inflating median `|D_q|` to 0.534 at margin == 1.0 against 0.14-0.27 in the lower bands --
    which reads as "dependence is strongest exactly where the crowd is unanimous", a striking
    dissociation against the near-zero `|dA|` there. It is an artifact of dividing two numbers
    that are both nearly zero. That apparent finding was dropped as a result.

PROVENANCE. The discovery-era output is frozen at
`results/tables/PROVENANCE_exp6c_D_estimators_discovery_20261003.json` and must not be
overwritten -- it is the record of the run that caught the defect. This script writes to a
separate, versioned file.

SUBSTRATE CAVEAT. This audit reads the per-cell `exp6c_boundary_*` dumps, which carry the
RETIRED probe-based easy/hard labels (`p2_probe.py:56` and `exp6c_boundary.py:239` disagree on
the display-shuffle `qi` convention). Treat the regime split here as an estimator comparison
only. Any easy/hard claim for the manuscript comes from the recovered strata in
`tier1_split_sample_strata.py`; correlations computed on probe labels are SUPERSEDED and must
not be reused.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from dependence_estimators import (DEGENERATE_DEN, ESTIMATOR_VERSION, boot_pooled_ratio,
                                   legacy_D, n_eff, pooled_D, terms_from_states)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
PROCESSED = ROOT / "data" / "processed"

SHARED = "shared_correct_strong"
INDEP = "indep_correct_strong"
EXCLUDED_BENCHMARKS = {"bbh"}
MIN_NEFF_DISPLAY = 15      # shown for orientation; the binding gate is in the primary script

CELL_RE = re.compile(r"exp6c_boundary_(easy|hard)_scaleup_([a-z0-9]+)_([a-z]+)_N(\d+)\.json$")


def load() -> list[dict]:
    """One record per question: both arms' (num, den), the margin, and dA."""
    out = []
    for path in sorted(TABLES.glob("exp6c_boundary_*_scaleup_*_N*.json")):
        m = CELL_RE.search(path.name)
        if not m:
            continue
        regime, model, bench, N = m[1], m[2], m[3], m[4]
        raw_p = PROCESSED / f"exp6c_raw_{regime}_scaleup_{model}_{bench}_N{N}.json"
        if not raw_p.exists():
            continue
        raw = json.loads(raw_p.read_text())["raw"]
        if SHARED not in raw or INDEP not in raw:
            continue
        cell = json.loads(path.read_text())
        for q in cell["per_q"]:
            qi = q["qi"]
            if qi >= len(raw[SHARED]) or qi >= len(raw[INDEP]):
                continue
            ns, ds = terms_from_states(raw[SHARED][qi])
            ni, di = terms_from_states(raw[INDEP][qi])
            A_s, A_i = q[SHARED].get("A_q"), q[INDEP].get("A_q")
            out.append({"model": model, "bench": bench, "regime": regime,
                        "excluded": bench in EXCLUDED_BENCHMARKS, "qi": qi,
                        "M": q["margin"]["runnerup_gap"],
                        "num_s": ns, "den_s": ds, "num_i": ni, "den_i": di,
                        "dA": (None if A_s is None or A_i is None else A_s - A_i)})
    return out


def report(rows, rng, n_boot: int) -> dict:
    live = [r for r in rows if not r["excluded"]]
    cells = sorted({(r["model"], r["bench"], r["regime"]) for r in live})
    print(f"\nestimator: {ESTIMATOR_VERSION}")
    print("AUDIT ONLY — no estimability decision is made here; the n_eff gate is in "
          "tier1_split_sample_strata.py")
    print("substrate: RETIRED probe labels — estimator comparison only, not a regime claim\n")

    hdr = (f"{'model':7} {'bench':14} {'reg':5} | {'n_dD':>4} {'ΔD legacy':>9} | "
           f"{'n_eff':>5} {'ΔD pooled':>9} {'95% CI':>18} {'degen':>5}")
    print(hdr); print("-" * len(hdr))
    out = []
    for model, bench, regime in cells:
        g = [r for r in live if (r["model"], r["bench"], r["regime"]) == (model, bench, regime)]
        leg = [legacy_D(r["num_s"], r["den_s"]) - legacy_D(r["num_i"], r["den_i"])
               for r in g if r["den_s"] > 1e-12 and r["den_i"] > 1e-12]
        lo, hi, pt = boot_pooled_ratio([r["num_s"] for r in g], [r["den_s"] for r in g],
                                       [r["num_i"] for r in g], [r["den_i"] for r in g],
                                       rng, n_boot)
        ne = n_eff([r["den_s"] + r["den_i"] for r in g])
        degen = sum(1 for r in g if 1e-12 < r["den_s"] < DEGENERATE_DEN
                    or 1e-12 < r["den_i"] < DEGENERATE_DEN)
        out.append({"model": model, "bench": bench, "regime": regime,
                    "n_dD_legacy": len(leg),
                    "dD_legacy": float(np.mean(leg)) if leg else float("nan"),
                    "n_q": len(g), "n_eff": ne, "dD_pooled": pt, "ci": [lo, hi],
                    "n_degenerate": degen})
        print(f"{model:7} {bench:14} {regime:5} | {len(leg):4d} "
              f"{(np.mean(leg) if leg else float('nan')):+9.3f} | {ne:5.1f} {pt:+9.3f} "
              f"[{lo:+7.3f},{hi:+7.3f}] {degen:5d}")
    print("-" * len(hdr))
    tot_deg = sum(c["n_degenerate"] for c in out)
    print(f"{tot_deg} question-arms had den<{DEGENERATE_DEN}; under the legacy per-question "
          f"ratio each votes ±1")
    disagree = [c for c in out
                if (c["n_dD_legacy"] >= MIN_NEFF_DISPLAY) != (c["n_eff"] >= MIN_NEFF_DISPLAY)]
    if disagree:
        print(f"\nn_dD and n_eff disagree at the {MIN_NEFF_DISPLAY} mark in {len(disagree)} "
              f"cell(s) — n_q overstates the information behind the ratio:")
        for c in disagree:
            print(f"  {c['model']}/{c['bench']}/{c['regime']}: "
                  f"n_dD={c['n_dD_legacy']} vs n_eff={c['n_eff']:.1f}")

    print("\n=== the dropped dissociation: legacy |D| inflation by margin ===")
    print(f"{'margin band':>13} {'nQ':>4} {'legacy med|D|':>13} {'pooled D':>9} {'n_eff':>6} "
          f"{'den=0':>6} {'degen':>5} {'mean|dA|':>9}")
    print("-" * 76)
    grad = []
    for lo_, hi_ in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .999), (.999, 1.01)]:
        g = [r for r in live if lo_ <= r["M"] < hi_]
        if not g:
            continue
        lg = [abs(legacy_D(r["num_s"], r["den_s"])) for r in g if r["den_s"] > 1e-12]
        da = [abs(r["dA"]) for r in g if r["dA"] is not None]
        row = {"lo": lo_, "hi": hi_, "n": len(g),
               "legacy_median_absD": float(np.median(lg)) if lg else float("nan"),
               "pooled_D": pooled_D([(r["num_s"], r["den_s"]) for r in g]),
               "n_eff": n_eff([r["den_s"] for r in g]),
               "n_den0": sum(1 for r in g if r["den_s"] <= 1e-12),
               "n_degenerate": sum(1 for r in g if 1e-12 < r["den_s"] < DEGENERATE_DEN),
               "mean_absdA": float(np.mean(da)) if da else float("nan")}
        grad.append(row)
        print(f"{lo_:.2f}-{hi_:<7.2f} {row['n']:4d} {row['legacy_median_absD']:13.3f} "
              f"{row['pooled_D']:9.3f} {row['n_eff']:6.1f} {row['n_den0']:6d} "
              f"{row['n_degenerate']:5d} {row['mean_absdA']:9.3f}")
    print("\nRead the last row with care: n_q=241 but n_eff≈20, and 204 of those questions "
          "carry\nzero denominator. The legacy median |D| of 0.534 there is the ±1 artifact, "
          "not a finding.")
    return {"cells": out, "gradient": grad}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--json", default=str(TABLES / "AUDIT_exp6c_D_estimators.json"),
                    help="versioned audit output; the discovery run is frozen under "
                         "PROVENANCE_exp6c_D_estimators_discovery_20261003.json")
    args = ap.parse_args()
    rows = load()
    if not rows:
        print("no raw state dumps found — nothing to audit")
        return
    rng = np.random.default_rng(args.seed)
    print(f"recovered {len(rows)} questions "
          f"({len({(r['model'], r['bench'], r['regime']) for r in rows})} cells) from raw dumps")
    res = report(rows, rng, args.n_boot)
    Path(args.json).write_text(json.dumps({
        "role": "audit-only; no estimability decision; estimators imported from "
                "dependence_estimators",
        "estimator_version": ESTIMATOR_VERSION,
        "substrate": "RETIRED probe easy/hard labels — estimator comparison only",
        "provenance_of_discovery_run":
            "results/tables/PROVENANCE_exp6c_D_estimators_discovery_20261003.json",
        "seed": args.seed, "n_boot": args.n_boot, "degenerate_den": DEGENERATE_DEN, **res,
    }, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
