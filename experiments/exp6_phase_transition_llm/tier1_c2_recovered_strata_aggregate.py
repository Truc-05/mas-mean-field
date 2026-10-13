"""C2 (susceptibility) macro-average over the recovered split-sample strata. Offline, zero
GPU, zero new metric. Pools the already-frozen per-stratum `c2_slope` column written by
`tier1_split_sample_strata.py` (OLS slope of |ΔA| on the baseline margin M, bootstrapped
within-stratum over questions) — does not recompute, does not touch the raw dumps.

ESTIMAND, FIXED BEFORE INSPECTING OUTPUT (2026-10-12). Mirrors the `dC` macro-average:
    beta_bar_regime = (1/K_regime) * sum_k c2_slope_k
equal weight per (model, benchmark) stratum, regardless of question count, for the same
reason as `dC`: the claim is whether the susceptibility relationship generalizes across
model-benchmark configurations, not what it averages to over this project's particular mix
of benchmark sizes. Uncertainty is a stratum-level bootstrap (resample strata, not questions
-- strata are the unit of replication, matching `tier1_dY_decision_churn_aggregate.py` and
paper.md's own declared Statistical Analysis convention). No alternative susceptibility
metric, no question-weighting, no new gate is introduced.

PRIMARY REGIME IS HARD, NOT POOLED, BY A PRE-EXISTING RULE (README, "STRATIFICATION DEFECT +
REPAIR", 2026-10-01/2026-10-03) -- NOT a choice made after seeing this script's output: the
high-competence ("easy") stratum is near-saturated (A_none approx 1.0 in most cells), so
|DeltaA| is identically 0 for every question in many easy strata and the OLS slope is then
forced to exactly 0 by construction (zero-variance y), not because the data show no
susceptibility relationship. "C1/C2 quoted off hard ... never averaged" predates this run.
We therefore report easy only as diagnostic context, flagging the exact-zero degeneracy, and
never fold it into the primary estimate.

Claim license: a negative macro-average slope licenses "lower collective decision margin is
associated with greater susceptibility to dependence perturbations" -- an association, not a
causal or uniqueness claim (margin is correlated with vote entropy/disagreement; see
`analysis_A_c2_controls.py`). Does not license "margin causes susceptibility" or "margin
uniquely predicts susceptibility".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
SRC = TABLES / "tier1_split_sample_strata.json"
EXCLUDED_BENCHMARKS = {"bbh"}
N_BOOT = 20000          # matches tier1_dA_regime_test.py's stratum-bootstrap scale
SEED = 20261012


def load_c2(regime: str) -> tuple[np.ndarray, list[str], int]:
    blob = json.loads(SRC.read_text())
    vals, skipped, n_zero = [], [], 0
    for cell in blob["cells"]:
        model, bench = cell["model"], cell["benchmark"]
        if bench in EXCLUDED_BENCHMARKS:
            skipped.append(f"{model}/{bench}: excluded benchmark")
            continue
        st = cell["strata"].get(regime, {})
        if st.get("insufficient") or st.get("c2_slope") is None:
            skipped.append(f"{model}/{bench}/{regime}: insufficient (n_q={st.get('n_q')})")
            continue
        v = float(st["c2_slope"])
        if abs(v) < 1e-12:
            n_zero += 1
        vals.append(v)
    return np.array(vals, float), skipped, n_zero


def macro_bootstrap(vals: np.ndarray, rng: np.random.Generator) -> dict:
    K = vals.size
    idx = rng.integers(0, K, size=(N_BOOT, K))
    draws = vals[idx].mean(1)
    lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    pt = float(vals.mean())
    return {"K": K, "beta_bar": pt, "ci": [lo, hi],
            "sig": bool(hi - lo > 1e-12 and (lo > 0 or hi < 0)),
            "n_negative": int((vals < 0).sum()), "n_positive": int((vals > 0).sum()),
            "n_exact_zero": int((np.abs(vals) < 1e-12).sum())}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing — run tier1_split_sample_strata.py first")
    rng = np.random.default_rng(SEED)

    out = {"estimand": "beta_bar = (1/K) sum_k c2_slope_k, equal-weight per stratum, "
                       "stratum-level bootstrap", "n_boot": N_BOOT, "seed": SEED,
          "primary_regime": "hard", "regimes": {}}
    print("=== C2 macro-average, |dA| ~ M slope, pooled over recovered strata ===\n")
    for regime in ("hard", "easy"):
        vals, skipped, n_zero = load_c2(regime)
        r = macro_bootstrap(vals, rng)
        r["skipped"] = skipped
        r["n_exact_zero_from_degenerate_dA"] = n_zero
        out["regimes"][regime] = r
        tag = "PRIMARY" if regime == "hard" else "CONTEXT ONLY — near-saturated, not pooled"
        print(f"[{regime}] ({tag})")
        print(f"    K={r['K']} strata  beta_bar={r['beta_bar']:+.4f}  "
              f"95% CI [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
              f"{'excludes 0' if r['sig'] else 'covers 0'}")
        print(f"    sign split: {r['n_negative']} negative / {r['n_positive']} positive / "
              f"{r['n_exact_zero']} exactly zero (of which {n_zero} forced by |dA|==0 "
              f"saturation in every question of that stratum)")
        if skipped:
            print(f"    excluded ({len(skipped)}): " + "; ".join(skipped))
        print()

    dst = TABLES / "tier1_c2_recovered_strata_aggregate.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
