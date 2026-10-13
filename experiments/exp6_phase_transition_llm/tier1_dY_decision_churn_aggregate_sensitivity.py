"""AUDIT-ONLY sensitivity check on the aggregation semantics of
`tier1_dY_decision_churn_aggregate.py` (2026-10-12). Does NOT touch, rerun, or overwrite that
script or its output; reads the same frozen `tier1_dY_decision_churn.json` and reports a
SECOND aggregate next to the frozen one, for comparison only. Not used to pick a headline.

QUESTION BEING AUDITED. `tier1_dY_decision_churn_aggregate.py`'s `dC_mean` is
`np.array([x["dC"] for x in g]).mean()` — an EQUAL-WEIGHT average over the K (model,
benchmark, regime) strata, each stratum counted once regardless of how many held-out
questions fed its point estimate. Strata are unequal in `n_q` (confirmed: easy 15-62,
hard 16-55), so this differs from a question-count-weighted average whenever the two are
correlated with the stratum's dC value. The two estimands are NOT the same population
quantity:

    macro    = (1/K) * sum_k dC_k                              -- average-STRATUM effect
    weighted = (sum_k n_k * dC_k) / (sum_k n_k)                -- average-QUESTION effect

Both resample STRATA (not questions) with replacement for the bootstrap, matching the
paper's own declared convention ("for cell-level summaries we resample cells, since cells
rather than questions are the unit of replication across models and benchmarks" —
paper.md, Statistical Analysis). They differ ONLY in how a fixed resampled multiset of
strata is averaged (plain mean vs n_q-weighted mean), never in which unit is treated as
exchangeable. No per-question value is ever resampled as if it were an independent
observation in either estimator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tier1_dY_decision_churn import N_BOOT, SEED  # noqa: E402  frozen constants, imported

ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
SRC = TABLES / "tier1_dY_decision_churn.json"


def macro_and_weighted(dC: np.ndarray, n: np.ndarray, rng: np.random.Generator):
    K = dC.size
    idx = rng.integers(0, K, size=(N_BOOT, K))          # resample STRATA, same unit both ways
    macro_draws = dC[idx].mean(1)
    weighted_draws = (dC[idx] * n[idx]).sum(1) / n[idx].sum(1)
    macro_pt, weighted_pt = float(dC.mean()), float((dC * n).sum() / n.sum())

    def ci(draws):
        lo, hi = np.quantile(draws, [0.025, 0.975])
        return (float(lo), float(hi)) if hi - lo > 1e-12 else (float("nan"), float("nan"))
    return (macro_pt, ci(macro_draws)), (weighted_pt, ci(weighted_draws))


def main() -> None:
    blob = json.loads(SRC.read_text())
    cells = blob["cells"]
    rng = np.random.default_rng(SEED)

    print("=== SENSITIVITY: macro (equal-weight-per-stratum) vs question-weighted dC ===")
    print("Both resample STRATA for the bootstrap; they differ only in the averaging weight.\n")
    out = {"source": str(SRC.relative_to(ROOT)), "n_boot": N_BOOT, "seed": SEED,
          "purpose": "sensitivity check only; frozen headline stays the macro aggregate "
                      "in tier1_dY_decision_churn_aggregate.json", "rows": []}
    for tag in ("strict", "tiebreak"):
        for regime in ("easy", "hard"):
            g = [c[tag] for c in cells if c["regime"] == regime]
            dC = np.array([x["dC"] for x in g], float)
            n = np.array([x["n_q"] for x in g], float)
            (m_pt, m_ci), (w_pt, w_ci) = macro_and_weighted(dC, n, rng)
            print(f"[{tag}/{regime}] K={len(g)}  n_q range [{int(n.min())},{int(n.max())}] "
                  f"sum={int(n.sum())}")
            print(f"    macro (equal-weight-per-stratum): dC={m_pt:+.4f}  "
                  f"95% CI [{m_ci[0]:+.4f},{m_ci[1]:+.4f}]")
            print(f"    weighted (question-count-weighted): dC={w_pt:+.4f}  "
                  f"95% CI [{w_ci[0]:+.4f},{w_ci[1]:+.4f}]")
            print(f"    agree in sign & both exclude 0: "
                  f"{bool((m_ci[0] > 0 or m_ci[1] < 0) and (w_ci[0] > 0 or w_ci[1] < 0) and np.sign(m_pt) == np.sign(w_pt))}\n")
            out["rows"].append({"regime": regime, "tie_convention": tag, "n_strata": len(g),
                                "n_q_total": int(n.sum()), "n_q_range": [int(n.min()), int(n.max())],
                                "macro": {"dC": m_pt, "ci": list(m_ci)},
                                "weighted": {"dC": w_pt, "ci": list(w_ci)}})
    dst = TABLES / "tier1_dY_decision_churn_aggregate_sensitivity.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
