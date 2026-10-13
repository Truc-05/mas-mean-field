"""AUDIT (2026-10-12): does raw |dA| in C2's `c2_slope` (|dA| ~ margin M) suffer the same
within-arm noise inflation that raw ΔY did before the dC correction? Offline, zero GPU.

THE CONCERN, STATED PRECISELY. `dA = A_shared - A_indep` is a difference of two means, each
unbiased for its arm's true accuracy -- unlike a disagreement RATE (ΔY), a difference of
means is not inflated in EXPECTATION by independent redraws. But C2 regresses |dA|, and by
Jensen's inequality E[|dA|] >= |E[dA]|, with the gap driven by dA's sampling VARIANCE. If that
variance is itself larger for low-margin questions (plausible: a close baseline vote is
plausibly a noisier regime for the plurality decision generally, independent of the
shared/indep manipulation), a NEGATIVE slope of raw |dA| on M can appear even with zero true
susceptibility effect -- the mirror image of C_within inflating raw ΔY.

THE AUDIT. For each question, estimate the within-arm sampling-noise floor for a
difference-of-two-group-means statistic directly: split the same arm's 4 held-out evaluation
replicates into two balanced halves (3 ways to split 4 into 2+2), compute |mean(half1) -
mean(half2)| for each split, and average -- this is exactly the magnitude a dA-shaped
statistic would show from pure replicate noise if shared and indep were literally the same
arm. Call the shared-arm and indep-arm versions diff_s, diff_i;
C2_within = (diff_s + diff_i) / 2 (mirrors C_within's definition exactly). The noise-adjusted
outcome is dA_excess = |dA| - C2_within, and the audited question is whether its regression
slope on M differs materially (sign, magnitude, significance) from raw |dA|'s.

Reuses the canonical loaders/estimators (`discover`, `load_cell`, `phat_split`, `margin_M`,
`boot_slope`) from `tier1_split_sample_strata.py` verbatim -- only the OUTCOME fed to the
already-frozen `boot_slope` changes, not the regression machinery itself.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp6c_shared_context as e6c  # noqa: E402
from mfg_llm.llm_debate.benchmarks import plurality_prediction  # noqa: E402
from tier1_split_sample_strata import (  # noqa: E402  canonical, imported not reimplemented
    EASY_HI, HARD_LO, MIN_Q, R_BIN, boot_slope, discover, load_cell, margin_M, phat_split)

ROOT = HERE.parents[1]
TABLES = ROOT / "results" / "tables"
NONE, SHARED, INDEP = "none", "shared_correct_strong", "indep_correct_strong"
EXCLUDED_BENCHMARKS = {"bbh"}
N_BOOT = 10000
SEED = 20261012

HALVES = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]   # all 3 balanced 2+2 splits of 4


def per_rep_correct(x_reps: list[np.ndarray], qi: int, cond: str, reps: list[int]) -> np.ndarray:
    ci = e6c.CONDITIONS.index(cond)
    return np.array([int(plurality_prediction(
        x_reps[r], np.random.default_rng([7, qi, r, ci])) == 0) for r in reps], float)


def within_noise(correct: np.ndarray) -> float:
    """Mean |half1-half2| over all 3 balanced splits of the 4 held-out replicates."""
    diffs = []
    for h1, h2 in HALVES:
        diffs.append(abs(correct[list(h1)].mean() - correct[list(h2)].mean()))
    return float(np.mean(diffs))


def process_cell(model: str, bench: str, paths: dict) -> list[dict]:
    rows, seen = [], {}
    for regime in ("easy", "hard"):
        if regime not in paths:
            continue
        for rec in load_cell(paths[regime]):
            if rec["question"] in seen:
                continue
            seen[rec["question"]] = regime
            rows.append(rec)
    if not rows:
        return []
    R = rows[0]["R"]
    bin_reps, eval_reps = range(R_BIN), list(range(R_BIN, R))
    out = []
    for rec in rows:
        rec["phat"] = phat_split(rec["x"][NONE], bin_reps)
        new_regime = ("easy" if rec["phat"] >= EASY_HI else
                      "hard" if rec["phat"] <= HARD_LO else "mid")
        if new_regime == "mid":
            continue
        qi = rec["qi"]
        cs = per_rep_correct(rec["x"][SHARED], qi, SHARED, eval_reps)
        ci = per_rep_correct(rec["x"][INDEP], qi, INDEP, eval_reps)
        dA = float(cs.mean() - ci.mean())
        c2w = (within_noise(cs) + within_noise(ci)) / 2
        M = margin_M(rec["x"][NONE], range(R))   # frozen default margin_half="all"
        out.append({"model": model, "bench": bench, "regime": new_regime,
                   "dA_abs_raw": abs(dA), "C2_within": c2w,
                   "dA_excess": abs(dA) - c2w, "M": M})
    return out


def main() -> None:
    found = discover(None, None)
    rows = []
    for (model, bench, N), paths in sorted(found.items()):
        if bench in EXCLUDED_BENCHMARKS or len(paths) < 2:
            continue
        rows += process_cell(model, bench, paths)

    print(f"=== C2 noise audit: {len(rows)} questions pooled (pre MIN_Q stratum gate) ===\n")
    rng = np.random.default_rng(SEED)
    by_regime: dict[str, list[dict]] = {}
    for r in rows:
        by_regime.setdefault(r["regime"], []).append(r)

    out = {"n_boot": N_BOOT, "seed": SEED, "regimes": {}}
    for regime in ("hard", "easy"):
        g = by_regime.get(regime, [])
        if len(g) < MIN_Q:
            continue
        M = np.array([r["M"] for r in g], float)
        raw = np.array([r["dA_abs_raw"] for r in g], float)
        exc = np.array([r["dA_excess"] for r in g], float)
        c2w = np.array([r["C2_within"] for r in g], float)
        lo_r, hi_r, pt_r = boot_slope(M, raw, rng, N_BOOT)
        lo_e, hi_e, pt_e = boot_slope(M, exc, rng, N_BOOT)
        print(f"[{regime}] n_q={len(g)}  mean|dA|_raw={raw.mean():.4f}  "
              f"mean C2_within={c2w.mean():.4f}  mean dA_excess={exc.mean():.4f}")
        print(f"    slope(raw |dA| ~ M)     = {pt_r:+.4f}  CI [{lo_r:+.4f},{hi_r:+.4f}]"
              f"{'  *' if (lo_r > 0 or hi_r < 0) else ''}")
        print(f"    slope(dA_excess ~ M)    = {pt_e:+.4f}  CI [{lo_e:+.4f},{hi_e:+.4f}]"
              f"{'  *' if (lo_e > 0 or hi_e < 0) else ''}")
        print(f"    raw mean|dA| vs mean C2_within: "
              f"{'raw signal mostly ABOVE noise floor' if raw.mean() > c2w.mean() * 1.5 else 'raw signal COMPARABLE TO OR BELOW noise floor -- same pathology as naive ΔY'}\n")
        out["regimes"][regime] = {
            "n_q": len(g), "mean_dA_abs_raw": float(raw.mean()),
            "mean_C2_within": float(c2w.mean()), "mean_dA_excess": float(exc.mean()),
            "slope_raw": {"pt": pt_r, "ci": [lo_r, hi_r]},
            "slope_excess": {"pt": pt_e, "ci": [lo_e, hi_e]},
        }

    dst = TABLES / "tier1_c2_noise_audit.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
