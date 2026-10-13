"""C2, CORRECTED (2026-10-12). Supersedes `tier1_c2_recovered_strata_aggregate.py`'s headline:
the noise audit (`tier1_c2_noise_audit.py`) showed raw |dA| ~ M regresses mostly within-arm
sampling noise, not a shared-vs-indep susceptibility effect -- in the primary (hard) regime
the slope even FLIPS SIGN once that noise is subtracted. This script computes the per-stratum
slope of the noise-adjusted outcome `dA_excess = |dA| - C2_within` on margin M (same `M`,
same `boot_slope`, same stratum pooling/MIN_Q gate as the original C2 -- only the OUTCOME
fed to the regression changes), then macro-averages across strata exactly as before
(equal weight per stratum, stratum-level bootstrap -- unchanged philosophy, not re-chosen
after seeing this result).

The old `tier1_c2_recovered_strata_aggregate.json` is NOT deleted (provenance) but its
headline (-0.0627 in hard) must not be quoted in the manuscript: it is now understood to be a
noise-floor artifact, not a susceptibility finding.
"""
from __future__ import annotations

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
MACRO_N_BOOT = 20000
SEED = 20261012
HALVES = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]


def per_rep_correct(x_reps, qi, cond, reps):
    ci = e6c.CONDITIONS.index(cond)
    return np.array([int(plurality_prediction(
        x_reps[r], np.random.default_rng([7, qi, r, ci])) == 0) for r in reps], float)


def within_noise(correct):
    return float(np.mean([abs(correct[list(h1)].mean() - correct[list(h2)].mean())
                          for h1, h2 in HALVES]))


def pool_cell(model, bench, paths):
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
        return [], 0
    R = rows[0]["R"]
    for rec in rows:
        rec["phat"] = phat_split(rec["x"][NONE], range(R_BIN))
        rec["new_regime"] = ("easy" if rec["phat"] >= EASY_HI else
                             "hard" if rec["phat"] <= HARD_LO else "mid")
    return rows, R


def stratum_slopes(rows, R, rng):
    """Per-question raw |dA|, C2_within, M -> (slope_raw, slope_excess) for this stratum."""
    eval_reps = list(range(R_BIN, R))
    raw, exc, M = [], [], []
    for rec in rows:
        qi = rec["qi"]
        cs = per_rep_correct(rec["x"][SHARED], qi, SHARED, eval_reps)
        ci = per_rep_correct(rec["x"][INDEP], qi, INDEP, eval_reps)
        dA = float(cs.mean() - ci.mean())
        c2w = (within_noise(cs) + within_noise(ci)) / 2
        raw.append(abs(dA)); exc.append(abs(dA) - c2w)
        M.append(margin_M(rec["x"][NONE], range(R)))   # frozen default margin_half="all"
    M, raw, exc = np.array(M), np.array(raw), np.array(exc)
    lo_r, hi_r, pt_r = boot_slope(M, raw, rng, N_BOOT)
    lo_e, hi_e, pt_e = boot_slope(M, exc, rng, N_BOOT)
    return {"slope_raw": pt_r, "slope_raw_ci": [lo_r, hi_r],
            "slope_excess": pt_e, "slope_excess_ci": [lo_e, hi_e], "n_q": len(rows)}


def macro(vals, rng):
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return {"K": int(vals.size), "mean": None, "ci": [None, None], "sig": False}
    K = vals.size
    idx = rng.integers(0, K, size=(MACRO_N_BOOT, K))
    draws = vals[idx].mean(1)
    lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    return {"K": K, "mean": float(vals.mean()), "ci": [lo, hi],
            "sig": bool(hi - lo > 1e-12 and (lo > 0 or hi < 0))}


def main() -> None:
    found = discover(None, None)
    rng = np.random.default_rng(SEED)
    by_regime: dict[str, list[dict]] = {"easy": [], "hard": []}
    for (model, bench, N), paths in sorted(found.items()):
        if bench in EXCLUDED_BENCHMARKS or len(paths) < 2:
            continue
        rows, R = pool_cell(model, bench, paths)
        if not rows:
            continue
        for regime in ("easy", "hard"):
            sub = [r for r in rows if r["new_regime"] == regime]
            if len(sub) < MIN_Q:
                continue
            s = stratum_slopes(sub, R, rng)
            s.update(model=model, bench=bench)
            by_regime[regime].append(s)
            print(f"[{model}/{bench}/{regime}] n_q={s['n_q']}  "
                  f"raw={s['slope_raw']:+.3f} {s['slope_raw_ci']}  "
                  f"excess={s['slope_excess']:+.3f} {s['slope_excess_ci']}")

    out = {"note": "slope_excess (noise-adjusted) supersedes slope_raw as the C2 estimand",
          "regimes": {}}
    print("\n=== MACRO-AVERAGE across strata, equal weight (primary = hard) ===")
    for regime in ("hard", "easy"):
        g = by_regime[regime]
        raw = np.array([s["slope_raw"] for s in g], float)
        exc = np.array([s["slope_excess"] for s in g], float)
        m_raw, m_exc = macro(raw, rng), macro(exc, rng)
        print(f"[{regime}] K={len(g)}")
        print(f"    raw |dA|~M     macro beta_bar = {m_raw['mean']:+.4f}  CI {m_raw['ci']}  "
              f"{'excludes 0' if m_raw['sig'] else 'covers 0'}  <- SUPERSEDED, noise artifact")
        print(f"    dA_excess~M    macro beta_bar = {m_exc['mean']:+.4f}  CI {m_exc['ci']}  "
              f"{'excludes 0' if m_exc['sig'] else 'covers 0'}  <- CORRECTED ESTIMAND")
        out["regimes"][regime] = {"K": len(g), "raw_superseded": m_raw, "excess_corrected": m_exc,
                                  "strata": g}

    dst = TABLES / "tier1_c2_noise_adjusted_aggregate.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
