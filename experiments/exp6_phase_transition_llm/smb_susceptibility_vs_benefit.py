"""SMB -- Susceptibility vs. Benefit (offline, NO LLM calls). See MMRT_SMB_PREREG.md.

Question: does the KEEP-realization vote margin predict WHETHER the independence-restoring
re-reason changes the answer (susceptibility S) more strongly than it predicts WHETHER that
change helps (signed net benefit B)? Direct empirical separation of the two constructs the
paper's theory already separates algebraically (Theorems 5/6).

Reuses sgir_common.build_episodes() (the full, already-frozen SGIR episode table) unmodified.
No fitting, no split: a straight per-pair OLS slope of each outcome on margin, macro-averaged
with a PAIRED (same 9 pairs) resample-pairs bootstrap, mirroring the macro-aggregation
convention used throughout Tier-3 (tier3_matched_d.macro_bootstrap).

Writes results/tables/smb_susceptibility_vs_benefit.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sgir_common import TABLES, build_episodes, eligible_mask  # noqa: E402

N_BOOT = 20000


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Simple OLS slope of y on x (intercept + slope); x, y already finite/aligned."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


def per_pair_slopes(episodes: list[dict]) -> dict[tuple[str, str], dict]:
    """(model,bench) -> {slope_S, slope_B, n}, using every eligible episode in that pair
    (pooling regimes/replicates), per MMRT_SMB_PREREG.md."""
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for e in episodes:
        by_pair.setdefault((e["model"], e["bench"]), []).append(e)
    out = {}
    for key, rows in by_pair.items():
        margin = np.array([r["margin"] if r["margin"] is not None else np.nan for r in rows], float)
        elig = eligible_mask(margin)
        m = margin[elig]
        keep_label = np.array([r["keep_label"] for r in rows], int)[elig]
        indep_label = np.array([r["indep_label"] for r in rows], int)[elig]
        keep_correct = np.array([r["keep_correct"] for r in rows], bool)[elig]
        indep_correct = np.array([r["indep_correct"] for r in rows], bool)[elig]
        s_outcome = (keep_label != indep_label).astype(float)
        b_outcome = indep_correct.astype(float) - keep_correct.astype(float)
        out[key] = {"n": int(elig.sum()), "slope_S": ols_slope(m, s_outcome),
                    "slope_B": ols_slope(m, b_outcome),
                    "mean_S": float(s_outcome.mean()), "mean_B": float(b_outcome.mean())}
    return out


def macro_paired(per_pair: dict[tuple[str, str], dict], rng: np.random.Generator) -> dict:
    keys = sorted(per_pair.keys())
    K = len(keys)
    abs_s = np.array([abs(per_pair[k]["slope_S"]) for k in keys])
    abs_b = np.array([abs(per_pair[k]["slope_B"]) for k in keys])
    diff = abs_s - abs_b                      # PAIRED per pair, same 9 pairs both sides
    idx = rng.integers(0, K, size=(N_BOOT, K))
    draws_s, draws_b, draws_d = abs_s[idx].mean(1), abs_b[idx].mean(1), diff[idx].mean(1)

    def ci(x):
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    return {
        "pairs": keys, "K": K,
        "mean_abs_slope_S": {"point": float(abs_s.mean()), "ci": ci(draws_s)},
        "mean_abs_slope_B": {"point": float(abs_b.mean()), "ci": ci(draws_b)},
        "paired_diff_absS_minus_absB": {"point": float(diff.mean()), "ci": ci(draws_d)},
    }


def main() -> None:
    episodes = build_episodes()
    per_pair = per_pair_slopes(episodes)
    print("per (model,benchmark) pair:")
    for key in sorted(per_pair):
        v = per_pair[key]
        print(f"  {key[0]:6s}|{key[1]:12s} n={v['n']:4d}  slope_S={v['slope_S']:+.4f}  "
              f"slope_B={v['slope_B']:+.4f}")

    rng = np.random.default_rng(20261108)
    macro = macro_paired(per_pair, rng)
    print("\n=== MACRO (equal weight across 9 pairs, paired resample-pairs bootstrap) ===")
    print(f"  mean |slope_S| = {macro['mean_abs_slope_S']['point']:+.4f}  "
          f"CI={macro['mean_abs_slope_S']['ci']}")
    print(f"  mean |slope_B| = {macro['mean_abs_slope_B']['point']:+.4f}  "
          f"CI={macro['mean_abs_slope_B']['ci']}")
    print(f"  paired diff |slope_S|-|slope_B| = "
          f"{macro['paired_diff_absS_minus_absB']['point']:+.4f}  "
          f"CI={macro['paired_diff_absS_minus_absB']['ci']}")

    out = {"per_pair": {f"{k[0]}|{k[1]}": v for k, v in per_pair.items()}, "macro": macro}
    dst = TABLES / "smb_susceptibility_vs_benefit.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
