"""C3 global directional test — one p-value, no magnitude inference. Closes C3.

FROZEN PRE-SPECIFICATION (written before the statistic was ever evaluated on real strata).

  Why a global directional test and nothing else. `calibrate_estimability.py` showed that at
  N=11 with R=8 (R=4 per half in the split-sample analysis) the pooled dependence ratio is
  upward-biased and its bootstrap interval under-covers (0.25-0.76 against a nominal 0.95),
  with reliability governed by error prevalence rather than by n_eff -- and BOTH real regimes
  (easy: error ~0.00; hard: error ~0.86-0.99) fall outside the only adequately covered band
  (intermediate prevalence, which this design never samples). No n_eff cut satisfies the frozen
  tolerances, so per-cell significance cannot be certified and per-cell magnitudes must not be
  interpreted. Building a bias-corrected or calibration-adjusted interval was considered and
  REJECTED: it would add a methodological branch the paper would then have to validate and
  defend, and it is not this paper's contribution.

  What rescues a directional claim. The upward bias is a bias under the ALTERNATIVE, not under
  the null. Simulated at rho = 0 across error rates 0.03-0.97 and both R, P(ΔD̂ > 0) lands in
  0.480-0.542 with mean ΔD̂ within ±0.007 of zero -- sign is close to a fair coin under H0. So a
  statistic built only from SIGNS is admissible where one built from magnitudes is not. The
  residual deviation (worst case 0.542 at R=4, error 0.97) is handled by SIMULATING the null
  distribution under each stratum's own design parameters rather than assuming Binomial(0.5).

  Hypothesis, one test, one p-value:
      H0: SHARED does not systematically increase error dependence over matched INDEP.
  Statistic, frozen:
      S = #{ strata with ΔD̂ > 0 }   over the K strata with ΔD̂ != 0.
  Null distribution: parametric bootstrap. For each real stratum take ONLY its design
  parameters -- n_q, replicate count R, and error prevalence (1 - A_none) -- and simulate both
  arms at rho = 0 using the same generative model as the calibration. Recompute S. Repeat M
  times. One-sided p = (1 + #{S_null >= S_obs}) / (1 + M).

  The observed ΔD values enter ONLY through their signs. Their magnitudes are never used, and
  the null is calibrated to the design, not to the result.

  Reporting rule, frozen. Report S/K, the simulated null mean, and the p-value. Report NO
  per-stratum ΔD point estimate or interval, and NO "k/n significant" count. Type-I error of
  the per-cell bootstrap is better behaved than its interval coverage but remains mildly
  anti-conservative (null rejection 0.093-0.095 against nominal 0.05), which is a further
  reason the per-cell counts are not reported.

  Withholding. Refuses a verdict below --min-strata, like PREREG_dA_regime.md's battery.

C3's role after this test is a manipulation check / causal-instrument validation inside
    controlled mechanism -> dependence direction -> behavioral consequence,
not a quantitative headline. Quantitative weight sits on ΔY, dA, break/rescue/repairability,
and the Tier-3 matched-D arm if it replicates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_estimability import one_stratum

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
STRATA = TABLES / "tier1_split_sample_strata.json"
EXCLUDED_BENCHMARKS = {"bbh"}
R_SPLIT = 4          # recovered strata evaluate on held-out replicates 4-7


def load_design_and_signs(path: Path):
    """Design parameters (used for the null) and observed SIGNS (used for S). No magnitudes."""
    blob = json.loads(path.read_text())
    rows, skipped = [], []
    for cell in blob.get("cells", []):
        model, bench = cell.get("model"), cell.get("benchmark")
        if bench in EXCLUDED_BENCHMARKS:
            skipped.append(f"{model}/{bench}: excluded benchmark")
            continue
        for regime, st in (cell.get("strata") or {}).items():
            if regime not in ("easy", "hard"):
                continue
            dD, nq, a_none = st.get("dD_pooled"), st.get("n_q"), st.get("A_none")
            if st.get("insufficient") or dD is None or nq is None or a_none is None:
                skipped.append(f"{model}/{bench}/{regime}: insufficient (n_q={nq})")
                continue
            if abs(dD) <= 1e-12:
                skipped.append(f"{model}/{bench}/{regime}: ΔD exactly 0, no sign")
                continue
            rows.append({"model": model, "bench": bench, "regime": regime,
                         "n_q": int(nq),
                         "err_rate": float(np.clip(1.0 - a_none, 1e-3, 1 - 1e-3)),
                         "sign_positive": bool(dD > 0)})
    return rows, skipped


def simulate_null_S(rows, rng, n_sim: int, n_boot: int) -> np.ndarray:
    """Null distribution of S, each stratum simulated at its own (n_q, err_rate), rho = 0."""
    S = np.zeros(n_sim, dtype=int)
    for m in range(n_sim):
        s = 0
        for r in rows:
            _, pt, _, _ = one_stratum(rng, r["err_rate"], 0.0, r["n_q"], R_SPLIT, n_boot)
            if pt == pt and pt > 0:
                s += 1
        S[m] = s
    return S


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata", default=str(STRATA))
    ap.add_argument("--min-strata", type=int, default=48,
                    help="withhold the verdict below this many strata (2/3 of the 72-cell grid)")
    ap.add_argument("--n-sim", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=50,
                    help="only the point estimate is needed under the null, so this is small")
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--json", default=str(TABLES / "c3_global_directional_test.json"))
    args = ap.parse_args()

    p = Path(args.strata)
    if not p.exists():
        raise SystemExit(f"{p} missing — run tier1_split_sample_strata.py first")
    rows, skipped = load_design_and_signs(p)
    if not rows:
        raise SystemExit(f"no usable strata; skipped: {skipped}")
    rng = np.random.default_rng(args.seed)

    K = len(rows)
    S_obs = sum(r["sign_positive"] for r in rows)
    print("\n=== C3 GLOBAL DIRECTIONAL TEST (frozen; magnitudes never used) ===")
    print("H0: SHARED does not systematically increase error dependence over matched INDEP")
    print(f"strata K={K}  (easy {sum(1 for r in rows if r['regime']=='easy')}, "
          f"hard {sum(1 for r in rows if r['regime']=='hard')})")
    if skipped:
        print(f"not entering the test ({len(skipped)}):")
        for s in skipped:
            print(f"  - {s}")
    print(f"\nS_obs = {S_obs}/{K} strata with ΔD̂ > 0")
    print(f"simulating the null at each stratum's own (n_q, error prevalence), R={R_SPLIT}, "
          f"rho=0 ... {args.n_sim} draws")

    S_null = simulate_null_S(rows, rng, args.n_sim, args.n_boot)
    pval = (1 + int((S_null >= S_obs).sum())) / (1 + args.n_sim)
    print(f"\nnull S: mean {S_null.mean():.2f}  sd {S_null.std():.2f}  "
          f"median {np.median(S_null):.0f}  "
          f"q95 {np.quantile(S_null, 0.95):.0f}  max {S_null.max()}")
    print(f"one-sided p = (1 + #{{S_null >= {S_obs}}}) / (1 + {args.n_sim}) = {pval:.5f}")

    withheld = K < args.min_strata
    print("\n=== VERDICT ===")
    if withheld:
        print(f"  WITHHELD: K={K} < --min-strata {args.min_strata}. "
              f"(provisional p={pval:.5f}; do not report it)")
    elif pval < 0.05:
        print(f"  C3 SUPPORTED, directionally: shared evidence induces greater error dependence")
        print(f"  than matched independent evidence (p={pval:.5f}). The present replicate budget")
        print(f"  does not support calibrated estimation of the MAGNITUDE of this increase.")
    else:
        print(f"  C3 NOT SUPPORTED at the 0.05 level (p={pval:.5f}).")
    print("\n  Report S/K, the null mean, and this p-value. Do NOT report per-stratum ΔD point")
    print("  estimates, bootstrap intervals, or a 'k/n significant' count.")

    Path(args.json).write_text(json.dumps({
        "role": "C3 global directional test; signs only, null calibrated to the design",
        "hypothesis": "H0: SHARED does not systematically increase dependence over INDEP",
        "statistic": "S = #{strata with dD_hat > 0}",
        "K": K, "S_obs": S_obs,
        "null": {"mean": float(S_null.mean()), "sd": float(S_null.std()),
                 "median": float(np.median(S_null)), "q95": float(np.quantile(S_null, .95)),
                 "max": int(S_null.max()), "n_sim": args.n_sim},
        "p_one_sided": pval, "withheld": withheld, "min_strata": args.min_strata,
        "seed": args.seed, "R_simulated": R_SPLIT,
        "strata": rows, "skipped": skipped,
        "reporting_rule": "no per-stratum magnitudes, no bootstrap intervals, no k/n counts",
    }, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
