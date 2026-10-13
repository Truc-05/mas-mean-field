"""Frozen pre-specified test battery for the regime dependence of dA. See PREREG_dA_regime.md.

Reads the recovered split-sample strata (tier1_split_sample_strata.json), NOT the retired
probe labels. Offline, no GPU, safe to run while the sweep is live -- but it withholds a
verdict until --min-cells strata exist, so running it early cannot be mistaken for a result.

Four pre-registered tests, in reporting order:
  H1  E[dA] < 0           one-sided; mixed-model intercept + distribution-free sign test
  H2  |E[dA]| < EPS       TOST, reported separately for easy and hard; "n.s." is NOT equivalence
  H3  beta1 in dA = b0 + b1*1[easy] + u_model + v_bench + eps, by PAIRED stratum bootstrap
  H4  corr(A_none - P_maj(N, A_none, 0), dA) > 0   the only test that can SUPPORT Theorem 4

EPS = 0.05 is the band the paper already uses for this quantity (paper.md:157, 212;
README.md:268) -- adopted, not chosen here.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
STRATA = TABLES / "tier1_split_sample_strata.json"

EPS = 0.05                 # frozen: paper.md:157/212, README.md:268
ALPHA = 0.05
N_AGENTS = 11
EXCLUDED_BENCHMARKS = {"bbh"}
MIN_NEFF = 15              # gate for interpreting a stratum's dD (not its dA)


def p_maj(N: int, p: float) -> float:
    """Condorcet majority accuracy under independence, P_maj(N, p, 0)."""
    if not 0.0 <= p <= 1.0:
        return float("nan")
    return float(sum(math.comb(N, k) * p ** k * (1 - p) ** (N - k)
                     for k in range(N // 2 + 1, N + 1)))


def load_strata(path: Path) -> tuple[list[dict], list[str]]:
    """Flatten tier1_split_sample_strata.json: cells[] each carry strata{easy,hard}.

    A stratum marked `insufficient` (below the script's MIN_Q question floor) has no dA at all
    and is skipped — but it is NAMED in the returned list, because a silently shrinking
    denominator is exactly how a pre-registered battery stops being one.
    """
    blob = json.loads(path.read_text())
    out, skipped = [], []
    for cell in blob.get("cells", []):
        bench = cell.get("benchmark")
        model = cell.get("model")
        if bench in EXCLUDED_BENCHMARKS:
            skipped.append(f"{model}/{bench}: excluded benchmark")
            continue
        for regime, st in (cell.get("strata") or {}).items():
            if regime not in ("easy", "hard"):
                continue
            if st.get("insufficient") or st.get("dA") is None:
                skipped.append(f"{model}/{bench}/{regime}: "
                               f"insufficient (n_q={st.get('n_q')})")
                continue
            out.append({
                "model": model, "bench": bench, "regime": regime,
                "dA": float(st["dA"]), "dA_ci": st.get("dA_ci"),
                "A_none": (None if st.get("A_none") is None else float(st["A_none"])),
                "n_q": st.get("n_q"), "n_D": st.get("n_D"),
                "dD_pooled": st.get("dD_pooled"),
                "dD_pooled_sig": st.get("dD_pooled_sig"),
                "mean_M": st.get("mean_M"),
            })
    for s in blob.get("skipped", []):
        skipped.append(str(s))
    return out, skipped


def boot_mean_ci(v, rng, B=20000, alpha=ALPHA):
    a = np.asarray(v, float)
    if a.size < 2:
        return (float("nan"), float("nan"))
    d = a[rng.integers(0, a.size, (B, a.size))].mean(1)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi)) if hi - lo > 1e-12 else (float("nan"), float("nan"))


def tost(v, eps, rng, B=20000):
    """Equivalence by a 90% bootstrap CI (= two one-sided tests at alpha=.05) inside +-eps."""
    a = np.asarray(v, float)
    if a.size < 2:
        return {"n": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
                "lo90": float("nan"), "hi90": float("nan"), "equivalent": False,
                "reason": "n<2"}
    d = a[rng.integers(0, a.size, (B, a.size))].mean(1)
    lo, hi = (float(x) for x in np.quantile(d, [0.05, 0.95]))
    if hi - lo < 1e-12:
        return {"n": int(a.size), "mean": float(a.mean()), "lo90": float("nan"),
                "hi90": float("nan"), "equivalent": False, "reason": "degenerate bootstrap"}
    return {"n": int(a.size), "mean": float(a.mean()), "lo90": lo, "hi90": hi,
            "equivalent": bool(-eps < lo and hi < eps), "reason": ""}


def beta1_paired(rows, rng, B=20000):
    """beta1 = E[dA | easy] - E[dA | hard] by resampling (model, bench) PAIRS.

    The crossed random effects in the pre-registered formula are absorbed by differencing
    within a (model, benchmark) pair: u_model and v_bench cancel, so the contrast is
    identified without fitting REML variance components on ~12 clusters where they would be
    unstable. Pairs lacking one regime are reported and excluded.
    """
    key = lambda r: (r["model"], r["bench"])
    pairs, incomplete = [], []
    for k, grp in groupby(sorted(rows, key=key), key=key):
        g = {r["regime"]: r["dA"] for r in grp}
        if "easy" in g and "hard" in g:
            pairs.append((k, g["easy"] - g["hard"]))
        else:
            incomplete.append((k, next(iter(g))))
    if len(pairs) < 2:
        return {"n_pairs": len(pairs), "beta1": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "incomplete": incomplete}
    d = np.array([x for _, x in pairs], float)
    draws = d[rng.integers(0, d.size, (B, d.size))].mean(1)
    lo, hi = (float(x) for x in np.quantile(draws, [ALPHA / 2, 1 - ALPHA / 2]))
    return {"n_pairs": len(pairs), "beta1": float(d.mean()),
            "lo": lo, "hi": hi, "excludes_zero": bool(hi < 0 or lo > 0),
            "incomplete": incomplete}


def h4_corr(rows, rng, B=20000):
    g = [r for r in rows if r["A_none"] is not None]
    if len(g) < 4:
        return {"n": len(g), "corr": float("nan"), "lo": float("nan"), "hi": float("nan")}
    pred = np.array([r["A_none"] - p_maj(N_AGENTS, r["A_none"]) for r in g], float)
    dA = np.array([r["dA"] for r in g], float)
    ok = np.isfinite(pred) & np.isfinite(dA)
    pred, dA = pred[ok], dA[ok]
    if pred.size < 4 or pred.std() < 1e-12 or dA.std() < 1e-12:
        return {"n": int(pred.size), "corr": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "note": "no variation in predictor or outcome"}
    point = float(np.corrcoef(pred, dA)[0, 1])
    idx = rng.integers(0, pred.size, (B, pred.size))
    cs = []
    for i in idx:
        x, y = pred[i], dA[i]
        if x.std() > 1e-12 and y.std() > 1e-12:
            cs.append(np.corrcoef(x, y)[0, 1])
    if len(cs) < 2:
        return {"n": int(pred.size), "corr": point, "lo": float("nan"), "hi": float("nan")}
    lo, hi = (float(x) for x in np.quantile(cs, [ALPHA / 2, 1 - ALPHA / 2]))
    return {"n": int(pred.size), "corr": point, "lo": lo, "hi": hi,
            "mean_pred": float(pred.mean())}


def verdict(eq_easy, eq_hard, b1, h1_sign_p) -> tuple[str, str]:
    h1 = h1_sign_p is not None and h1_sign_p < ALPHA
    b1_excl = bool(b1.get("excludes_zero"))
    if eq_easy["equivalent"] and eq_hard["equivalent"] and not b1_excl:
        return ("R1", "regime-invariant negligibility — 'dependence can increase substantially "
                      "without materially changing accuracy'; cite Corollary 4.1, NOT Thm 4 pt 1")
    if eq_hard["equivalent"] and not eq_easy["equivalent"] and b1_excl and b1["beta1"] < 0:
        return ("R2", "regime-dependent cancellation — high dependence does NOT imply dA≈0; "
                      "promote to a finding (Thm 4 pt 1 becoming visible as p rises)")
    if h1 and not eq_easy["equivalent"] and not eq_hard["equivalent"]:
        return ("R3", "systematic accuracy cost — drop any 'dA≈0' headline; keep Thm 4 at its "
                      "own assumptions. C1/C2/C3/C5 stand independently")
    if not h1 and eq_easy["equivalent"] and eq_hard["equivalent"]:
        return ("R4", "null-but-bounded — dA within ±0.05 with no reliable sign; "
                      "no mechanism claim either way")
    return ("MIXED", "no pre-registered cell matches; report all four tests verbatim and "
                     "do NOT pick a narrative post hoc")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strata", default=str(STRATA))
    ap.add_argument("--min-cells", type=int, default=48,
                    help="withhold the verdict below this many strata (default 48 = 2/3 of 72)")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--json", default=str(TABLES / "tier1_dA_regime_test.json"))
    args = ap.parse_args()

    p = Path(args.strata)
    if not p.exists():
        raise SystemExit(f"{p} missing — run tier1_split_sample_strata.py first")
    rows, skipped = load_strata(p)
    rng = np.random.default_rng(args.seed)

    easy = [r["dA"] for r in rows if r["regime"] == "easy"]
    hard = [r["dA"] for r in rows if r["regime"] == "hard"]
    allv = [r["dA"] for r in rows]
    if not rows:
        raise SystemExit(f"{p} parsed to 0 usable strata — schema mismatch or all insufficient; "
                         f"skipped: {skipped}")

    print(f"\n=== PRE-REGISTERED dA REGIME BATTERY (PREREG_dA_regime.md) ===")
    print(f"substrate: recovered split-sample strata, bbh excluded, eps={EPS}")
    print(f"strata: {len(rows)} ({len(easy)} easy, {len(hard)} hard)")
    sig = sum(1 for r in rows if r.get("dD_pooled_sig"))
    print(f"        ΔD_pooled significant in {sig}/{len(rows)} (context only — dA is not gated "
          f"on ΔD)")
    if skipped:
        print(f"        {len(skipped)} stratum/strata not entering the battery:")
        for s in skipped:
            print(f"          - {s}")

    # H1 -- sign-matched to Thm 4: sign(dA) should equal sign(1/2 - p), NOT be globally negative.
    # See the specification note in PREREG_dA_regime.md §3.
    matched, nz = 0, 0
    for r in rows:
        if r["A_none"] is None or abs(r["dA"]) <= 1e-12:
            continue
        pred = r["A_none"] - p_maj(N_AGENTS, r["A_none"])
        if abs(pred) <= 1e-12:
            continue                      # degenerate at p->1: theory predicts nothing to match
        nz += 1
        matched += int(np.sign(r["dA"]) == np.sign(pred))
    sign_p = (float(stats.binomtest(matched, nz, 0.5, alternative="greater").pvalue)
              if nz else None)
    lo, hi = boot_mean_ci(allv, rng)
    print(f"\nH1  sign(dA) == sign(1/2 - p)   (one-sided, Thm 4 pt 1 for M1, sign-matched)")
    print(f"    mean dA = {np.mean(allv):+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  (context, not the test)")
    print(f"    sign agreement: {matched}/{nz} strata (ties and p->1 degenerates dropped), "
          f"p = {sign_p if sign_p is None else f'{sign_p:.4f}'}"
          f"{'  SUPPORTED' if sign_p is not None and sign_p < ALPHA else '  not supported'}")
    if nz < 8:
        print(f"    NOTE n={nz} is underpowered for a sign test; see the declared design "
              f"limitation (strata sit at p~1 and p~0, where p-P_maj is degenerate)")

    # H2
    print(f"\nH2  |E[dA]| < {EPS}   (TOST, 90% CI inside the band; reported per regime)")
    eq = {}
    for lab, v in (("easy", easy), ("hard", hard), ("all", allv)):
        t = tost(v, EPS, rng); eq[lab] = t
        tag = ("EQUIVALENT" if t["equivalent"]
               else f"NOT equivalent{(' (' + t['reason'] + ')') if t['reason'] else ''}")
        print(f"    {lab:5} n={t['n']:3d}  mean={t['mean']:+.4f}  "
              f"90% CI [{t['lo90']:+.4f},{t['hi90']:+.4f}]  {tag}")

    # H3
    b1 = beta1_paired(rows, rng)
    print(f"\nH3  beta1 = E[dA|easy] - E[dA|hard]   (paired (model,bench) bootstrap)")
    print(f"    n_pairs={b1['n_pairs']}  beta1={b1['beta1']:+.4f}  "
          f"95% CI [{b1['lo']:+.4f},{b1['hi']:+.4f}]  "
          f"{'excludes 0' if b1.get('excludes_zero') else 'covers 0'}")
    if b1["incomplete"]:
        print(f"    excluded (one regime only): "
              + ", ".join(f"{m}/{b}:{r}" for (m, b), r in b1["incomplete"]))

    # H4
    h4 = h4_corr(rows, rng)
    print(f"\nH4  corr(A_none - P_maj(11,A_none), dA) > 0   (can SUPPORT Thm 4, least powered)")
    if h4.get("note"):
        print(f"    n={h4['n']}  {h4['note']}")
    else:
        print(f"    n={h4['n']}  corr={h4['corr']:+.3f}  95% CI [{h4['lo']:+.3f},{h4['hi']:+.3f}]"
              f"  mean predicted Delta = {h4.get('mean_pred', float('nan')):+.4f}")
        print(f"    {'SUPPORTED' if h4['lo'] > 0 else 'no support (not a refutation)'}")

    # verdict
    code, text = verdict(eq["easy"], eq["hard"], b1, sign_p)
    print(f"\n=== VERDICT ===")
    if len(rows) < args.min_cells:
        print(f"  WITHHELD: {len(rows)} strata < --min-cells {args.min_cells}.")
        print(f"  (provisional would be {code}; do not report it)")
    else:
        print(f"  {code}: {text}")

    Path(args.json).write_text(json.dumps({
        "eps": EPS, "seed": args.seed, "n_strata": len(rows),
        "min_cells": args.min_cells, "withheld": len(rows) < args.min_cells,
        "H1": {"mean": float(np.mean(allv)) if allv else None, "ci": [lo, hi],
               "n_sign_matched": matched, "n_testable": nz, "sign_p": sign_p,
               "spec": "sign(dA)==sign(1/2-p); see PREREG §3 specification note"},
        "H2": eq, "H3": {k: v for k, v in b1.items() if k != "incomplete"},
        "H4": h4, "verdict": code, "verdict_text": text,
    }, indent=2))
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
