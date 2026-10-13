"""P2 cross-model / cross-benchmark aggregator — reports C1 / C2 / C3 SEPARATELY per run
group, never collapsing to one accuracy or one correlation statistic.

Inputs (produced by `exp6c_boundary.py --model-tag <mtag>`): per-regime analysis JSONs
`exp6c_boundary_{regime}[_{mtag}]_N{N}.json` (+ the sibling CSV for single-agent `p`).
A "group" = one model × benchmark, identified by `model_tag` (falls back to `model` for
the legacy qwen run). Regimes are pooled within a group; every group is reported —
**failures are kept, never dropped**.

Per group it emits, with FROZEN expectations set before any cross-model number is seen:

  C1  THEORY-IMPLIED mechanism contrast — a GENERALIZATION TEST of Theorem 4's ρ→1
      prediction, NOT a direct empirical observation of two mechanisms. The two limits are
      *computed* from the `none` arm (so the correct-leaning shared evidence never enters),
      they are NOT separately rolled out here:
        copying limit  A_M2 = P_maj = plurality_acc(none)
        common  limit  A_M1 = p     = p_single(none)
        Condorcet gap  = P_maj − p        (the predicted ρ→1 split)
      PASS: sign(gap) == sign(p − ½), and |gap| > 0. This tests whether the Theorem-4
      prediction holds across models/benchmarks. The MAIN empirical Mechanism
      Non-Equivalence remains the controlled ablation (copying vs common-shock rollout,
      Exp6/6B); do NOT report this block as an observed two-mechanism result. Realized
      shared/indep arms are descriptive only (evidence-quality-confounded).

  C2  Collective decision-boundary amplification (Theorem 5): slope of |ΔA| on margin M,
      where ΔA = A_shared − A_indep, M = runner-up gap. Saves {slope, CI, 3 margin bins, N}
      so the Theorem-5 prediction (slope steepens with N) is directly testable across runs.
      PASS: slope < 0 with bootstrap CI excluding 0.

  C3  Matched dependence instrument (unchanged validated estimand): paired
      ΔD = mean_q[ D_shared(q) − D_indep(q) ] on the shared∩indep mask, question bootstrap CI.
      PASS: ΔD > 0 with CI excluding 0.

CPU-only. Usage:
  python p2_aggregate.py                         # glob all runs in results/tables
  python p2_aggregate.py --n 11                  # restrict to N=11 runs
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TABLES = HERE.parents[1] / "results" / "tables"
NONE, SHARED, INDEP = "none", "shared_correct_strong", "indep_correct_strong"


# ── small stats helpers (question-level bootstrap; seeded, deterministic) ─────────
def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean()
    denom = float((xm * xm).sum())
    return float((xm * (y - y.mean())).sum() / denom) if denom > 1e-12 else float("nan")


def _boot(fn, arrays, n=10000, seed=(7,)):
    rng = np.random.default_rng(list(seed))
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        vals.append(fn(*[a[idx] for a in arrays]))
    vals = np.array([v for v in vals if v == v])
    if not len(vals):
        return (float("nan"), float("nan"))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _bins(M: np.ndarray, absA: np.ndarray, k=3) -> list[dict]:
    order = np.argsort(M)
    out = []
    for chunk in np.array_split(order, k):
        if not len(chunk):
            continue
        out.append({"n": int(len(chunk)), "M_lo": float(M[chunk].min()),
                    "M_hi": float(M[chunk].max()), "mean_absA": float(absA[chunk].mean())})
    return out


# ── load one boundary JSON + its sibling CSV (for p_single) ───────────────────────
def _load_csv_cond(regime: str, mtag: str, N: int) -> dict:
    suf = f"_{mtag}" if mtag else ""
    cands = glob.glob(str(TABLES / f"exp6c_*_{regime}{suf}_N{N}.csv"))
    for path in cands:
        rows = {}
        for row in csv.DictReader(open(path)):
            if row.get("condition"):
                rows[row["condition"]] = row
        if NONE in rows:                     # a boundary CSV (has the 3 conditions)
            return rows
    return {}


def _per_q_arrays(per_q: list[dict]):
    M, dA, dD_pairs = [], [], []
    for r in per_q:
        aS, aI = r.get(SHARED, {}).get("A_q"), r.get(INDEP, {}).get("A_q")
        if aS is not None and aI is not None:
            M.append(r["margin"]["runnerup_gap"]); dA.append(aS - aI)
        dS, dI = r.get(SHARED, {}).get("D_q"), r.get(INDEP, {}).get("D_q")
        if dS is not None and dI is not None:
            dD_pairs.append(dS - dI)
    return np.array(M, float), np.array(dA, float), np.array(dD_pairs, float)


# ── the three claim computations (each on a pooled set of per_q rows) ─────────────
def c2_slope(M, dA) -> dict:
    absA = np.abs(dA)
    slope = _ols_slope(M, absA)
    ci = _boot(lambda m, a: _ols_slope(m, a), [M, absA])
    return {"n": int(len(M)), "slope": slope, "ci95": ci, "bins": _bins(M, absA),
            "PASS": bool(slope < 0 and ci[1] < 0)}


def c3_deltaD(dD_pairs) -> dict:
    if not len(dD_pairs):
        return {"n": 0, "deltaD": float("nan"), "ci95": (float("nan"),) * 2, "PASS": False}
    mean = float(dD_pairs.mean())
    ci = _boot(lambda a: float(a.mean()), [dD_pairs])
    return {"n": int(len(dD_pairs)), "deltaD": mean, "ci95": ci, "PASS": bool(mean > 0 and ci[0] > 0)}


def c1_mechanism(csv_by_regime: dict) -> dict:
    """Aggregate the Theorem-4 ρ→1 limit contrast over regimes (each regime its own p, P_maj)."""
    per = {}
    for regime, rows in csv_by_regime.items():
        if NONE not in rows:
            continue
        p = float(rows[NONE]["p_single"]); pmaj = float(rows[NONE]["plurality_acc"])
        gap = pmaj - p
        sign_ok = (p == 0.5) or (math.copysign(1, gap) == math.copysign(1, p - 0.5)) or abs(gap) < 1e-9
        per[regime] = {
            "p_single": p, "P_maj_copying_limit": pmaj, "condorcet_gap": gap,
            "sign_matches_condorcet": bool(sign_ok),
            "A_shared_realized": float(rows.get(SHARED, {}).get("plurality_acc", "nan")),
            "A_indep_realized": float(rows.get(INDEP, {}).get("plurality_acc", "nan")),
        }
    # non-equivalence holds if every regime with |p−½| not tiny shows the predicted sign
    testable = [v for v in per.values() if abs(v["p_single"] - 0.5) > 0.02]
    PASS = bool(testable) and all(v["sign_matches_condorcet"] for v in testable)
    return {"per_regime": per, "PASS": PASS}


# ── group discovery ───────────────────────────────────────────────────────────────
def discover(nfilter: int | None):
    groups = defaultdict(lambda: {"regimes": {}, "csv": {}, "model": None, "N": None})
    for path in sorted(glob.glob(str(TABLES / "exp6c_boundary_*_N*.json"))):
        d = json.load(open(path))
        N = int(d["N"])
        if nfilter and N != nfilter:
            continue
        regime = d["regime"]
        mtag = d.get("model_tag") or ""
        gid = mtag if mtag else d.get("model", "legacy")
        g = groups[(gid, N)]
        g["model"], g["N"], g["gid"] = d.get("model"), N, gid
        g["regimes"][regime] = d["per_q"]
        g["csv"][regime] = _load_csv_cond(regime, mtag, N)
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="restrict to this N (else all)")
    ap.add_argument("--out", default=str(TABLES / "p2_aggregate_summary.json"))
    args = ap.parse_args(argv)

    groups = discover(args.n)
    if not groups:
        raise SystemExit(f"no exp6c_boundary_*_N*.json found under {TABLES}")

    summary = {}
    c2_nscaling = []                                            # (gid, N, slope, ci) for Thm5
    print("=" * 92)
    print("P2 CROSS-MODEL AGGREGATOR — C1 mechanism / C2 boundary / C3 ΔD (failures kept)")
    print("=" * 92)
    for (gid, N), g in sorted(groups.items()):
        # pooled per_q across regimes
        pooled = [r for pq in g["regimes"].values() for r in pq]
        rec = {"model": g["model"], "model_tag": gid, "N": N,
               "regimes": sorted(g["regimes"]), "C1": c1_mechanism(g["csv"]),
               "C2": {}, "C3": {}}
        # C2 / C3 per-regime + pooled
        for label, rows in list(g["regimes"].items()) + [("pooled", pooled)]:
            M, dA, dD = _per_q_arrays(rows)
            rec["C2"][label] = c2_slope(M, dA)
            rec["C3"][label] = c3_deltaD(dD)
        summary[f"{gid}|N{N}"] = rec

        c2p, c3p, c1 = rec["C2"]["pooled"], rec["C3"]["pooled"], rec["C1"]
        c2_nscaling.append((gid, N, c2p["slope"], c2p["ci95"]))
        print(f"\n▶ {gid}  (model={g['model']}, N={N}, regimes={rec['regimes']})")
        # C1 — theory-implied contrast (generalization test of Thm4 prediction; NOT a rollout)
        print(f"  C1 theory-implied mechanism contrast (Thm4 pred; main empirical=ablation) "
              f"[{'PASS' if c1['PASS'] else 'FAIL'}]:")
        for rg, v in c1["per_regime"].items():
            print(f"       {rg:5s} p={v['p_single']:.3f}  P_maj(copying)={v['P_maj_copying_limit']:.3f}  "
                  f"gap={v['condorcet_gap']:+.3f}  sign_ok={v['sign_matches_condorcet']}  "
                  f"[realized shared={v['A_shared_realized']:.3f} indep={v['A_indep_realized']:.3f}]")
        # C2
        b = "  ".join(f"{x['mean_absA']:.3f}" for x in c2p["bins"])
        print(f"  C2 |ΔA|~M pooled [{'PASS' if c2p['PASS'] else 'FAIL'}]: slope={c2p['slope']:+.3f} "
              f"CI[{c2p['ci95'][0]:+.3f},{c2p['ci95'][1]:+.3f}]  bins(low→high M)=[{b}]  n={c2p['n']}")
        # C3
        print(f"  C3 ΔD paired pooled [{'PASS' if c3p['PASS'] else 'FAIL'}]: ΔD={c3p['deltaD']:+.3f} "
              f"CI[{c3p['ci95'][0]:+.3f},{c3p['ci95'][1]:+.3f}]  n={c3p['n']}")

    # Theorem-5 N-scaling view (slope should become MORE negative as N grows)
    print("\n" + "-" * 92)
    print("C2 N-scaling view (Theorem 5 prediction: slope ↓ as N ↑) — one row per run:")
    for gid, N, sl, ci in sorted(c2_nscaling, key=lambda t: (t[0], t[1])):
        print(f"   {gid:22s} N={N:<3d} slope={sl:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
    if len({gid for gid, *_ in c2_nscaling}) and max(n for _, n, *_ in c2_nscaling) == \
       min(n for _, n, *_ in c2_nscaling):
        print("   (only one N present — run ≥2 N values per model to test the N-scaling slope)")

    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    # headline PASS matrix
    print("\n" + "=" * 92)
    print(f"{'group':26s} {'C1':^6s} {'C2':^6s} {'C3':^6s}   (generalization matrix)")
    for k, r in summary.items():
        f = lambda b: " ✓ " if b else " ✗ "
        print(f"{k:26s} {f(r['C1']['PASS']):^6s} {f(r['C2']['pooled']['PASS']):^6s} "
              f"{f(r['C3']['pooled']['PASS']):^6s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
