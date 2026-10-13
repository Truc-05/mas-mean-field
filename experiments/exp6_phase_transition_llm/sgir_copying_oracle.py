"""SGIR oracle — COPYING axis (offline, NO LLM calls). See thm6_repair_headroom.md §5.

Susceptibility-Gated Independent Resampling: M(q)>tau accept the (copied) collective,
M(q)<=tau discard it and resample a FRESH INDEPENDENT ensemble. Uses exp6 g-sweep states.

  A_current  = plurality at g=COPY final round     (herded/copying, rho->1)
  A_recovery = plurality at g=0    final round      (fresh independent)
  M(q)       = round-0 independent runner-up gap     (leakage-free, deployable gate)
Gate: PASS if low-margin recovery gain >= +0.05, else KILL (no actionability).
"""
from __future__ import annotations
import glob, re, collections
from pathlib import Path
import numpy as np
from repair_headroom import plurality, PROC  # reuse exact plurality rule

rng = np.random.default_rng([20260901])
PAT = re.compile(r"_g([0-9.]+)_q(\d+)_([0-9a-f]+)\.npz$")


def margin(x_round):
    info = x_round.max(1) > x_round.min(1); xi = x_round[info]
    if len(xi) == 0:
        return 0.0
    c = np.bincount(np.argmax(xi, 1), minlength=x_round.shape[1]); o = np.argsort(c)[::-1]
    return (c[o[0]] - (c[o[1]] if len(o) > 1 else 0)) / len(xi)


def boot_mean(x, nb=20000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    d = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)])
    return float(x.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def rows_for(by, copy_g):
    rows = []
    for h, gm in by.items():
        if "0.00" not in gm or copy_g not in gm:
            continue
        s0 = np.load(gm["0.00"], allow_pickle=True)["states"]
        sc = np.load(gm[copy_g], allow_pickle=True)["states"]
        rows.append(dict(M=margin(s0[0]),
                         A_rec=int(plurality(s0[-1]) == 0),
                         A_cur=int(plurality(sc[-1]) == 0)))
        rows[-1]["gain"] = rows[-1]["A_rec"] - rows[-1]["A_cur"]
    return rows


def report(by, copy_g):
    rows = rows_for(by, copy_g)
    n = len(rows); g = np.array([r["gain"] for r in rows]); M = np.array([r["M"] for r in rows])
    print("=" * 78)
    print(f"SGIR COPYING-AXIS ORACLE — indep(g=0) vs COPYING(g={copy_g})   n={n}")
    print("=" * 78)
    pt, lo, hi = boot_mean(g)
    print(f"  A_current={np.mean([r['A_cur'] for r in rows]):.3f}  "
          f"A_recovery={np.mean([r['A_rec'] for r in rows]):.3f}  "
          f"overall gain={pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
    thr = np.quantile(M, 1 / 3); low = [r["gain"] for r in rows if r["M"] <= thr]
    pt, lo, hi = boot_mean(low)
    verdict = "PASS" if pt >= 0.05 else ("marginal" if hi >= 0.05 else "KILL")
    print(f"  LOW-margin (M<={thr:.2f}, n={len(low)}): gain={pt:+.3f} [{lo:+.3f},{hi:+.3f}] -> {verdict}")


if __name__ == "__main__":
    import sys
    bench = sys.argv[1] if len(sys.argv) > 1 else "truthfulqa"
    by = collections.defaultdict(dict)
    for f in glob.glob(str(PROC / f"exp6_{bench}_N11_g*.npz")):
        m = PAT.search(f)
        if m:
            by[m.group(3)][m.group(1)] = f
    for cg in ["1.00", "2.00"]:
        report(by, cg)
