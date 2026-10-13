"""SGIR oracle — SHARED-EVIDENCE axis (offline, NO LLM calls). See thm6_repair_headroom.md.

Recovery from shared-evidence dependence using the matched instrument already in cache
(exp6c_boundary_*_N11.json): A_current = shared_correct_strong (D~0.7),
A_recovery = indep_correct_strong (D~0, matched evidence), gate M = runner-up gap of `none`.
Gate: PASS if low-margin recovery gain (A_indep - A_shared) >= +0.05, else KILL.
"""
from __future__ import annotations
import glob, json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
rng = np.random.default_rng([20260901])
SHARED, INDEP = "shared_correct_strong", "indep_correct_strong"


def load(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        for q in d["per_q"]:
            if SHARED not in q or INDEP not in q:
                continue
            mg = q["margin"]
            rows.append(dict(model=os.path.basename(p), regime=d["regime"],
                             A_cur=q[SHARED]["A_q"], A_rec=q[INDEP]["A_q"],
                             gain=q[INDEP]["A_q"] - q[SHARED]["A_q"], M=mg["runnerup_gap"]))
    return rows


def boot_mean(x, nb=20000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return 0.0, 0.0, 0.0
    d = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)])
    return float(x.mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))


def gate(rows, label):
    n = len(rows); M = np.array([r["M"] for r in rows])
    print("=" * 78); print(f"SGIR SHARED-EV ORACLE — {label}   n={n}"); print("=" * 78)
    pt, lo, hi = boot_mean([r["gain"] for r in rows])
    print(f"  A_current={np.mean([r['A_cur'] for r in rows]):.3f}  "
          f"A_recovery={np.mean([r['A_rec'] for r in rows]):.3f}  "
          f"overall gain={pt:+.3f} [{lo:+.3f},{hi:+.3f}]")
    thr = np.quantile(M, 1 / 3); low = [r["gain"] for r in rows if r["M"] <= thr]
    pt, lo, hi = boot_mean(low)
    verdict = "PASS" if pt >= 0.05 else ("marginal" if hi >= 0.05 else "KILL")
    print(f"  LOW-margin (M<={thr:.2f}, n={len(low)}): gain={pt:+.3f} [{lo:+.3f},{hi:+.3f}] -> {verdict}")


if __name__ == "__main__":
    paths = sorted(glob.glob(str(TABLES / "exp6c_boundary_*_N11.json")))
    rows = load(paths)
    gate(rows, "POOLED N=11, all model configs, easy+hard")
    for reg in ("easy", "hard"):
        gate([r for r in rows if r["regime"] == reg], f"N=11 {reg} (all models)")
