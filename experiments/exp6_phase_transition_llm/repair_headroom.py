"""Theorem 6 (repair headroom) empirical backbone — offline, NO LLM calls.

Reads the exp6 g-sweep raw states (data/processed/exp6_<bench>_N*_g*.npz), truth=index 0,
and computes, per coupling g (paired on question text with g=0):

  p_+- = P(Y_0=Y*, Y_g!=Y*)   copying BREAKS a correct plurality  (gross damage G)
  p_-+ = P(Y_0!=Y*, Y_g=Y*)   copying RESCUES a wrong plurality
  H_repair = p_+- - p_-+       net recoverable headroom of any IRIC intervention
  eps      = P(Y_g!=Y_0)       winner-change rate (decision susceptibility ΔY)

Checks the two exact statements of theory/05_theorems/thm6_repair_headroom.md:
  Proposition 6.1 (identity):  A_0 - A_g == p_+- - p_-+     (assumption-free)
  Theorem 6.2 (bound):         |A_0 - A_g| <= eps
Writes results/tables/repair_headroom.json.
"""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
PAT = re.compile(r"_g([0-9.]+)_q(\d+)_([0-9a-f]+)\.npz$")


def plurality(x_round: np.ndarray) -> int:
    """Plurality option over informative agents; top-tie -> -1 (scored wrong). Truth=0."""
    info = x_round.max(axis=1) > x_round.min(axis=1)
    xi = x_round[info]
    if len(xi) == 0:
        return -1
    counts = np.bincount(np.argmax(xi, axis=1), minlength=x_round.shape[1])
    order = np.argsort(counts)[::-1]
    if len(order) > 1 and counts[order[0]] == counts[order[1]]:
        return -1
    return int(order[0])


def index(glob_pat: str):
    by = collections.defaultdict(dict)     # hash -> {g: path}
    for f in glob.glob(str(PROC / glob_pat)):
        m = PAT.search(f)
        if m:
            by[m.group(3)][m.group(1)] = f
    return by


def contingency(by, g: str) -> dict | None:
    a0 = ag = pmp = ppm = eps = n = 0
    for h, gm in by.items():
        if "0.00" not in gm or g not in gm:
            continue
        y0 = plurality(np.load(gm["0.00"], allow_pickle=True)["states"][-1])
        yg = plurality(np.load(gm[g], allow_pickle=True)["states"][-1])
        n += 1
        c0, cg = int(y0 == 0), int(yg == 0)
        a0 += c0; ag += cg
        pmp += int(c0 and not cg)
        ppm += int(cg and not c0)
        eps += int(y0 != yg)
    if n == 0:
        return None
    A0, Ag = a0 / n, ag / n
    p_pm, p_mp, e = pmp / n, ppm / n, eps / n
    return dict(g=float(g), n=n, A_0=A0, A_g=Ag, dA=A0 - Ag,
                p_break=p_pm, p_rescue=p_mp, H_repair=p_pm - p_mp, G_damage=p_pm, eps=e,
                identity_ok=abs((A0 - Ag) - (p_pm - p_mp)) < 1e-9,
                bound_ok=abs(A0 - Ag) <= e + 1e-9)


def main(bench="truthfulqa"):
    by = index(f"exp6_{bench}_N11_g*.npz")
    gs = sorted({g for gm in by.values() for g in gm} - {"0.00"}, key=float)
    rows = [r for g in gs if (r := contingency(by, g))]
    print(f"{'g':>5} {'n':>4} {'A_0':>5} {'A_g':>5} {'A0-Ag':>7} | {'p_brk':>6} {'p_rsc':>6} "
          f"{'H_rep':>6} {'eps':>5} | {'id':>3} {'bnd':>4}")
    for r in rows:
        print(f"{r['g']:>5.2f} {r['n']:>4} {r['A_0']:5.3f} {r['A_g']:5.3f} {r['dA']:+7.3f} | "
              f"{r['p_break']:6.3f} {r['p_rescue']:6.3f} {r['H_repair']:+6.3f} {r['eps']:5.3f} | "
              f"{'✓' if r['identity_ok'] else '✗':>3} {'✓' if r['bound_ok'] else '✗':>4}")
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "repair_headroom.json"
    out.write_text(json.dumps({"bench": bench, "truth_index": 0, "rows": rows}, indent=2))
    print(f"\nH_repair = net recoverable headroom of a perfect IRIC intervention (Thm 6).")
    print(f"Identity (Prop 6.1) and bound (Thm 6.2) exact regardless of n. wrote {out}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "truthfulqa")
