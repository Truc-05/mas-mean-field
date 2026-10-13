"""Analysis B — same-D, different-mechanism (empirical C1 / Theorem 4). Offline, NO LLM calls.

Two mechanisms generating inter-agent dependence, on a COMMON dependence scale
D = mean pairwise vote-agreement of the correlated arm (computed identically for both):
  UPSTREAM  shared-evidence : exp6c_raw qwen (R=8 reps); corr arm = shared_correct_strong,
                              independent baseline = indep_correct_strong.
  DOWNSTREAM copying         : exp6 g-sweep qwen (1 real/q); corr arm = g>0, baseline = g=0.

Per question, mechanism effect vs its own independent baseline:
  dA   = A_corr - A_indep          (signed accuracy effect)
  ΔY   = P(Y_corr != Y_indep)      (decision change)
  p+-  = P(Y_indep=Y*, Y_corr!=Y*)  ;  p-+ = P(Y_indep!=Y*, Y_corr=Y*)
  H_repair = A_indep - A_corr = p+- - p-+   (recoverable by restoring independence)
Then compare the two mechanisms in MATCHED D bands. C1 smoking gun: at D_1≈D_2 the
consequences (ΔY / sign of dA / H_repair) differ.
"""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def plur(state):
    x = np.asarray(state, float); info = x.max(1) > x.min(1); xi = x[info]
    if len(xi) == 0:
        return -1
    c = np.bincount(np.argmax(xi, 1), minlength=x.shape[1]); o = np.argsort(c)[::-1]
    if len(o) > 1 and c[o[0]] == c[o[1]]:
        return -1
    return int(o[0])


def agreement(state):
    x = np.asarray(state, float); info = x.max(1) > x.min(1)
    v = np.argmax(x[info], 1)
    if len(v) < 2:
        return float("nan")
    return float(np.mean([v[i] == v[j] for i in range(len(v)) for j in range(i + 1, len(v))]))


def shared_rows():
    """exp6c_raw qwen (default-model files only): per-q averaged over R replicates."""
    rows = []
    for f in [PROC / "exp6c_raw_easy_N11.json", PROC / "exp6c_raw_hard_N11.json"]:
        d = json.load(open(f)); raw = d["raw"]
        for q in range(len(raw["none"])):
            D, dY, pmp, ppm, Ac, Ai = [], [], [], [], [], []
            for r in range(len(raw["shared_correct_strong"][q])):
                ys = plur(raw["shared_correct_strong"][q][r])
                yi = plur(raw["indep_correct_strong"][q][r])
                D.append(agreement(raw["shared_correct_strong"][q][r]))
                cs, ci = int(ys == 0), int(yi == 0)
                Ac.append(cs); Ai.append(ci); dY.append(int(ys != yi))
                pmp.append(int(ci and not cs)); ppm.append(int(cs and not ci))
            rows.append(dict(mech="shared", D=np.nanmean(D), dA=np.mean(Ac) - np.mean(Ai),
                             dY=np.mean(dY), p_break=np.mean(pmp), p_rescue=np.mean(ppm),
                             H=np.mean(pmp) - np.mean(ppm)))
    return rows


def copy_rows():
    """exp6 g-sweep qwen TruthfulQA: per-q per-g (1 realization)."""
    pat = re.compile(r"_g([0-9.]+)_q(\d+)_([0-9a-f]+)\.npz$")
    by = collections.defaultdict(dict)
    for f in glob.glob(str(PROC / "exp6_truthfulqa_N11_g*.npz")):
        m = pat.search(f)
        if m:
            by[m.group(3)][m.group(1)] = f
    rows = []
    for h, gm in by.items():
        if "0.00" not in gm:
            continue
        s0 = np.load(gm["0.00"], allow_pickle=True)["states"][-1]
        y0 = plur(s0); c0 = int(y0 == 0)
        for g, p in gm.items():
            if g == "0.00":
                continue
            sg = np.load(p, allow_pickle=True)["states"][-1]
            yg = plur(sg); cg = int(yg == 0)
            rows.append(dict(mech="copy", g=float(g), D=agreement(sg), dA=cg - c0,
                             dY=int(yg != y0), p_break=int(c0 and not cg),
                             p_rescue=int(cg and not c0), H=int(c0 and not cg) - int(cg and not c0)))
    return rows


def band_report(rows, edges):
    D = np.array([r["D"] for r in rows])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [r for r in rows if lo <= r["D"] < hi]
        if len(sub) < 5:
            print(f"    D∈[{lo:.2f},{hi:.2f}) n={len(sub):3d}  (too few)")
            continue
        print(f"    D∈[{lo:.2f},{hi:.2f}) n={len(sub):3d}  "
              f"dA={np.mean([r['dA'] for r in sub]):+.3f}  ΔY={np.mean([r['dY'] for r in sub]):.3f}  "
              f"H_rep={np.mean([r['H'] for r in sub]):+.3f}  "
              f"p+-={np.mean([r['p_break'] for r in sub]):.3f} p-+={np.mean([r['p_rescue'] for r in sub]):.3f}")


if __name__ == "__main__":
    sh, cp = shared_rows(), copy_rows()
    print(f"shared (upstream) rows: {len(sh)}   copy (downstream) rows: {len(cp)}")
    print(f"D range  shared [{min(r['D'] for r in sh):.2f},{max(r['D'] for r in sh):.2f}]  "
          f"copy [{np.nanmin([r['D'] for r in cp]):.2f},{np.nanmax([r['D'] for r in cp]):.2f}]")
    edges = [0.5, 0.7, 0.8, 0.9, 0.95, 1.001]
    print("\n== UPSTREAM shared-evidence — consequence vs dependence D ==");  band_report(sh, edges)
    print("\n== DOWNSTREAM copying — consequence vs dependence D ==");        band_report(cp, edges)
    print("\n== MATCHED-D contrast (same D, different mechanism) ==")
    for lo, hi in [(0.85, 0.95), (0.90, 1.001)]:
        s = [r for r in sh if lo <= r["D"] < hi]; c = [r for r in cp if lo <= r["D"] < hi]
        if len(s) >= 5 and len(c) >= 5:
            print(f"  D∈[{lo:.2f},{hi:.2f}):  shared(n={len(s)}) dA={np.mean([r['dA'] for r in s]):+.3f} "
                  f"ΔY={np.mean([r['dY'] for r in s]):.3f} H={np.mean([r['H'] for r in s]):+.3f}  |  "
                  f"copy(n={len(c)}) dA={np.mean([r['dA'] for r in c]):+.3f} "
                  f"ΔY={np.mean([r['dY'] for r in c]):.3f} H={np.mean([r['H'] for r in c]):+.3f}")
    TABLES.joinpath("analysis_B_matched_mechanism.json").write_text(
        json.dumps({"shared": sh, "copy": cp}, indent=2))
    print(f"\nwrote {TABLES/'analysis_B_matched_mechanism.json'}")
