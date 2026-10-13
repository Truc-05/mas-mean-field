"""Analysis C — matched-D falsification of C1 (offline, NO LLM calls).

Sharpens Analysis B: is the mechanism (upstream shared-evidence vs downstream copying)
consequential *after* conditioning on dependence magnitude D? Three independent tests, all
on the SAME per-question rows produced by Analysis B's extractors (reused, not re-run):

  (1) Caliper matched-D contrast at eps in {.02,.05,.10}: for each copying row, average the
      consequences of shared rows within |D_copy - D_shared| <= eps; report the mechanism gap
      in dY / dA / p+- / p-+ on the matched set, with a paired bootstrap CI. If the gap
      survives tight calipers, it is NOT a dependence-magnitude effect.
  (2) Nested linear-probability models on dY: M0: dY~D ; M1: +mechanism ; M2: +D*mechanism.
      Nested F-tests (numpy/scipy; statsmodels absent) test whether mechanism / interaction
      retains explanatory power after conditioning on D. This is the statistical form of C1.
  (3) Nearest-neighbour matching on the confounder triple (D, A0, margin): estimate
      E[dY|copy] - E[dY|shared] in the matched set, killing the objection that the copying arm
      merely has a different baseline-difficulty / margin / D distribution.

Reuses plur()/agreement() semantics from analysis_B_matched_mechanism.py; adds A0 (baseline
independent accuracy) and margin (baseline plurality vote-margin) per row. analysis_B stays
byte-identical (frozen). Writes results/tables/analysis_C_matched_D_falsification.json.
"""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path
import numpy as np
from scipy.stats import f as fdist
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
RNG = np.random.default_rng(0)


# ---- shared geometry (same informative-agent filter as analysis_B) -----------------
def _info(state):
    x = np.asarray(state, float)
    return x, x.max(1) > x.min(1)


def plur(state):
    x, info = _info(state); xi = x[info]
    if len(xi) == 0:
        return -1
    c = np.bincount(np.argmax(xi, 1), minlength=x.shape[1]); o = np.argsort(c)[::-1]
    if len(o) > 1 and c[o[0]] == c[o[1]]:
        return -1
    return int(o[0])


def agreement(state):
    x, info = _info(state); v = np.argmax(x[info], 1)
    if len(v) < 2:
        return float("nan")
    return float(np.mean([v[i] == v[j] for i in range(len(v)) for j in range(i + 1, len(v))]))


def margin(state):
    """Plurality vote margin (n_1 - n_2)/n_informative of the baseline arm."""
    x, info = _info(state); xi = x[info]
    if len(xi) == 0:
        return float("nan")
    c = np.sort(np.bincount(np.argmax(xi, 1), minlength=x.shape[1]))[::-1]
    n = len(xi)
    return float((c[0] - (c[1] if len(c) > 1 else 0)) / n)


# ---- row extractors (augmented with A0 + margin of the baseline/independent arm) ----
def shared_rows():
    rows = []
    for f in [PROC / "exp6c_raw_easy_N11.json", PROC / "exp6c_raw_hard_N11.json"]:
        raw = json.load(open(f))["raw"]
        for q in range(len(raw["none"])):
            D, dY, pmp, ppm, Ac, Ai, Mg = [], [], [], [], [], [], []
            for r in range(len(raw["shared_correct_strong"][q])):
                ys = plur(raw["shared_correct_strong"][q][r])
                yi = plur(raw["indep_correct_strong"][q][r])
                D.append(agreement(raw["shared_correct_strong"][q][r]))
                Mg.append(margin(raw["indep_correct_strong"][q][r]))
                cs, ci = int(ys == 0), int(yi == 0)
                Ac.append(cs); Ai.append(ci); dY.append(int(ys != yi))
                pmp.append(int(ci and not cs)); ppm.append(int(cs and not ci))
            rows.append(dict(mech="shared", D=float(np.nanmean(D)), A0=float(np.mean(Ai)),
                             margin=float(np.nanmean(Mg)), dA=float(np.mean(Ac) - np.mean(Ai)),
                             dY=float(np.mean(dY)), p_break=float(np.mean(pmp)),
                             p_rescue=float(np.mean(ppm)), H=float(np.mean(pmp) - np.mean(ppm))))
    return rows


def copy_rows():
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
        y0 = plur(s0); c0 = int(y0 == 0); mg0 = margin(s0)
        for g, p in gm.items():
            if g == "0.00":
                continue
            sg = np.load(p, allow_pickle=True)["states"][-1]
            yg = plur(sg); cg = int(yg == 0)
            rows.append(dict(mech="copy", g=float(g), D=float(agreement(sg)), A0=float(c0),
                             margin=float(mg0), dA=float(cg - c0), dY=float(int(yg != y0)),
                             p_break=float(int(c0 and not cg)), p_rescue=float(int(cg and not c0)),
                             H=float(int(c0 and not cg) - int(cg and not c0))))
    return rows


# ---- (1) caliper matched-D contrast ------------------------------------------------
def caliper_contrast(sh, cp, eps, metric, n_boot=2000):
    Ds = np.array([r["D"] for r in sh]); Vs = np.array([r[metric] for r in sh])
    diffs = []  # per copy-row: V_copy - mean(V_shared within eps)
    for r in cp:
        if not np.isfinite(r["D"]):
            continue
        sel = np.abs(Ds - r["D"]) <= eps
        if sel.sum() == 0:
            continue
        diffs.append(r[metric] - float(np.nanmean(Vs[sel])))
    diffs = np.array([d for d in diffs if np.isfinite(d)])
    if len(diffs) < 5:
        return dict(eps=eps, metric=metric, n=int(len(diffs)), gap=None, ci=None)
    boot = [float(np.mean(RNG.choice(diffs, len(diffs), replace=True))) for _ in range(n_boot)]
    return dict(eps=eps, metric=metric, n=int(len(diffs)), gap=float(diffs.mean()),
                ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])


# ---- (2) nested linear-probability models -----------------------------------------
def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(((y - X @ beta) ** 2).sum())
    return beta, rss


def nested_models(sh, cp):
    rows = [r for r in sh + cp if np.isfinite(r["D"])]
    y = np.array([r["dY"] for r in rows]); D = np.array([r["D"] for r in rows])
    M = np.array([1.0 if r["mech"] == "copy" else 0.0 for r in rows])
    n = len(y); one = np.ones(n)
    X0 = np.column_stack([one, D])
    X1 = np.column_stack([one, D, M])
    X2 = np.column_stack([one, D, M, D * M])
    out = {}
    prev = None
    for name, X in [("M0_D", X0), ("M1_D+mech", X1), ("M2_D+mech+DxM", X2)]:
        beta, rss = _ols(X, y); p = X.shape[1]
        r2 = 1 - rss / float(((y - y.mean()) ** 2).sum())
        rec = dict(k=p, rss=rss, r2=float(r2), beta=[float(b) for b in beta])
        if prev is not None:
            F = ((prev["rss"] - rss) / (p - prev["k"])) / (rss / (n - p))
            rec["F_vs_prev"] = float(F)
            rec["p_vs_prev"] = float(fdist.sf(F, p - prev["k"], n - p))
        out[name] = rec; prev = rec
    out["n"] = n
    out["mech_coef_given_D"] = out["M1_D+mech"]["beta"][2]  # effect of copy after conditioning on D
    return out


# ---- (3) nearest-neighbour matching on (D, A0, margin) -----------------------------
def nn_matching(sh, cp, n_boot=2000):
    def mat(rows):
        return np.array([[r["D"], r["A0"], r["margin"]] for r in rows], float)
    Zs, Zc = mat(sh), mat(cp)
    keep_s = np.all(np.isfinite(Zs), 1); keep_c = np.all(np.isfinite(Zc), 1)
    Zs, sh = Zs[keep_s], [r for r, k in zip(sh, keep_s) if k]
    Zc, cp = Zc[keep_c], [r for r, k in zip(cp, keep_c) if k]
    mu = np.vstack([Zs, Zc]).mean(0); sd = np.vstack([Zs, Zc]).std(0) + 1e-9
    Zs, Zc = (Zs - mu) / sd, (Zc - mu) / sd
    nn_s = NearestNeighbors(n_neighbors=1).fit(Zs)   # copy -> nearest shared
    _, idx_cs = nn_s.kneighbors(Zc)
    att = np.array([cp[i]["dY"] - sh[j[0]]["dY"] for i, j in enumerate(idx_cs)])  # E[dY|copy]-E[dY|shared]
    boot = [float(att[RNG.integers(0, len(att), len(att))].mean()) for _ in range(n_boot)]
    return dict(n_copy=len(cp), n_shared=len(sh), att_copy_minus_shared=float(att.mean()),
                ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])


if __name__ == "__main__":
    sh, cp = shared_rows(), copy_rows()
    print(f"shared rows: {len(sh)}   copy rows: {len(cp)}")
    print(f"D range  shared [{min(r['D'] for r in sh):.2f},{max(r['D'] for r in sh):.2f}]  "
          f"copy [{np.nanmin([r['D'] for r in cp]):.2f},{np.nanmax([r['D'] for r in cp]):.2f}]")

    print("\n== (1) CALIPER matched-D contrast (copy - shared at matched D) ==")
    calip = {}
    for eps in (0.02, 0.05, 0.10):
        calip[f"eps_{eps}"] = {}
        print(f"  eps={eps}:")
        for metric in ("dY", "dA", "p_break", "p_rescue"):
            r = caliper_contrast(sh, cp, eps, metric)
            calip[f"eps_{eps}"][metric] = r
            g = f"{r['gap']:+.3f}" if r["gap"] is not None else "  n/a"
            ci = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]" if r["ci"] else ""
            sig = "*" if (r["ci"] and (r["ci"][0] > 0 or r["ci"][1] < 0)) else " "
            print(f"    {metric:9s} n={r['n']:4d}  gap={g} {ci} {sig}")

    print("\n== (2) NESTED linear-probability models on dY ==")
    nm = nested_models(sh, cp)
    for k in ("M0_D", "M1_D+mech", "M2_D+mech+DxM"):
        rec = nm[k]
        extra = (f"  F={rec['F_vs_prev']:.2f} p={rec['p_vs_prev']:.2e}"
                 if "F_vs_prev" in rec else "")
        print(f"  {k:16s} R2={rec['r2']:.4f}{extra}")
    print(f"  mechanism(copy) coef after conditioning on D: {nm['mech_coef_given_D']:+.3f}")

    print("\n== (3) NN matching on (D, A0, margin) ==")
    nn = nn_matching(sh, cp)
    print(f"  E[dY|copy]-E[dY|shared] (matched) = {nn['att_copy_minus_shared']:+.3f} "
          f"[{nn['ci'][0]:+.3f},{nn['ci'][1]:+.3f}]  (n_copy={nn['n_copy']}, n_shared={nn['n_shared']})")

    out = dict(n_shared=len(sh), n_copy=len(cp), caliper=calip, nested_models=nm, nn_matching=nn)
    TABLES.joinpath("analysis_C_matched_D_falsification.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {TABLES/'analysis_C_matched_D_falsification.json'}")
