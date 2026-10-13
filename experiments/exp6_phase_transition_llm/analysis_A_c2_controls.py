"""Analysis A — C2 falsification: is the margin gate just uncertainty/disagreement/confidence?

Offline, NO LLM calls. Substrate = exp6c_raw_*_N11.json (the canonical C2 boundary dataset:
per-(condition,question,replicate) agent belief states, 5 model configs x 2 regimes).

Per question, OUTCOME = |ΔA_q| = |A_shared − A_indep| (the C2 susceptibility estimand).
PREDICTORS (all from the pre-intervention `none` condition, averaged over replicates):
  x1 margin           = normalized runner-up gap of the vote counts   (collective vote-boundary)
  x2 vote_entropy     = Shannon entropy of the vote distribution       (collective uncertainty)
  x3 disagreement     = 1 − Simpson = expected pairwise vote disagreement
  x4 confidence       = mean_i max_k softmax(belief_i)                 (AGENT-level certainty)
  x5 confidence_gap   = mean_i (top1 − top2 of softmax(belief_i))      (AGENT-level certainty)
x1–x3 are vote-distribution signals (collinear); x4–x5 are the *different axis* (agent
epistemic certainty) — the Bellina / 2605.27288 objection. The load-bearing test: does the
collective margin beat AGENT confidence?

Univariate: Spearman ρ and standardized OLS slope of each predictor vs |ΔA|.
Multivariate: |ΔA| ~ z(x1..x5); does β_margin stay < 0 (CI excludes 0) with controls?
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
rng = np.random.default_rng([20260901])


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def none_predictors(states_qr):
    """states_qr: list over replicates of (N,K) belief arrays (the `none` condition)."""
    margins, ents, disags, confs, cgaps = [], [], [], [], []
    for x in states_qr:
        x = np.asarray(x, float)
        info = x.max(1) > x.min(1)
        xi = x[info]
        if len(xi) == 0:
            continue
        votes = np.argmax(xi, 1)
        counts = np.bincount(votes, minlength=x.shape[1]).astype(float)
        p = counts / counts.sum()
        order = np.sort(counts)[::-1]
        margins.append((order[0] - (order[1] if len(order) > 1 else 0)) / len(xi))
        nz = p[p > 0]
        ents.append(float(-(nz * np.log(nz)).sum()))
        disags.append(float(1 - (p ** 2).sum()))
        probs = np.array([softmax(a) for a in xi])
        confs.append(float(np.mean(probs.max(1))))
        srt = np.sort(probs, 1)[:, ::-1]
        cgaps.append(float(np.mean(srt[:, 0] - srt[:, 1])))
    if not margins:
        return None
    return dict(margin=np.mean(margins), vote_entropy=np.mean(ents),
                disagreement=np.mean(disags), confidence=np.mean(confs),
                confidence_gap=np.mean(cgaps))


def plur_correct(state):
    x = np.asarray(state, float)
    info = x.max(1) > x.min(1); xi = x[info]
    if len(xi) == 0:
        return 0
    c = np.bincount(np.argmax(xi, 1), minlength=x.shape[1]); o = np.argsort(c)[::-1]
    if len(o) > 1 and c[o[0]] == c[o[1]]:
        return 0
    return int(o[0] == 0)


def load_rows():
    rows = []
    for f in sorted(glob.glob(str(PROC / "exp6c_raw_*_N11.json"))):
        d = json.load(open(f))
        raw = d["raw"]
        if not all(k in raw for k in ("none", "shared_correct_strong", "indep_correct_strong")):
            continue
        model = Path(f).name.replace("exp6c_raw_", "").replace("_N11.json", "")
        nq = len(raw["none"])
        for q in range(nq):
            pred = none_predictors(raw["none"][q])
            if pred is None:
                continue
            A_sh = np.mean([plur_correct(s) for s in raw["shared_correct_strong"][q]])
            A_in = np.mean([plur_correct(s) for s in raw["indep_correct_strong"][q]])
            rows.append(dict(model=model, dA=A_sh - A_in, absdA=abs(A_sh - A_in), **pred))
    return rows


def z(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / s if s > 0 else a * 0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ols_beta(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


PREDS = ["margin", "vote_entropy", "disagreement", "confidence", "confidence_gap"]


def boot_ci(fn, n, nb=5000):
    pt = fn(np.arange(n))
    d = np.array([fn(rng.integers(0, n, n)) for _ in range(nb)])
    d = d[np.isfinite(d)]
    return pt, float(np.quantile(d, .025)), float(np.quantile(d, .975))


def run(rows, label):
    n = len(rows)
    y = np.array([r["absdA"] for r in rows])
    Xz = {p: z([r[p] for r in rows]) for p in PREDS}
    print("=" * 84)
    print(f"ANALYSIS A — {label}   n={n}   mean|ΔA|={y.mean():.3f}")
    print("=" * 84)
    print("  UNIVARIATE (standardized slope on |ΔA|, and Spearman ρ):")
    for p in PREDS:
        x = Xz[p]
        sl, lo, hi = boot_ci(lambda idx: ols_beta(np.column_stack([np.ones(len(idx)), x[idx]]), y[idx])[1], n)
        rho = spearman([r[p] for r in rows], y)
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"    {p:15s} slope={sl:+.4f} [{lo:+.4f},{hi:+.4f}]{star}  spearman={rho:+.3f}")
    # multivariate
    Xmat = np.column_stack([np.ones(n)] + [Xz[p] for p in PREDS])
    print("  MULTIVARIATE  |ΔA| ~ z(all 5):")
    for k, p in enumerate(PREDS):
        sl, lo, hi = boot_ci(lambda idx: ols_beta(Xmat[idx], y[idx])[k + 1], n)
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"    β[{p:15s}] = {sl:+.4f} [{lo:+.4f},{hi:+.4f}]{star}")
    # the load-bearing dissociation: margin controlling ONLY for agent confidence
    print("  DISSOCIATION  |ΔA| ~ z(margin)+z(confidence)+z(confidence_gap):")
    sub = ["margin", "confidence", "confidence_gap"]
    Xs = np.column_stack([np.ones(n)] + [Xz[p] for p in sub])
    for k, p in enumerate(sub):
        sl, lo, hi = boot_ci(lambda idx: ols_beta(Xs[idx], y[idx])[k + 1], n)
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"    β[{p:15s}] = {sl:+.4f} [{lo:+.4f},{hi:+.4f}]{star}")


if __name__ == "__main__":
    rows = load_rows()
    run(rows, "POOLED N=11 (5 model configs, easy+hard)")
    for reg in ("easy", "hard"):
        run([r for r in rows if reg in r["model"] or r["model"].startswith(reg)], f"{reg} only")
    TABLES.joinpath("analysis_A_c2_controls.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=2))
    print(f"\nwrote {TABLES/'analysis_A_c2_controls.json'}")
