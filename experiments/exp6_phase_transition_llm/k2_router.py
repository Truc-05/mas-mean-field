"""K2 — observability gate for CDCR: can a router on PRE-INTERVENTION OBSERVABLE features
capture the honest per-question headroom K1 said exists (cross-fit H ≈ +0.037 pooled)?

K1 answered "is there headroom to control?" (conditional yes). K2 answers the harder,
deployment-relevant question: "is the observable state rich enough to EXPLOIT it?" — i.e.
does an action a* chosen from features available BEFORE any intervention beat the best fixed
policy, and how much of the oracle gap does it close.

    V̂_a(x) = P(Y_a = 1 | x)   fit per action a on pre-intervention features x
    a*(x)   = argmax_a [ V̂_a(x) − λ C_a ]         (λ=0 ⇒ pure accuracy; λ>0 ⇒ cost-aware)

This is the final CDCR controller structure and gives the accuracy–cost Pareto directly.

FEATURES x_q (OBSERVABLE, no leakage). Computed from the exp6C `none` (no-evidence,
independent) belief cache — the same probe a deployed controller would run first. Every
feature is PERMUTATION-INVARIANT in the option order, so the truth index (canonical 0) is
never used → the pre-intervention state cannot leak the label:
    plurality_share, runnerup_gap (margin M), vote_entropy, n_unique_frac, informative_frac,
    conf_gap_mean, conf_peak_mean, within_rep_agreement, plurality_stability, K.
We deliberately DROP the boundary table's A_none / p_correct_votes / correctness_lead /
dist_to_boundary — those read the truth index and would be deployment leakage.
Post-action D (error dependence) is also excluded: it is only known AFTER acting.

LABELS Y_qa come from K1 (k1_yqa_{tag}_{regime}_N{N}.json). Features (from the boundary
`none` run) and labels (from K1) are INDEPENDENT stochastic draws of the same no-evidence
distribution on the SAME question set — so x_q is a genuine noisy pre-intervention signature,
never the same realization that produced the indep_vote/single label (no within-draw leakage).

HONEST EVALUATION. Leave-one-question-out cross-fit: for each held-out q, fit V̂_a on the
other questions, predict, route, then SCORE a* on q's held-out K1 accuracy. Compared against:
  A_bestfixed      best single fixed policy (the thing to beat)
  A_oracle (cf)    leave-one-replicate-out per-question oracle (honest upper bound)
  headroom captured = (A_router − A_bestfixed) / (A_oracle_cf − A_bestfixed)
Primary learner = RidgeCV (pre-registered; low variance at Q≈40–80). GradBoost / kNN /
difficulty-only (entropy+margin) reported as sensitivity. Paired question bootstrap CIs.

Usage (CPU only, no LLM/GPU):
  python k2_router.py --tag v1                 # pooled over regimes with ready features
  python k2_router.py --tag v1 --regimes easy  # single regime
Features auto-include a regime once its exp6c_belief_cache_{regime}.jsonl `none` cells are
complete (N*R per question); until HARD boundary finishes, pooled = easy (+ any ready hard q).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

import k1_analyze as K
from k1_hm_audit import crossfit_oracle_subset  # reused honest oracle

PROC = K.ROOT / "data" / "processed"
ALL = K.ALL                                     # single,indep_vote,diversity,shared_ev,debate,stop
FEATURES = ["plurality_share", "runnerup_gap", "vote_entropy", "n_unique_frac",
            "informative_frac", "conf_gap_mean", "conf_peak_mean",
            "within_rep_agreement", "plurality_stability", "K"]
DIFFICULTY_ONLY = ["vote_entropy", "runnerup_gap"]   # the cheap difficulty signal


# ── pre-intervention OBSERVABLE features from the `none` belief cache ────────────
def question_features(recs_none: list[dict]) -> dict | None:
    """recs_none = all cache rows for one (regime, qi) with cond='none'. Returns the
    observable feature dict, or None if coverage is incomplete. All measures pool over
    the R replicates × N agents and are permutation-invariant (never touch truth@0)."""
    if not recs_none:
        return None
    Kopt = len(recs_none[0]["shuffled"])
    votes, conf_gap, conf_peak = [], [], []
    n_info = n_tot = 0
    per_rep = defaultdict(list)                  # r -> list of informative votes that replicate
    for rec in recs_none:
        o = np.asarray(rec["opinion_shuffled"], float)
        n_tot += 1
        if o.max() > o.min():                    # informative (non-flat) belief
            n_info += 1
            v = int(np.argmax(o))
            votes.append(v)
            per_rep[rec["r"]].append(v)
            s = np.sort(o)[::-1]
            conf_gap.append(float(s[0] - s[1]))
            conf_peak.append(float(s[0]))
    if not votes:
        return None
    counts = np.bincount(votes, minlength=Kopt).astype(float)
    shares = counts / counts.sum()
    order = np.sort(shares)[::-1]
    nz = shares[shares > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    # within-replicate agreement: mean over reps of the top vote-share INSIDE that replicate
    wr = []
    rep_plur = []
    for r, vs in per_rep.items():
        c = np.bincount(vs, minlength=Kopt).astype(float)
        wr.append(float(c.max() / c.sum()))
        rep_plur.append(int(np.argmax(c)))
    within_rep_agreement = float(np.mean(wr)) if wr else 0.0
    # plurality stability: fraction of replicates whose within-rep plurality == modal one
    if rep_plur:
        modal = np.bincount(rep_plur).argmax()
        plurality_stability = float(np.mean([p == modal for p in rep_plur]))
    else:
        plurality_stability = 0.0
    return {
        "plurality_share": float(order[0]),
        "runnerup_gap": float(order[0] - (order[1] if order.size > 1 else 0.0)),
        "vote_entropy": entropy,
        "n_unique_frac": float((counts > 0).sum()) / Kopt,
        "informative_frac": n_info / n_tot,
        "conf_gap_mean": float(np.mean(conf_gap)),
        "conf_peak_mean": float(np.mean(conf_peak)),
        "within_rep_agreement": within_rep_agreement,
        "plurality_stability": plurality_stability,
        "K": float(Kopt),
        "_n_cells": n_tot,
    }


def load_features(regime: str) -> dict[int, dict]:
    """qi -> observable feature dict, ONLY for questions with complete `none` coverage."""
    path = PROC / f"exp6c_belief_cache_{regime}.jsonl"
    if not path.exists():
        return {}
    by_q: dict[int, list[dict]] = defaultdict(list)
    agents_per_q: dict[int, set] = defaultdict(set)
    reps_per_q: dict[int, set] = defaultdict(set)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:                                          # tolerate a torn final line if the
            rec = json.loads(line)                    # boundary is concurrently appending
        except json.JSONDecodeError:
            continue
        if rec.get("cond") != "none":
            continue
        qi = rec["qi"]
        by_q[qi].append(rec)
        agents_per_q[qi].add(rec["agent"])
        reps_per_q[qi].add(rec["r"])
    out = {}
    for qi, rows in by_q.items():
        n_agents, n_reps = len(agents_per_q[qi]), len(reps_per_q[qi])
        if n_agents * n_reps != len(rows):       # dedup/partial → skip to be safe
            continue
        feat = question_features(rows)
        # require the full N*R grid so the feature is a stable per-question signature
        if feat and feat["_n_cells"] >= n_agents * n_reps and n_reps >= 4:
            out[qi] = feat
    return out


# ── assemble aligned (X, Y) over questions that have BOTH features and K1 labels ─
def load_regime(regime: str, tag: str):
    """Return (X (Q,F), Ymean (Q,A), recs subset aligned to X rows, qids)."""
    feats = load_features(regime)
    kpath = K.TABLES / f"k1_yqa_{tag}_{regime}_N11.json"
    if not kpath.exists():
        return None
    recs, meta = K.load([str(kpath)])            # recs[qi] aligns to question qi (file order)
    rows, recs_keep, qids = [], [], []
    for qi, r in enumerate(recs):
        if qi in feats:
            rows.append([feats[qi][f] for f in FEATURES])
            recs_keep.append(r)
            qids.append((regime, qi))
    if not rows:
        return None
    return np.array(rows, float), np.array([[r["y"][a] for a in ALL] for r in recs_keep], float), \
        recs_keep, qids, meta


# ── cross-fit value predictions V̂_a(x) via leave-one-question-out ───────────────
def make_learner(kind: str):
    if kind == "ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    if kind == "gboost":
        return GradientBoostingRegressor(n_estimators=120, max_depth=2,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=0)
    if kind == "knn":
        return KNeighborsRegressor(n_neighbors=7, weights="distance")
    raise ValueError(kind)


def loqo_vhat(X: np.ndarray, Ymean: np.ndarray, kind: str,
              feat_idx: list[int] | None = None) -> np.ndarray:
    """(Q,A) cross-fit predicted value: for each held-out q, fit V̂_a on the other
    questions (scaler fit on train only) and predict a's success prob at x_q."""
    Q, A = Ymean.shape
    cols = feat_idx if feat_idx is not None else list(range(X.shape[1]))
    Xc = X[:, cols]
    Vhat = np.zeros((Q, A))
    for q in range(Q):
        tr = np.arange(Q) != q
        sc = StandardScaler().fit(Xc[tr])
        Xtr, xq = sc.transform(Xc[tr]), sc.transform(Xc[q:q + 1])
        for ai in range(A):
            m = make_learner(kind)
            if kind == "knn":                    # cap neighbours to the (Q−1) training rows
                m.set_params(n_neighbors=max(1, min(7, Xtr.shape[0])))
            m.fit(Xtr, Ymean[tr, ai])
            Vhat[q, ai] = float(m.predict(xq)[0])
    return Vhat


def route(Vhat: np.ndarray, cost_vec: np.ndarray, lam: float) -> np.ndarray:
    """Per-question action pick = argmax_a[V̂_a − λ C_a], ties broken toward CHEAPER a
    (matches k1_analyze.oracle_frequency)."""
    score = Vhat - lam * cost_vec[None, :]
    picks = np.zeros(len(Vhat), int)
    for q in range(len(Vhat)):
        best = score[q].max()
        cand = np.where(score[q] >= best - 1e-12)[0]
        picks[q] = cand[np.argmin(cost_vec[cand])]
    return picks


def boot_ci(diff: np.ndarray, n=10000, seed=(7,)):
    rng = np.random.default_rng(list(seed))
    d = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)])
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def evaluate(X, Ymean, recs, cost_vec, label: str) -> dict:
    Q, A = Ymean.shape
    A_bestfixed_col = Ymean.mean(0)
    bf_i = int(np.argmax(A_bestfixed_col))
    A_bestfixed = float(A_bestfixed_col[bf_i])
    bf_outcomes = Ymean[:, bf_i]                          # per-q outcome of the fixed policy
    A_oracle_plugin = float(Ymean.max(1).mean())
    A_oracle_cf = crossfit_oracle_subset(recs, ALL)
    denom = A_oracle_cf - A_bestfixed

    routers = {}
    for kind in ("ridge", "gboost", "knn"):
        Vhat = loqo_vhat(X, Ymean, kind)
        picks = route(Vhat, cost_vec, 0.0)
        out = Ymean[np.arange(Q), picks]
        A_router = float(out.mean())
        lo, hi = boot_ci(out - bf_outcomes)
        routers[kind] = {"A_router": A_router, "cost": float(cost_vec[picks].mean()),
                         "gain": A_router - A_bestfixed, "gain_ci": (lo, hi),
                         "captured": (A_router - A_bestfixed) / denom if denom > 1e-9 else float("nan"),
                         "pick_hist": {ALL[i]: int((picks == i).sum()) for i in range(A)},
                         "_Vhat": Vhat, "_picks": picks, "_out": out}
    # difficulty-only router (entropy + margin) on the primary learner
    fi = [FEATURES.index(f) for f in DIFFICULTY_ONLY]
    Vd = loqo_vhat(X, Ymean, "ridge", feat_idx=fi)
    dp = route(Vd, cost_vec, 0.0)
    do = Ymean[np.arange(Q), dp]
    routers["ridge_difficulty_only"] = {
        "A_router": float(do.mean()), "cost": float(cost_vec[dp].mean()),
        "gain": float(do.mean()) - A_bestfixed, "gain_ci": boot_ci(do - bf_outcomes),
        "captured": (float(do.mean()) - A_bestfixed) / denom if denom > 1e-9 else float("nan"),
        "pick_hist": {ALL[i]: int((dp == i).sum()) for i in range(A)}}

    # λ-sweep Pareto on the primary (ridge) cross-fit values
    Vp = routers["ridge"]["_Vhat"]
    pareto = []
    for lam in [0.0, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5]:
        pk = route(Vp, cost_vec, lam)
        pareto.append({"lam": lam, "acc": float(Ymean[np.arange(Q), pk].mean()),
                       "cost": float(cost_vec[pk].mean())})
    return {"label": label, "Q": Q, "A_bestfixed": A_bestfixed, "best_action": ALL[bf_i],
            "A_oracle_plugin": A_oracle_plugin, "A_oracle_cf": A_oracle_cf,
            "denom_cf": denom, "routers": routers, "pareto": pareto,
            "_bf_outcomes": bf_outcomes}


def report(ev: dict, cost_vec: np.ndarray) -> None:
    print("\n" + "=" * 80)
    print(f"K2 OBSERVABILITY — {ev['label']}   Q={ev['Q']}")
    print("=" * 80)
    print(f"A_bestfixed      = {ev['A_bestfixed']:.3f}  ({ev['best_action']})   ← policy to beat")
    print(f"A_oracle plug-in = {ev['A_oracle_plugin']:.3f}")
    print(f"A_oracle cross-f = {ev['A_oracle_cf']:.3f}   ← honest upper bound")
    print(f"controllable headroom (cross-fit) = A_oracle_cf − A_bestfixed = {ev['denom_cf']:+.3f}")
    print("-" * 80)
    print("router                    A_router  gain vs fixed   [95% CI]        headroom captured")
    for kind in ("ridge", "gboost", "knn", "ridge_difficulty_only"):
        r = ev["routers"][kind]
        cap = r["captured"]
        caps = f"{cap*100:5.0f}%" if cap == cap else "  n/a"
        star = "*" if (r["gain_ci"][0] > 0 or r["gain_ci"][1] < 0) else " "
        print(f"  {kind:22s}  {r['A_router']:.3f}    {r['gain']:+.3f}     "
              f"[{r['gain_ci'][0]:+.3f},{r['gain_ci'][1]:+.3f}]{star}   {caps}")
    print("  (primary = ridge; * = CI excludes 0. captured = gain / cross-fit controllable headroom)")
    print("-" * 80)
    ph = ev["routers"]["ridge"]["pick_hist"]
    print("ridge router action mix:", "  ".join(f"{a}={ph.get(a,0)}" for a in ALL))
    print("-" * 80)
    print("cost–accuracy Pareto (ridge V̂, sweep λ):   cost = mean LLM calls/question")
    for p in ev["pareto"]:
        print(f"  λ={p['lam']:<5g}  acc={p['acc']:.3f}  cost={p['cost']:5.1f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--regimes", default="easy,hard",
                    help="comma list; a regime is included only if its `none` features are ready")
    args = ap.parse_args(argv)
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    MIN_Q = 10                                    # below this a cross-fit eval is just noise
    loaded = {rg: load_regime(rg, args.tag) for rg in regimes}
    for rg, v in list(loaded.items()):
        if v is not None and v[0].shape[0] < MIN_Q:
            print(f"  [skip] {rg}: only {v[0].shape[0]} questions with ready features "
                  f"(< {MIN_Q}) — waiting for boundary run to fill it")
            loaded[rg] = None
    loaded = {rg: v for rg, v in loaded.items() if v is not None}
    if not loaded:
        raise SystemExit("no regime has BOTH ready features (≥%d q) and K1 labels" % MIN_Q)
    any_meta = next(iter(loaded.values()))[4]
    cost = K.cost_model(any_meta)
    cost_vec = np.array([cost[a] for a in ALL], float)

    print("K2 router — feature source: exp6c `none` belief cache (observable, leakage-free)")
    for rg, v in loaded.items():
        print(f"  {rg}: {v[0].shape[0]} questions with complete features + K1 labels")
    print(f"  features ({len(FEATURES)}): {FEATURES}")
    print(f"  cost (LLM calls): " + "  ".join(f"{a}={cost[a]}" for a in ALL))

    out_all = {}
    # per-regime
    for rg, (X, Ymean, recs, qids, meta) in loaded.items():
        ev = evaluate(X, Ymean, recs, cost_vec, f"regime={rg}")
        report(ev, cost_vec)
        out_all[rg] = {k: v for k, v in ev.items() if not k.startswith("_")}
        for r in out_all[rg]["routers"].values():
            for kk in ("_Vhat", "_picks", "_out"):
                r.pop(kk, None)
    # pooled (only if ≥2 regimes ready)
    if len(loaded) >= 2:
        X = np.vstack([v[0] for v in loaded.values()])
        Ymean = np.vstack([v[1] for v in loaded.values()])
        recs = [r for v in loaded.values() for r in v[2]]
        ev = evaluate(X, Ymean, recs, cost_vec, "POOLED " + "+".join(loaded))
        report(ev, cost_vec)
        pooled = {k: v for k, v in ev.items() if not k.startswith("_")}
        for r in pooled["routers"].values():
            for kk in ("_Vhat", "_picks", "_out"):
                r.pop(kk, None)
        out_all["pooled"] = pooled

    outp = K.TABLES / "k2_router_summary.json"
    outp.write_text(json.dumps(out_all, ensure_ascii=False, indent=2, default=float))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
