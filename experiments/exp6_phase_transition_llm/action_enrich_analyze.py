"""Action-enrichment analysis — the CDCR actionability gate (no LLM calls).

Reads the per-question/per-replicate A1 correctness emitted by action_enrich.py (merges
shards), and computes the FROZEN report (README §5 / action_enrich.py header):

    A0 = {single, indep_vote, diversity, shared_ev, debate, stop}
    A1 = A0 ∪ {a_copy, a_common, a_dissent}

    Δ_fixed  = A_bestfixed(A1) − A_bestfixed(A0)     uniform lift (good intervention, maybe no CDCR)
    Δ_oracle = A_oracle(A1)    − A_oracle(A0)
    H^CV_new = A_oracle^cf(A1) − A_bestfixed(A1)     routing headroom on A1 — the CDCR number

Cross-fit oracle = leave-one-replicate-out (k1_analyze.crossfit_oracle, generalized to an
action subset) — removes the winner's-curse of max_a over noisy ŷ. Question-level bootstrap CI
on H^CV_new. LOCKED GATE on H^CV_hard,new:
    < 0.02        → KILL CDCR (phenomenon real, no intervention leverage)
    0.02 – 0.05   → weak / conditional (continue only if complementarity is interpretable)
    ≥ 0.05        → actionability restored → fresh mechanism probes worthwhile → K3
    0.08 – 0.10+  → strong

SECONDARY WIN (standalone algorithm, no router): a NEW operator is the A1 best-fixed policy and
Δ_fixed > 0 ⇒ an always-apply intervention that beats every A0 baseline.

FROZEN mechanism-matching (pre-registered). Per-q label from A0 means: copying-heavy =
|y_debate−y_indep_vote| (downstream copying moved the vote), common-heavy = |y_shared_ev−y_indep_vote|
(upstream common evidence moved it). gain_op(q) = y_op(q) − max_{a∈A0} y_a(q). Prediction:
copying-heavy ⇒ a_copy gains most; common-heavy ⇒ a_common gains most (a targeted-action ×
mechanism interaction, not just marginal accuracy).

Run:  python action_enrich_analyze.py [--glob 'actionenrich_*_hard_N*.json']
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
TABLES = _HERE.parents[1] / "results" / "tables"

A0 = ["single", "indep_vote", "diversity", "shared_ev", "debate", "stop"]
NEW = ["a_copy", "a_common", "a_dissent"]
A1 = A0 + NEW


def load(paths: list[str]) -> tuple[list[dict], dict]:
    """Merge shard files → per-question records (deduped by qid, sorted). stop = all-zeros."""
    by_qid: dict[int, dict] = {}
    meta0 = None
    for p in sorted(paths):
        blob = json.loads(Path(p).read_text())
        meta0 = meta0 or blob["meta"]
        for q in blob["questions"]:
            R = len(next(iter(q["reps"].values())))
            reps = {a: list(map(float, q["reps"][a])) for a in q["reps"]}
            reps["stop"] = [0.0] * R
            by_qid[q["qid"]] = {"qid": q["qid"], "K": q["K"], "R": R, "reps": reps,
                                "y": {a: float(np.mean(reps[a])) for a in reps}}
    return [by_qid[k] for k in sorted(by_qid)], meta0


def ymat(recs: list[dict], actions: list[str]) -> np.ndarray:
    return np.array([[r["y"][a] for a in actions] for r in recs], float)


def best_fixed(recs: list[dict], actions: list[str]) -> tuple[float, str]:
    col = ymat(recs, actions).mean(0)
    i = int(np.argmax(col))
    return float(col[i]), actions[i]


def oracle(recs: list[dict], actions: list[str]) -> float:
    return float(ymat(recs, actions).max(1).mean())


def crossfit_perq(recs: list[dict], actions: list[str]) -> np.ndarray:
    """Per-question leave-one-replicate-out oracle value (select action on R−1 reps, score on
    the held-out rep; average over held-outs). Generalizes k1_analyze.crossfit_oracle to a
    subset. Falls back to plug-in max when R<2."""
    out = []
    for r in recs:
        R = r["R"]
        rep = {a: np.array(r["reps"][a], float) for a in actions}
        if R < 2:
            out.append(max(r["y"][a] for a in actions))
            continue
        tot = {a: rep[a].sum() for a in actions}
        acc = 0.0
        for t in range(R):
            pick = {a: (tot[a] - rep[a][t]) / (R - 1) for a in actions}
            a_star = max(pick, key=pick.get)
            acc += rep[a_star][t]
        out.append(acc / R)
    return np.array(out, float)


def hcv(recs: list[dict], actions: list[str]) -> tuple[float, float, float]:
    """(A_oracle_crossfit, A_best_fixed, H^CV) for an action subset."""
    a_cf = float(crossfit_perq(recs, actions).mean())
    a_bf, _ = best_fixed(recs, actions)
    return a_cf, a_bf, a_cf - a_bf


def boot_hcv(recs: list[dict], actions: list[str], n_boot=10000, seed=(202,)) -> tuple[float, float]:
    """Question-level bootstrap CI for H^CV = mean(crossfit_perq) − max_a mean(y_a)."""
    cfq = crossfit_perq(recs, actions)
    Y = ymat(recs, actions)
    rng = np.random.default_rng(list(seed))
    Q = len(recs)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, Q, Q)
        draws.append(cfq[idx].mean() - Y[idx].mean(0).max())
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def boot_delta(recs: list[dict], fn, n_boot=10000, seed=(303,)) -> tuple[float, float]:
    """Generic question-level bootstrap CI for a scalar fn(subset_of_recs)."""
    rng = np.random.default_rng(list(seed))
    Q = len(recs)
    draws = [fn([recs[i] for i in rng.integers(0, Q, Q)]) for _ in range(n_boot)]
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def gate(h: float) -> str:
    if h < 0.02:
        return "KILL CDCR — phenomenon real, no intervention leverage (retain K2/K2.5 chain)"
    if h < 0.05:
        return "WEAK / CONDITIONAL — continue only if complementarity is interpretable"
    if h < 0.08:
        return "ACTIONABILITY RESTORED — fresh mechanism probes worthwhile → K3"
    return "STRONG — clear actionable headroom"


def mechanism_matching(recs: list[dict]) -> dict:
    """Pre-registered targeted-action × mechanism interaction (see header)."""
    A0core = [a for a in A0 if a != "stop"]
    rows = []
    for r in recs:
        y = r["y"]
        best_a0 = max(y[a] for a in A0core)
        cop_score = abs(y["debate"] - y["indep_vote"])       # downstream copying moved the vote
        com_score = abs(y["shared_ev"] - y["indep_vote"])    # upstream common evidence moved it
        rows.append({
            "qid": r["qid"],
            "gain_copy": y["a_copy"] - best_a0,
            "gain_common": y["a_common"] - best_a0,
            "gain_dissent": y["a_dissent"] - best_a0,
            "cop_score": cop_score, "com_score": com_score,
            "label": "copying" if cop_score > com_score else "common",
        })
    cop = [x for x in rows if x["label"] == "copying"]
    com = [x for x in rows if x["label"] == "common"]
    mean = lambda L, k: float(np.mean([x[k] for x in L])) if L else float("nan")   # noqa: E731
    # interaction: on copying-heavy q, does a_copy out-gain a_common (and vice-versa)?
    inter_cop = [x["gain_copy"] - x["gain_common"] for x in cop]
    inter_com = [x["gain_common"] - x["gain_copy"] for x in com]
    boot = lambda v: (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if v else (float("nan"),) * 2  # noqa: E731
    rng = np.random.default_rng([404])

    def _bmean(v):
        v = np.array(v, float)
        if not len(v):
            return (float("nan"), float("nan"))
        d = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(10000)]
        return (float(np.quantile(d, .025)), float(np.quantile(d, .975)))

    return {
        "n_copying": len(cop), "n_common": len(com),
        "copying_heavy": {"a_copy": mean(cop, "gain_copy"), "a_common": mean(cop, "gain_common"),
                          "interaction(copy−common)": mean(cop, "gain_copy") - mean(cop, "gain_common"),
                          "interaction_ci": _bmean(inter_cop)},
        "common_heavy": {"a_common": mean(com, "gain_common"), "a_copy": mean(com, "gain_copy"),
                         "interaction(common−copy)": mean(com, "gain_common") - mean(com, "gain_copy"),
                         "interaction_ci": _bmean(inter_com)},
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="actionenrich_*_hard_N*.json")
    ap.add_argument("--out", default=str(TABLES / "actionenrich_gate.json"))
    args = ap.parse_args(argv)
    paths = glob.glob(str(TABLES / args.glob))
    if not paths:
        raise SystemExit(f"no files match {TABLES / args.glob}")
    recs, meta = load(paths)
    Q = len(recs)
    print("=" * 78)
    print(f"ACTION-ENRICHMENT GATE — regime={meta['regime']}  N={meta['N']}  R={meta['R']}  "
          f"Q={Q}  model={meta['model']}")
    print(f"files={len(paths)}  A0={[a for a in A0 if a!='stop']}+stop  NEW={NEW}")
    print("=" * 78)

    a0_cf, a0_bf, a0_h = hcv(recs, A0)
    a1_cf, a1_bf, a1_h = hcv(recs, A1)
    bf0_v, bf0_a = best_fixed(recs, A0)
    bf1_v, bf1_a = best_fixed(recs, A1)
    or0, or1 = oracle(recs, A0), oracle(recs, A1)
    d_fixed = bf1_v - bf0_v
    d_oracle = or1 - or0
    ci_hnew = boot_hcv(recs, A1)
    ci_dfixed = boot_delta(recs, lambda rr: best_fixed(rr, A1)[0] - best_fixed(rr, A0)[0])

    print("\nPER-ACTION mean accuracy (fixed policy):")
    col = ymat(recs, A1).mean(0)
    for a, v in sorted(zip(A1, col), key=lambda t: -t[1]):
        tag = "  <-- NEW" if a in NEW else ("  <-- A0 best" if a == bf0_a else "")
        print(f"   {a:12s} {v:.3f}{tag}")

    print("\nFROZEN REPORT (all three):")
    print(f"   A_bestfixed(A0) = {bf0_v:.3f}  ({bf0_a})")
    print(f"   A_bestfixed(A1) = {bf1_v:.3f}  ({bf1_a})")
    print(f"   Δ_fixed         = {d_fixed:+.3f}   CI[{ci_dfixed[0]:+.3f},{ci_dfixed[1]:+.3f}]"
          f"   {'(NEW operator is best-fixed → standalone win)' if bf1_a in NEW else ''}")
    print(f"   A_oracle(A0)    = {or0:.3f}     A_oracle(A1) = {or1:.3f}     Δ_oracle = {d_oracle:+.3f}")
    print(f"   H^CV(A0)        = {a0_h:+.3f}   [A_or_cf={a0_cf:.3f} − A_bf={a0_bf:.3f}]  (K1-prior ≈ +0.003 hard)")
    print(f"   H^CV_new(A1)    = {a1_h:+.3f}   [A_or_cf={a1_cf:.3f} − A_bf={a1_bf:.3f}]  CI[{ci_hnew[0]:+.3f},{ci_hnew[1]:+.3f}]")
    print(f"\n   >>> GATE (H^CV_hard,new = {a1_h:+.3f}):  {gate(a1_h)}")
    if d_fixed > 0 and bf1_a in NEW:
        print(f"   >>> SECONDARY WIN: '{bf1_a}' beats every A0 baseline by {d_fixed:+.3f} "
              "→ standalone algorithm contribution (validate cross-model next).")

    mm = mechanism_matching(recs)
    print("\nPRE-REGISTERED mechanism-matching (targeted-action × mechanism):")
    print(f"   copying-heavy q (n={mm['n_copying']}): a_copy gain={mm['copying_heavy']['a_copy']:+.3f}  "
          f"a_common gain={mm['copying_heavy']['a_common']:+.3f}  "
          f"Δ(copy−common)={mm['copying_heavy']['interaction(copy−common)']:+.3f} "
          f"CI[{mm['copying_heavy']['interaction_ci'][0]:+.3f},{mm['copying_heavy']['interaction_ci'][1]:+.3f}]")
    print(f"   common-heavy  q (n={mm['n_common']}): a_common gain={mm['common_heavy']['a_common']:+.3f}  "
          f"a_copy gain={mm['common_heavy']['a_copy']:+.3f}  "
          f"Δ(common−copy)={mm['common_heavy']['interaction(common−copy)']:+.3f} "
          f"CI[{mm['common_heavy']['interaction_ci'][0]:+.3f},{mm['common_heavy']['interaction_ci'][1]:+.3f}]")
    pred_ok = (mm['copying_heavy']['interaction(copy−common)'] > 0 and
               mm['common_heavy']['interaction(common−copy)'] > 0)
    print(f"   prediction (both Δ>0) {'HOLDS (directional)' if pred_ok else 'does NOT hold'} "
          "— interaction is the publishable claim, weigh above marginal accuracy.")

    out = {"meta": meta, "Q": Q,
           "A_bestfixed_A0": bf0_v, "best_A0": bf0_a,
           "A_bestfixed_A1": bf1_v, "best_A1": bf1_a,
           "A_oracle_A0": or0, "A_oracle_A1": or1,
           "delta_fixed": d_fixed, "delta_fixed_ci": ci_dfixed,
           "delta_oracle": d_oracle,
           "Hcv_A0": a0_h, "Hcv_new_A1": a1_h, "Hcv_new_ci": ci_hnew,
           "gate": gate(a1_h), "per_action_mean": dict(zip(A1, map(float, col))),
           "mechanism_matching": mm}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
