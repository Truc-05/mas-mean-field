"""K1 analysis — pool regimes, compute the oracle-headroom gate (no LLM calls).

Reads the per-question / per-replicate correctness emitted by k1_oracle_headroom.py
(results/tables/k1_yqa_{tag}_{regime}_N{N}.json) and reports the go/no-go quantities:

    A_best_fixed = max_a mean_q y_qa            best single FIXED policy
    A_oracle     = mean_q max_a y_qa            per-question best action (upper bound)
    H            = A_oracle − A_best_fixed        headroom a perfect per-q router buys
    H_M          = A_oracle(full) − A_oracle(no-mech)   [no-mech = single,indep_vote,stop]
                   does debate/diversity/shared_ev open headroom the base actions cannot?

Plus: question-level bootstrap CIs; a CROSS-FIT oracle (splits replicates to remove the
max-of-noisy-estimates winner's-curse); oracle action-selection frequency; an action
COMPLEMENTARITY matrix (who solves what the others miss); a cost-Pareto; and the gate:

    H ≲ 0.03            → NO-GO for an accuracy headline (don't build CDCR for accuracy)
    0.03 ≲ H ≲ 0.08     → CONDITIONAL (weigh cost-Pareto and H_M)
    H ≳ 0.08–0.10, CI ok → GO: proceed to K2

Pooled vs within-regime H separates cheap difficulty-routing headroom (easy↔hard, a mere
difficulty detector) from genuine per-question mechanism selection within a regime.

Usage:
  python .../k1_analyze.py [--tag v1] [file1.json file2.json ...]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from mfg_llm.metrics.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"

BASE = ["single", "indep_vote", "diversity", "shared_ev", "debate"]   # emitted by runner
ALL = BASE + ["stop"]
NO_MECH = ["single", "indep_vote", "stop"]                            # base (no mechanism)
MECH = ["diversity", "shared_ev", "debate"]
SOLVE_THRESH = 0.5              # a question is "solved" by a if ≥half its replicates correct


def load(paths: list[str]) -> tuple[list[dict], dict]:
    """Return (per-question records, first meta). Each record: {regime, K, y{action→mean},
    reps{action→[per-replicate]}} with stop added as all-zeros."""
    recs, meta0 = [], None
    for p in paths:
        blob = json.loads(Path(p).read_text())
        meta = blob["meta"]
        meta0 = meta0 or meta
        for q in blob["questions"]:
            R = len(next(iter(q["reps"].values())))
            reps = {a: list(map(float, q["reps"][a])) for a in BASE}
            reps["stop"] = [0.0] * R
            recs.append({"regime": meta["regime"], "K": q["K"], "R": R,
                         "reps": reps, "y": {a: float(np.mean(reps[a])) for a in ALL}})
    return recs, meta0


def ymat(recs: list[dict], actions: list[str]) -> np.ndarray:
    """(Q, |actions|) matrix of per-question mean accuracy."""
    return np.array([[r["y"][a] for a in actions] for r in recs], float)


def best_fixed(M: np.ndarray) -> tuple[float, int]:
    col = M.mean(0)
    i = int(np.argmax(col))
    return float(col[i]), i


def oracle(M: np.ndarray) -> float:
    return float(M.max(1).mean())


def headroom(recs: list[dict]) -> dict:
    Mall = ymat(recs, ALL)
    Mnm = ymat(recs, NO_MECH)
    A_best, bi = best_fixed(Mall)
    A_or = oracle(Mall)
    A_or_nm = oracle(Mnm)
    return {"A_best": A_best, "best_action": ALL[bi], "A_oracle": A_or,
            "A_oracle_nomech": A_or_nm, "H": A_or - A_best, "H_M": A_or - A_or_nm}


def boot_headroom(recs: list[dict], n_boot: int = 10000, seed=(101,)) -> dict:
    """Question-level paired bootstrap CIs for A_oracle, A_best, H, H_M."""
    Mall, Mnm = ymat(recs, ALL), ymat(recs, NO_MECH)
    rng = np.random.default_rng(list(seed))
    Q = len(recs)
    keys = ("A_oracle", "A_best", "H", "H_M")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, Q, Q)
        a_best = Mall[idx].mean(0).max()
        a_or = Mall[idx].max(1).mean()
        a_nm = Mnm[idx].max(1).mean()
        draws["A_oracle"].append(a_or)
        draws["A_best"].append(a_best)
        draws["H"].append(a_or - a_best)
        draws["H_M"].append(a_or - a_nm)
    q = lambda v, p: float(np.quantile(v, p))                          # noqa: E731
    return {k: (q(v, 0.025), q(v, 0.975)) for k, v in draws.items()}


def crossfit_oracle(recs: list[dict]) -> float:
    """Honest oracle via LEAVE-ONE-REPLICATE-OUT: for each held-out replicate t, pick the
    per-question action using the OTHER R−1 replicates' means, then SCORE that action on t;
    average over t and questions. This removes the winner's-curse inflation of max_a over
    noisy ŷ (selection and scoring use disjoint data) while keeping selection noise low
    (R−1 reps drive the pick, vs a 50/50 split's R/2). Falls back to plug-in when R<2."""
    vals = []
    for r in recs:
        R = r["R"]
        rep = {a: np.array(r["reps"][a], float) for a in ALL}
        if R < 2:
            vals.append(max(r["y"][a] for a in ALL))
            continue
        tot = {a: rep[a].sum() for a in ALL}
        acc = 0.0
        for t in range(R):
            pick = {a: (tot[a] - rep[a][t]) / (R - 1) for a in ALL}    # select on R−1 reps
            a_star = max(pick, key=pick.get)
            acc += rep[a_star][t]                                      # score on held-out t
        vals.append(acc / R)
    return float(np.mean(vals))


def oracle_frequency(recs: list[dict], cost: dict) -> dict:
    """How often each action is the per-question oracle pick (ties → cheaper action)."""
    order = sorted(ALL, key=lambda a: cost[a])                         # cheap-first tie-break
    counts = {a: 0 for a in ALL}
    for r in recs:
        best = max(r["y"].values())
        winners = [a for a in ALL if abs(r["y"][a] - best) < 1e-9]
        counts[min(winners, key=lambda a: order.index(a))] += 1
    Q = len(recs)
    return {a: counts[a] / Q for a in ALL}


def complementarity(recs: list[dict]) -> dict:
    """solves[a] = y_qa ≥ threshold. Matrix[a][b] = P(a solves & b fails); diagonal = solve
    rate of a. Also coverage (solved by someone) and unique-rescue per action."""
    S = {a: np.array([r["y"][a] >= SOLVE_THRESH for r in recs]) for a in ALL}
    M = {a: {b: float(np.mean(S[a] & ~S[b])) for b in ALL} for a in ALL}
    anyone = np.zeros(len(recs), bool)
    for a in ALL:
        anyone |= S[a]
    unique = {}
    for a in ALL:
        others = np.zeros(len(recs), bool)
        for b in ALL:
            if b != a:
                others |= S[b]
        unique[a] = float(np.mean(S[a] & ~others))
    return {"matrix": M, "solve_rate": {a: float(S[a].mean()) for a in ALL},
            "coverage": float(anyone.mean()), "unique_rescue": unique}


def cost_model(meta: dict) -> dict:
    N = meta["N"]
    rounds = meta.get("debate_rounds", 3)
    return {"stop": 0, "single": 1, "indep_vote": N, "diversity": N,
            "shared_ev": N, "debate": N * rounds}


def pareto(points: list[tuple[str, float, float]]) -> set[str]:
    """Non-dominated (max accuracy, min cost). points = [(name, cost, acc)]."""
    nd = set()
    for name, c, a in points:
        if not any(c2 <= c and a2 >= a and (c2, a2) != (c, a) for _, c2, a2 in points):
            nd.add(name)
    return nd


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:+.3f},{ci[1]:+.3f}]"


def report(recs: list[dict], meta: dict) -> None:
    cost = cost_model(meta)
    regimes = sorted({r["regime"] for r in recs})
    print("=" * 74)
    print(f"K1 ORACLE HEADROOM — pooled over {regimes}   Q={len(recs)}   "
          f"N={meta['N']} R={meta['R']} rounds={meta.get('debate_rounds')} model={meta['model']}")
    print(f"actions={ALL}   mechanism={MECH}   shared_ev={meta.get('shared_ev_cond')}")
    print("=" * 74)

    # ── per-action fixed accuracy + cost ────────────────────────────────────────
    Mall = ymat(recs, ALL)
    col = Mall.mean(0)
    print("\nper-action FIXED accuracy (mean_q y_qa)   [cost = LLM calls]:")
    for i, a in enumerate(ALL):
        print(f"  {a:11s} acc={col[i]:.3f}   cost={cost[a]}")

    # ── headroom + bootstrap CIs ────────────────────────────────────────────────
    hr = headroom(recs)
    ci = boot_headroom(recs)
    cf = crossfit_oracle(recs)
    print("\n" + "-" * 74)
    print(f"A_best_fixed   = {hr['A_best']:.3f}  ({hr['best_action']})   {_fmt_ci(ci['A_best'])}")
    print(f"A_oracle       = {hr['A_oracle']:.3f}   {_fmt_ci(ci['A_oracle'])}"
          f"     (cross-fit honest = {cf:.3f})")
    print(f"A_oracle_nomech= {hr['A_oracle_nomech']:.3f}   [single/indep_vote/stop only]")
    print(f"H  = A_oracle − A_best_fixed   = {hr['H']:+.3f}   {_fmt_ci(ci['H'])}")
    print(f"     cross-fit H (winner's-curse-corrected) = {cf - hr['A_best']:+.3f}")
    print(f"H_M = A_oracle(full) − A_oracle(no-mech) = {hr['H_M']:+.3f}   {_fmt_ci(ci['H_M'])}"
          f"   [headroom UNIQUE to mechanism actions]")

    # ── pooled vs within-regime (difficulty-routing vs genuine per-q selection) ──
    print("\nwithin-regime headroom (isolates genuine per-q selection from easy↔hard routing):")
    for rg in regimes:
        sub = [r for r in recs if r["regime"] == rg]
        h = headroom(sub)
        print(f"  {rg:5s} Q={len(sub):3d}  A_best={h['A_best']:.3f} ({h['best_action']:10s})  "
              f"A_oracle={h['A_oracle']:.3f}  H={h['H']:+.3f}  H_M={h['H_M']:+.3f}")
    within_mean = np.mean([headroom([r for r in recs if r["regime"] == rg])["H"] for rg in regimes])
    print(f"  pooled H={hr['H']:+.3f} vs mean within-regime H={within_mean:+.3f}  "
          f"→ difficulty-routing share ≈ {hr['H'] - within_mean:+.3f}")

    # ── oracle action-selection frequency ───────────────────────────────────────
    freq = oracle_frequency(recs, cost)
    print("\noracle action-selection frequency (share of Q where action is the per-q best):")
    for a in ALL:
        print(f"  {a:11s} {freq[a]*100:5.1f}%")

    # ── complementarity ─────────────────────────────────────────────────────────
    comp = complementarity(recs)
    print(f"\ncoverage (solved by ≥1 action, y≥{SOLVE_THRESH}) = {comp['coverage']:.3f}")
    print("solve-rate / unique-rescue per action:")
    for a in ALL:
        print(f"  {a:11s} solve={comp['solve_rate'][a]:.3f}  unique_rescue={comp['unique_rescue'][a]:.3f}")
    print("\ncomplementarity matrix  M[row][col] = P(row solves & col fails):")
    print("row\\col   " + " ".join(f"{a[:8]:>8s}" for a in ALL))
    for a in ALL:
        print(f"{a:9s} " + " ".join(f"{comp['matrix'][a][b]:8.3f}" for b in ALL))

    # ── cost-Pareto ─────────────────────────────────────────────────────────────
    pts = [(a, cost[a], col[i]) for i, a in enumerate(ALL)]
    pts.append(("ORACLE", float(np.mean([cost[max(r["y"], key=r["y"].get)] for r in recs])),
                hr["A_oracle"]))
    nd = pareto(pts)
    print("\ncost-accuracy Pareto (★ = non-dominated):")
    for name, c, a in sorted(pts, key=lambda t: t[1]):
        print(f"  {'★' if name in nd else ' '} {name:11s} cost={c:6.1f}  acc={a:.3f}")

    # ── gate verdict ────────────────────────────────────────────────────────────
    # The DECISION number is the CROSS-FIT (winner's-curse-corrected) H: max_a of noisy ŷ_qa
    # inflates the plug-in oracle, and at small R that bias is large (it is the honest
    # per-q-routing gain that survives on held-out replicates). Plug-in H + its bootstrap CI
    # bound the uncertainty; we require the plug-in CI to also clear 0 so the signal is real.
    cfH = cf - hr["A_best"]
    H, Hlo = hr["H"], ci["H"][0]
    ci_real = Hlo > 0.0
    print("\n" + "=" * 74)
    if cfH < 0.03 or not ci_real:
        why = "" if ci_real else "  (plug-in CI includes 0 → headroom not distinguishable from noise)"
        verdict = ("NO-GO for an accuracy headline — do NOT build CDCR to chase accuracy. "
                   "Reframe around cost-Pareto / calibration, or a different headline." + why)
    elif cfH < 0.08:
        verdict = ("CONDITIONAL — honest headroom is modest. Proceed only if (a) H_M shows "
                   "mechanism actions add unique headroom AND/OR (b) the cost-Pareto favours "
                   "routing. Otherwise the difficulty-router is the whole story.")
    else:
        verdict = "GO → build the CDCR controller and run K2. Cross-fit headroom clears the bar."
    print(f"GATE (decision = cross-fit H): cfH={cfH:+.3f}   "
          f"[support: plug-in H={H:+.3f} CI={_fmt_ci(ci['H'])}, H_M={hr['H_M']:+.3f}]")
    print(f"VERDICT: {verdict}")
    print("NOTE: shared_ev uses the CORRECT-leaning realization → its contribution to H/H_M "
          "is an UPPER BOUND on a real, quality-uncertain retriever.")
    print("=" * 74)

    out = TABLES / "k1_headroom_summary.json"
    out.write_text(json.dumps({"meta": meta, "Q": len(recs), "regimes": regimes,
                               "headroom": hr, "ci": ci, "crossfit_oracle": cf,
                               "oracle_freq": freq, "complementarity": comp,
                               "cost": cost}, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("files", nargs="*", help="explicit k1_yqa_*.json (else glob by tag)")
    args = ap.parse_args(argv)
    paths = args.files or sorted(glob.glob(str(TABLES / f"k1_yqa_{args.tag}_*_N*.json")))
    if not paths:
        raise SystemExit(f"no K1 json found for tag={args.tag} in {TABLES}")
    print("loading:", *[Path(p).name for p in paths])
    recs, meta = load(paths)
    report(recs, meta)


if __name__ == "__main__":
    main()
