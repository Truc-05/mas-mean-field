"""CI damage–repair gate (no LLM calls). Reads ci_enrich.py output, computes CI_PROTOCOL_FROZEN.md.

Design A′ — damage–repair on a controlled copying-induction:
    A_indep(q) = y_indep_vote(q)   (independent state)
    A_copy(q)  = y_debate(q)       (coupled-debate / contaminated state)
    A_CI(q)    = y_a_CI(q)         (counterfactual isolation)
    C_q        = A_indep − A_copy                          # copying-HARM contrast (>0 ⇒ copying hurt)
    S_copy     = { q : C_q > 0  ∧  A_copy ≤ 0.5 }          # copying hurt AND consensus wrong ≥ ½ reps

FROZEN report:
    Δ_CI = mean_{S_copy}(A_CI − A_copy)                    # PRIMARY: recovery vs contaminated state
    RR   = (mean A_CI − mean A_copy) / (mean A_indep − mean A_copy)  over S_copy   # Recovery Ratio
    slope of (A_CI − A_common) on C_q  > 0                 # SECONDARY mechanism-match (continuous)
    I    = mean_{S_copy}(A_CI−A_common) − mean_{S_common}(A_CI−A_common)           # discrete DiD echo

LOCKED GATE on Δ_CI: <0.02 KILL branch · 0.02–0.05 weak · ≥0.05 (&≥2 models) GO · ≥0.08 (&slope>0) STRONG.
Necessary for any mechanism-matched claim: slope>0 / I>0 AND blunt-flip guardrails pass.
Deployment strength: RR ≥ ~100% (A_CI ≥ A_indep).

Run:  python ci_enrich_analyze.py [--glob 'ci_enrich_*_hard_N*.json']
      python ci_enrich_analyze.py --selftest        # no-GPU synthetic known-answer validation
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
TABLES = _HERE.parents[1] / "results" / "tables"

A0 = ["single", "indep_vote", "diversity", "shared_ev", "debate"]


def load(paths: list[str]) -> tuple[list[dict], dict]:
    by_qid: dict[int, dict] = {}
    meta0 = None
    for p in sorted(paths):
        blob = json.loads(Path(p).read_text())
        meta0 = meta0 or blob["meta"]
        for q in blob["questions"]:
            reps = {a: list(map(float, v)) for a, v in q["reps"].items()}
            by_qid[q["qid"]] = {"qid": q["qid"], "K": q["K"], "R": len(next(iter(reps.values()))),
                                "reps": reps, "y": {a: float(np.mean(v)) for a, v in reps.items()}}
    return [by_qid[k] for k in sorted(by_qid)], meta0


def _bootstrap(vals: np.ndarray, stat, n_boot=10000, seed=(505,)) -> tuple[float, float]:
    """Question-level bootstrap CI of stat(resampled_vals). vals is a (Q, …) array of per-q rows."""
    rng = np.random.default_rng(list(seed))
    Q = len(vals)
    if Q == 0:
        return (float("nan"), float("nan"))
    draws = [stat(vals[rng.integers(0, Q, Q)]) for _ in range(n_boot)]
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def gate(d: float) -> str:
    if d < 0.02:
        return "KILL algorithm branch — ship C1–C4, no operator #5 (copying cell stays empty)"
    if d < 0.05:
        return "WEAK — do NOT build the controller (report as a note)"
    if d < 0.08:
        return "GO (if same-sign in ≥2 models) — copying cell filled"
    return "STRONG (if slope/I>0) — build the two-tier adaptive controller"


def analyze(recs: list[dict]) -> dict:
    q = np.array([r["qid"] for r in recs])
    A_ind = np.array([r["y"]["indep_vote"] for r in recs])
    A_cop = np.array([r["y"]["debate"] for r in recs])
    A_ci = np.array([r["y"]["a_CI"] for r in recs])
    A_com = np.array([r["y"]["a_common"] for r in recs])
    A0max = np.array([max(r["y"][a] for a in A0) for r in recs])
    C = A_ind - A_cop                                              # copying-harm contrast

    s_copy = (C > 0) & (A_cop <= 0.5)                              # copying hurt AND consensus wrong ≥½
    # common-heavy label (existing convention) among non-copy-harmed questions
    A_sev = np.array([r["y"]["shared_ev"] for r in recs])
    common_heavy = np.abs(A_sev - A_ind) > np.abs(A_cop - A_ind)
    s_common = common_heavy & ~s_copy

    diff = A_ci - A_com                                            # gain_CI − gain_common (A0max cancels)

    # ---- PRIMARY: recovery Δ_CI on S_copy ----
    rec = (A_ci - A_cop)[s_copy]
    d_ci = float(rec.mean()) if rec.size else float("nan")
    d_ci_ci = _bootstrap(rec, lambda v: v.mean(), seed=(1,))

    # ---- Recovery Ratio (pooled over S_copy) ----
    def _rr(rows):                                                # rows: (n,3) = [A_ci, A_cop, A_ind]
        num = rows[:, 0].mean() - rows[:, 1].mean()
        den = rows[:, 2].mean() - rows[:, 1].mean()
        return num / den if abs(den) > 1e-9 else float("nan")
    rr_rows = np.stack([A_ci[s_copy], A_cop[s_copy], A_ind[s_copy]], axis=1) if s_copy.any() \
        else np.zeros((0, 3))
    rr = _rr(rr_rows) if rr_rows.size else float("nan")
    rr_ci = _bootstrap(rr_rows, _rr, seed=(2,))

    # ---- SECONDARY: continuous mechanism-match — slope of (A_CI−A_common) on C_q ----
    def _slope(rows):                                             # rows: (n,2) = [C, diff]
        x, yv = rows[:, 0], rows[:, 1]
        if len(x) < 2 or np.ptp(x) < 1e-9:
            return float("nan")
        return float(np.polyfit(x, yv, 1)[0])
    sl_rows = np.stack([C, diff], axis=1)
    slope = _slope(sl_rows)
    slope_ci = _bootstrap(sl_rows, _slope, seed=(3,))

    # ---- discrete DiD echo ----
    did_copy = float(diff[s_copy].mean()) if s_copy.any() else float("nan")
    did_common = float(diff[s_common].mean()) if s_common.any() else float("nan")
    I = did_copy - did_common

    # ---- blunt-flip guardrails ----
    # (a) a_CI must NOT beat a_common on S_common
    g1 = (A_ci - A_com)[s_common]
    g1_ci = _bootstrap(g1, lambda v: v.mean(), seed=(4,))
    # (b) a_CI must NOT hurt vs debate on non-copy controls (C_q ≤ 0)
    ctrl = C <= 0
    g2 = (A_ci - A_cop)[ctrl]
    g2_ci = _bootstrap(g2, lambda v: v.mean(), seed=(5,))

    guard_a_ok = not (g1_ci[0] > 0)          # CI not significantly > a_common on common-heavy
    guard_b_ok = not (g2_ci[1] < 0)          # CI not significantly < debate on non-copy controls
    slope_pos = slope_ci[0] > 0

    return {
        "Q": len(recs), "n_S_copy": int(s_copy.sum()), "n_S_common": int(s_common.sum()),
        "n_ctrl_noncopy": int(ctrl.sum()),
        "means_S_copy": {"A_indep": float(A_ind[s_copy].mean()) if s_copy.any() else float("nan"),
                         "A_copy": float(A_cop[s_copy].mean()) if s_copy.any() else float("nan"),
                         "A_CI": float(A_ci[s_copy].mean()) if s_copy.any() else float("nan"),
                         "A_common": float(A_com[s_copy].mean()) if s_copy.any() else float("nan"),
                         "A_a_copy": float(np.array([r["y"]["a_copy"] for r in recs])[s_copy].mean())
                         if s_copy.any() else float("nan")},
        "delta_CI": d_ci, "delta_CI_ci": list(d_ci_ci),
        "recovery_ratio": rr, "recovery_ratio_ci": list(rr_ci),
        "slope_C_q": slope, "slope_C_q_ci": list(slope_ci), "slope_positive": bool(slope_pos),
        "interaction_DiD": I, "did_copy": did_copy, "did_common": did_common,
        "guardrail_a_CI_vs_common_on_Scommon": {"mean": float(g1.mean()) if g1.size else float("nan"),
                                                "ci": list(g1_ci), "ok": bool(guard_a_ok)},
        "guardrail_a_CI_vs_debate_on_noncopy": {"mean": float(g2.mean()) if g2.size else float("nan"),
                                                "ci": list(g2_ci), "ok": bool(guard_b_ok)},
        "gate": gate(d_ci),
        "mechanism_matched": bool(slope_pos and guard_a_ok and guard_b_ok),
        "deployment_strength_RR_ge_100": bool(rr >= 1.0),
    }


def report(res: dict, meta: dict | None = None) -> None:
    m = meta or {}
    print("=" * 80)
    print(f"CI DAMAGE–REPAIR GATE — regime={m.get('regime','?')} N={m.get('N','?')} "
          f"R={m.get('R','?')} coupling={m.get('coupling','?')} model={m.get('model','?')}  Q={res['Q']}")
    print("=" * 80)
    print(f"S_copy (C_q>0 ∧ A_copy≤0.5): n={res['n_S_copy']}   "
          f"S_common: n={res['n_S_common']}   non-copy controls (C_q≤0): n={res['n_ctrl_noncopy']}")
    mc = res["means_S_copy"]
    print(f"\nOn S_copy:  A_indep={mc['A_indep']:.3f}  A_copy={mc['A_copy']:.3f}  "
          f"A_CI={mc['A_CI']:.3f}  A_common={mc['A_common']:.3f}  a_copy(dead)={mc['A_a_copy']:.3f}")
    print("\nPRIMARY (recovery):")
    print(f"   Δ_CI = A_CI − A_copy  = {res['delta_CI']:+.3f}  "
          f"CI[{res['delta_CI_ci'][0]:+.3f},{res['delta_CI_ci'][1]:+.3f}]")
    print(f"   Recovery Ratio        = {res['recovery_ratio']*100:6.1f}%  "
          f"CI[{res['recovery_ratio_ci'][0]*100:.1f}%,{res['recovery_ratio_ci'][1]*100:.1f}%]"
          f"   {'← DEPLOYMENT-STRENGTH (A_CI≥A_indep)' if res['deployment_strength_RR_ge_100'] else '(partial repair)'}")
    print("\nSECONDARY (mechanism-match):")
    print(f"   slope (A_CI−A_common) on C_q = {res['slope_C_q']:+.3f}  "
          f"CI[{res['slope_C_q_ci'][0]:+.3f},{res['slope_C_q_ci'][1]:+.3f}]   "
          f"{'✓ positive' if res['slope_positive'] else '✗ not > 0'}")
    print(f"   discrete DiD I = {res['interaction_DiD']:+.3f}  "
          f"(copy diff {res['did_copy']:+.3f} − common diff {res['did_common']:+.3f})")
    print("\nBLUNT-FLIP GUARDRAILS:")
    ga, gb = res["guardrail_a_CI_vs_common_on_Scommon"], res["guardrail_a_CI_vs_debate_on_noncopy"]
    print(f"   (a) a_CI−a_common on S_common = {ga['mean']:+.3f} CI[{ga['ci'][0]:+.3f},{ga['ci'][1]:+.3f}]"
          f"   {'OK (not anti-consensus everywhere)' if ga['ok'] else '✗ FAIL — beats a_common off-target'}")
    print(f"   (b) a_CI−debate on non-copy    = {gb['mean']:+.3f} CI[{gb['ci'][0]:+.3f},{gb['ci'][1]:+.3f}]"
          f"   {'OK (no harm to good consensus)' if gb['ok'] else '✗ FAIL — destroys good consensus'}")
    print(f"\n>>> GATE (Δ_CI={res['delta_CI']:+.3f}): {res['gate']}")
    print(f">>> mechanism-matched control justified: {res['mechanism_matched']} "
          f"(needs slope>0 AND both guardrails)")


def _synth() -> list[dict]:
    """No-GPU known-answer scenario. 40 q: 16 copying-harmed (CI partially repairs, a_common doesn't),
    16 common-heavy (a_common helps, CI ~neutral), 8 non-copy controls (CI harmless). R=8."""
    def mk(qid, y):                                               # y: dict action→mean (mult of 1/8)
        R = 8
        reps = {a: [1.0] * int(round(v * R)) + [0.0] * (R - int(round(v * R))) for a, v in y.items()}
        return {"qid": qid, "K": 4, "R": R, "reps": reps, "y": {a: float(np.mean(v)) for a, v in reps.items()}}
    recs = []
    base = dict(single=.3, diversity=.4, a_dissent=.35, a_copy=.375, shared_ev=.5)  # A0/others
    for i in range(16):        # copying-harmed: indep .625 > copy .25 (C=+.375, copy≤.5) → S_copy
        recs.append(mk(i, {**base, "indep_vote": .625, "debate": .25,
                           "a_CI": .5, "a_common": .375}))          # CI recovers .25→.5 (RR .667); common flat
    for i in range(16, 32):    # common-heavy: C≤0, a_common helps, CI neutral/below
        recs.append(mk(i, {**base, "indep_vote": .5, "debate": .5, "shared_ev": .75,
                           "a_CI": .5, "a_common": .625}))
    for i in range(32, 40):    # non-copy control: copying didn't hurt (C≤0), CI harmless
        recs.append(mk(i, {**base, "indep_vote": .5, "debate": .625, "shared_ev": .5,
                           "a_CI": .625, "a_common": .5}))
    return recs


def _selftest() -> None:
    recs = _synth()
    res = analyze(recs)
    report(res, {"regime": "SELFTEST", "N": 11, "R": 8, "coupling": 1.0, "model": "synthetic"})
    print("\n--- known-answer checks ---")
    ok = True
    checks = [
        ("n_S_copy == 16", res["n_S_copy"] == 16),
        ("Δ_CI ≈ +0.25", abs(res["delta_CI"] - 0.25) < 1e-6),
        ("RR ≈ 66.7%", abs(res["recovery_ratio"] - (0.5 - 0.25) / (0.625 - 0.25)) < 1e-6),
        ("slope(C_q) > 0", res["slope_C_q"] > 0 and res["slope_positive"]),
        ("DiD I > 0", res["interaction_DiD"] > 0),
        ("guardrail (a) OK", res["guardrail_a_CI_vs_common_on_Scommon"]["ok"]),
        ("guardrail (b) OK", res["guardrail_a_CI_vs_debate_on_noncopy"]["ok"]),
        ("gate == STRONG", res["gate"].startswith("STRONG")),
        ("mechanism_matched True", res["mechanism_matched"] is True),
        ("RR<100% ⇒ not deployment-strength", res["deployment_strength_RR_ge_100"] is False),
    ]
    for name, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\nSELFTEST {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="ci_enrich_*_hard_N*.json")
    ap.add_argument("--out", default=str(TABLES / "ci_gate.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        _selftest()
        return
    paths = glob.glob(str(TABLES / args.glob))
    if not paths:
        raise SystemExit(f"no files match {TABLES / args.glob}")
    recs, meta = load(paths)
    res = analyze(recs)
    report(res, meta)
    out = {"meta": meta, **res}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
