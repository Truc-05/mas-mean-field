"""K1 H_M audit — lock the exact definition of H_M and separate it from the deployable
headroom H, BEFORE H_M goes into the paper (a reviewer will otherwise ask why the
"mechanism headroom" 0.125 dwarfs the oracle-vs-best-fixed gap 0.064 / cross-fit 0.037).

WHAT H_M IS (locked here, no LLM calls; reads the same k1_yqa_*.json as k1_analyze):
    ŷ_qa   = (1/R) Σ_r y_qar            per-question EXPECTATION over the R stochastic
                                        replicates (NOT a single realization).
    A_oracle(S) = mean_q max_{a∈S} ŷ_qa    plug-in per-question oracle over action set S,
                                        the SAME 80-question ŷ table for every S.
    full    = {single,indep_vote,diversity,shared_ev,debate,stop}
    no-mech = {single,indep_vote,stop}
    H_M     = A_oracle(full) − A_oracle(no-mech).

WHY H_M ≠ the deployable headroom H. Both share A_oracle(full) but subtract different
baselines, so they are NOT comparable:
    H   = A_oracle(full) − A_bestfixed(full)      headroom a per-q ROUTER buys over the
                                                  best single FIXED policy   (deployable)
    H_M = A_oracle(full) − A_oracle(no-mech)      how much the per-q ORACLE gains from
                                                  OWNING mechanism actions   (diagnostic)
The no-mech ORACLE is itself an upper bound and can sit BELOW a single fixed mechanism
policy, so H_M over-credits mechanism. Exact identity (verified numerically below):

    H_M = H + Δ_fix − R_base ,   equivalently   H_M − H = A_bestfixed(full) − A_oracle(no-mech)
      Δ_fix  = A_bestfixed(full) − A_bestfixed(no-mech)   uplift from switching your ONE
                                                          fixed policy to a mechanism action
                                                          (a FIXED-policy benefit, no routing)
      R_base = A_oracle(no-mech) − A_bestfixed(no-mech)   per-q routing headroom WITHIN base

So the excess of H_M over the deployable H is (Δ_fix − R_base): almost all of H_M is the
FIXED-policy value of having e.g. shared_ev in the toolbox, NOT per-question routing — which
is exactly why best_fixed = shared_ev and cross-fit H ≈ 0 within the hard regime.

ESTIMATOR. H_M is reported PLUG-IN (max over noisy ŷ) AND cross-fit (leave-one-replicate-out,
selection and scoring on disjoint replicates) so the paper can quote the winner's-curse-honest
number. Plug-in inflates full more than no-mech (more actions to max over) → plug-in H_M is an
upper bound on the honest H_M.

Usage:  python k1_hm_audit.py [--tag v1] [file1.json file2.json ...]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import k1_analyze as K  # sibling; reuse the SAME loader so numbers match the gate exactly

ALL = K.ALL              # single,indep_vote,diversity,shared_ev,debate,stop
NO_MECH = K.NO_MECH      # single,indep_vote,stop
MECH = K.MECH            # diversity,shared_ev,debate


def crossfit_oracle_subset(recs: list[dict], actions: list[str]) -> float:
    """Leave-one-replicate-out honest oracle RESTRICTED to `actions` (generalises
    k1_analyze.crossfit_oracle, which is hard-wired to ALL). Selection on R−1 reps,
    scoring on the held-out rep; averaged over reps and questions."""
    vals = []
    for r in recs:
        R = r["R"]
        rep = {a: np.array(r["reps"][a], float) for a in actions}
        if R < 2:
            vals.append(max(r["y"][a] for a in actions))
            continue
        tot = {a: rep[a].sum() for a in actions}
        acc = 0.0
        for t in range(R):
            pick = {a: (tot[a] - rep[a][t]) / (R - 1) for a in actions}
            a_star = max(pick, key=pick.get)
            acc += rep[a_star][t]
        vals.append(acc / R)
    return float(np.mean(vals))


def audit(recs: list[dict], label: str) -> dict:
    Mall = K.ymat(recs, ALL)
    Mnm = K.ymat(recs, NO_MECH)
    A_or_full = K.oracle(Mall)
    A_or_nm = K.oracle(Mnm)
    A_bf_full, bi_full = K.best_fixed(Mall)
    A_bf_nm, bi_nm = K.best_fixed(Mnm)

    H = A_or_full - A_bf_full
    H_M = A_or_full - A_or_nm
    Delta_fix = A_bf_full - A_bf_nm
    R_base = A_or_nm - A_bf_nm
    # exact identity check: H_M == H + Delta_fix - R_base  (== A_bf_full - A_or_nm for the gap)
    identity_resid = H_M - (H + Delta_fix - R_base)

    # cross-fit (winner's-curse-honest) versions
    A_or_full_cf = crossfit_oracle_subset(recs, ALL)
    A_or_nm_cf = crossfit_oracle_subset(recs, NO_MECH)
    H_cf = A_or_full_cf - A_bf_full
    H_M_cf = A_or_full_cf - A_or_nm_cf

    # per-question gain from owning mechanism actions, and WHICH action realises the pick
    gain_q = Mall.max(1) - Mnm.max(1)                       # ≥ 0 by construction
    full_pick = [ALL[int(i)] for i in Mall.argmax(1)]
    # on questions where mechanism strictly helps (gain>0), how concentrated is the winner?
    mech_win = [(full_pick[i], gain_q[i]) for i in range(len(recs)) if gain_q[i] > 1e-9]
    winner_share = {a: 0 for a in ALL}
    for a, _ in mech_win:
        winner_share[a] += 1
    n_gain = len(mech_win)
    top_mech = max(MECH, key=lambda a: winner_share[a]) if n_gain else None

    return {
        "label": label, "Q": len(recs),
        "A_bestfixed_nomech": A_bf_nm, "best_nomech": NO_MECH[bi_nm],
        "A_bestfixed_full": A_bf_full, "best_full": ALL[bi_full],
        "A_oracle_nomech": A_or_nm, "A_oracle_full": A_or_full,
        "A_oracle_nomech_cf": A_or_nm_cf, "A_oracle_full_cf": A_or_full_cf,
        "H": H, "H_cf": H_cf, "H_M": H_M, "H_M_cf": H_M_cf,
        "Delta_fix": Delta_fix, "R_base": R_base, "identity_resid": identity_resid,
        "gap_HM_minus_H": A_bf_full - A_or_nm,              # == Delta_fix - R_base
        "n_gain_questions": n_gain, "winner_share": winner_share, "top_mech": top_mech,
    }


def _bar(x, w=28, lo=0.0, hi=1.0):
    n = int(round((x - lo) / (hi - lo) * w))
    return "█" * max(0, min(w, n))


def report(a: dict) -> None:
    L = a["label"]
    print("\n" + "=" * 78)
    print(f"H_M AUDIT — {L}   (Q={a['Q']})")
    print("=" * 78)
    print("action-set accuracies (ŷ = mean over R reps; oracle = mean_q max_a ŷ_qa):")
    print(f"  best FIXED  no-mech = {a['A_bestfixed_nomech']:.3f}  ({a['best_nomech']})")
    print(f"  best FIXED  full    = {a['A_bestfixed_full']:.3f}  ({a['best_full']})")
    print(f"  ORACLE      no-mech = {a['A_oracle_nomech']:.3f}   (cross-fit {a['A_oracle_nomech_cf']:.3f})")
    print(f"  ORACLE      full    = {a['A_oracle_full']:.3f}   (cross-fit {a['A_oracle_full_cf']:.3f})")
    print("-" * 78)
    print("the two quantities a reviewer will conflate:")
    print(f"  H   = A_oracle(full) − A_bestfixed(full) = {a['H']:+.3f}   "
          f"(cross-fit {a['H_cf']:+.3f})   ← DEPLOYABLE (router vs best fixed)")
    print(f"  H_M = A_oracle(full) − A_oracle(no-mech) = {a['H_M']:+.3f}   "
          f"(cross-fit {a['H_M_cf']:+.3f})   ← DIAGNOSTIC (oracle vs oracle)")
    print("-" * 78)
    print("exact decomposition  H_M = H + Δ_fix − R_base   (H_M − H = A_bestfixed(full) − A_oracle(no-mech)):")
    print(f"  Δ_fix  (fixed mechanism switch) = A_bf(full) − A_bf(no-mech) = {a['Delta_fix']:+.3f}")
    print(f"  R_base (routing within base)    = A_or(no-mech) − A_bf(no-mech) = {a['R_base']:+.3f}")
    print(f"  ⇒ H_M − H = Δ_fix − R_base = {a['gap_HM_minus_H']:+.3f}   "
          f"(identity residual {a['identity_resid']:+.1e})")
    frac_fixed = a["Delta_fix"] / a["H_M"] if a["H_M"] else float("nan")
    print(f"  ⇒ of H_M={a['H_M']:+.3f}, a FIXED policy already captures Δ_fix={a['Delta_fix']:+.3f} "
          f"({frac_fixed*100:.0f}%); only H={a['H']:+.3f} needs a router.")
    print("-" * 78)
    print(f"per-question: mechanism strictly raises the oracle on {a['n_gain_questions']}/{a['Q']} questions.")
    print("  winning full-oracle action on those questions (concentration ⇒ 'fixed action', not routing):")
    tot = max(1, a["n_gain_questions"])
    for act in ALL:
        c = a["winner_share"][act]
        if c:
            tag = " ←MECH" if act in MECH else ""
            print(f"    {act:11s} {c:3d} ({c/tot*100:4.0f}%) {_bar(c/tot)}{tag}")
    print(f"  dominant mechanism action = {a['top_mech']}  "
          f"→ H_M is largely 'own {a['top_mech']} as a FIXED policy', not per-question mechanism routing.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args(argv)
    paths = args.files or sorted(glob.glob(str(K.TABLES / f"k1_yqa_{args.tag}_*_N*.json")))
    if not paths:
        raise SystemExit(f"no K1 json for tag={args.tag} in {K.TABLES}")
    print("loading:", *[Path(p).name for p in paths])

    # pooled + each regime separately (per-regime is where the story is starkest)
    recs_all, meta = K.load(paths)
    audits = [audit(recs_all, "POOLED easy+hard")]
    for rg in sorted({r["regime"] for r in recs_all}):
        sub = [r for r in recs_all if r["regime"] == rg]
        audits.append(audit(sub, f"regime={rg}"))
    for a in audits:
        report(a)

    out = K.TABLES / "k1_hm_audit.json"
    out.write_text(json.dumps(audits, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    print("\nHEADLINE FOR THE PAPER:")
    P = audits[0]
    print(f"  Report H (cross-fit {P['H_cf']:+.3f}) as the controllable headroom. H_M={P['H_M']:+.3f} is a")
    print(f"  motivation diagnostic: {P['Delta_fix']/P['H_M']*100:.0f}% of it ({P['Delta_fix']:+.3f}) is the FIXED value")
    print(f"  of owning mechanism actions, not per-question routing. Do NOT headline H_M as controllable.")


if __name__ == "__main__":
    main()
