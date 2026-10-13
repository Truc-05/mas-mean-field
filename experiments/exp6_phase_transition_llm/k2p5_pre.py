"""K2.5-pre — LEAKY-OPTIMISTIC necessary-condition screen (0 GPU).

Question: with a RICHER, deliberately LEAKY feature set built entirely from the
existing 3-condition belief cache, can a router capture the honest headroom on the
CURRENT action set — or is the bottleneck actionability, not observability?

Features z_q (leaky on purpose — this is an UPPER BOUND, not a controller):
    margin      = runner-up gap M            (from boundary summary; difficulty axis)
    chi         = decision susceptibility χ̂  (from boundary summary; difficulty axis)
    d_shared    = TV(P_collective[shared_correct_strong], P_collective[none])   ← social/common-evidence elasticity proxy
    d_indep     = TV(P_collective[indep_correct_strong],  P_collective[none])   ← independent-evidence elasticity proxy

d_shared / d_indep are computed from the SAME shared/indep conditions that also
produce the Y_qa labels for actions shared_ev / indep_vote / diversity → the feature
peeks at the treatment realisation. That leakage makes any gain here an OPTIMISTIC
CEILING: if even this cannot capture the headroom, a clean fresh-probe version cannot.

Everything else (Ymean labels, recs, action set ALL, cost_vec, LOQO cross-fit, honest
cross-fit oracle) is IDENTICAL to K2 — only X is swapped. denom_cf therefore reproduces
K2 exactly (easy +0.024, hard +0.003, pooled +0.037) as a consistency check.

NOT a paper result. Necessary-condition screen only. Writes k2p5_pre_summary.json
(separate file — main k2_router_summary.json untouched).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import k1_analyze as K
import k2_router as k2

TAG = "v1"
REGIMES = ["easy", "hard"]
PROC = K.ROOT / "data" / "processed"
ALL = k2.ALL

# treatment conditions in the cache, mapped to the user's probe axes
SOCIAL_COND = "shared_correct_strong"   # common/shared evidence injection  ~ P_S proxy
INDEP_COND = "indep_correct_strong"     # independent evidence resample     ~ P_E proxy


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def collective_beliefs(regime: str) -> dict[int, dict[str, tuple[list, np.ndarray]]]:
    """(qi, cond) -> (canonical option keys, mean softmax belief over agents×reps).
    Canonical order = option TEXT sorted, so conditions of the same question align."""
    path = PROC / f"exp6c_belief_cache_{regime}.jsonl"
    acc: dict[tuple[int, str], dict] = defaultdict(lambda: {"keys": None, "P": []})
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        sc = r.get("scores")
        if not isinstance(sc, dict) or not sc:
            continue
        keys = sorted(sc)
        p = _softmax(np.array([sc[k] for k in keys], float))
        d = acc[(r["qi"], r["cond"])]
        d["keys"] = keys
        d["P"].append(p)
    out: dict[int, dict[str, tuple[list, np.ndarray]]] = defaultdict(dict)
    for (qi, cond), d in acc.items():
        out[qi][cond] = (d["keys"], np.mean(np.array(d["P"]), axis=0))
    return out


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def delta_pair(cb_q: dict, cond: str) -> float:
    """TV distance between the treatment-condition collective belief and `none`."""
    if "none" not in cb_q or cond not in cb_q:
        return 0.0
    kn, pn = cb_q["none"]
    kc, pc = cb_q[cond]
    if kc != kn:                                   # realign by option text if needed
        idx = [kc.index(k) for k in kn]
        pc = pc[np.array(idx)]
    return _tv(pn, pc)


# ── thin evaluate: same primitives as K2, X swapped, no FEATURES-indexed baseline ──
def evaluate_leaky(X, Ymean, recs, cost_vec, label, variants):
    Q, A = Ymean.shape
    bf_col = Ymean.mean(0)
    bf_i = int(np.argmax(bf_col))
    A_bf = float(bf_col[bf_i])
    bf_out = Ymean[:, bf_i]
    A_or_pl = float(Ymean.max(1).mean())
    A_or_cf = k2.crossfit_oracle_subset(recs, ALL)      # IDENTICAL honest oracle as K2
    denom = A_or_cf - A_bf

    routers = {}
    pareto_vhat = None
    for name, (kind, cols) in variants.items():
        Vh = k2.loqo_vhat(X, Ymean, kind, feat_idx=cols)
        pk = k2.route(Vh, cost_vec, 0.0)
        out = Ymean[np.arange(Q), pk]
        Ar = float(out.mean())
        lo, hi = k2.boot_ci(out - bf_out)
        ph = {ALL[i]: int((pk == i).sum()) for i in range(A)}
        collapse = max(ph.values()) / Q                 # fraction on the single most-picked action
        routers[name] = {"A_router": Ar, "cost": float(cost_vec[pk].mean()),
                         "gain": Ar - A_bf, "gain_ci": (lo, hi),
                         "captured": (Ar - A_bf) / denom if denom > 1e-9 else float("nan"),
                         "pick_hist": ph, "collapse_frac": collapse}
        if name == "ridge_full":
            pareto_vhat = Vh

    pareto = []
    for lam in [0.0, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5]:
        pk = k2.route(pareto_vhat, cost_vec, lam)
        pareto.append({"lam": lam, "acc": float(Ymean[np.arange(Q), pk].mean()),
                       "cost": float(cost_vec[pk].mean())})

    return {"label": label, "Q": Q, "A_bestfixed": A_bf, "best_action": ALL[bf_i],
            "A_oracle_plugin": A_or_pl, "A_oracle_cf": A_or_cf, "denom_cf": denom,
            "routers": routers, "pareto": pareto}


def report(ev):
    print("\n" + "=" * 84)
    print(f"K2.5-pre (LEAKY UPPER BOUND) — {ev['label']}   Q={ev['Q']}")
    print("=" * 84)
    print(f"A_bestfixed      = {ev['A_bestfixed']:.3f}  ({ev['best_action']})   ← policy to beat")
    print(f"A_oracle cross-f = {ev['A_oracle_cf']:.3f}   ← honest upper bound (same as K2)")
    print(f"controllable headroom (cross-fit) = {ev['denom_cf']:+.3f}   [K2 check: easy+0.024 hard+0.003 pool+0.037]")
    print("-" * 84)
    print("router (feat set)          A_router  gain     [95% CI]         captured   collapse")
    for name, r in ev["routers"].items():
        star = "*" if (r["gain_ci"][0] > 0 or r["gain_ci"][1] < 0) else " "
        cap = r["captured"]
        caps = f"{cap*100:6.0f}%" if cap == cap else "   n/a"
        print(f"  {name:24s}  {r['A_router']:.3f}   {r['gain']:+.3f}   "
              f"[{r['gain_ci'][0]:+.3f},{r['gain_ci'][1]:+.3f}]{star}  {caps}   "
              f"{r['collapse_frac']*100:3.0f}% on 1 act")
    print("  (* = CI excludes 0.  captured = gain / headroom.  collapse = mass on single most-picked action)")
    ph = ev["routers"]["ridge_full"]["pick_hist"]
    print("  ridge_full action mix:", "  ".join(f"{a}={ph.get(a,0)}" for a in ALL))
    print("  cost–acc Pareto (ridge_full):  " +
          "  ".join(f"λ{p['lam']:g}:{p['acc']:.3f}@{p['cost']:.1f}" for p in ev["pareto"][:7]))


def main():
    bsum = json.loads((K.TABLES / "exp6c_boundary_summary.json").read_text())
    Msum = {(row["regime"], row["qi"]): (float(row["M"]), float(row["chi"]))
            for row in bsum["rows"]}

    # feature-set variants (name -> (learner, column indices into z4))
    # z4 columns: 0=margin 1=chi 2=d_shared 3=d_indep
    VARIANTS = {
        "ridge_full":            ("ridge", None),
        "gboost_full":           ("gboost", None),
        "knn_full":              ("knn", None),
        "ridge_difficulty_only": ("ridge", [0, 1]),   # margin+chi (the axis K2 already showed fails)
        "ridge_mechanism_only":  ("ridge", [2, 3]),   # d_shared+d_indep (does the probe add anything?)
    }

    loaded = {}
    for rg in REGIMES:
        v = k2.load_regime(rg, TAG)                    # (Xpassive, Ymean, recs, qids, meta) — keep all but Xpassive
        if v is None:
            print(f"[skip] {rg}: no aligned features+labels")
            continue
        _, Ymean, recs, qids, meta = v
        cb = collective_beliefs(rg)
        rows = []
        for (r_, qi) in qids:
            M, chi = Msum[(r_, qi)]
            dsh = delta_pair(cb.get(qi, {}), SOCIAL_COND)
            din = delta_pair(cb.get(qi, {}), INDEP_COND)
            rows.append([M, chi, dsh, din])
        X = np.array(rows, float)
        loaded[rg] = (X, Ymean, recs, qids, meta)

    if not loaded:
        raise SystemExit("no regime loaded")
    meta = next(iter(loaded.values()))[4]
    cost = K.cost_model(meta)
    cost_vec = np.array([cost[a] for a in ALL], float)

    print("K2.5-pre — leaky features z4 = [margin, chi, d_shared(TV), d_indep(TV)]")
    print(f"  d_shared from cond '{SOCIAL_COND}',  d_indep from cond '{INDEP_COND}'  (vs 'none')")
    for rg, v in loaded.items():
        print(f"  {rg}: Q={v[0].shape[0]}   z-range margin[{v[0][:,0].min():.2f},{v[0][:,0].max():.2f}] "
              f"d_shared[{v[0][:,2].min():.2f},{v[0][:,2].max():.2f}] d_indep[{v[0][:,3].min():.2f},{v[0][:,3].max():.2f}]")

    out_all = {}
    for rg, (X, Ymean, recs, qids, meta) in loaded.items():
        ev = evaluate_leaky(X, Ymean, recs, cost_vec, f"regime={rg}", VARIANTS)
        report(ev)
        out_all[rg] = ev
    if len(loaded) >= 2:
        X = np.vstack([v[0] for v in loaded.values()])
        Ymean = np.vstack([v[1] for v in loaded.values()])
        recs = [r for v in loaded.values() for r in v[2]]
        ev = evaluate_leaky(X, Ymean, recs, cost_vec, "POOLED " + "+".join(loaded), VARIANTS)
        report(ev)
        out_all["pooled"] = ev

    # ── programmatic verdict per the locked gate (necessary-condition, leaky = upper bound) ──
    print("\n" + "#" * 84)
    print("# VERDICT (necessary-condition screen; leaky ⇒ these are OPTIMISTIC ceilings)")
    print("#" * 84)
    for scope in ("hard", "pooled"):
        if scope not in out_all:
            continue
        rs = out_all[scope]["routers"]
        best_name = max(rs, key=lambda n: rs[n]["captured"] if rs[n]["captured"] == rs[n]["captured"] else -9)
        best = rs[best_name]
        cap = best["captured"]
        ridge = rs["ridge_full"]
        mech = rs["ridge_mechanism_only"]
        print(f"[{scope}] headroom {out_all[scope]['denom_cf']:+.3f} | "
              f"best-leaky router = {best_name} captured {cap*100:.0f}% "
              f"(gain {best['gain']:+.3f} CI[{best['gain_ci'][0]:+.3f},{best['gain_ci'][1]:+.3f}])")
        print(f"        ridge_full captured {ridge['captured']*100:.0f}% | "
              f"mechanism-only(d_shared,d_indep) captured {mech['captured']*100:.0f}% "
              f"| ridge_full collapse {ridge['collapse_frac']*100:.0f}% on 1 action")
        sig = best["gain_ci"][0] > 0
        if best["gain"] <= 0 or cap < 0.20:
            v = "KILL routing on current action set (even leaky can't) → go straight to action-enrichment test"
        elif cap < 0.50:
            v = "signal too weak vs leakage advantage → NOT worth fresh-probe GPU; action-enrichment first"
        elif not sig or ridge["collapse_frac"] > 0.90:
            v = "≥50% captured but unstable/collapsed → still weak; action-enrichment first, probes only if it firms up"
        else:
            v = "≥50% captured, stable, non-collapsed → fresh leakage-free probes JUSTIFIED"
        print(f"        → {v}")

    outp = K.TABLES / "k2p5_pre_summary.json"
    outp.write_text(json.dumps(out_all, ensure_ascii=False, indent=2, default=float))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
