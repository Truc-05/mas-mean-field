"""MMRT -- Mechanism-Matched Repair Test (offline, NO LLM calls). See MMRT_SMB_PREREG.md.

Question: at MATCHED observed dependence D, does the repair gain H = Acc(INDEP) -
Acc(mechanism state) from the SAME independence-restoring re-reason differ between the
shared-evidence and copying mechanisms? H is already computed, with an identical sign
convention, by both tier3_matched_d.shared_rows_tier1 and .copy_rows_tier3 (verified in the
prereg): P(mechanism-wrong, INDEP-correct) - P(mechanism-correct, INDEP-wrong).

Reuses, UNMODIFIED: caliper_contrast (already generic over `metric`, so calling it with
metric="H" needs no new matching code), shared_rows_tier1, copy_rows_tier3, macro_bootstrap,
_ols, RNG. Generalizes ONLY nested_models/nn_matching (hard-coded to "dY" in the frozen
analysis_C module) to an explicit `metric` argument, reusing their internals unchanged.

Writes results/tables/mmrt_matched_repair.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as fdist
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis_C_matched_D_falsification import RNG, _ols, caliper_contrast  # noqa: E402
from tier3_matched_d import (  # noqa: E402
    BENCHMARKS, MODELS, TABLES, copy_rows_tier3, macro_bootstrap, shared_rows_tier1)

N_BOOT = 2000
CALIPERS = (0.02, 0.05, 0.10)
PRIMARY_CALIPER = 0.05
PRIMARY_METRIC = "H"


def nested_models_metric(sh: list[dict], cp: list[dict], metric: str) -> dict:
    """Generalization of analysis_C.nested_models to an arbitrary outcome metric; internals
    (_ols, the nested F-test) are reused verbatim, only the outcome column changes."""
    rows = [r for r in sh + cp if np.isfinite(r["D"]) and np.isfinite(r[metric])]
    y = np.array([r[metric] for r in rows]); D = np.array([r["D"] for r in rows])
    M = np.array([1.0 if r["mech"] == "copy" else 0.0 for r in rows])
    n = len(y); one = np.ones(n)
    X0 = np.column_stack([one, D])
    X1 = np.column_stack([one, D, M])
    X2 = np.column_stack([one, D, M, D * M])
    out = {}
    prev = None
    for name, X in [("M0_D", X0), ("M1_D+mech", X1), ("M2_D+mech+DxM", X2)]:
        beta, rss = _ols(X, y); p = X.shape[1]
        denom = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - rss / denom if denom > 0 else float("nan")
        rec = dict(k=p, rss=float(rss), r2=float(r2), beta=[float(b) for b in beta])
        if prev is not None and n - p > 0:
            F = ((prev["rss"] - rss) / (p - prev["k"])) / (rss / (n - p))
            rec["F_vs_prev"] = float(F)
            rec["p_vs_prev"] = float(fdist.sf(F, p - prev["k"], n - p))
        out[name] = rec; prev = rec
    out["n"] = n
    out["mech_coef_given_D"] = out["M1_D+mech"]["beta"][2]
    return out


def nn_matching_metric(sh: list[dict], cp: list[dict], metric: str, n_boot: int = N_BOOT) -> dict:
    """Generalization of analysis_C.nn_matching to an arbitrary outcome metric; the matching
    coordinates (D, A0, margin) and the nearest-neighbour logic are reused unchanged."""
    def mat(rows):
        return np.array([[r["D"], r["A0"], r["margin"]] for r in rows], float)
    Zs, Zc = mat(sh), mat(cp)
    keep_s = np.all(np.isfinite(Zs), 1) & np.array([np.isfinite(r[metric]) for r in sh])
    keep_c = np.all(np.isfinite(Zc), 1) & np.array([np.isfinite(r[metric]) for r in cp])
    Zs, sh2 = Zs[keep_s], [r for r, k in zip(sh, keep_s) if k]
    Zc, cp2 = Zc[keep_c], [r for r, k in zip(cp, keep_c) if k]
    if len(sh2) == 0 or len(cp2) == 0:
        return {"n_copy": len(cp2), "n_shared": len(sh2), "att_copy_minus_shared": None, "ci": [None, None]}
    mu = np.vstack([Zs, Zc]).mean(0); sd = np.vstack([Zs, Zc]).std(0) + 1e-9
    Zs, Zc = (Zs - mu) / sd, (Zc - mu) / sd
    nn_s = NearestNeighbors(n_neighbors=1).fit(Zs)
    _, idx_cs = nn_s.kneighbors(Zc)
    att = np.array([cp2[i][metric] - sh2[j[0]][metric] for i, j in enumerate(idx_cs)])
    boot = [float(att[RNG.integers(0, len(att), len(att))].mean()) for _ in range(n_boot)]
    return dict(n_copy=len(cp2), n_shared=len(sh2), att_copy_minus_shared=float(att.mean()),
                ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])


def main() -> None:
    rng = np.random.default_rng(20261108)
    pairs, pending = [], []
    for model in MODELS:
        for bench in BENCHMARKS:
            sh = shared_rows_tier1(model, bench)
            cp = copy_rows_tier3(model, bench)
            if len(cp) < 5 or len(sh) < 5:
                pending.append({"model": model, "bench": bench, "n_shared": len(sh),
                                "n_copy": len(cp), "status": "pending"})
                continue
            caliper = {}
            for eps in CALIPERS:
                res = caliper_contrast(sh, cp, eps, PRIMARY_METRIC, n_boot=N_BOOT)
                n_match = res["n"]
                caliper[f"eps_{eps}"] = {**res, "n_matched": n_match, "n_copy_total": len(cp),
                                         "coverage": n_match / len(cp) if len(cp) else 0.0}
                print(f"[{model}/{bench}] eps={eps}  H-gap(copy-shared)={res['gap']}  "
                      f"ci={res['ci']}  n_matched={n_match}/{len(cp)}")
            nested = nested_models_metric(sh, cp, PRIMARY_METRIC)
            nn = nn_matching_metric(sh, cp, PRIMARY_METRIC, n_boot=N_BOOT)
            pairs.append({"model": model, "bench": bench, "n_shared": len(sh), "n_copy": len(cp),
                         "caliper": caliper, "nested_models_H": nested, "nn_matching_H": nn})

    out = {"primary_metric": PRIMARY_METRIC, "primary_caliper": PRIMARY_CALIPER,
          "grid": {"models": MODELS, "benchmarks": BENCHMARKS, "n_pairs": len(pairs)},
          "pending": pending, "pairs": pairs}

    if pairs:
        print("\n=== MACRO-AVERAGE across (model, benchmark) pairs, equal weight ===")
        for eps in CALIPERS:
            gaps = np.array([p["caliper"][f"eps_{eps}"]["gap"] for p in pairs
                             if p["caliper"][f"eps_{eps}"]["gap"] is not None], float)
            r = macro_bootstrap(gaps, rng)
            tag = "PRIMARY" if eps == PRIMARY_CALIPER else "sensitivity"
            print(f"  [{tag}] caliper eps={eps}  H gap(copy-shared): K={r['K']} mean={r['mean']}  "
                  f"ci={r['ci']}  sig={r['sig']}")
            out[f"macro_caliper_H_eps_{eps}"] = r
        att = np.array([p["nn_matching_H"]["att_copy_minus_shared"] for p in pairs
                        if p["nn_matching_H"]["att_copy_minus_shared"] is not None], float)
        r = macro_bootstrap(att, rng)
        print(f"  NN-matching ATT(copy-shared) H: K={r['K']} mean={r['mean']}  ci={r['ci']}  sig={r['sig']}")
        out["macro_nn_att_H"] = r
        mech = np.array([p["nested_models_H"]["mech_coef_given_D"] for p in pairs], float)
        r = macro_bootstrap(mech, rng)
        print(f"  nested-LPM mechanism coef on H given D: K={r['K']} mean={r['mean']}  ci={r['ci']}  sig={r['sig']}")
        out["macro_nested_mech_coef_H"] = r
    else:
        print("No pair has both shared and copy data -- nothing to report.")

    if pending:
        print(f"\npending ({len(pending)}): " +
              "; ".join(f"{p['model']}/{p['bench']}" for p in pending))

    dst = TABLES / "mmrt_matched_repair.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
