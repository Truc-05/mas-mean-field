#!/usr/bin/env python
"""OFFLINE go/no-go for the susceptibility-gated a_common algorithm.

ONE question only (per Alex's firewall): is the a_common treatment effect
    Δ_q = A_q(a_common) − A_q(baseline)
CONCENTRATED at low round-0 margin M_q (contested/susceptible questions)?

We need Δ_low-M > Δ_high-M (heterogeneity) consistently cross-model. We do NOT
pick a threshold τ or report an adaptive accuracy here — that is the NEXT step
(threshold + LOO-CV) and only runs if this passes.

Sources (validated same-dataset join, TruthfulQA hard, K=4, N=11):
  Δ    ← results/tables/actionenrich_{tag}_hard_N11.json   (reps per action, R=8)
  M    ← results/tables/exp6c_boundary_hard_{model}_N11.json (margin.runnerup_gap
         from the `none` = independent no-evidence poll — D-free, correctness-free)
Join key = qid (position-verified 40/40 vs select_questions('hard',40)).
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np

TABLES = Path("results/tables")

# model -> (action_enrich tag glob, boundary tag)
MODELS = {
    "gemma9b":   ("gemma9b",   "gemma9b"),
    "llama8b":   ("llama8b",   "llama8b"),
    "mistral7b": ("mistral7b", "mistral7b"),
}
BASELINES = ["indep_vote", "shared_ev"]   # Δ1 vs STOP-path, Δ2 vs common-clue (decorrelation)


def load_actionenrich(tag: str) -> dict[int, dict]:
    """merge shards s0/s1 -> {qid: {action: mean over reps}}"""
    out = {}
    for f in sorted(glob.glob(str(TABLES / f"actionenrich_{tag}_s*_hard_N11.json"))):
        d = json.load(open(f))
        for q in d["questions"]:
            reps = q["reps"]
            out[q["qid"]] = {a: float(np.mean(v)) for a, v in reps.items()}
    return out


def load_margin(model_tag: str) -> dict[int, dict]:
    d = json.load(open(TABLES / f"exp6c_boundary_hard_{model_tag}_N11.json"))
    out = {}
    for i, q in enumerate(d["per_q"]):
        mg = q["margin"]
        out[i] = {"M": mg["runnerup_gap"], "chi": mg["dist_to_boundary"],
                  "entropy": mg["vote_entropy"], "plurality_share": mg["plurality_share"],
                  "A_none": mg["A_none"]}
    return out


def boot_ci(vals: np.ndarray, rng, fn, n_boot=10000):
    idx = np.arange(len(vals))
    stats = np.array([fn(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n_boot)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def boot_ci_pairs(x: np.ndarray, y: np.ndarray, rng, fn, n_boot=10000):
    n = len(x)
    stats = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        stats.append(fn(x[s], y[s]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def slope(x, y):
    x = x - x.mean()
    d = (x * x).sum()
    return float((x * (y - y.mean())).sum() / d) if d > 0 else 0.0


def analyze_model(model: str, verbose=True) -> dict:
    ae_tag, m_tag = MODELS[model]
    ae = load_actionenrich(ae_tag)
    mg = load_margin(m_tag)
    qids = sorted(set(ae) & set(mg))
    rng = np.random.default_rng(20260820)

    M   = np.array([mg[q]["M"] for q in qids])
    chi = np.array([mg[q]["chi"] for q in qids])
    A_common = np.array([ae[q]["a_common"] for q in qids])
    res = {"model": model, "n": len(qids), "qids": qids,
           "corr_M_Aindep": None, "baselines": {}}

    # confound diagnostic: is M just a proxy for baseline correctness?
    A_indep = np.array([ae[q]["indep_vote"] for q in qids])
    res["corr_M_Aindep"] = float(np.corrcoef(M, A_indep)[0, 1])

    lines = [f"\n{'='*74}\n{model}   n={len(qids)} questions (hard, N=11, R=8, TruthfulQA)"]
    lines.append(f"  corr(M, A_indep) = {res['corr_M_Aindep']:+.2f}   "
                 f"(near 0 ⇒ margin is NOT just difficulty; the susceptibility read is clean)")

    for base in BASELINES:
        A_base = np.array([ae[q][base] for q in qids])
        dA = A_common - A_base                      # treatment effect per question

        # ---- (1) 3 equal-count bins by M ----
        order = np.argsort(M)
        nb = 3
        bins = np.array_split(order, nb)
        bin_rows = []
        for b, sidx in enumerate(bins):
            dsub = dA[sidx]
            lo, hi = boot_ci(dsub, rng, np.mean)
            bin_rows.append({"M_lo": float(M[sidx].min()), "M_hi": float(M[sidx].max()),
                             "n": int(len(sidx)), "dA": float(dsub.mean()),
                             "ci": [lo, hi], "A_base": float(A_base[sidx].mean()),
                             "A_common": float(A_common[sidx].mean())})
        # heterogeneity contrast: low-M bin minus high-M bin
        lowidx, highidx = bins[0], bins[-1]
        def contrast(_):  # resample within each bin
            sl = lowidx[rng.integers(0, len(lowidx), len(lowidx))]
            sh = highidx[rng.integers(0, len(highidx), len(highidx))]
            return dA[sl].mean() - dA[sh].mean()
        c_pt = dA[lowidx].mean() - dA[highidx].mean()
        c_stats = np.array([contrast(0) for _ in range(10000)])
        c_lo, c_hi = float(np.percentile(c_stats, 2.5)), float(np.percentile(c_stats, 97.5))

        # ---- (2) slope Δ ~ M ----
        s_pt = slope(M, dA)
        s_lo, s_hi = boot_ci_pairs(M, dA, rng, slope)
        # slope vs chi (dist to boundary) as robustness
        sc_pt = slope(chi, dA); sc_lo, sc_hi = boot_ci_pairs(chi, dA, rng, slope)

        res["baselines"][base] = {
            "dA_overall": float(dA.mean()),
            "bins": bin_rows,
            "contrast_low_minus_high": {"pt": float(c_pt), "ci": [c_lo, c_hi]},
            "slope_M": {"pt": s_pt, "ci": [s_lo, s_hi]},
            "slope_chi": {"pt": sc_pt, "ci": [sc_lo, sc_hi]},
        }

        verdict = ("HETEROGENEITY ✓ (low-M lift > high-M)" if c_lo > 0
                   else "trend ✓ (dir. right, CI touches 0)" if c_pt > 0
                   else "FLAT/opposite ✗")
        lines.append(f"\n  Δ = a_common − {base}   overall Δ̄={dA.mean():+.3f}")
        for b, r in enumerate(bin_rows):
            tagb = ["LOW-M ", "mid-M ", "HIGH-M"][b]
            lines.append(f"    {tagb} M∈[{r['M_lo']:.2f},{r['M_hi']:.2f}] n={r['n']:2d}  "
                         f"Δ={r['dA']:+.3f} CI[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]  "
                         f"(A_base={r['A_base']:.2f}→A_common={r['A_common']:.2f})")
        lines.append(f"    → contrast Δ(LOW−HIGH) = {c_pt:+.3f} CI[{c_lo:+.3f},{c_hi:+.3f}]   {verdict}")
        lines.append(f"    → slope Δ~M   = {s_pt:+.3f} CI[{s_lo:+.3f},{s_hi:+.3f}]  "
                     f"(neg ⇒ smaller margin, larger lift)")
        lines.append(f"    → slope Δ~chi = {sc_pt:+.3f} CI[{sc_lo:+.3f},{sc_hi:+.3f}]")

    if verbose:
        print("\n".join(lines))
    res["_report"] = "\n".join(lines)
    return res


def main():
    print("="*74)
    print("SUSCEPTIBILITY-GATE OFFLINE GO/NO-GO — a_common treatment effect × round-0 margin")
    print("Firewall: heterogeneity-existence ONLY. No τ selection, no adaptive accuracy here.")
    print("="*74)
    all_res = {}
    for model in MODELS:
        # skip models with no action_enrich data yet
        if not glob.glob(str(TABLES / f"actionenrich_{MODELS[model][0]}_s*_hard_N11.json")):
            print(f"\n[{model}] no action_enrich file yet — skipping")
            continue
        all_res[model] = analyze_model(model)

    # cross-model summary
    print("\n" + "="*74)
    print("CROSS-MODEL SUMMARY  (primary baseline = indep_vote = the STOP path)")
    print("="*74)
    print(f"{'model':10s} {'n':>3s}  {'Δ̄':>7s}  {'Δ(LOW-HIGH)':>22s}  {'slope Δ~M':>20s}  corr(M,Ai)")
    n_pass = 0
    for model, r in all_res.items():
        b = r["baselines"]["indep_vote"]
        c = b["contrast_low_minus_high"]; s = b["slope_M"]
        het = c["ci"][0] > 0
        trend = c["pt"] > 0
        n_pass += het
        mark = "✓sig" if het else ("~dir" if trend else "✗")
        print(f"{model:10s} {r['n']:>3d}  {b['dA_overall']:+.3f}  "
              f"{c['pt']:+.3f}[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] {mark:>4s}  "
              f"{s['pt']:+.3f}[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]  {r['corr_M_Aindep']:+.2f}")
    print(f"\nheterogeneity CI-significant in {n_pass}/{len(all_res)} models (indep_vote baseline)")

    out = TABLES / "suscept_gate_offline.json"
    out.write_text(json.dumps(all_res, ensure_ascii=False, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
