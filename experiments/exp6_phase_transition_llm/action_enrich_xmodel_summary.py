"""Cross-model validation of the action-enrichment STANDALONE WIN (`a_common`).

The qwen HARD run gave a WEAK/CONDITIONAL router gate (H^CV_new=+0.035) but a clean
SECONDARY WIN: the NEW operator `a_common` was the A1 best-fixed policy, beating every A0
baseline by Δ_fixed=+0.038. That standalone-algorithm claim is only interesting if it
REPLICATES across models — this script tabulates it for qwen + gemma + llama + mistral
(same 40 HARD questions, N=11, R=8, same fixed qwen clue pool = the P2 convention).

For each model it recomputes, PAIRED per question, the a_common lift over the model's own
best-fixed A0 baseline (usually shared_ev), with a question-level bootstrap CI — the precise
"beats every A0 baseline" statistic (delta_fixed only isolates a_common when a_common IS the
A1 argmax). Then a POOLED estimate across all models that produced shards.

Reads shard files actionenrich_<glob>.json + per-model gate JSON. No LLM calls.
Run:  python action_enrich_xmodel_summary.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from action_enrich_analyze import A0, NEW, TABLES, best_fixed, load

# model_tag -> (display, shard glob, gate json)
MODELS = [
    ("qwen2.5:7b", "actionenrich_s[01]_hard_N*.json", "actionenrich_gate.json"),
    ("gemma2:9b", "actionenrich_gemma9b_*_hard_N*.json", "actionenrich_gate_gemma9b.json"),
    ("llama3.1:8b", "actionenrich_llama8b_*_hard_N*.json", "actionenrich_gate_llama8b.json"),
    ("mistral:7b", "actionenrich_mistral7b_*_hard_N*.json", "actionenrich_gate_mistral7b.json"),
]


def paired_lift(recs, op="a_common"):
    """Per-question lift of `op` over the FIXED best-A0 policy (chosen on the full set),
    plus question-level bootstrap CI. Returns (point, lo, hi, best_a0_name, op_rank)."""
    _, bf_a = best_fixed(recs, A0)                       # fixed A0 baseline (e.g. shared_ev)
    op_y = np.array([r["y"][op] for r in recs], float)
    a0_y = np.array([r["y"][bf_a] for r in recs], float)
    lift = op_y - a0_y
    rng = np.random.default_rng([505])
    Q = len(lift)
    draws = [lift[rng.integers(0, Q, Q)].mean() for _ in range(10000)]
    # rank of op among ALL actions by mean (1 = best)
    col = {a: float(np.mean([r["y"][a] for r in recs])) for a in (A0 + NEW)}
    rank = 1 + sum(1 for a, v in col.items() if v > col[op])
    return (float(lift.mean()), float(np.quantile(draws, .025)),
            float(np.quantile(draws, .975)), bf_a, rank, lift)


def main():
    print("=" * 90)
    print("ACTION-ENRICHMENT CROSS-MODEL VALIDATION — standalone win `a_common` (HARD, N=11, R=8)")
    print("paired a_common lift over each model's own best-fixed A0 baseline; fixed qwen clue pool")
    print("=" * 90)
    print(f"\n{'model':13s} {'Q':>3s} {'bestA0':>10s} {'A0':>6s} {'a_common':>8s} "
          f"{'lift':>7s} {'CI95':>18s} {'rank':>4s} {'Hcv_new':>8s}")
    rows, pooled_lifts = [], []
    for disp, gpat, gate_name in MODELS:
        paths = sorted(glob.glob(str(TABLES / gpat)))
        if not paths:
            print(f"{disp:13s}  -- no shard files ({gpat}) --")
            continue
        recs, meta = load(paths)
        pt, lo, hi, bf_a, rank, lift = paired_lift(recs)
        a0v = float(np.mean([r["y"][bf_a] for r in recs]))
        acv = float(np.mean([r["y"]["a_common"] for r in recs]))
        gp = TABLES / gate_name
        hcv = json.loads(gp.read_text()).get("Hcv_new_A1") if gp.exists() else None
        hcv_s = f"{hcv:+.3f}" if hcv is not None else "  n/a"
        sig = "*" if lo > 0 else " "
        print(f"{disp:13s} {len(recs):3d} {bf_a:>10s} {a0v:6.3f} {acv:8.3f} "
              f"{pt:+7.3f} [{lo:+.3f},{hi:+.3f}]{sig} {rank:4d} {hcv_s:>8s}")
        rows.append({"model": disp, "Q": len(recs), "best_A0": bf_a, "A0_acc": a0v,
                     "a_common_acc": acv, "lift": pt, "lift_ci": [lo, hi],
                     "a_common_rank": rank, "Hcv_new": hcv, "sig": lo > 0})
        pooled_lifts.append(lift)

    if pooled_lifts:
        alll = np.concatenate(pooled_lifts)
        rng = np.random.default_rng([606])
        n = len(alll)
        draws = [alll[rng.integers(0, n, n)].mean() for _ in range(10000)]
        plo, phi = float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
        print("-" * 90)
        print(f"{'POOLED':13s} {n:3d} {'(paired)':>10s} {'':6s} {'':8s} "
              f"{float(alll.mean()):+7.3f} [{plo:+.3f},{phi:+.3f}]{'*' if plo>0 else ' '}")
        n_rep = sum(1 for r in rows if r["lift"] > 0)
        n_sig = sum(1 for r in rows if r["sig"])
        print(f"\nreplication: a_common lift > 0 in {n_rep}/{len(rows)} models "
              f"({n_sig}/{len(rows)} with CI excluding 0); a_common is top-ranked operator in "
              f"{sum(1 for r in rows if r['a_common_rank']==1)}/{len(rows)}.")
        out = {"models": rows, "pooled": {"n_q": n, "lift": float(alll.mean()),
               "ci": [plo, phi], "n_replicate": n_rep, "n_sig": n_sig}}
        (TABLES / "actionenrich_xmodel_summary.json").write_text(json.dumps(out, indent=2))
        print(f"\nwrote {TABLES / 'actionenrich_xmodel_summary.json'}")


if __name__ == "__main__":
    main()
