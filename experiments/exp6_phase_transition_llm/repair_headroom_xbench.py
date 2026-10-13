"""Cross-benchmark generalization of Theorem 6 (offline, NO LLM calls, NO GPU).

Reuses repair_headroom.contingency()/plurality() UNCHANGED (estimator untouched); only the
file-indexing differs so it can also read the AQuA g-sweep (older no-hash naming, paired by
qi). Adds a dependence-proxy column D_g = mean final-round pairwise vote-agreement over
informative agents (rises toward 1 as rho->1) to show the D↑↑-but-H_repair≈0 phenomenon.

g-sweep copying states available offline are qwen-only:
  - TruthfulQA: exp6_truthfulqa_N11_g*_q*_<hash>.npz   (hash-keyed)
  - AQuA:       exp6_aqua_g*_q*.npz                     (qi-keyed, g in {0,0.5,1.0})
So this is a cross-BENCHMARK check (same model), not cross-model (no gemma/llama/mistral
g-sweep exists — would need GPU, deliberately not run).
"""
from __future__ import annotations
import glob, json, re, collections
from pathlib import Path
import numpy as np
from repair_headroom import plurality, contingency, PROC, TABLES  # estimator reused verbatim


def dep_proxy(x_round: np.ndarray) -> float:
    """Mean pairwise vote-agreement over informative agents (dependence proxy, ->1 as rho->1)."""
    info = x_round.max(1) > x_round.min(1)
    v = np.argmax(x_round[info], axis=1)
    if len(v) < 2:
        return float("nan")
    agree = np.mean([v[i] == v[j] for i in range(len(v)) for j in range(i + 1, len(v))])
    return float(agree)


def index_tqa():
    pat = re.compile(r"_g([0-9.]+)_q(\d+)_([0-9a-f]+)\.npz$")
    by = collections.defaultdict(dict)
    for f in glob.glob(str(PROC / "exp6_truthfulqa_N11_g*.npz")):
        m = pat.search(f)
        if m:
            by[m.group(3)][m.group(1)] = f
    return by


def index_aqua():
    """qi-keyed; verify question-text matches across g (guards the pre-hash cache bug)."""
    pat = re.compile(r"_g([0-9.]+)_q(\d+)\.npz$")
    by = collections.defaultdict(dict)
    for f in glob.glob(str(PROC / "exp6_aqua_g*.npz")):
        m = pat.search(f)
        if m:
            by["q" + m.group(2)][m.group(1)] = f
    # drop any qi whose question text disagrees between g=0 and g=1
    clean, dropped = collections.defaultdict(dict), 0
    for k, gm in by.items():
        qs = {g: str(np.load(p, allow_pickle=True)["question"]) for g, p in gm.items()}
        if len({v for v in qs.values()}) == 1:
            clean[k] = gm
        else:
            dropped += 1
    if dropped:
        print(f"[aqua] dropped {dropped} qi with mismatched question text across g")
    return clean


def dep_by_g(by, g):
    vals = []
    for h, gm in by.items():
        if g in gm:
            vals.append(dep_proxy(np.load(gm[g], allow_pickle=True)["states"][-1]))
    return float(np.nanmean(vals)) if vals else float("nan")


def table(by, name):
    gs = sorted({g for gm in by.values() for g in gm} - {"0.00"}, key=float)
    print(f"\n=== {name} (qwen2.5:7b, copying g-sweep) ===")
    print(f"{'g':>5} {'n':>4} {'D_g':>5} {'A_0':>5} {'A_g':>5} {'H_rep':>6} {'p+-':>5} {'p-+':>5} "
          f"{'eps=ΔY':>7} | {'id':>2} {'bnd':>3}")
    rows = []
    d0 = dep_by_g(by, "0.00")
    for g in gs:
        r = contingency(by, g)          # <-- estimator reused verbatim
        if not r:
            continue
        r["D_g"] = dep_by_g(by, g)
        r["D_0"] = d0
        rows.append(r)
        print(f"{r['g']:>5.2f} {r['n']:>4} {r['D_g']:5.2f} {r['A_0']:5.3f} {r['A_g']:5.3f} "
              f"{r['H_repair']:+6.3f} {r['p_break']:5.3f} {r['p_rescue']:5.3f} {r['eps']:7.3f} | "
              f"{'✓' if r['identity_ok'] else '✗':>2} {'✓' if r['bound_ok'] else '✗':>3}")
    print(f"      (independent baseline dependence D_0 = {d0:.2f})")
    return rows


if __name__ == "__main__":
    out = {"truth_index": 0,
           "truthfulqa": table(index_tqa(), "TruthfulQA"),
           "aqua": table(index_aqua(), "AQuA")}
    p = TABLES / "repair_headroom_xbench.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nPhenomenon to check: D_g ↑↑ while H_repair ≈ 0, cross-benchmark. wrote {p}")
