"""Offline-only, post-hoc (2026-10-13): compute the baseline-arm ('none'
condition) within-arm churn C_within_base for each of the 9 Tier-3 (model,
benchmark) pairs, pooling easy+hard regimes exactly as tier3_matched_d.py
does for the shared side. Reuses plur() verbatim (copied inline below to
avoid any drift) from analysis_C_matched_D_falsification.py (frozen).
No new model queries -- reads only already-stored raw tensors
(data/processed/exp6c_raw_*.json). Written in response to a manuscript-review
request to turn "copy-side noise is unidentifiable" from a stated limitation
into an actual computed sensitivity check: C_within_base is NOT the same
quantity as the (still genuinely unknowable) copy-arm's own within-condition
noise at a fixed coupling g, but is a defensible, computable floor since
copying's reference arm is this same baseline ('none') condition. See
tier3_doubly_adjusted.py for how this is used, and paper.md Limitations for
the caveat on what this is and is not.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

MODELS = ["qwen", "llama", "gemma"]
BENCHMARKS = ["truthfulqa", "strategyqa", "aqua"]
REGIMES = ["easy", "hard"]


def _info(state):
    x = np.asarray(state, float)
    return x, x.max(1) > x.min(1)


def plur(state):
    x, info = _info(state); xi = x[info]
    if len(xi) == 0:
        return -1
    c = np.bincount(np.argmax(xi, 1), minlength=x.shape[1]); o = np.argsort(c)[::-1]
    if len(o) > 1 and c[o[0]] == c[o[1]]:
        return -1
    return int(o[0])


def within_churn(ys):
    n = len(ys)
    if n < 2:
        return float("nan")
    return float(np.mean([int(ys[i] != ys[j]) for i in range(n) for j in range(i + 1, n)]))


results = []
for model in MODELS:
    for bench in BENCHMARKS:
        base_ys_all = []
        n_q_total = 0
        for regime in REGIMES:
            f = PROC / f"exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json"
            if not f.exists():
                continue
            raw = json.loads(f.read_text())["raw"]
            none_cond = raw["none"]
            for q in range(len(none_cond)):
                ys = [plur(none_cond[q][r]) for r in range(len(none_cond[q]))]
                base_ys_all.append(ys)
                n_q_total += 1
        c_withins = [within_churn(ys) for ys in base_ys_all]
        c_withins = [c for c in c_withins if np.isfinite(c)]
        mean_c_within_base = float(np.mean(c_withins)) if c_withins else float("nan")
        results.append(dict(model=model, bench=bench, n_q=n_q_total,
                             mean_C_within_base=mean_c_within_base))
        print(f"{model}/{bench}: n_q={n_q_total}  C_within_base={mean_c_within_base:.4f}")

out = {"role": "copy-side noise floor candidate, offline recompute from stored raw tensors, "
               "2026-10-13 post-hoc analysis (not part of the frozen Tier-3 pipeline)",
       "note": "C_within_base is the 'none'/baseline condition's own within-arm churn across "
               "its R=8 replicates, pooled easy+hard -- the same role C_within_shared plays "
               "for the shared arm, computed identically via plur()+pairwise disagreement.",
       "pairs": results}
dst = TABLES / "tier3_copy_noise_floor.json"
dst.write_text(json.dumps(out, indent=2))
print("wrote", dst)
