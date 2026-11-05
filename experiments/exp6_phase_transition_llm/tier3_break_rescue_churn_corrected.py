"""Offline-only, post-hoc (2026-10-13, REVISED same day after a reviewer-caught
bug): does the "churn is the largest single component of the caliper Delta Y
gap" claim (paper.md Sec. 6.2 / Fig. anatomy) survive noise correction, or is
it an artifact of reading the decomposition off the RAW gap?

BUG IN THE FIRST VERSION OF THIS SCRIPT, FOUND BY REVIEW (do not repeat it):
the first pass applied a single PAIR-AVERAGED correctness-noise floor
(CN_shared, one scalar per (model,bench) pair) to every shared row uniformly.
That is inconsistent with how the headline Delta Y correction actually works
(tier3_doubly_adjusted.py's sh_adj = dict(r, dY=r["dC"]) for r in sh), which
uses EACH ROW'S OWN dC = dY - C_within -- a per-QUESTION quantity, not a
pair-level average, because C_within itself is computed per question inside
shared_rows_tier1. Applying a pair-level average to the break/rescue split
while the headline dY correction uses per-row values breaks the identity
dY = p_break + p_rescue + p_churn at the row level, so after caliper matching
the three re-derived gaps no longer have to sum consistently with the actual
(correctly, per-row corrected) dY gap -- which is exactly why the first
version produced churn shares (-6%, -105%) outside the range implied by
simple bounds on the correction. Fixed below by computing the correctness-flip
noise floor PER QUESTION, matching shared_rows_tier1's own per-row C_within
exactly in granularity, so the row-level identity
  dC_row = p_break_adj_row + p_rescue_adj_row + p_churn_adj_row
holds by construction (p_break_adj_row = p_break_row - CN_row/2,
p_rescue_adj_row = p_rescue_row - CN_row/2, and consequently the implied
p_churn_adj_row = p_churn_row - (C_within_row - CN_row), since
dC_row = dY_row - C_within_row and dY_row = p_break_row+p_rescue_row+p_churn_row
exactly per row already). Caliper-matched gaps are a linear (matched-mean)
functional of per-row values, so this row-level identity is expected to carry
through to the gap level as well, up to how each metric's own missing/NaN D
rows are handled by caliper_contrast (same row set for all three metrics here,
so this is not a concern).

On the copy side, rows are single-draw (no repeated structure), so -- exactly
as the headline dY correction already does (cp_adj subtracts a uniform
per-pair C_within_base from every copy row's dY) -- we apply a uniform
per-pair correctness-noise floor (CN_base) to every copy row. This was NOT
the bug (the copy-side correction in the first version was already uniform,
consistent with how the headline copy correction works) and is unchanged
here.

ALLOCATION ASSUMPTION (unchanged, stated not hidden): the correctness-flip
noise (CN_row or CN_base) is split 50/50 between break and rescue, since the
replicate data cannot itself say which direction a given noise-induced
correctness flip would have gone -- matching the symmetric-noise argument
already made in paper.md's Introduction. This is an assumption, not a
measurement; the churn residual inherits it.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "exp6_phase_transition_llm"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
sys.path.insert(0, str(EXP))

from analysis_C_matched_D_falsification import caliper_contrast  # noqa: E402
from tier3_matched_d import shared_rows_tier1, copy_rows_tier3, MODELS, BENCHMARKS, REGIMES  # noqa: E402

da = json.loads((TABLES / "tier3_doubly_adjusted.json").read_text())
dY_by_pair = {(p["model"], p["bench"]): p for p in da["pairs"]}


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


def pairwise_disagree(xs):
    n = len(xs)
    if n < 2:
        return float("nan")
    return float(np.mean([int(xs[i] != xs[j]) for i in range(n) for j in range(i + 1, n)]))


def shared_rows_CN(model, bench):
    """Per-row (per-question) correctness-flip noise floor, SAME loop order as
    tier3_matched_d.shared_rows_tier1, so the i-th value here lines up with the
    i-th row returned by shared_rows_tier1(model, bench)."""
    out = []
    for regime in REGIMES:
        f = PROC / f"exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json"
        if not f.exists():
            continue
        raw = json.loads(f.read_text())["raw"]
        cond = raw["shared_correct_strong"]
        for q in range(len(cond)):
            cs = [int(plur(cond[q][r]) == 0) for r in range(len(cond[q]))]
            out.append(pairwise_disagree(cs))
    return out


def base_correctness_noise(model, bench):
    """Copy side: single draw per row, so (as the headline correction already
    does) one uniform per-pair floor, pooled over easy+hard."""
    cvals = []
    for regime in REGIMES:
        f = PROC / f"exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json"
        if not f.exists():
            continue
        raw = json.loads(f.read_text())["raw"]
        none_cond = raw["none"]
        for q in range(len(none_cond)):
            cs = [int(plur(none_cond[q][r]) == 0) for r in range(len(none_cond[q]))]
            c = pairwise_disagree(cs)
            if np.isfinite(c):
                cvals.append(c)
    return float(np.mean(cvals)) if cvals else float("nan")


pair_results = []
for model in MODELS:
    for bench in BENCHMARKS:
        sh = shared_rows_tier1(model, bench)
        cp = copy_rows_tier3(model, bench)
        if len(sh) < 5 or len(cp) < 5:
            continue
        key = (model, bench)
        d = dY_by_pair[key]

        cn_rows = shared_rows_CN(model, bench)
        assert len(cn_rows) == len(sh), (len(cn_rows), len(sh))
        CN_shared_mean = float(np.nanmean(cn_rows))
        mean_c_within_shared = float(np.mean([r["C_within"] for r in sh]))
        CN_base = base_correctness_noise(model, bench)
        c_within_base = d["c_within_base"]
        half_base = CN_base / 2.0

        sh_adj = []
        for r, cn in zip(sh, cn_rows):
            half = (cn / 2.0) if np.isfinite(cn) else 0.0
            sh_adj.append(dict(r, p_break=r["p_break"] - half, p_rescue=r["p_rescue"] - half))
        cp_adj = [dict(r, p_break=r["p_break"] - half_base, p_rescue=r["p_rescue"] - half_base) for r in cp]

        eps = 0.05
        pb_raw = caliper_contrast(sh, cp, eps, "p_break", n_boot=2000)
        pr_raw = caliper_contrast(sh, cp, eps, "p_rescue", n_boot=2000)
        pb_shared_only = caliper_contrast(sh_adj, cp, eps, "p_break", n_boot=2000)
        pr_shared_only = caliper_contrast(sh_adj, cp, eps, "p_rescue", n_boot=2000)
        pb_copy_only = caliper_contrast(sh, cp_adj, eps, "p_break", n_boot=2000)
        pr_copy_only = caliper_contrast(sh, cp_adj, eps, "p_rescue", n_boot=2000)
        pb_both = caliper_contrast(sh_adj, cp_adj, eps, "p_break", n_boot=2000)
        pr_both = caliper_contrast(sh_adj, cp_adj, eps, "p_rescue", n_boot=2000)

        def churn(reading):
            dy = dict(raw=d["caliper_gap_raw"], shared_only=d["caliper_gap_shared_only"],
                      copy_only=d["caliper_gap_copy_only"], both=d["caliper_gap_doubly_adj"])[reading]
            pb = dict(raw=pb_raw, shared_only=pb_shared_only, copy_only=pb_copy_only, both=pb_both)[reading]["gap"]
            pr = dict(raw=pr_raw, shared_only=pr_shared_only, copy_only=pr_copy_only, both=pr_both)[reading]["gap"]
            return dy - pb - pr

        pair_results.append(dict(
            model=model, bench=bench,
            CN_shared_mean=CN_shared_mean, CN_base=CN_base,
            mean_c_within_shared=mean_c_within_shared, c_within_base=c_within_base,
            p_break_gap_raw=pb_raw["gap"], p_break_gap_shared_only=pb_shared_only["gap"],
            p_break_gap_copy_only=pb_copy_only["gap"], p_break_gap_both=pb_both["gap"],
            p_rescue_gap_raw=pr_raw["gap"], p_rescue_gap_shared_only=pr_shared_only["gap"],
            p_rescue_gap_copy_only=pr_copy_only["gap"], p_rescue_gap_both=pr_both["gap"],
            churn_raw=churn("raw"), churn_shared_only=churn("shared_only"),
            churn_copy_only=churn("copy_only"), churn_both=churn("both"),
        ))
        print(f"{model:8s}/{bench:12s} CN_shared_mean={CN_shared_mean:.4f} (of C_within {mean_c_within_shared:.4f})  "
              f"CN_base={CN_base:.4f} (of C_within_base {c_within_base:.4f})  "
              f"churn raw/shared/copy/both = {churn('raw'):.4f}/{churn('shared_only'):.4f}/"
              f"{churn('copy_only'):.4f}/{churn('both'):.4f}")


def macro(key):
    return float(np.mean([p[key] for p in pair_results]))


macro_out = dict(
    p_break_gap_raw=macro("p_break_gap_raw"), p_break_gap_shared_only=macro("p_break_gap_shared_only"),
    p_break_gap_copy_only=macro("p_break_gap_copy_only"), p_break_gap_both=macro("p_break_gap_both"),
    p_rescue_gap_raw=macro("p_rescue_gap_raw"), p_rescue_gap_shared_only=macro("p_rescue_gap_shared_only"),
    p_rescue_gap_copy_only=macro("p_rescue_gap_copy_only"), p_rescue_gap_both=macro("p_rescue_gap_both"),
    churn_raw=macro("churn_raw"), churn_shared_only=macro("churn_shared_only"),
    churn_copy_only=macro("churn_copy_only"), churn_both=macro("churn_both"),
)
print()
print("MACRO (break, rescue, churn) by reading:")
for reading in ("raw", "shared_only", "copy_only", "both"):
    pb, pr, ch = (macro_out[f"p_break_gap_{reading}"], macro_out[f"p_rescue_gap_{reading}"],
                  macro_out[f"churn_{reading}"])
    dy = dict(raw=da["macro"]["caliper_raw"], shared_only=da["macro"]["caliper_shared_only"],
              copy_only=da["macro"]["caliper_copy_only"], both=da["macro"]["caliper_doubly_adj"])[reading]
    print(f"  {reading:12s}: dY_gap={dy:.4f}  break={pb:.4f}  rescue={pr:.4f}  "
          f"directional={pb+pr:.4f}  churn={ch:.4f}  churn_share={ch/dy if dy else float('nan'):.1%}")

# additive-linearity check, same logic as tier3_doubly_adjusted.py
print()
print("additive check (both == shared_only + copy_only - raw):")
print(f"  p_break: {macro_out['p_break_gap_shared_only'] + macro_out['p_break_gap_copy_only'] - macro_out['p_break_gap_raw']:.5f} "
      f"vs both={macro_out['p_break_gap_both']:.5f}")
print(f"  p_rescue: {macro_out['p_rescue_gap_shared_only'] + macro_out['p_rescue_gap_copy_only'] - macro_out['p_rescue_gap_raw']:.5f} "
      f"vs both={macro_out['p_rescue_gap_both']:.5f}")

out = dict(role="noise-corrected break/rescue/churn decomposition of the caliper Delta Y gap, "
                "2026-10-13 post-hoc, offline only (REVISED same day: shared-side "
                "correctness-flip noise is now applied per-row, matching the granularity "
                "of the headline per-row dC = dY - C_within correction, fixing an earlier "
                "pair-averaged version that was inconsistent with it)",
           note="correctness-flip noise is allocated 50/50 to break/rescue, an assumption "
                "(symmetric noise, no measurement basis to split further), not a measurement; "
                "churn_reading = dY_gap_reading - p_break_gap_reading - p_rescue_gap_reading",
           pairs=pair_results, macro=macro_out)
dst = TABLES / "tier3_break_rescue_churn_corrected.json"
dst.write_text(json.dumps(out, indent=2))
print("wrote", dst)
