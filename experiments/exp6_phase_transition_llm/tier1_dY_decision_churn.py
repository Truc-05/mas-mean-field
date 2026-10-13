"""Tier-1 breadth — decision-churn (ΔY) from the raw dumps. OFFLINE, no LLM calls, no GPU.

WHY THIS EXISTS. C1 was re-sold on decision change (ΔY), not net accuracy (README framing
lock 2026-09-22), but `exp6c_boundary` only writes dA per cell. The breadth tier was therefore
dA-only and could not speak to the ΔY form at all. Every quantity below is recoverable from
`data/processed/exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json`, which stores the full
(N,K) canonical belief matrix for every (condition, question, replicate) — so this costs zero
generation.

REVISION (2026-10-12, after the English-rebuild sweep completed at 72/72). Three fixes agreed
before this run, none changed after seeing its output:
  1. SUBSTRATE. The probe `easy`/`hard` labels are retired (`p2_probe.py:56` vs
     `exp6c_boundary.py:239` disagree on the display-shuffle `qi` convention, so they do not
     reproduce — see `tier1_split_sample_strata.py`). This script now recovers the same
     split-sample strata itself, reusing the canonical loaders (`discover`, `load_cell`,
     `phat_split`, the frozen `.80`/`.30` cuts) rather than re-deriving them, so the regime
     label here is identical to the one `tier1_split_sample_strata.py` reports.
  2. EVALUATION HALF. All per-question statistics below are computed on the held-out
     evaluation replicates (`R_BIN..R`) only, never on the binning replicates (`0..R_BIN`) —
     matching the split-sample discipline; computing ΔY on the binning half would let the
     regime assignment leak into the measured contrast.
  3. BBH EXCLUDED. `bbh`'s 80 items are a single `logical_deduction` subtask; closed-premise
     questions admit no informative evidence and correct-pointing clues LOWER accuracy by 0.21
     (see `tier1_breadth.BENCHMARKS`). Its qwen-only cells are dropped before pooling.

NEW: WITHIN-ARM CHURN BASELINE. `ΔY_cross = P(Y_shared,r != Y_indep,r)` at matched replicate
`r` answers "does swapping shared evidence for independent evidence change the decision" — but
part of that rate is just replicate-to-replicate sampling noise that would appear even with NO
arm swap at all. We therefore also compute, per question, the all-pairs decision-churn rate
WITHIN a single arm over the same held-out replicates:
    C_s = mean over distinct eval-replicate pairs (r, r') of  1[Y_shared,r != Y_shared,r']
    C_i = mean over distinct eval-replicate pairs (r, r') of  1[Y_indep,r  != Y_indep,r' ]
    C_within = (C_s + C_i) / 2
    dC        = ΔY_cross − C_within
`dC` is the churn attributable to the shared-vs-independent evidence swap NET of the baseline
noise a single arm already exhibits across replicates. This is the primary audit for whether
C1-at-breadth's `ΔY > 0` headline reflects the treatment or just resampling variance.

SCOPE — READ BEFORE QUOTING. This does NOT make Tier-1 test C1 in its *matched-D
mechanism-non-equivalence* form. That claim needs TWO dependence mechanisms on a common D
scale (upstream shared-evidence vs downstream copying); Tier-1 has no copying arm, only
none / shared_correct_strong / indep_correct_strong. The matched-D falsification stays with
`analysis_B` + `analysis_C` on qwen and is NOT replicated at breadth. What Tier-1 can now show
is the weaker, still load-bearing statement: raising dependence churns decisions (ΔY > 0)
while leaving net accuracy flat (dA ≈ 0) — "Dependence ≠ Harm" at breadth.

ESTIMATORS ARE THE FROZEN ONES. `plurality_prediction` is imported from the frozen module;
`_agreement` reproduces `analysis_B_matched_mechanism.agreement` (mean pairwise vote agreement
over informative agents). Per-question definitions are copied verbatim from
`analysis_B_matched_mechanism.py:51-61`:
    dA       = A_shared - A_indep
    ΔY       = P(Y_shared != Y_indep)        at matched replicate
    p_break  = P(Y_indep=truth, Y_shared!=truth)
    p_rescue = P(Y_indep!=truth, Y_shared=truth)
    H_repair = p_break - p_rescue
Truth is pinned at canonical index 0. Regime recovery (`discover`/`load_cell`/`phat_split`)
and the `.80`/`.30`/`R_BIN`/`MIN_Q` constants are imported from `tier1_split_sample_strata`,
never re-implemented.

TIE CONVENTION. `analysis_B` calls plurality with no rng, so a genuine tie returns -1 and is
scored incorrect. `exp6c_boundary` passes a seeded rng and breaks ties uniformly. These differ,
so BOTH are computed and reported side by side: `strict` (rng=None, the analysis_B convention,
PRIMARY because the ΔY claim is analysis_B's) and `tiebreak` (seeded, the boundary-runner
convention). If they diverge materially, that is a finding, not a nuisance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from mfg_llm.llm_debate.benchmarks import plurality_prediction  # noqa: E402
from tier1_split_sample_strata import (  # noqa: E402  canonical recovered-strata loader
    EASY_HI, HARD_LO, MIN_Q, R_BIN, discover, load_cell, phat_split)

PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

NONE, SHARED, INDEP = "none", "shared_correct_strong", "indep_correct_strong"
CONDITIONS = [NONE, SHARED, INDEP]          # tiebreak rng index — unchanged from the frozen form
EXCLUDED_BENCHMARKS = {"bbh"}                # logical_deduction: see tier1_breadth.BENCHMARKS
N_BOOT = 5000
SEED = 20261001


def _agreement(state) -> float:
    """Mean pairwise vote agreement over informative agents (analysis_B.agreement)."""
    x = np.asarray(state, float)
    v = np.argmax(x[x.max(1) > x.min(1)], 1)
    if len(v) < 2:
        return float("nan")
    return float(np.mean([v[i] == v[j] for i in range(len(v)) for j in range(i + 1, len(v))]))


def _y(state, qi: int, r: int, cond: str, strict: bool) -> int:
    x = np.asarray(state, float)
    if strict:
        return int(plurality_prediction(x))                      # tie -> -1, scored wrong
    rng = np.random.default_rng([7, qi, r, CONDITIONS.index(cond)])
    return int(plurality_prediction(x, rng))


def _within_churn(ys: list[int]) -> float:
    """All-pairs decision-disagreement rate within one arm over the held-out replicates."""
    n = len(ys)
    if n < 2:
        return float("nan")
    diffs = [int(ys[i] != ys[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(diffs))


def _boot_ci(vals: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    v = vals[~np.isnan(vals)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    bs = v[idx].mean(1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def per_question(rec: dict, eval_reps: range, strict: bool) -> dict:
    """One recovered-stratum question, evaluated on the held-out replicates only."""
    qi = rec["qi"]
    Y = {c: [_y(rec["x"][c][r], qi, r, c, strict) for r in eval_reps] for c in CONDITIONS}
    Acs = [int(y == 0) for y in Y[SHARED]]
    Aci = [int(y == 0) for y in Y[INDEP]]
    dY = [int(ys != yi) for ys, yi in zip(Y[SHARED], Y[INDEP])]
    dYsn = [int(ys != yn) for ys, yn in zip(Y[SHARED], Y[NONE])]
    dYin = [int(yi != yn) for yi, yn in zip(Y[INDEP], Y[NONE])]
    pb = [int(ci and not cs) for cs, ci in zip(Acs, Aci)]
    pr = [int(cs and not ci) for cs, ci in zip(Acs, Aci)]
    D = [_agreement(rec["x"][SHARED][r]) for r in eval_reps]
    C_s, C_i = _within_churn(Y[SHARED]), _within_churn(Y[INDEP])
    C_within = float(np.nanmean([C_s, C_i]))
    dY_cross = float(np.mean(dY))
    return dict(
        dA=float(np.mean(Acs) - np.mean(Aci)), dY=dY_cross,
        dY_shared_vs_none=float(np.mean(dYsn)), dY_indep_vs_none=float(np.mean(dYin)),
        p_break=float(np.mean(pb)), p_rescue=float(np.mean(pr)),
        H_repair=float(np.mean(pb) - np.mean(pr)),
        A_shared=float(np.mean(Acs)), A_indep=float(np.mean(Aci)),
        D_shared=float(np.nanmean(D)),
        C_shared_within=C_s, C_indep_within=C_i, C_within=C_within,
        dC=dY_cross - C_within,
    )


def summarise(rows: list[dict], rng: np.random.Generator) -> dict:
    out = {}
    for k in ("dA", "dY", "dY_shared_vs_none", "dY_indep_vs_none",
              "p_break", "p_rescue", "H_repair", "A_shared", "A_indep", "D_shared",
              "C_shared_within", "C_indep_within", "C_within", "dC"):
        v = np.array([r[k] for r in rows], float)
        m = float(np.nanmean(v))
        lo, hi = _boot_ci(v, rng)
        out[k] = m
        out[k + "_ci"] = [lo, hi]
        out[k + "_sig"] = bool(not np.isnan(lo) and (lo > 0 or hi < 0))
    out["n_q"] = len(rows)
    return out


def pool_cell(model: str, bench: str, paths: dict) -> tuple[list[dict], int]:
    """Pool easy+hard raw regimes into the 80-question split-sample pool (mirrors
    `tier1_split_sample_strata.process_cell`'s pooling step — imported loaders, not copied
    logic), then recover `new_regime` per question from the binning half."""
    rows: list[dict] = []
    seen: dict[str, str] = {}
    for regime in ("easy", "hard"):
        if regime not in paths:
            continue
        for rec in load_cell(paths[regime]):
            if rec["question"] in seen:
                continue
            seen[rec["question"]] = regime
            rows.append(rec)
    if not rows:
        return [], 0
    R = rows[0]["R"]
    bin_reps = range(R_BIN)
    for rec in rows:
        rec["phat"] = phat_split(rec["x"][NONE], bin_reps)
        rec["new_regime"] = ("easy" if rec["phat"] >= EASY_HI else
                             "hard" if rec["phat"] <= HARD_LO else "mid")
    return rows, R


def main() -> None:
    found = discover(None, None)
    rng = np.random.default_rng(SEED)
    cells = []
    for (model, bench, N), paths in sorted(found.items()):
        if bench in EXCLUDED_BENCHMARKS:
            print(f"  SKIP {model}/{bench}: excluded benchmark")
            continue
        if len(paths) < 2:
            print(f"  SKIP {model}/{bench}: only regime {'+'.join(sorted(paths))} on disk")
            continue
        rows, R = pool_cell(model, bench, paths)
        if not rows:
            continue
        eval_reps = range(R_BIN, R)
        for new_regime in ("easy", "hard"):
            sub = [r for r in rows if r["new_regime"] == new_regime]
            if len(sub) < MIN_Q:
                print(f"[{model}/{bench}/{new_regime}] n_q={len(sub)}  INSUFFICIENT (< {MIN_Q})")
                continue
            rec = dict(model=model, benchmark=bench, regime=new_regime, N=N, R=R,
                      n_pooled_questions=len(rows))
            for tag, strict in (("strict", True), ("tiebreak", False)):
                prows = [per_question(r, eval_reps, strict) for r in sub]
                rec[tag] = summarise(prows, rng)
                rec[tag]["per_q"] = prows
            cells.append(rec)
            s, t = rec["strict"], rec["tiebreak"]
            print(f"[{model}/{bench}/{new_regime}] n_q={s['n_q']}  "
                  f"strict: dA={s['dA']:+.3f}{'*' if s['dA_sig'] else ' '} "
                  f"ΔY={s['dY']:.3f}{'*' if s['dY_sig'] else ' '} "
                  f"C_within={s['C_within']:.3f} "
                  f"dC={s['dC']:+.3f}{'*' if s['dC_sig'] else ' '} "
                  f"H={s['H_repair']:+.3f}{'*' if s['H_repair_sig'] else ' '} "
                  f"| tiebreak: dA={t['dA']:+.3f} ΔY={t['dY']:.3f} dC={t['dC']:+.3f}")

    # ── the C1-at-breadth question: is ΔY large while dA is ~0, and does ΔY survive the
    #    within-arm noise baseline (dC)? ──
    lines = ["# Tier-1 decision churn (ΔY) — the ΔY form of the C1 contrast at breadth", "",
             f"Cells: **{len(cells)}** regime-strata, pooled over the English-rebuild 72-cell "
             f"grid (`bbh` excluded). Substrate: recovered split-sample strata "
             f"(`tier1_split_sample_strata.discover`/`load_cell`/`phat_split`, frozen "
             f".80/.30 cuts). Evaluated on the held-out replicates only "
             f"(bin {R_BIN} / eval the rest). Estimators verbatim from "
             f"`analysis_B_matched_mechanism.py:51-61`; `plurality_prediction` imported from "
             f"the frozen module.", "",
             "**SCOPE.** Tier-1 has no copying arm, so this does NOT replicate C1's matched-D "
             "mechanism-non-equivalence (that stays with `analysis_B`/`analysis_C` on qwen). It "
             "establishes the weaker claim: raising dependence churns decisions while leaving net "
             "accuracy flat.", "",
             "**C_within / dC.** `C_within = (C_shared_within + C_indep_within)/2` is the "
             "all-pairs decision-churn rate WITHIN a single arm across the held-out replicates — "
             "the churn rate sampling noise alone produces with no shared/indep swap at all. "
             "`dC = ΔY − C_within` is the churn attributable to the swap net of that baseline; "
             "this is the primary audit for whether the `ΔY > 0` headline reflects the treatment "
             "or just resampling variance.", "",
             "`strict` = tie→-1 (analysis_B convention, **primary**). `tiebreak` = seeded random "
             "tie-break (boundary-runner convention).", ""]
    for tag in ("strict", "tiebreak"):
        groups = [c for c in cells if c["regime"] == "easy"], [c for c in cells if c["regime"] == "hard"]
        lines += [f"## Pooled over cells — `{tag}`", "",
                  "| regime | cells | mean dA | mean ΔY | ΔY sig>0 | mean C_within | "
                  "mean dC | dC sig>0 | mean H_repair |", "|---|---|---|---|---|---|---|---|---|"]
        for name, grp in zip(("easy", "hard"), groups):
            if not grp:
                continue
            g = [c[tag] for c in grp]
            lines.append(
                f"| {name} | {len(g)} | {np.mean([x['dA'] for x in g]):+.3f} | "
                f"{np.mean([x['dY'] for x in g]):.3f} | "
                f"{sum(1 for x in g if x['dY_sig'] and x['dY'] > 0)}/{len(g)} | "
                f"{np.mean([x['C_within'] for x in g]):.3f} | "
                f"{np.mean([x['dC'] for x in g]):+.3f} | "
                f"{sum(1 for x in g if x['dC_sig'] and x['dC'] > 0)}/{len(g)} | "
                f"{np.mean([x['H_repair'] for x in g]):+.3f} |")
        lines.append("")
    lines += ["## Per cell", "",
              "| model | benchmark | regime | n_q | dA | ΔY | C_within | dC | dC 95% CI | "
              "H_repair | ΔY tiebreak | dC tiebreak |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        s = c["strict"]
        lines.append(
            f"| {c['model']} | {c['benchmark']} | {c['regime']} | {s['n_q']} | "
            f"{s['dA']:+.3f}{'*' if s['dA_sig'] else ''} | {s['dY']:.3f}{'*' if s['dY_sig'] else ''} | "
            f"{s['C_within']:.3f} | {s['dC']:+.3f}{'*' if s['dC_sig'] else ''} | "
            f"[{s['dC_ci'][0]:+.3f},{s['dC_ci'][1]:+.3f}] | "
            f"{s['H_repair']:+.3f}{'*' if s['H_repair_sig'] else ''} | "
            f"{c['tiebreak']['dY']:.3f} | {c['tiebreak']['dC']:+.3f} |")

    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tier1_dY_decision_churn.md").write_text("\n".join(lines) + "\n")
    slim = [{k: v for k, v in c.items() if k not in ("strict", "tiebreak")}
            | {t: {k: v for k, v in c[t].items() if k != "per_q"} for t in ("strict", "tiebreak")}
            | {"per_q_strict": c["strict"]["per_q"]} for c in cells]
    (TABLES / "tier1_dY_decision_churn.json").write_text(
        json.dumps(dict(n_cells=len(cells), n_boot=N_BOOT, seed=SEED,
                        excluded_benchmarks=sorted(EXCLUDED_BENCHMARKS),
                        bin_reps=R_BIN, primary_tie_convention="strict", cells=slim),
                  indent=1))
    print(f"\nwrote {TABLES / 'tier1_dY_decision_churn.md'}")
    print(f"wrote {TABLES / 'tier1_dY_decision_churn.json'}")


if __name__ == "__main__":
    main()
