"""Tier-3 matched-D consumer — FROZEN 2026-10-12, BEFORE any Tier-3 cell has run.

Extends Analysis B/C's qwen-only matched-D comparison (upstream shared-evidence vs
downstream copying, on a common dependence scale D) to the 3-model x 3-benchmark Tier-3
grid, using the SAME three estimators, imported verbatim from
`analysis_C_matched_D_falsification.py` (never re-implemented): caliper matching at
eps in {.02,.05,.10}, nested linear-probability models (D, +mechanism, +D*mechanism) with
nested F-tests, and nearest-neighbour matching on (D, A0, margin).

WHY THE "SHARED" SIDE IS NOT analysis_B/C's ORIGINAL FILE (data integrity finding,
2026-10-12). `analysis_B_matched_mechanism.py` / `analysis_C_matched_D_falsification.py`
read `data/processed/exp6c_raw_{easy,hard}_N11.json`, last written 2026-09-08 -- BEFORE the
2026-10-01 language-confound discovery and the 2026-10-03 English clue-cache regeneration.
That file was never regenerated against the English substrate, so analysis_B/C's qwen
"shared" rows are the SAME vintage as the Vietnamese-era data the whole English-rebuild
campaign exists to replace; they are not known-good, merely not yet re-audited. There is
also no analogue of that file for llama/gemma, so it could not cover the full Tier-3 grid
even if trusted. We therefore source the "shared" side from the English-rebuild Tier-1
breadth raw dumps instead --
`data/processed/exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json` -- for exactly the
(model, benchmark) pairs Tier-3 covers (qwen/llama/gemma x truthfulqa/strategyqa/aqua,
pooling both regimes, matching analysis_B/C's own easy+hard pooling). This is already-frozen,
English, audited data (the same substrate behind C1/C2/C3/dC above) -- not a new run, not a
new estimator, just a different, cleaner frozen source for the same role analysis_B/C's file
played. `analysis_B.py`/`analysis_C.py` are left byte-identical; this script supersedes their
coverage, it does not edit them.

GRID, FROZEN, EXACTLY 18 CELLS (`tier3_copying.py`, do not extend): models
{qwen, llama, gemma} x benchmarks {truthfulqa, strategyqa, aqua} x regimes {easy, hard},
coupling g in {0, 1, 2}, N=11. No additional model, benchmark, regime, coupling value,
replicate, caliper width, or matching rule may be added after this script is run once Tier-3
data exists. If Tier-3 does not replicate the qwen matched-D effect, that is reported as a
negative result, not grounds to add a 19th cell or a 4th caliper width.

AGGREGATION ACROSS THE 9 (MODEL, BENCHMARK) PAIRS. Each pair's rows (pooling easy+hard, as
analysis_B/C do) feed the three tests independently, exactly reproducing the qwen-only
procedure once per pair -- 9 independent replications, not one 9x-larger pool, so one
config's row count cannot dominate another's (same equal-weight-per-stratum philosophy
already used for `dC` and C2 above, for the same reason: the question is whether the
mechanism gap generalizes across configurations). A SINGLE pooled headline per test, if
wanted, is the unweighted mean of the 9 pairs' point estimates with a stratum-level bootstrap
(resample PAIRS, not rows) -- never a question-weighted alternative chosen after seeing which
gives a cleaner result.

PER-MECHANISM WITHIN-ARM NOISE -- ASYMMETRIC BY DESIGN, STATED BEFORE ANY RESULT EXISTS
(2026-10-12 audit, in response to the finding that raw cross-arm dY/|dA| are inflated by
replicate noise -- see `tier1_dY_decision_churn.py`'s dC and `tier1_c2_noise_adjusted_aggregate.py`).
  SHARED side: the Tier-1 breadth source has R=8 independent replicates per question, so
    `C_within_shared` is computable exactly as `dC` was -- mean pairwise disagreement of the
    arm's OWN decision across all C(8,2)=28 replicate pairs -- giving
    `dC_shared = dY_shared_vs_indep - C_within_shared` per row.
  COPY side: `tier3_copying.py` / `run.py` draw exactly ONE coupling realization per
    (question, g); the `R` in `run.py` is the number of SEQUENTIAL belief-update rounds
    inside that one debate, not independent replicate redraws. There is therefore NO
    repeated-draw structure at fixed g from which to estimate a within-copy-arm noise floor,
    and `C_within_copy` / `dC_copy` are reported as `null` ("not estimable from this design"),
    never silently imputed, never set to the shared side's value, and never used to claim the
    copy mechanism has zero noise. The three matched-D tests below therefore necessarily
    compare `dC_shared` (noise-corrected) against raw `dY_copy` (single-draw, uncorrected) on
    the shared side vs copy side respectively -- an acknowledged asymmetry, reported as such in
    every output row, not resolved by inventing a copy-side correction post hoc.

COMMON SUPPORT, REPORTED NOT EXTRAPOLATED. Copying is expected to saturate `D` near 1 (the
qwen g=2 cell in the Vietnamese-era precedent did); the shared arm may rarely reach that D
region. Every caliper result below reports `n_matched / n_copy_total` (the fraction of copy
rows that found ANY shared row within the caliper) and the D-range of each arm, so a caliper
width with near-zero common support reads as "could not test here", not as a null finding.

Writes `results/tables/tier3_matched_d.json`. Safe to run before any Tier-3 cell exists: pairs
with zero copy-rows are reported as "pending", not silently dropped or null-filled.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis_C_matched_D_falsification import (  # noqa: E402  frozen, imported verbatim
    agreement, caliper_contrast, margin, nested_models, nn_matching, plur)

ROOT = HERE.parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

MODELS = ["qwen", "llama", "gemma"]
BENCHMARKS = ["truthfulqa", "strategyqa", "aqua"]
REGIMES = ["easy", "hard"]
MODEL_TAG = {"qwen": "", "llama": "llama8b", "gemma": "gemma9b"}      # tier3_copying.py, frozen
N_BOOT = 2000           # matches analysis_C's own n_boot for caliper/NN; not re-tuned
MACRO_N_BOOT = 20000     # matches the dC/C2 macro-average scale


def _within_churn(ys: list[int]) -> float:
    n = len(ys)
    if n < 2:
        return float("nan")
    return float(np.mean([int(ys[i] != ys[j]) for i in range(n) for j in range(i + 1, n)]))


def shared_rows_tier1(model: str, bench: str) -> list[dict]:
    """Pool easy+hard from the English-rebuild Tier-1 breadth raw dump for (model, bench).
    Same row schema as analysis_C.shared_rows() (D/A0/margin/dA/dY/p_break/p_rescue/H), PLUS
    `C_within` and `dC` (= dY - C_within): dY here is already analysis_B/C's own cross-arm
    quantity P(Y_shared != Y_indep), so it is the exact same noise-inflation pattern `dC`
    corrected at breadth. R=8 independent replicates per question make this computable."""
    rows = []
    for regime in REGIMES:
        f = PROC / f"exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json"
        if not f.exists():
            continue
        raw = json.loads(f.read_text())["raw"]
        for q in range(len(raw["none"])):
            D, dY, pmp, ppm, Ac, Ai, Mg, Ys = [], [], [], [], [], [], [], []
            for r in range(len(raw["shared_correct_strong"][q])):
                ys = plur(raw["shared_correct_strong"][q][r])
                yi = plur(raw["indep_correct_strong"][q][r])
                D.append(agreement(raw["shared_correct_strong"][q][r]))
                Mg.append(margin(raw["indep_correct_strong"][q][r]))
                cs, ci = int(ys == 0), int(yi == 0)
                Ac.append(cs); Ai.append(ci); dY.append(int(ys != yi)); Ys.append(ys)
                pmp.append(int(ci and not cs)); ppm.append(int(cs and not ci))
            dY_cross = float(np.mean(dY))
            c_within = _within_churn(Ys)
            rows.append(dict(mech="shared", D=float(np.nanmean(D)), A0=float(np.mean(Ai)),
                             margin=float(np.nanmean(Mg)), dA=float(np.mean(Ac) - np.mean(Ai)),
                             dY=dY_cross, p_break=float(np.mean(pmp)),
                             p_rescue=float(np.mean(ppm)), H=float(np.mean(pmp) - np.mean(ppm)),
                             C_within=c_within, dC=dY_cross - c_within))
    return rows


def copy_rows_tier3(model: str, bench: str) -> list[dict]:
    """Pool easy+hard Tier-3 g-sweep npz for (model, bench). Same schema/semantics as
    analysis_C.copy_rows(), generalized over model via the model-tag npz namespace.
    `C_within`/`dC` are `None`: ONE coupling realization per (question, g) -- no repeated
    draws to estimate a within-copy-arm noise floor from. Reported explicitly, not imputed."""
    mtag = f"_{MODEL_TAG[model]}" if MODEL_TAG[model] else ""
    pat = re.compile(rf"exp6_{re.escape(bench)}{re.escape(mtag)}_N11_g([0-9.]+)_q(\d+)_([0-9a-f]+)\.npz$")
    by: dict[str, dict[str, str]] = {}
    for f in glob.glob(str(PROC / f"exp6_{bench}{mtag}_N11_g*.npz")):
        m = pat.search(f)
        if m:
            by.setdefault(m.group(3), {})[m.group(1)] = f
    rows = []
    for _, gm in by.items():
        if "0.00" not in gm:
            continue
        s0 = np.load(gm["0.00"], allow_pickle=True)["states"][-1]
        y0, c0, mg0 = plur(s0), None, margin(s0)
        c0 = int(y0 == 0)
        for g, p in gm.items():
            if g == "0.00":
                continue
            sg = np.load(p, allow_pickle=True)["states"][-1]
            yg = plur(sg); cg = int(yg == 0)
            rows.append(dict(mech="copy", g=float(g), D=float(agreement(sg)), A0=float(c0),
                             margin=float(mg0), dA=float(cg - c0), dY=float(int(yg != y0)),
                             p_break=float(int(c0 and not cg)), p_rescue=float(int(cg and not c0)),
                             H=float(int(c0 and not cg) - int(cg and not c0)),
                             C_within=None, dC=None))
    return rows


def d_range(rows: list[dict]) -> list[float]:
    d = [r["D"] for r in rows if np.isfinite(r["D"])]
    return [float(min(d)), float(max(d))] if d else [None, None]


def macro_bootstrap(vals: np.ndarray, rng: np.random.Generator) -> dict:
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return {"K": int(vals.size), "mean": float(vals.mean()) if vals.size else None,
                "ci": [None, None], "sig": False}
    K = vals.size
    idx = rng.integers(0, K, size=(MACRO_N_BOOT, K))
    draws = vals[idx].mean(1)
    lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    return {"K": K, "mean": float(vals.mean()), "ci": [lo, hi],
            "sig": bool(hi - lo > 1e-12 and (lo > 0 or hi < 0))}


def main() -> None:
    rng = np.random.default_rng(20261012)
    pairs, pending = [], []
    for model in MODELS:
        for bench in BENCHMARKS:
            sh = shared_rows_tier1(model, bench)
            cp = copy_rows_tier3(model, bench)
            if len(cp) < 5 or len(sh) < 5:
                pending.append({"model": model, "bench": bench, "n_shared": len(sh),
                                "n_copy": len(cp), "status": "pending (Tier-3 not run yet)"})
                continue
            print(f"[{model}/{bench}] shared n={len(sh)}  copy n={len(cp)}")
            d_sh, d_cp = d_range(sh), d_range(cp)
            print(f"    D range: shared {d_sh}  copy {d_cp}  (common support check)")
            caliper = {}
            for eps in (0.02, 0.05, 0.10):
                per_metric = {m: caliper_contrast(sh, cp, eps, m, n_boot=N_BOOT)
                             for m in ("dY", "dA", "p_break", "p_rescue")}
                n_match = per_metric["dY"]["n"]
                coverage = n_match / len(cp) if len(cp) else 0.0
                caliper[f"eps_{eps}"] = {**per_metric, "n_matched": n_match,
                                         "n_copy_total": len(cp), "coverage": coverage}
                print(f"    eps={eps}: matched {n_match}/{len(cp)} copy rows "
                      f"(coverage={coverage:.2f})"
                      + (" -- LOW COMMON SUPPORT, do not extrapolate" if coverage < 0.5 else ""))
            nested = nested_models(sh, cp)
            nn = nn_matching(sh, cp, n_boot=N_BOOT)
            sh_dC = np.array([r["dC"] for r in sh], float)
            sh_dY = np.array([r["dY"] for r in sh], float)
            cp_dY = np.array([r["dY"] for r in cp], float)
            pairs.append({"model": model, "bench": bench, "n_shared": len(sh), "n_copy": len(cp),
                         "D_range_shared": d_sh, "D_range_copy": d_cp,
                         "caliper": caliper, "nested_models": nested, "nn_matching": nn,
                         "shared_noise_correction": {
                             "mean_dY_raw": float(sh_dY.mean()),
                             "mean_C_within": float(np.nanmean([r["C_within"] for r in sh])),
                             "mean_dC": float(sh_dC.mean())},
                         "copy_noise_correction": "not estimable (single draw per (q,g)) "
                                                   "-- reporting raw dY only, see module docstring",
                         "mean_dY_copy_raw": float(cp_dY.mean())})

    out = {"grid": {"models": MODELS, "benchmarks": BENCHMARKS, "regimes": REGIMES,
                    "coupling": [0, 1, 2], "n_cells_frozen": 18},
          "shared_side_source": "English-rebuild Tier-1 breadth raw dumps "
                                "(exp6c_raw_{regime}_scaleup_{model}_{bench}_N11.json), "
                                "NOT analysis_B/C's original (stale, pre-English) file",
          "pending": pending, "pairs": pairs}

    if pairs:
        print("\n=== MACRO-AVERAGE across (model, benchmark) pairs, equal weight ===")
        for eps in (0.02, 0.05, 0.10):
            gaps = np.array([p["caliper"][f"eps_{eps}"]["dY"]["gap"] for p in pairs
                             if p["caliper"][f"eps_{eps}"]["dY"]["gap"] is not None], float)
            r = macro_bootstrap(gaps, rng)
            print(f"  caliper eps={eps}  dY gap: K={r['K']} mean={r['mean']}  ci={r['ci']}")
            out[f"macro_caliper_dY_eps_{eps}"] = r
        att = np.array([p["nn_matching"]["att_copy_minus_shared"] for p in pairs], float)
        r = macro_bootstrap(att, rng)
        print(f"  NN-matching ATT(copy-shared) dY: K={r['K']} mean={r['mean']}  ci={r['ci']}")
        out["macro_nn_att_dY"] = r
        mech = np.array([p["nested_models"]["mech_coef_given_D"] for p in pairs], float)
        r = macro_bootstrap(mech, rng)
        print(f"  nested-LPM mechanism coef given D: K={r['K']} mean={r['mean']}  ci={r['ci']}")
        out["macro_nested_mech_coef"] = r
    else:
        print("No (model,benchmark) pair has Tier-3 data yet -- all 9 pending. "
              "This is expected before the 18-cell sweep runs; nothing to report.")

    if pending:
        print(f"\npending ({len(pending)}): " +
              "; ".join(f"{p['model']}/{p['bench']} (shared={p['n_shared']}, copy={p['n_copy']})"
                       for p in pending))

    dst = TABLES / "tier3_matched_d.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
