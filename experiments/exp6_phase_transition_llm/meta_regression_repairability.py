"""Meta-regression for the diversity→repairability law — offline, ZERO GPU.

TESTS:  H_repair ~ β0 + β1·(1 − D₀) + u_model + v_task      (README §5.0 "Meta" row)

WHY THIS IS FITTED AT QUESTION LEVEL, NOT CELL LEVEL
----------------------------------------------------
The README spec is written at cell level (one row per model×task×g). That specification is
NOT IDENTIFIED on any data this project can produce, and the failure is structural rather
than a power problem:

  • D₀ is a property of the *baseline arm of a cell*, so at cell level it takes exactly one
    value per (model, task). Regressing on (1 − D₀) while also including v_task therefore
    puts a task-constant regressor next to task fixed effects → perfect collinearity.
    Measured on the g-sweep data that exists (qwen × {truthfulqa, aqua}): 7 rows, 2 distinct
    D₀ values (0.8323, 0.5250), design [1, 1−D₀, task_FE] has rank 2 of 3. β1 unidentified.
  • Tier-3 as scoped (3 models × 3 tasks) would give 9 cells; model FE + task FE consume
    1 + 2 + 2 = 5 df, leaving 3 residual df for β1. Fragile even in the best case.

Fitting per QUESTION fixes the identification problem at its source: D₀ varies *within* a
cell across questions, so β1 is estimated from within-cell variation and survives both sets
of fixed effects. This is the stronger form of the claim — "for a given model on a given
task, the more diverse the crowd is on a question, the more repair headroom that question
carries" — and it is what the raw dumps were persisted for.

  H_repair per question:   h_q = P(break) − P(rescue), averaged over paired replicates
                                 = 1[Y_base correct ∧ Y_treat wrong] − 1[Y_base wrong ∧ Y_treat correct]
  Identity (Prop 6.1):     mean_q h_q == A_base − A_treat     (asserted, assumption-free)
  D₀ per question:         `repair_headroom_xbench.dep_proxy` on the BASELINE arm — mean
                                 pairwise vote agreement over informative agents, imported
                                 verbatim so it matches the C5 tables. Always defined (unlike
                                 `pooled_pair_corr`, which is NaN on saturated questions), and
                                 it keeps the two data sources on one scale.

TWO SOURCES (report both; they are different interventions, not substitutes)
---------------------------------------------------------------------------
  --source c5     copying g-sweep, the intervention C5 is actually about. Faithful but
                  narrow: qwen only, 2 benchmarks (no other model's g-sweep exists — it was
                  deliberately not run). Fits task FE; model FE impossible.
  --source tier1  shared/independent correct-evidence arms of the Tier-1 breadth sweep.
                  Wide (3 models × 7 benchmarks now, 6 × 7 when the sweep finishes) but the
                  intervention supplies *correct* evidence, so it is usually repair-positive
                  (h < 0 = net rescue) rather than damaging. Tests the same law in the
                  opposite direction.

Uses ALL questions and ALL replicates — no easy/hard stratification, so this analysis is
independent of the display-shuffle strata bug and its split-sample repair.

  python experiments/exp6_phase_transition_llm/meta_regression_repairability.py
  python ... --source c5 --n-boot 5000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp6c_shared_context as e6c  # noqa: E402
from mfg_llm.llm_debate.benchmarks import plurality_prediction  # noqa: E402
from repair_headroom import plurality  # noqa: E402  (C5 estimator, verbatim)
from repair_headroom_xbench import dep_proxy, index_aqua, index_tqa  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

NONE = "none"
ARMS = ["shared_correct_strong", "indep_correct_strong"]
RAW_RE = re.compile(r"exp6c_raw_(easy|hard)_scaleup_([a-z0-9.]+)_([a-z0-9]+)_N(\d+)\.json$")


# ── row building ───────────────────────────────────────────────────────────────
def rows_tier1(models: list[str], benchmarks: list[str]) -> list[dict]:
    """One row per (model, task, arm, question) from the Tier-1 raw belief dumps."""
    by_cell: dict[tuple[str, str], list[Path]] = {}
    for p in sorted(PROC.glob("exp6c_raw_*_scaleup_*_N*.json")):
        m = RAW_RE.search(p.name)
        if not m:
            continue
        _, model, bench = m.group(1), m.group(2), m.group(3)
        if (models and model not in models) or (benchmarks and bench not in benchmarks):
            continue
        by_cell.setdefault((model, bench), []).append(p)

    rows, dropped = [], 0
    for (model, bench), paths in sorted(by_cell.items()):
        # accuracies accumulated over the RETAINED questions only, per arm, so the
        # cell-level identity check below compares like with like
        acc: dict[str, list[tuple[float, float]]] = {}
        for path in paths:                                  # pool the easy+hard files
            blob = json.loads(path.read_text())
            raw, R = blob["raw"], blob["R"]
            arms = [a for a in ARMS if a in raw]
            ci_b = e6c.CONDITIONS.index(NONE)
            for qi in range(len(raw[NONE])):
                xs_b = [np.asarray(x, float) for x in raw[NONE][qi]]
                dps = [dep_proxy(x) for x in xs_b]
                if not any(np.isfinite(d) for d in dps):
                    dropped += 1                            # no replicate had ≥2 informative agents
                    continue
                D0 = float(np.nanmean(dps))
                c_b = [plurality_prediction(x, np.random.default_rng([7, qi, r, ci_b])) == 0
                       for r, x in enumerate(xs_b)]
                for arm in arms:
                    ci_t = e6c.CONDITIONS.index(arm)
                    xs_t = [np.asarray(x, float) for x in raw[arm][qi]]
                    c_t = [plurality_prediction(x, np.random.default_rng([7, qi, r, ci_t])) == 0
                           for r, x in enumerate(xs_t)]
                    h = float(np.mean([int(b and not t) - int(t and not b)
                                       for b, t in zip(c_b, c_t)]))
                    # Prop 6.1 identity, per question (assumption-free → must hold exactly)
                    assert abs(h - (np.mean(c_b) - np.mean(c_t))) < 1e-12, "identity broke"
                    acc.setdefault(arm, []).append((float(np.mean(c_b)), float(np.mean(c_t))))
                    rows.append({"source": "tier1", "model": model, "task": bench,
                                 "arm": arm, "regime": blob["regime"], "qi": qi,
                                 "R": R, "h": h, "D0": D0, "div": 1.0 - D0,
                                 "A_base": float(np.mean(c_b))})
        # cell-level identity: mean_q h_q == A_base − A_arm over the same retained questions
        for arm, pairs in acc.items():
            hs = [r["h"] for r in rows if r["model"] == model and r["task"] == bench
                  and r["arm"] == arm]
            gap = np.mean([b for b, _ in pairs]) - np.mean([t for _, t in pairs])
            assert abs(np.mean(hs) - gap) < 1e-9, f"cell identity broke {model}/{bench}/{arm}"
    if dropped:
        print(f"  (dropped {dropped} questions with no informative agent pair in the "
              f"baseline arm — D₀ undefined)")
    return rows


def rows_c5(g_min: float) -> list[dict]:
    """One row per (task, g, question) from the copying g-sweep npz states (qwen only)."""
    rows = []
    for task, by in (("truthfulqa", index_tqa()), ("aqua", index_aqua())):
        gs = sorted({g for gm in by.values() for g in gm} - {"0.00"}, key=float)
        for g in gs:
            if float(g) < g_min:
                continue
            for h_key, gm in by.items():
                if "0.00" not in gm or g not in gm:
                    continue
                x0 = np.load(gm["0.00"], allow_pickle=True)["states"][-1]
                xg = np.load(gm[g], allow_pickle=True)["states"][-1]
                c0, cg = int(plurality(x0) == 0), int(plurality(xg) == 0)
                D0 = dep_proxy(x0)
                if not np.isfinite(D0):
                    continue
                rows.append({"source": "c5", "model": "qwen", "task": task,
                             "arm": f"copy_g{g}", "regime": "all", "qi": h_key, "R": 1,
                             "h": float(int(c0 and not cg) - int(cg and not c0)),
                             "D0": float(D0), "div": 1.0 - float(D0),
                             "A_base": float(c0)})
    return rows


# ── estimation ─────────────────────────────────────────────────────────────────
MIN_CLUSTERS = 3          # below this a cell-cluster bootstrap CI is meaningless, not just wide


def design(rows: list[dict], task_fe: bool, model_fe: bool,
           a_base: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """[1, diversity, (A_base), (task dummies), (model dummies)] — first level dropped.

    `a_base` adds the baseline accuracy of the question. This control is decisive, not
    cosmetic: diversity is mechanically tied to baseline competence (a unanimous crowd has
    diversity 0 and, on an easy question, A_base = 1, so there is nothing left to rescue and
    h can only be ≥ 0). Without A_base, β1 largely measures "hard questions have more room
    to move" rather than any effect of diversity itself.
    """
    n = len(rows)
    cols = [np.ones(n), np.array([r["div"] for r in rows], float)]
    if a_base:
        cols.append(np.array([r["A_base"] for r in rows], float))
    for use, key in ((task_fe, "task"), (model_fe, "model")):
        if not use:
            continue
        levels = sorted({r[key] for r in rows})
        for lv in levels[1:]:
            cols.append(np.array([1.0 if r[key] == lv else 0.0 for r in rows]))
    return np.column_stack(cols), np.array([r["h"] for r in rows], float)


def beta1(rows: list[dict], task_fe: bool, model_fe: bool, a_base: bool = False) -> float:
    """OLS coefficient on diversity; NaN if the design cannot identify it."""
    X, y = design(rows, task_fe, model_fe, a_base)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return float("nan")
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def fit(rows: list[dict], task_fe: bool, model_fe: bool, rng, n_boot: int,
        a_base: bool = False) -> dict:
    """β1 with a CELL-cluster bootstrap (primary — cells are the random-effect unit) and a
    question-level bootstrap (narrow, ignores within-cell correlation).

    With fewer than MIN_CLUSTERS cells the cluster bootstrap is reported as unusable rather
    than as a tight interval: resampling 1 cell always returns that same cell, which would
    otherwise print a zero-width CI and look significant.
    """
    X, _ = design(rows, task_fe, model_fe, a_base)
    pt = beta1(rows, task_fe, model_fe, a_base)
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["task"]), []).append(r)
    keys = sorted(cells)

    cl, qb = [], []
    if len(keys) >= MIN_CLUSTERS:
        for _ in range(n_boot):
            draw = [r for k in (keys[i] for i in rng.integers(0, len(keys), len(keys)))
                    for r in cells[k]]
            b = beta1(draw, task_fe, model_fe, a_base)
            if np.isfinite(b):
                cl.append(b)
    for _ in range(n_boot):
        draw = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        b = beta1(draw, task_fe, model_fe, a_base)
        if np.isfinite(b):
            qb.append(b)

    def ci(v):
        return ([float(np.quantile(v, .025)), float(np.quantile(v, .975))]
                if len(v) >= 2 else [float("nan")] * 2)
    return {"beta1": pt, "n_rows": len(rows), "n_cells": len(keys),
            "n_models": len({r["model"] for r in rows}), "n_tasks": len({r["task"] for r in rows}),
            "design_cols": int(X.shape[1]), "design_rank": int(np.linalg.matrix_rank(X)),
            "ci_cell_cluster": ci(cl), "ci_question": ci(qb),
            "cluster_ci_usable": len(keys) >= MIN_CLUSTERS,
            "boot_ok_cell": len(cl), "boot_ok_question": len(qb)}


def cell_level_diagnostic(rows: list[dict]) -> dict:
    """Reproduce the README's cell-level spec and report WHY it fails."""
    cells: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["task"], r["arm"]), []).append(r)
    agg = [{"model": k[0], "task": k[1], "arm": k[2],
            "h": float(np.mean([r["h"] for r in v])),
            "div": float(np.mean([r["div"] for r in v]))} for k, v in sorted(cells.items())]
    X, _ = design(agg, task_fe=True, model_fe=len({a["model"] for a in agg}) > 1)
    rank, ncol = int(np.linalg.matrix_rank(X)), int(X.shape[1])
    return {"n_cell_rows": len(agg), "design_cols": ncol, "design_rank": rank,
            "identified": rank >= ncol,
            "distinct_div_values": len({round(a["div"], 6) for a in agg}),
            "n_tasks": len({a["task"] for a in agg}), "n_models": len({a["model"] for a in agg}),
            "note": ("β1 identified" if rank >= ncol else
                     "RANK DEFICIENT — (1−D₀) is task-constant at cell level, collinear with "
                     "task fixed effects; this is why the question-level fit is used")}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["tier1", "c5", "both"], default="both")
    ap.add_argument("--models", default="")
    ap.add_argument("--benchmarks", default="")
    ap.add_argument("--g-min", type=float, default=0.0, help="c5: skip couplings below this")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="meta_regression_repairability")
    args = ap.parse_args(argv)

    models = [s for s in args.models.split(",") if s]
    benchmarks = [s for s in args.benchmarks.split(",") if s]
    sources = ["tier1", "c5"] if args.source == "both" else [args.source]

    out = {"spec": "H_repair ~ b0 + b1*(1 - D0) + u_model + v_task",
           "fitted_at": "question level (cell level is rank-deficient — see cell_diagnostic)",
           "D0": "mean pairwise vote agreement in the baseline arm "
                 "(repair_headroom_xbench.dep_proxy, imported verbatim)",
           "n_boot": args.n_boot, "seed": args.seed, "sources": {}}

    for src in sources:
        print("=" * 78)
        print(f"SOURCE = {src}")
        rows = (rows_tier1(models, benchmarks) if src == "tier1" else rows_c5(args.g_min))
        if not rows:
            print("  no rows — skipped")
            continue
        arms = sorted({r["arm"] for r in rows})
        print(f"  {len(rows)} question-rows | models={sorted({r['model'] for r in rows})} "
              f"| tasks={sorted({r['task'] for r in rows})} | arms={arms}")
        diag = cell_level_diagnostic(rows)
        print(f"  cell-level spec: {diag['n_cell_rows']} rows, design {diag['design_rank']}"
              f"/{diag['design_cols']} → {diag['note']}")

        per_arm = {}
        for arm in arms:
            sub = [r for r in rows if r["arm"] == arm]
            n_models = len({r["model"] for r in sub})
            specs = [("M0 pooled", False, False, False), ("M1 +task FE", True, False, False)]
            if n_models > 1:
                specs.append(("M2 +task+model FE", True, True, False))
            specs.append((f"M3 {'M2' if n_models > 1 else 'M1'}+A_base", True, n_models > 1, True))
            print(f"\n  [{arm}]  n={len(sub)}  mean H_repair={np.mean([r['h'] for r in sub]):+.4f}"
                  f"  mean diversity={np.mean([r['div'] for r in sub]):.3f}"
                  f"  mean A_base={np.mean([r['A_base'] for r in sub]):.3f}"
                  f"  corr(div, A_base)={np.corrcoef([r['div'] for r in sub], [r['A_base'] for r in sub])[0, 1]:+.3f}")
            fits = {}
            for si, (label, tfe, mfe, ab) in enumerate(specs):
                rng = np.random.default_rng([args.seed, si])   # deterministic (no str hash)
                f = fit(sub, tfe, mfe, rng, args.n_boot, ab)
                fits[label] = f
                if f["cluster_ci_usable"]:
                    lo, hi = f["ci_cell_cluster"]
                    cl = f"cell-cluster CI [{lo:+.4f},{hi:+.4f}]{'*' if (lo > 0 or hi < 0) else ' '}"
                else:
                    cl = f"cell-cluster CI n/a (only {f['n_cells']} cell(s) < {MIN_CLUSTERS})"
                print(f"    {label:20s} β1={f['beta1']:+.4f}  {cl}  "
                      f"question CI [{f['ci_question'][0]:+.4f},{f['ci_question'][1]:+.4f}]")
            if n_models == 1:
                print("    (model FE impossible — only one model in this source)")
            per_arm[arm] = fits
        out["sources"][src] = {"cell_diagnostic": diag, "n_rows": len(rows),
                               "models": sorted({r["model"] for r in rows}),
                               "tasks": sorted({r["task"] for r in rows}), "fits": per_arm}

    TABLES.mkdir(parents=True, exist_ok=True)
    p = TABLES / f"{args.tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("\n" + "=" * 78)
    print(f"wrote {p}")
    print("β1 > 0 ⇒ more diverse crowds carry MORE repair headroom (the hypothesised law). "
          "Cell-cluster CI is primary; with few cells it is wide by construction, which is "
          "honest, not a bug.")


if __name__ == "__main__":
    main()
