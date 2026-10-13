"""Tier-1 SPLIT-SAMPLE strata recovery — offline, ZERO GPU.

WHY THIS EXISTS
---------------
The `easy`/`hard` labels in the Tier-1 sweep are not reproducible. Both the probe and the
boundary runner seed the option-display shuffle with `default_rng([12345, qi])`, but `qi`
means different things in the two files:

    p2_probe.py:56       for qi, (q, opts) in enumerate(pool)        → qi = dataset index 0..399
    exp6c_boundary.py:239 for qi, (q, opts) in enumerate(questions)  → qi = position in the
                                                                        selected 40-q subset

Same question, different option permutation → the model's position bias flips answers, so a
question probed at p̂≥0.80 can re-measure at A_none≈0.0 in the run. The *within-cell*
contrasts (none / shared / indep share one question set and one display order) are unaffected
and stay valid — only the regime LABELS are compromised.

The code is deliberately NOT fixed: re-seeding mid-sweep would make the remaining cells
incomparable with the ones already finished (~78 cells ≈ 8 GPU-days to redo). Instead the
strata are recovered offline from the raw belief dumps, which hold every (condition,
question, replicate) canonical (N,K) matrix with truth@0.

RECIPE (split-sample, so the binning variable is independent of the evaluation data)
-----------------------------------------------------------------------------------
Per (model, benchmark): pool the easy+hard cells → 80 questions.
  BIN   on `none` replicates 0..3  → p̂_split = vote-level accuracy over N×4 = 44 votes,
        the exact same statistic and budget as the old probe (`p2_probe._correct`), with the
        frozen cuts easy ≥ 0.80 / hard ≤ 0.30 (mid-band questions are dropped, as in the probe).
  EVAL  on `none`/`shared`/`indep` replicates 4..7 → A_q, D_q per condition.
Because the bin is decided on replicates the evaluation never sees, there is no winner's
curse: a question is not selected *because* it happened to score high on the data used to
score it. Cost: effective R=4 for evaluation ⇒ wider error bars than the R=8 protocol.

Estimators are IMPORTED from the frozen modules (never re-implemented) so they cannot drift:
`plurality_prediction`, `pooled_pair_corr`, `CONDITIONS` (for the rng seed `ci`).
Per-replicate correctness reproduces the runner exactly:
    plurality_prediction(x, default_rng([7, qi, r, ci])) == 0

MARGIN M_q for the C2 regression is computed from the `none` condition only (as in the frozen
protocol — ΔA is a shared−indep contrast, so a none-derived regressor cannot leak the
treatment). `--margin-half` picks which replicates feed it; `all` (default) reproduces the
frozen 88-vote definition, `bin` uses only 0..3 for a fully split-sample covariate.

  python experiments/exp6_phase_transition_llm/tier1_split_sample_strata.py
  python ... --models qwen --benchmarks truthfulqa --n-boot 20000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp6c_shared_context as e6c  # noqa: E402  (defs only; main is __main__-guarded)
from dependence_estimators import (  # noqa: E402  canonical estimators, one implementation
    ESTIMATOR_VERSION, boot_pooled_ratio, n_eff, pair_corr_terms)
from mfg_llm.llm_debate.benchmarks import plurality_prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

SHARED = "shared_correct_strong"
INDEP = "indep_correct_strong"
NONE = "none"

# frozen selection constants — same numbers as p2_probe / REGIMES, do not tune
EASY_HI = 0.80
HARD_LO = 0.30
R_BIN = 4          # `none` replicates used for binning (matches the old probe's R=4)
MIN_Q = 15         # a bin below this is reported as insufficient, not silently used
MIN_NEFF = 15      # pooled-ratio estimability gate (2026-10-03). Questions carry unequal
                   # denominator weight, so n_q >= MIN_Q does not imply the ratio is backed by
                   # MIN_Q questions; n_eff is. This is the project's single estimability gate.

RAW_RE = re.compile(r"exp6c_raw_(easy|hard)_scaleup_([a-z0-9.]+)_([a-z0-9]+)_N(\d+)\.json$")


# ── loading ────────────────────────────────────────────────────────────────────
def discover(models: list[str] | None, benchmarks: list[str] | None) -> dict:
    """→ {(model, bench, N): {regime: raw_path}} for every raw dump on disk."""
    cells: dict[tuple[str, str, int], dict[str, Path]] = {}
    for p in sorted(PROC.glob("exp6c_raw_*_scaleup_*_N*.json")):
        m = RAW_RE.search(p.name)
        if not m:
            continue
        regime, model, bench, N = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if models and model not in models:
            continue
        if benchmarks and bench not in benchmarks:
            continue
        cells.setdefault((model, bench, N), {})[regime] = p
    return cells


def load_cell(raw_path: Path) -> list[dict]:
    """One (model, bench, regime) cell → list of per-question records with raw matrices.

    Joins the raw dump (by position) to the boundary analysis JSON, which carries the
    question text. Positional join is exact: both are written in the same loop order by
    exp6c_boundary.main, from the same `questions` list.
    """
    blob = json.loads(raw_path.read_text())
    ana_path = TABLES / raw_path.name.replace("exp6c_raw_", "exp6c_boundary_")
    if not ana_path.exists():
        raise FileNotFoundError(f"missing companion analysis JSON {ana_path.name}")
    ana = json.loads(ana_path.read_text())
    raw, conds, R = blob["raw"], blob["conditions"], blob["R"]
    per_q = ana["per_q"]
    nq = len(raw[NONE])
    if len(per_q) != nq:
        raise ValueError(f"{raw_path.name}: {nq} raw questions vs {len(per_q)} in analysis JSON")
    if R <= R_BIN:
        raise ValueError(f"{raw_path.name}: R={R} leaves no held-out replicates (need R>{R_BIN})")
    out = []
    for qi in range(nq):
        rec = {
            "old_regime": blob["regime"], "qi": qi, "R": R,
            "question": per_q[qi]["question"], "n_options": per_q[qi]["n_options"],
            "model": ana["model"], "model_tag": ana["model_tag"],
            "x": {c: [np.asarray(x, float) for x in raw[c][qi]] for c in conds},
        }
        out.append(rec)
    return out


# ── the two halves ─────────────────────────────────────────────────────────────
def _vote_correct(row: np.ndarray) -> bool:
    """p2_probe._correct verbatim: a vote counts only if informative, truth is index 0."""
    return bool(row.max() > row.min() and int(np.argmax(row)) == 0)


def phat_split(x_none: list[np.ndarray], reps: range) -> float:
    """Binning statistic: vote-level accuracy over N × len(reps) `none` votes."""
    votes = [_vote_correct(row) for r in reps for row in x_none[r]]
    return float(np.mean(votes)) if votes else 0.0


# MOVED 2026-10-03 to dependence_estimators.pair_corr_terms -- single canonical
# implementation, imported rather than copied. The self-check assert below still
# verifies num/den rebuilds the frozen e6c.pooled_pair_corr ratio on every call.
_pair_corr_terms = pair_corr_terms


def eval_condition(x_reps: list[np.ndarray], qi: int, cond: str, reps: range) -> dict:
    """Held-out A_q (plurality accuracy) and D_q (replicate-level pooled error corr)."""
    ci = e6c.CONDITIONS.index(cond)
    plur, E_rows = [], []
    for r in reps:
        x = x_reps[r]
        plur.append(int(plurality_prediction(x, np.random.default_rng([7, qi, r, ci])) == 0))
        E_rows.append((x.argmax(1) != 0).astype(float))
    E = np.array(E_rows)
    D = e6c.pooled_pair_corr(E)
    num, den = _pair_corr_terms(E)
    if den > 1e-12:                       # self-check: our terms must rebuild the frozen ratio
        assert abs(num / den - D) < 1e-9, f"pair-corr term mismatch at q{qi} {cond}"
    return {"A_q": float(np.mean(plur)), "D_q": (None if np.isnan(D) else float(D)),
            "D_num": num, "D_den": den}


def margin_M(x_none: list[np.ndarray], reps: range) -> float:
    """Frozen primary margin: runner-up gap of the pooled informative `none` vote shares."""
    votes, K = [], x_none[0].shape[1]
    for r in reps:
        x = x_none[r]
        info = x.max(1) > x.min(1)
        if info.any():
            votes.extend(int(t) for t in np.argmax(x[info], 1))
    if not votes:
        return 0.0
    shares = np.bincount(votes, minlength=K).astype(float)
    shares /= shares.sum()
    order = np.sort(shares)[::-1]
    return float(order[0] - (order[1] if len(order) > 1 else 0.0))


# ── bootstrap (vectorized; question-level resampling) ───────────────────────────
def _idx(rng, n: int, n_boot: int) -> np.ndarray:
    return rng.integers(0, n, (n_boot, n))


def boot_mean(v: np.ndarray, rng, n_boot: int) -> tuple[float, float, float]:
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float("nan"),) * 3
    draws = v[_idx(rng, v.size, n_boot)].mean(1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), float(v.mean())


# MOVED 2026-10-03 to dependence_estimators.boot_pooled_ratio (carries the
# degenerate-bootstrap guard). Imported, not reimplemented.


def boot_slope(x: np.ndarray, y: np.ndarray, rng, n_boot: int) -> tuple[float, float, float]:
    """OLS slope of y on x with an intercept = cov/var, bootstrapped over questions."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return (float("nan"),) * 3

    def sl(xs, ys):
        """Row-wise OLS slope for a (B, n) resample matrix; nan where x has no spread."""
        xm, ym = xs.mean(1, keepdims=True), ys.mean(1, keepdims=True)
        var = ((xs - xm) ** 2).sum(1)
        cov = ((xs - xm) * (ys - ym)).sum(1)
        return np.where(var > 1e-12, cov / np.where(var > 1e-12, var, 1.0), np.nan)
    I = _idx(rng, x.size, n_boot)
    draws = sl(x[I], y[I])
    draws = draws[np.isfinite(draws)]
    if draws.size < 2:
        return (float("nan"),) * 3
    pt = float(sl(x[None, :], y[None, :])[0])
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), pt


# ── per-cell analysis ──────────────────────────────────────────────────────────
def analyze_bin(rows: list[dict], seed: list[int], n_boot: int) -> dict:
    """C1/C2/C3 on one recovered stratum. Same estimators as exp6c_boundary_analyze."""
    A_none = np.array([r["eval"][NONE]["A_q"] for r in rows])
    A_s = np.array([r["eval"][SHARED]["A_q"] for r in rows])
    A_i = np.array([r["eval"][INDEP]["A_q"] for r in rows])
    D_s = np.array([np.nan if r["eval"][SHARED]["D_q"] is None else r["eval"][SHARED]["D_q"]
                    for r in rows])
    D_i = np.array([np.nan if r["eval"][INDEP]["D_q"] is None else r["eval"][INDEP]["D_q"]
                    for r in rows])
    M = np.array([r["M"] for r in rows])
    dA, dD = A_s - A_i, D_s - D_i

    rng = np.random.default_rng(seed)
    dd_lo, dd_hi, dd_pt = boot_mean(dD, rng, n_boot)
    da_lo, da_hi, da_pt = boot_mean(dA, rng, n_boot)
    c2_lo, c2_hi, c2_pt = boot_slope(M, np.abs(dA), rng, n_boot)

    # competing ΔD estimator: pool the pair-corr terms across questions (keeps saturated qs)
    terms = {c: (np.array([r["eval"][c]["D_num"] for r in rows], float),
                 np.array([r["eval"][c]["D_den"] for r in rows], float))
             for c in (SHARED, INDEP)}
    pd_lo, pd_hi, pd_pt = boot_pooled_ratio(*terms[SHARED], *terms[INDEP], rng, n_boot)
    # Weight each question by the denominator it contributes to BOTH arms: that is the weight
    # the pooled ratio actually gives it.
    n_eff_pooled = n_eff(terms[SHARED][1] + terms[INDEP][1])
    return {
        "n_q": len(rows),
        "n_D": int(np.isfinite(dD).sum()),
        "estimator_version": ESTIMATOR_VERSION,
        "dD_pooled": pd_pt, "dD_pooled_ci": [pd_lo, pd_hi],
        # n_eff is the ONLY estimability gate in the project (decided 2026-10-03). n_q counts
        # questions; n_eff counts how many actually carry denominator weight. A stratum can
        # show n_q=44 with n_eff≈1, in which case the "pooled" D is one question wearing a
        # disguise. Significance therefore requires a finite CI AND n_eff >= MIN_NEFF; the
        # point estimate still prints so nothing is hidden, only unscored.
        "n_eff": n_eff_pooled,
        "n_eff_gate": MIN_NEFF,
        "dD_pooled_estimable": bool(n_eff_pooled >= MIN_NEFF),
        "dD_pooled_sig": bool(np.isfinite(pd_lo) and (pd_lo > 0 or pd_hi < 0)
                              and n_eff_pooled >= MIN_NEFF),
        "A_none": float(A_none.mean()), "A_shared": float(A_s.mean()), "A_indep": float(A_i.mean()),
        "gain_shared": float((A_s - A_none).mean()), "gain_indep": float((A_i - A_none).mean()),
        "dA": da_pt, "dA_ci": [da_lo, da_hi],
        "dD": dd_pt, "dD_ci": [dd_lo, dd_hi], "dD_sig": bool(dd_lo > 0 or dd_hi < 0),
        "c2_slope": c2_pt, "c2_ci": [c2_lo, c2_hi], "c2_sig": bool(c2_lo > 0 or c2_hi < 0),
        "mean_M": float(M.mean()),
        "phat_range": [float(min(r["phat"] for r in rows)), float(max(r["phat"] for r in rows))],
    }


def process_cell(model: str, bench: str, N: int, paths: dict[str, Path],
                 margin_half: str, n_boot: int, seed: int) -> dict:
    rows: list[dict] = []
    seen: dict[str, str] = {}
    for regime in ("easy", "hard"):
        if regime not in paths:
            continue
        for rec in load_cell(paths[regime]):
            if rec["question"] in seen:        # same question selected by both probes
                continue                        # (impossible under p̂≥.80 / p̂≤.30 — guard anyway)
            seen[rec["question"]] = regime
            rows.append(rec)

    R = rows[0]["R"]
    bin_reps, eval_reps = range(R_BIN), range(R_BIN, R)
    m_reps = {"all": range(R), "bin": bin_reps, "eval": eval_reps}[margin_half]

    for rec in rows:
        rec["phat"] = phat_split(rec["x"][NONE], bin_reps)
        rec["new_regime"] = ("easy" if rec["phat"] >= EASY_HI else
                             "hard" if rec["phat"] <= HARD_LO else "mid")
        rec["M"] = margin_M(rec["x"][NONE], m_reps)
        rec["eval"] = {c: eval_condition(rec["x"][c], rec["qi"], c, eval_reps) for c in rec["x"]}

    # bin migration — how badly the old labels were wrong
    migration = {old: {new: 0 for new in ("easy", "mid", "hard")} for old in ("easy", "hard")}
    for rec in rows:
        if rec["old_regime"] in migration:
            migration[rec["old_regime"]][rec["new_regime"]] += 1

    strata = {}
    for new_regime in ("easy", "hard"):
        sub = [r for r in rows if r["new_regime"] == new_regime]
        if len(sub) < MIN_Q:
            strata[new_regime] = {"n_q": len(sub), "insufficient": True}
            continue
        # deterministic per-cell seed (str.__hash__ is PYTHONHASHSEED-randomized — do not use)
        tag = f"{model}|{bench}|{new_regime}"
        strata[new_regime] = analyze_bin(
            sub, [seed, int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)], n_boot)
        strata[new_regime]["insufficient"] = False

    return {
        "model": model, "benchmark": bench, "N": N, "R": R,
        "pooled_regimes": sorted(paths), "n_pooled_questions": len(rows),
        "bin_reps": [bin_reps.start, bin_reps.stop], "eval_reps": [eval_reps.start, eval_reps.stop],
        "migration": migration, "strata": strata,
        "per_q": [{"question": r["question"], "old_regime": r["old_regime"], "qi": r["qi"],
                   "phat_split": r["phat"], "new_regime": r["new_regime"], "M": r["M"],
                   "eval": r["eval"]} for r in rows],
    }


# ── reporting ──────────────────────────────────────────────────────────────────
def _n(v, p: int = 3, sign: str = "+") -> str:
    """Format a statistic, rendering an undefined (NaN) value as an em dash."""
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:{sign}.{p}f}"


def _ci(ci) -> str:
    if ci is None or not all(np.isfinite(c) for c in ci):
        return "—"
    return f"[{ci[0]:+.3f},{ci[1]:+.3f}]"


def markdown(cells: list[dict], margin_half: str) -> str:
    L = ["# Tier-1 strata recovered by split-sample re-binning (offline, 0 GPU)", "",
         f"Bin on `none` replicates 0–{R_BIN - 1} (p̂ over N×{R_BIN} votes, frozen cuts "
         f"easy ≥ {EASY_HI:.2f} / hard ≤ {HARD_LO:.2f}); evaluate all conditions on the "
         f"held-out replicates {R_BIN}–{cells[0]['R'] - 1}. Margin M from `none` "
         f"replicates = `{margin_half}`. Bins with < {MIN_Q} questions are reported, not used.", "",
         "Supersedes the probe-based `easy`/`hard` labels, which do not reproduce "
         "(display-shuffle `qi` convention differs between `p2_probe.py:56` and "
         "`exp6c_boundary.py:239`). Within-cell contrasts were never affected.", "",
         "## Recovered strata", "",
         "**ΔD pooled is PRIMARY** (ratified 2026-09-30): it sums the pair-correlation "
         "numerator and denominator over ALL questions, so a saturated question contributes "
         "num = den = 0 and drops out arithmetically. `ΔD mean-of-ratios` — the frozen "
         "per-question estimator — is kept as the legacy comparison, but in the easy strata it "
         "is computable on as few as 4 of 45 questions (`n_D`), and that surviving subset is "
         "selected on error variance, the very quantity D measures. Both are shown so the "
         "choice stays visible. The frozen 4×2 substrate keeps mean-of-ratios for continuity.",
         "",
         "| model | benchmark | regime | n_q | A_none | A_shared | A_indep | shared−none | "
         "indep−none | dA (C1) | **ΔD pooled** | 95% CI | sig | ΔD mean-of-ratios (legacy) | "
         "95% CI | sig | n_D | \\|ΔA\\|~M (C2) | 95% CI |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        for reg in ("easy", "hard"):
            s = c["strata"][reg]
            if s.get("insufficient"):
                L.append(f"| {c['model']} | {c['benchmark']} | {reg} | {s['n_q']} | "
                         f"*insufficient (< {MIN_Q})* | | | | | | | | | | | | | | |")
                continue
            L.append(
                f"| {c['model']} | {c['benchmark']} | {reg} | {s['n_q']} | "
                f"{s['A_none']:.3f} | {s['A_shared']:.3f} | {s['A_indep']:.3f} | "
                f"{_n(s['gain_shared'])} | {_n(s['gain_indep'])} | {_n(s['dA'])} | "
                f"**{_n(s['dD_pooled'])}** | {_ci(s['dD_pooled_ci'])} | "
                f"{'✓' if s['dD_pooled_sig'] else '–'} | "
                f"{_n(s['dD'])} | {_ci(s['dD_ci'])} | "
                f"{'✓' if s['dD_sig'] else '–'} | {s['n_D']} | {_n(s['c2_slope'])} | "
                f"{_ci(s['c2_ci'])} |")

    L += ["", "## Bin migration — old probe label → recovered label", "",
          "| model | benchmark | old easy → easy/mid/hard | old hard → easy/mid/hard |",
          "|---|---|---|---|"]
    for c in cells:
        e, h = c["migration"]["easy"], c["migration"]["hard"]
        L.append(f"| {c['model']} | {c['benchmark']} | {e['easy']}/{e['mid']}/{e['hard']} | "
                 f"{h['easy']}/{h['mid']}/{h['hard']} |")

    # per-model pooled ΔD / C2, for comparison with the frozen 4×2 substrate
    L += ["", "## Pooled per model (question-weighted mean over recovered strata)", "",
          "| model | cells | n_q | mean A_none | mean dA | **ΔD pooled** | "
          "ΔD mean-of-ratios (legacy) | mean C2 slope |",
          "|---|---|---|---|---|---|---|---|"]
    by_model: dict[str, list[dict]] = {}
    for c in cells:
        for reg in ("easy", "hard"):
            s = c["strata"][reg]
            if not s.get("insufficient"):
                by_model.setdefault(c["model"], []).append(s)
    for model, ss in sorted(by_model.items()):
        w_all = np.array([s["n_q"] for s in ss], float)

        def wm(k):
            """Question-weighted mean over the strata where the statistic is defined
            (a saturated arm leaves D_q — hence ΔD — genuinely unmeasurable)."""
            v = np.array([s[k] for s in ss], float)
            ok = np.isfinite(v)
            return float(np.average(v[ok], weights=w_all[ok])) if ok.any() else float("nan")
        L.append(f"| {model} | {len(ss)} | {int(w_all.sum())} | {wm('A_none'):.3f} | "
                 f"{_n(wm('dA'))} | **{_n(wm('dD_pooled'))}** | {_n(wm('dD'))} | "
                 f"{_n(wm('c2_slope'))} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma list; default = all on disk")
    ap.add_argument("--benchmarks", default="", help="comma list; default = all on disk")
    ap.add_argument("--margin-half", choices=["all", "bin", "eval"], default="all",
                    help="`none` replicates feeding M_q (all = frozen 88-vote definition)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--allow-partial", action="store_true",
                    help="also process (model,benchmark) pairs with only one regime on disk "
                         "(the 80-question pool is then one-sided — interpret with care)")
    ap.add_argument("--tag", default="tier1_split_sample_strata")
    args = ap.parse_args(argv)

    models = [s for s in args.models.split(",") if s]
    benchmarks = [s for s in args.benchmarks.split(",") if s]
    found = discover(models, benchmarks)
    if not found:
        raise SystemExit("no exp6c_raw_*_scaleup_* dumps matched")

    cells, skipped = [], []
    for (model, bench, N), paths in sorted(found.items()):
        if len(paths) < 2 and not args.allow_partial:
            skipped.append(f"{model}/{bench} (only {'+'.join(sorted(paths))})")
            continue
        print(f"[{model}/{bench}] pooling {'+'.join(sorted(paths))} ...", flush=True)
        cells.append(process_cell(model, bench, N, paths, args.margin_half,
                                  args.n_boot, args.seed))
        c = cells[-1]
        for reg in ("easy", "hard"):
            s = c["strata"][reg]
            if s.get("insufficient"):
                print(f"    {reg:4s} n={s['n_q']:2d}  INSUFFICIENT (< {MIN_Q})")
            else:
                print(f"    {reg:4s} n={s['n_q']:2d}  A_none={s['A_none']:.3f} "
                      f"A_shared={s['A_shared']:.3f} A_indep={s['A_indep']:.3f}  "
                      f"dA={_n(s['dA'])}  ΔD_meanratio={_n(s['dD'])}"
                      f"{'*' if s['dD_sig'] else ' '}(n_D={s['n_D']})  "
                      f"ΔD_pooled={_n(s['dD_pooled'])} {_ci(s['dD_pooled_ci'])}"
                      f"{'*' if s['dD_pooled_sig'] else ' '}  C2={_n(s['c2_slope'])}")
    if skipped:
        print(f"\nskipped (one regime only, no --allow-partial): {', '.join(skipped)}")
    if not cells:
        raise SystemExit("nothing to analyze")

    TABLES.mkdir(parents=True, exist_ok=True)
    js = TABLES / f"{args.tag}.json"
    md = TABLES / f"{args.tag}.md"
    js.write_text(json.dumps({
        "recipe": "split-sample: bin on none reps 0..3 (p̂, frozen .80/.30 cuts), "
                  "evaluate on reps 4..7",
        "constants": {"EASY_HI": EASY_HI, "HARD_LO": HARD_LO, "R_BIN": R_BIN, "MIN_Q": MIN_Q},
        "primary_dD_estimator": "dD_pooled",
        "estimator_note": "dD_pooled (Σ num / Σ den over all questions) is primary; dD "
                          "(mean of per-question ratios) is the legacy frozen estimator, "
                          "undefined for saturated questions and therefore computed on an "
                          "error-variance-selected subset in the easy strata",
        "margin_half": args.margin_half, "n_boot": args.n_boot, "seed": args.seed,
        "skipped": skipped, "cells": cells,
    }, ensure_ascii=False, indent=2))
    md.write_text(markdown(cells, args.margin_half))
    print(f"\nwrote {js}\nwrote {md}")


if __name__ == "__main__":
    main()
