"""Calibrate the estimability rule for the pooled dependence ratio. BLIND to the C3 results.

WHY. `MIN_NEFF = 15` currently has no justification -- it was picked by analogy to `MIN_Q = 15`.
It is load-bearing: on R=8 probe cells it admits 10/18 strata, on the R=4 recovered strata only
3/19. A threshold that swings the headline by that much must be derived from the estimator's
sampling behaviour, never from the observed table. This script therefore simulates data with a
KNOWN true ΔD and measures how the estimator behaves, with no reference to any real stratum.

FROZEN PRE-SPECIFICATION (written before the first run; do not edit after seeing output).

  Estimand. C3's estimand is the PAIRED contrast
      ΔD = D_shared − D_indep,
  so that is what is calibrated. Per-arm D calibration is reported too, but the rule is set on
  ΔD.

  Generative model = Theorem 4's M₁ operator with replicate-level resampling. For question q,
  replicate r, agent i, with per-question error rate p_q and latent correlation ρ:
      shared arm:  L_{i,r} = √ρ·Z_{0,r} + √(1−ρ)·ε_{i,r}     (Z_{0,r} common to all agents)
      indep arm:   L_{i,r} = √ρ·Z_{i,r} + √(1−ρ)·ε_{i,r}     (own shock per agent)
      error E_{i,r} = 1[L_{i,r} ≤ Φ⁻¹(p_q)]
  Marginal error rate is p_q in both arms by construction, so the arms are matched on
  competence and differ only in evidence commonality -- the real contrast.

  True values. The pairwise ERROR correlation under the shared arm is the tetrachoric image
      τ_p(ρ) = [Φ₂(a,a;ρ) − p²] / [p(1−p)],  a = Φ⁻¹(p),
  (NOT ρ itself -- thresholding attenuates; see the 2026-10-01 correction in
  thm4_mechanism_non_equivalence.md §2). The indep arm has true correlation 0. Hence
      ΔD_true = τ_p(ρ).
  A self-check at large R verifies the estimator recovers τ_p(ρ), so the target is validated
  rather than assumed.

  Acceptance tolerances, fixed now:
    T1 COVERAGE     Wilson 95% lower bound on P(ΔD_true ∈ CI₉₅) ≥ 0.90.
                    Nominal is 0.95; 0.90 is the declared tolerance because Monte-Carlo
                    coverage is itself uncertain and demanding ≥0.95 point-wise would select
                    on simulation noise.
    T2 NULL RATE    at ρ = 0: Wilson 95% UPPER bound on P(CI excludes 0) ≤ 0.10.
                    Coverage alone is insufficient -- C3 is a one-sided existence claim, so a
                    bin with good coverage but inflated false-positive rate is unusable.
    T3 DEGENERACY   P(bootstrap returns no usable CI) ≤ 0.10.
    T4 BIAS         |mean(ΔD̂) − ΔD_true| ≤ 0.05, the paper's own materiality band.

  Decision rule, fixed now: report the acceptance status of every (n_eff bin × condition) cell.
  Then, and only then, read off the rule. If a single n_eff cut satisfies T1-T4 across all
  simulated p and ρ, that cut is the threshold. **If reliability depends jointly on n_eff and
  on error prevalence or true ΔD, the correct output is a multi-condition estimability rule, or
  a statement that cell-level significance cannot be certified and ΔD must be reported with
  uncertainty instead.** This script is not required to produce a single number, and must not
  be re-run with a different grid in order to find one.

  Blinding. Nothing here reads results/tables/. The (p, ρ) grid spans the design's plausible
  range on both sides of the observed values and was not tuned to any measured ΔD.

Usage:
    python calibrate_estimability.py --quick       # smoke test the machinery
    python calibrate_estimability.py              # full grid (CPU-bound, minutes)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, norm

from dependence_estimators import ESTIMATOR_VERSION, boot_pooled_ratio, n_eff, pair_corr_terms

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"

N_AGENTS = 11
R_EVAL = 4          # recovered strata evaluate on held-out replicates 4-7
# Grid spans both saturation ends and the middle the design never samples. Not tuned to data.
P_GRID = (0.05, 0.15, 0.30, 0.50, 0.70, 0.90)
RHO_GRID = (0.0, 0.10, 0.20, 0.40, 0.60, 0.80)
NQ_GRID = (20, 30, 40)
P_HETERO = 0.10     # per-question error rate jitter (sd on the logit), mimics real strata
NEFF_BINS = (0, 3, 5, 8, 10, 12, 15, 20, 25, 1e9)

T1_COVERAGE_MIN = 0.90
T2_NULL_MAX = 0.10
T3_DEGEN_MAX = 0.10
T4_BIAS_MAX = 0.05


def tetrachoric(p: float, rho: float) -> float:
    """True pairwise error correlation induced by latent correlation rho at error rate p."""
    if rho <= 0:
        return 0.0
    a = norm.ppf(p)
    phi2 = float(multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([a, a]))
    return (phi2 - p * p) / (p * (1 - p))


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval — used so Monte-Carlo uncertainty enters the decision."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def draw_arm(rng, p_q: np.ndarray, rho: float, R: int, shared: bool) -> list:
    """(num, den) per question for one arm. p_q is per-question error rate, length n_q."""
    n_q = p_q.size
    a = norm.ppf(p_q)[:, None, None]                       # (n_q,1,1)
    eps = rng.standard_normal((n_q, R, N_AGENTS))
    if shared:
        z = rng.standard_normal((n_q, R, 1))               # one shock per (question, replicate)
    else:
        z = rng.standard_normal((n_q, R, N_AGENTS))        # own shock per agent
    L = np.sqrt(rho) * z + np.sqrt(1 - rho) * eps
    E = (L <= a).astype(float)                             # (n_q,R,N) 1 = wrong
    return [pair_corr_terms(E[q]) for q in range(n_q)]


def one_stratum(rng, p_mean: float, rho: float, n_q: int, R: int, n_boot: int):
    logit = np.log(p_mean / (1 - p_mean)) + rng.normal(0, P_HETERO, n_q)
    p_q = np.clip(1 / (1 + np.exp(-logit)), 1e-4, 1 - 1e-4)
    sh = draw_arm(rng, p_q, rho, R, shared=True)
    ind = draw_arm(rng, p_q, rho, R, shared=False)
    lo, hi, pt = boot_pooled_ratio([x[0] for x in sh], [x[1] for x in sh],
                                   [x[0] for x in ind], [x[1] for x in ind], rng, n_boot)
    ne = n_eff([x[1] + y[1] for x, y in zip(sh, ind)])
    return ne, pt, lo, hi


def self_check(rng) -> dict:
    """Does the estimator recover tau_p(rho) at large R? Validates the target, not the rule."""
    out = []
    for p, rho in ((0.30, 0.40), (0.50, 0.60), (0.15, 0.20)):
        truth = tetrachoric(p, rho)
        sh = draw_arm(rng, np.full(400, p), rho, 400, shared=True)
        est = sum(x[0] for x in sh) / sum(x[1] for x in sh)
        out.append({"p": p, "rho": rho, "tau_true": truth, "pooled_est": est,
                    "abs_err": abs(est - truth)})
        print(f"  p={p:.2f} rho={rho:.2f}  tau_true={truth:+.4f}  estimated={est:+.4f}  "
              f"|err|={abs(est - truth):.4f}")
    return {"checks": out, "max_abs_err": max(c["abs_err"] for c in out)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=400, help="simulated strata per grid point")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7717)
    ap.add_argument("--R", type=int, default=R_EVAL,
                    help="replicates per question. 4 = recovered strata (split-sample), "
                         "8 = full run. Reported separately: finite-R bias does not cancel "
                         "in the num/den ratio, so R is a calibration axis, not a detail.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None, help="defaults to calibrate_estimability_R<R>.json")
    args = ap.parse_args()
    sims = 40 if args.quick else args.sims
    p_grid = (0.15, 0.50) if args.quick else P_GRID
    rho_grid = (0.0, 0.40) if args.quick else RHO_GRID
    nq_grid = (40,) if args.quick else NQ_GRID
    out_json = args.json or str(TABLES / f"calibrate_estimability_R{args.R}.json")
    rng = np.random.default_rng(args.seed)

    print(f"estimator: {ESTIMATOR_VERSION}")
    print(f"BLIND calibration — reads no results/tables/ input\n")
    print("self-check: does the pooled ratio recover the tetrachoric target at large R?")
    chk = self_check(rng)

    print(f"\nsimulating {sims} strata per (p, rho, n_q) over "
          f"{len(p_grid)}x{len(rho_grid)}x{len(nq_grid)} grid, N={N_AGENTS}, R={args.R}")
    recs = []
    for p in p_grid:
        for rho in rho_grid:
            truth = tetrachoric(p, rho)
            for n_q in nq_grid:
                for _ in range(sims):
                    ne, pt, lo, hi = one_stratum(rng, p, rho, n_q, args.R, args.n_boot)
                    ok = np.isfinite(lo) and np.isfinite(hi)
                    recs.append({
                        "p": p, "rho": rho, "n_q": n_q, "truth": truth, "n_eff": ne,
                        "point": pt, "degenerate": not ok,
                        "covered": bool(ok and lo <= truth <= hi),
                        "excl0": bool(ok and (lo > 0 or hi < 0)),
                    })
            print(f"  p={p:.2f} rho={rho:.2f} tau_true={truth:+.4f} done")

    # --- acceptance table by n_eff bin, split null vs non-null -----------------------------
    print(f"\n=== ACCEPTANCE BY n_eff BIN (tolerances frozen in the docstring) ===")
    print(f"T1 coverage Wilson-lo>={T1_COVERAGE_MIN}  T2 null-rate Wilson-hi<={T2_NULL_MAX}  "
          f"T3 degeneracy<={T3_DEGEN_MAX}  T4 |bias|<={T4_BIAS_MAX}")
    hdr = (f"{'n_eff bin':>12} {'n':>6} | {'cover':>6} {'Wil-lo':>7} {'T1':>3} | "
           f"{'degen':>6} {'T3':>3} | {'bias':>7} {'T4':>3} | {'nullN':>6} {'rate':>6} "
           f"{'Wil-hi':>7} {'T2':>3} | {'PASS':>5}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for lo_b, hi_b in zip(NEFF_BINS[:-1], NEFF_BINS[1:]):
        g = [r for r in recs if lo_b <= r["n_eff"] < hi_b]
        if not g:
            continue
        nn = [r for r in g if r["rho"] > 0]
        nul = [r for r in g if r["rho"] == 0]
        usable = [r for r in nn if not r["degenerate"]]
        cov = sum(r["covered"] for r in usable)
        cov_rate = cov / len(usable) if usable else float("nan")
        cl, _ = wilson(cov, len(usable)) if usable else (float("nan"),) * 2
        degen = sum(r["degenerate"] for r in g) / len(g)
        bias = (float(np.mean([r["point"] - r["truth"] for r in usable]))
                if usable else float("nan"))
        nu = [r for r in nul if not r["degenerate"]]
        fp = sum(r["excl0"] for r in nu)
        fp_rate = fp / len(nu) if nu else float("nan")
        _, fh = wilson(fp, len(nu)) if nu else (float("nan"),) * 2
        t1 = bool(cl == cl and cl >= T1_COVERAGE_MIN)
        t2 = bool(fh == fh and fh <= T2_NULL_MAX)
        t3 = bool(degen <= T3_DEGEN_MAX)
        t4 = bool(bias == bias and abs(bias) <= T4_BIAS_MAX)
        allp = bool(t1 and t2 and t3 and t4)
        lbl = f"{lo_b:g}-{'inf' if hi_b > 1e8 else f'{hi_b:g}'}"
        rows.append({"bin": lbl, "lo": lo_b, "n": len(g), "n_usable": len(usable),
                     "coverage": float(cov_rate), "coverage_wilson_lo": cl, "T1": t1,
                     "degeneracy": float(degen), "T3": t3, "bias": bias, "T4": t4,
                     "n_null": len(nu), "null_rate": float(fp_rate), "null_wilson_hi": fh, "T2": t2,
                     "pass": allp})
        print(f"{lbl:>12} {len(g):6d} | {cov_rate:6.3f} {cl:7.3f} {'ok' if t1 else 'NO':>3} | "
              f"{degen:6.3f} {'ok' if t3 else 'NO':>3} | {bias:+7.4f} {'ok' if t4 else 'NO':>3} | "
              f"{len(nu):6d} {fp_rate:6.3f} {fh:7.3f} {'ok' if t2 else 'NO':>3} | "
              f"{'PASS' if allp else '----':>5}")

    # --- does a single cut work, or is the rule conditional? -------------------------------
    print(f"\n=== IS A SINGLE n_eff CUT SUFFICIENT? ===")
    print("Per (n_eff bin x error-rate band): coverage Wilson-lo. A single cut is only")
    print("defensible if every condition passes above it.")
    bands = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0)]
    print(f"{'n_eff bin':>12} " + " ".join(f"{f'p in {a}-{b}':>13}" for a, b in bands))
    print("-" * (13 + 14 * len(bands)))
    cond = []
    for lo_b, hi_b in zip(NEFF_BINS[:-1], NEFF_BINS[1:]):
        cells, line = [], []
        for a, b in bands:
            g = [r for r in recs if lo_b <= r["n_eff"] < hi_b and a <= r["p"] < b
                 and r["rho"] > 0 and not r["degenerate"]]
            if len(g) < 30:
                line.append(f"{'(n<30)':>13}"); cells.append(None); continue
            k = sum(r["covered"] for r in g)
            cl, _ = wilson(k, len(g))
            cells.append({"p_lo": a, "p_hi": b, "n": len(g), "cov_lo": cl,
                          "pass": bool(cl >= T1_COVERAGE_MIN)})
            line.append(f"{cl:9.3f}{'ok' if cl >= T1_COVERAGE_MIN else ' NO':>4}")
        lbl = f"{lo_b:g}-{'inf' if hi_b > 1e8 else f'{hi_b:g}'}"
        cond.append({"bin": lbl, "lo": lo_b, "cells": cells})
        print(f"{lbl:>12} " + " ".join(line))

    passing = [r["lo"] for r in rows if r["pass"]]
    print(f"\n=== READ-OFF (rule frozen before this line was ever printed) ===")
    if not passing:
        print("  NO n_eff bin satisfies T1-T4. Per the frozen decision rule the conclusion is")
        print("  that cell-level significance cannot be certified for this design; report ΔD")
        print("  with uncertainty instead of a per-cell significance verdict.")
    else:
        print(f"  bins satisfying T1-T4: {sorted(passing)}")
        print(f"  smallest qualifying n_eff cut = {min(passing):g}")
        # (conditional check is read from the table above; no inline filter needed)

        print("  CHECK the conditional table above before adopting it as a single cut: if any")
        print("  error-rate band fails at or above that cut, the rule is conditional, not a cut.")

    Path(out_json).write_text(json.dumps({
        "role": "blind calibration of the estimability rule; reads no measured results",
        "estimator_version": ESTIMATOR_VERSION,
        "prespec": {"T1_coverage_wilson_lo_min": T1_COVERAGE_MIN,
                    "T2_null_wilson_hi_max": T2_NULL_MAX,
                    "T3_degeneracy_max": T3_DEGEN_MAX, "T4_abs_bias_max": T4_BIAS_MAX},
        "design": {"N": N_AGENTS, "R": args.R, "p_grid": list(p_grid),
                   "rho_grid": list(rho_grid), "n_q_grid": list(nq_grid),
                   "p_hetero_logit_sd": P_HETERO, "sims_per_point": sims,
                   "n_boot": args.n_boot, "seed": args.seed},
        "self_check": chk, "by_neff_bin": rows, "conditional_coverage": cond,
        "bins_passing": sorted(passing),
    }, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
