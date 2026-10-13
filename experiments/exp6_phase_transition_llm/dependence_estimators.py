"""Canonical dependence estimators. ONE implementation, imported everywhere.

Created 2026-10-03 by consolidation. The pooled-ratio estimator and the effective-sample-size
gate had been written twice -- `tier1_split_sample_strata._pair_corr_terms`/`boot_pooled_ratio`
(ratified primary, 2026-09-30) and a second copy in `exp6c_D_estimators` (audit, 2026-10-03).
Two implementations of the same *primary* estimator is how two numbers quietly diverge six
months later, so both now import from here and neither keeps a local copy.

Division of responsibility, fixed:
  * `tier1_split_sample_strata.py` is the PRIMARY analysis and the ONLY place that decides
    estimability by gating on `n_eff`.
  * `exp6c_D_estimators.py` is an AUDIT-ONLY wrapper: legacy per-question ratio vs the
    canonical pooled ratio, side by side. It makes no estimability decision and holds no
    estimator of its own.

The quantities. For one question, with `E` the (R, N) binary error matrix (1 = agent wrong):

    num = Σ_{i≠j} Cov_ij     = Var(Σ_i E_i) − Σ_i Var(E_i)
    den = Σ_{i≠j} sd_i sd_j  = (Σ_i sd_i)² − Σ_i sd_i²

`exp6c_shared_context.pooled_pair_corr` returns `num/den` per question and NaN when
`den <= 1e-12`. Keeping the two terms apart lets a stratum pool them BEFORE dividing:

    D_pooled = (Σ_q num_q) / (Σ_q den_q)
    n_eff    = (Σ_q w_q)² / (Σ_q w_q²),     w_q = den_q

Why pooling, not averaging per-question ratios: a saturated question (every agent constant
across replicates) carries num = den = 0 and drops out arithmetically instead of becoming a
NaN that must be discarded -- and discarding it selects on error variance, the very quantity D
measures. Why `n_eff`: pooling fixes undefinedness but not thin information. A stratum can show
n_q = 40 while one question carries the whole denominator; then the "pooled" D *is* that one
question and the bootstrap collapses to zero width. `n_eff` is what `n_q` only pretends to be.
"""
from __future__ import annotations

import numpy as np

# Bump when the arithmetic changes, and stamp it into every output file so a table can always
# be traced to the estimator that produced it.
ESTIMATOR_VERSION = "pooled-ratio/n_eff v1 (2026-10-03)"

DEGENERATE_DEN = 0.3     # below this the per-question ratio pins at ±1; audit threshold only
EPS_DEN = 1e-12


def errors_from_states(states) -> np.ndarray:
    """(R, N, K) canonical opinion tensor (truth at index 0) -> (R, N) binary, 1 = wrong."""
    return (np.asarray(states, dtype=float).argmax(axis=2) != 0).astype(float)


def pair_corr_terms(E: np.ndarray) -> tuple[float, float]:
    """(num, den) of the frozen `pooled_pair_corr` ratio for one question.

    Moved verbatim from tier1_split_sample_strata._pair_corr_terms so the ratified arithmetic
    is preserved exactly; callers there still assert `num/den` rebuilds the frozen function.
    """
    E = np.asarray(E, dtype=float)
    if E.shape[0] < 2:
        return 0.0, 0.0
    var_i = E.var(axis=0)
    sd = np.sqrt(var_i)
    num = float(E.sum(axis=1).var() - var_i.sum())
    den = float(sd.sum() ** 2 - (sd ** 2).sum())
    return num, den


def terms_from_states(states) -> tuple[float, float]:
    return pair_corr_terms(errors_from_states(states))


def legacy_D(num: float, den: float) -> float:
    """Per-question ratio: the frozen estimator. NaN when the arm saturates."""
    return num / den if den > EPS_DEN else float("nan")


def pooled_D(pairs) -> float:
    """Σnum / Σden over questions. Defined whenever any question varies."""
    sn = sum(n for n, _ in pairs)
    sd = sum(d for _, d in pairs)
    return sn / sd if sd > EPS_DEN else float("nan")


def n_eff(weights) -> float:
    """Kish effective sample size (Σw)²/Σw² of the pooled ratio; w = den."""
    w = np.asarray(weights, dtype=float)
    s, s2 = w.sum(), (w ** 2).sum()
    return float(s * s / s2) if s2 > 0 else 0.0


def boot_pooled_ratio(num_s, den_s, num_i, den_i, rng, n_boot: int
                      ) -> tuple[float, float, float]:
    """ΔD = pooled_D(shared) − pooled_D(indep), question bootstrap preserving the pairing.

    Returns (lo, hi, point) to match the call signature already in use by the primary script.
    A zero-width interval is returned as (nan, nan, point): every resample reproduced the same
    ratio, which means one question holds all the weight. Reported as precision it is the
    opposite of precision. Before this guard, qwen/truthfulqa `easy` printed
    ΔD_pooled = −0.133 [−0.133, −0.133]* off n_D = 0 -- a SIGNIFICANT NEGATIVE ΔD, readable as
    a counterexample to C3.
    """
    num_s = np.asarray(num_s, float); den_s = np.asarray(den_s, float)
    num_i = np.asarray(num_i, float); den_i = np.asarray(den_i, float)

    def stat(I):
        bs, bi = den_s[I].sum(-1), den_i[I].sum(-1)
        ds = num_s[I].sum(-1) / np.where(bs > EPS_DEN, bs, np.nan)
        di = num_i[I].sum(-1) / np.where(bi > EPS_DEN, bi, np.nan)
        return ds - di

    n = len(num_s)
    if n < 2:
        return (float("nan"),) * 3
    pt = float(stat(np.arange(n)[None, :])[0])
    draws = stat(rng.integers(0, n, (n_boot, n)))
    draws = draws[np.isfinite(draws)]
    if draws.size < 2:
        return (float("nan"), float("nan"), pt)
    lo, hi = float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
    if hi - lo < 1e-9:
        return (float("nan"), float("nan"), pt)
    return lo, hi, pt
