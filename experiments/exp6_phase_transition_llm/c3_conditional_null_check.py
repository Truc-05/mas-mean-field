"""Offline-only, post-hoc (2026-10-13) robustness check on c3_global_directional_test.py.
Does NOT modify or re-run the frozen test (c3_global_directional_test.json stands as the
reported result). This checks a specific concern raised in manuscript review: the frozen
null scores an undefined simulated draw as "not positive" (see simulate_null_S), while the
K=66 real strata entering S_obs were pre-filtered to only those with a DEFINED dD_pooled
(see load_design_and_signs's `if ... dD is None: skipped...; continue`). That is an
apples-to-oranges comparison -- the null's denominator always stays at K=66 per trial even
when many per-stratum draws are undefined that trial, while the real data's denominator
already excludes the undefined ones by construction.

This script re-simulates the null, but for each trial computes the proportion
S_trial / K_effective_trial, where K_effective_trial excludes that trial's own undefined
draws (mirroring exactly how the real S_obs/K=63/66 was constructed), and compares the
observed proportion (not raw count) against this corrected null distribution.
"""
import sys
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "exp6_phase_transition_llm"
TABLES = ROOT / "results" / "tables"
sys.path.insert(0, str(EXP))

from c3_global_directional_test import load_design_and_signs, STRATA  # noqa: E402
from calibrate_estimability import one_stratum  # noqa: E402

rows, skipped = load_design_and_signs(Path(STRATA))
K = len(rows)
S_obs = sum(r["sign_positive"] for r in rows)
obs_frac = S_obs / K
print(f"K={K}  S_obs={S_obs}  observed fraction={obs_frac:.4f}")

rng = np.random.default_rng(31337)
R_SPLIT = 4
n_boot = 50
n_sim = 2000

frac_null = np.zeros(n_sim)
k_eff_record = np.zeros(n_sim, dtype=int)
for m in range(n_sim):
    s, k_eff = 0, 0
    for r in rows:
        _, pt, _, _ = one_stratum(rng, r["err_rate"], 0.0, r["n_q"], R_SPLIT, n_boot)
        if pt == pt:  # defined this trial -- include in the effective denominator
            k_eff += 1
            if pt > 0:
                s += 1
        # undefined this trial: excluded entirely, matching how real undefined strata
        # were excluded from K=66 rather than scored against it
    frac_null[m] = (s / k_eff) if k_eff > 0 else np.nan
    k_eff_record[m] = k_eff

valid = ~np.isnan(frac_null)
p_conditional = (1 + int((frac_null[valid] >= obs_frac).sum())) / (1 + valid.sum())
print(f"conditional null (excluding undefined draws from each trial's denominator):")
print(f"  mean K_effective per trial: {k_eff_record.mean():.1f} / {K}")
print(f"  mean fraction positive: {np.nanmean(frac_null):.4f}  "
      f"sd: {np.nanstd(frac_null):.4f}")
print(f"  one-sided p (fraction-based, conditional on definedness) = {p_conditional:.5f}")

out = dict(role="conditional-null robustness check for C3 (does not replace the frozen test)",
           K=K, S_obs=S_obs, obs_fraction=obs_frac,
           conditional_null=dict(mean_k_effective=float(k_eff_record.mean()),
                                  mean_fraction=float(np.nanmean(frac_null)),
                                  sd_fraction=float(np.nanstd(frac_null)),
                                  n_sim=n_sim, p_one_sided=p_conditional))
dst = TABLES / "c3_conditional_null_check.json"
dst.write_text(json.dumps(out, indent=2))
print("wrote", dst)
