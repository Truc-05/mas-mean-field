"""Offline-only, post-hoc (2026-10-13): test whether the observed
p_break/p_rescue pattern in the matched-dependence comparison is distinguishable
from what two UNCORRELATED arms at the same marginal accuracies would produce.
Under independence, p_break = A_shared*(1-A_copy) and p_rescue =
(1-A_shared)*A_copy per question; since A_shared~=A_copy empirically, these
are close to each other BY CONSTRUCTION under independence, regardless of any
real mechanism-level correlation. This computes both the observed (actual
paired) and independence-implied p_break/p_rescue per pair, pooled, to check
whether the observed values exceed what chance alone predicts.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "exp6_phase_transition_llm"
TABLES = ROOT / "results" / "tables"
sys.path.insert(0, str(EXP))

from tier3_matched_d import shared_rows_tier1, copy_rows_tier3, MODELS, BENCHMARKS  # noqa: E402

results = []
for model in MODELS:
    for bench in BENCHMARKS:
        sh = shared_rows_tier1(model, bench)
        cp = copy_rows_tier3(model, bench)
        if len(sh) < 5 or len(cp) < 5:
            continue
        A_shared = np.mean([r["A0"] for r in sh])  # shared arm's own accuracy (indep baseline per row def)
        # shared_rows_tier1's A0 is actually indep-arm accuracy (Ai); the shared arm's accuracy is Ac
        # recompute both explicitly from the correctness fields captured in p_break/p_rescue logic:
        # p_break = P(indep correct, shared wrong), p_rescue = P(shared correct, indep wrong)
        # => A_indep = mean(ci), A_shared_arm = mean(cs); reconstruct from row-level p_break/p_rescue + dA
        # dA = mean(cs) - mean(ci); so mean(cs) = A_indep + dA
        A_indep = np.mean([r["A0"] for r in sh])
        dA_sh = np.mean([r["dA"] for r in sh])
        A_shared_arm = A_indep + dA_sh
        p_break_obs_sh = np.mean([r["p_break"] for r in sh])
        p_rescue_obs_sh = np.mean([r["p_rescue"] for r in sh])
        p_break_indep_sh = A_indep * (1 - A_shared_arm)
        p_rescue_indep_sh = (1 - A_indep) * A_shared_arm

        A0_cp = np.mean([r["A0"] for r in cp])
        dA_cp = np.mean([r["dA"] for r in cp])
        A_copy_arm = A0_cp + dA_cp
        p_break_obs_cp = np.mean([r["p_break"] for r in cp])
        p_rescue_obs_cp = np.mean([r["p_rescue"] for r in cp])
        p_break_indep_cp = A0_cp * (1 - A_copy_arm)
        p_rescue_indep_cp = (1 - A0_cp) * A_copy_arm

        results.append(dict(
            model=model, bench=bench,
            shared=dict(A_ref=A_indep, A_arm=A_shared_arm,
                        p_break_obs=p_break_obs_sh, p_break_indep=p_break_indep_sh,
                        p_rescue_obs=p_rescue_obs_sh, p_rescue_indep=p_rescue_indep_sh),
            copy=dict(A_ref=A0_cp, A_arm=A_copy_arm,
                      p_break_obs=p_break_obs_cp, p_break_indep=p_break_indep_cp,
                      p_rescue_obs=p_rescue_obs_cp, p_rescue_indep=p_rescue_indep_cp),
        ))
        print(f"{model:8s}/{bench:12s} "
              f"shared: p_break {p_break_obs_sh:.3f} (indep-pred {p_break_indep_sh:.3f})  "
              f"p_rescue {p_rescue_obs_sh:.3f} (indep-pred {p_rescue_indep_sh:.3f})  |  "
              f"copy: p_break {p_break_obs_cp:.3f} (indep-pred {p_break_indep_cp:.3f})  "
              f"p_rescue {p_rescue_obs_cp:.3f} (indep-pred {p_rescue_indep_cp:.3f})")

# macro: is the GAP (copy-shared) in p_break/p_rescue close to what the gap in
# independence-implied values would be?
macro_gap_break_obs = np.mean([r["copy"]["p_break_obs"] - r["shared"]["p_break_obs"] for r in results])
macro_gap_rescue_obs = np.mean([r["copy"]["p_rescue_obs"] - r["shared"]["p_rescue_obs"] for r in results])
macro_gap_break_indep = np.mean([r["copy"]["p_break_indep"] - r["shared"]["p_break_indep"] for r in results])
macro_gap_rescue_indep = np.mean([r["copy"]["p_rescue_indep"] - r["shared"]["p_rescue_indep"] for r in results])
print()
print(f"macro p_break gap:  observed={macro_gap_break_obs:.4f}   independence-implied={macro_gap_break_indep:.4f}")
print(f"macro p_rescue gap: observed={macro_gap_rescue_obs:.4f}   independence-implied={macro_gap_rescue_indep:.4f}")

out = dict(role="independence-null check for the break/rescue 'cancellation' pattern, 2026-10-13 post-hoc",
           note="p_break_indep/p_rescue_indep are what two arms with NO real correctness correlation "
                "would produce at the same marginal accuracies (A_ref*(1-A_arm), (1-A_ref)*A_arm); "
                "comparison to p_*_obs tests whether the observed pattern exceeds this chance baseline",
           pairs=results,
           macro=dict(gap_break_obs=macro_gap_break_obs, gap_break_indep=macro_gap_break_indep,
                      gap_rescue_obs=macro_gap_rescue_obs, gap_rescue_indep=macro_gap_rescue_indep))
dst = TABLES / "tier3_independence_check.json"
dst.write_text(json.dumps(out, indent=2))
print("wrote", dst)
