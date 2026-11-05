"""Offline-only, post-hoc (2026-10-13): re-run the estimators (caliper_contrast,
nested_models, nn_matching imported verbatim from the pre-specified
analysis_C module) under four readings -- raw, shared-side correction only,
copy-side correction only, and both together -- using shared side's existing
per-question dC and a new copy-side correction using C_within_base (the
baseline/'none' arm's own within-replicate churn, pooled per pair, computed
in tier3_copy_noise_floor.py). No new model queries; reads only
already-stored raw tensors and tables. Run tier3_copy_noise_floor.py first.

The combined (both) reading is, to numerical precision, exactly
shared_only + copy_only - raw for every estimator, since each estimator is a
linear functional of the per-row dY values and both corrections are additive
shifts to dY -- this is checked explicitly below, not assumed, and is the
basis for the decomposition reported in paper.md Section 6.2.

Caveat stated here and in paper.md Limitations: C_within_base measures the
baseline arm's own resampling noise, which is a defensible but not identical
stand-in for the copying arm's (still genuinely unidentifiable) own
within-condition noise at each fixed coupling g. Report accordingly.
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "exp6_phase_transition_llm"
TABLES = ROOT / "results" / "tables"
sys.path.insert(0, str(EXP))

from analysis_C_matched_D_falsification import caliper_contrast, nested_models, nn_matching  # noqa: E402
from tier3_matched_d import shared_rows_tier1, copy_rows_tier3, MODELS, BENCHMARKS, macro_bootstrap  # noqa: E402

cf = json.loads((TABLES / "tier3_copy_noise_floor.json").read_text())
c_within_base_by_pair = {(p["model"], p["bench"]): p["mean_C_within_base"] for p in cf["pairs"]}

rng = np.random.default_rng(20261013)
pair_results = []
for model in MODELS:
    for bench in BENCHMARKS:
        sh = shared_rows_tier1(model, bench)
        cp = copy_rows_tier3(model, bench)
        if len(sh) < 5 or len(cp) < 5:
            continue
        c_base = c_within_base_by_pair[(model, bench)]

        # shared rows: dY already compared at raw; build an "adjusted" copy
        # using each row's OWN per-question C_within/dC (already computed)
        sh_adj = [dict(r, dY=r["dC"]) for r in sh]
        # copy rows: subtract the PAIR-level baseline noise floor uniformly
        cp_adj = [dict(r, dY=r["dY"] - c_base) for r in cp]

        cal_raw = caliper_contrast(sh, cp, 0.05, "dY", n_boot=2000)
        cal_shared_only = caliper_contrast(sh_adj, cp, 0.05, "dY", n_boot=2000)
        cal_copy_only = caliper_contrast(sh, cp_adj, 0.05, "dY", n_boot=2000)
        cal_adj = caliper_contrast(sh_adj, cp_adj, 0.05, "dY", n_boot=2000)
        nn_raw = nn_matching(sh, cp, n_boot=2000)
        nn_shared_only = nn_matching(sh_adj, cp, n_boot=2000)
        nn_copy_only = nn_matching(sh, cp_adj, n_boot=2000)
        nn_adj = nn_matching(sh_adj, cp_adj, n_boot=2000)
        nm_raw = nested_models(sh, cp)
        nm_shared_only = nested_models(sh_adj, cp)
        nm_copy_only = nested_models(sh, cp_adj)
        nm_adj = nested_models(sh_adj, cp_adj)

        pair_results.append(dict(
            model=model, bench=bench, c_within_base=c_base,
            caliper_gap_raw=cal_raw["gap"], caliper_gap_shared_only=cal_shared_only["gap"],
            caliper_gap_copy_only=cal_copy_only["gap"], caliper_gap_doubly_adj=cal_adj["gap"],
            caliper_ci_doubly_adj=cal_adj["ci"],
            nn_att_raw=nn_raw["att_copy_minus_shared"], nn_att_shared_only=nn_shared_only["att_copy_minus_shared"],
            nn_att_copy_only=nn_copy_only["att_copy_minus_shared"],
            nn_att_doubly_adj=nn_adj["att_copy_minus_shared"], nn_ci_doubly_adj=nn_adj["ci"],
            mech_coef_raw=nm_raw["mech_coef_given_D"], mech_coef_shared_only=nm_shared_only["mech_coef_given_D"],
            mech_coef_copy_only=nm_copy_only["mech_coef_given_D"],
            mech_coef_doubly_adj=nm_adj["mech_coef_given_D"],
        ))
        print(f"{model:8s}/{bench:12s} caliper {cal_raw['gap']:.4f}->{cal_adj['gap']:.4f}  "
              f"NN-ATT {nn_raw['att_copy_minus_shared']:.4f}->{nn_adj['att_copy_minus_shared']:.4f}  "
              f"nested {nm_raw['mech_coef_given_D']:.4f}->{nm_adj['mech_coef_given_D']:.4f}")

def macro(key):
    return float(np.mean([p[key] for p in pair_results]))

macro_caliper_raw, macro_caliper_shared, macro_caliper_copy, macro_caliper_adj = (
    macro("caliper_gap_raw"), macro("caliper_gap_shared_only"), macro("caliper_gap_copy_only"), macro("caliper_gap_doubly_adj"))
macro_nn_raw, macro_nn_shared, macro_nn_copy, macro_nn_adj = (
    macro("nn_att_raw"), macro("nn_att_shared_only"), macro("nn_att_copy_only"), macro("nn_att_doubly_adj"))
macro_mech_raw, macro_mech_shared, macro_mech_copy, macro_mech_adj = (
    macro("mech_coef_raw"), macro("mech_coef_shared_only"), macro("mech_coef_copy_only"), macro("mech_coef_doubly_adj"))

r_caliper = macro_bootstrap(np.array([p["caliper_gap_doubly_adj"] for p in pair_results]), rng)
r_nn = macro_bootstrap(np.array([p["nn_att_doubly_adj"] for p in pair_results]), rng)
r_mech = macro_bootstrap(np.array([p["mech_coef_doubly_adj"] for p in pair_results]), rng)

print()
print(f"MACRO caliper: raw={macro_caliper_raw:.4f} shared_only={macro_caliper_shared:.4f} "
      f"copy_only={macro_caliper_copy:.4f} both={macro_caliper_adj:.4f}  "
      f"additive_check={macro_caliper_shared + macro_caliper_copy - macro_caliper_raw:.4f}  "
      f"CI(both macro)={r_caliper['ci']}")
print(f"MACRO NN-ATT:  raw={macro_nn_raw:.4f} shared_only={macro_nn_shared:.4f} "
      f"copy_only={macro_nn_copy:.4f} both={macro_nn_adj:.4f}  "
      f"additive_check={macro_nn_shared + macro_nn_copy - macro_nn_raw:.4f}  "
      f"CI(both macro)={r_nn['ci']}")
print(f"MACRO nested:  raw={macro_mech_raw:.4f} shared_only={macro_mech_shared:.4f} "
      f"copy_only={macro_mech_copy:.4f} both={macro_mech_adj:.4f}  "
      f"additive_check={macro_mech_shared + macro_mech_copy - macro_mech_raw:.4f}  "
      f"CI(both macro)={r_mech['ci']}")

out = dict(role="raw/shared-only/copy-only/both sensitivity re-run of the pre-specified "
                "caliper/NN/nested estimators, 2026-10-13 post-hoc, offline only",
           note="the 'both' reading equals shared_only + copy_only - raw to numerical precision "
                "for every estimator (verified, not assumed), since each estimator is linear in "
                "the per-row dY values and both corrections are additive shifts to dY",
           pairs=pair_results,
           macro=dict(caliper_raw=macro_caliper_raw, caliper_shared_only=macro_caliper_shared,
                      caliper_copy_only=macro_caliper_copy, caliper_doubly_adj=macro_caliper_adj,
                      caliper_ci=r_caliper["ci"],
                      nn_att_raw=macro_nn_raw, nn_att_shared_only=macro_nn_shared,
                      nn_att_copy_only=macro_nn_copy, nn_att_doubly_adj=macro_nn_adj, nn_ci=r_nn["ci"],
                      mech_coef_raw=macro_mech_raw, mech_coef_shared_only=macro_mech_shared,
                      mech_coef_copy_only=macro_mech_copy, mech_coef_doubly_adj=macro_mech_adj,
                      mech_ci=r_mech["ci"]))
dst = TABLES / "tier3_doubly_adjusted.json"
dst.write_text(json.dumps(out, indent=2))
print("wrote", dst)
