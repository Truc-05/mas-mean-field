"""Verified quantities for the figure suite. Every number here is either
loaded directly from a frozen results/tables/*.json file, or -- where no
single JSON row holds the macro-level number -- transcribed from paper.md's
audited prose (itself cross-checked against the JSON files in the same pass
that produced this script). Each constant below says which.

No retired dD magnitudes, no C4, no unverified numbers.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "tables"))


def _load(name):
    with open(os.path.join(ROOT, name)) as f:
        return json.load(f)


# ----------------------------------------------------------- F1 / instrument
_c3 = _load("c3_global_directional_test.json")
C3 = dict(
    K=_c3["K"], S_obs=_c3["S_obs"],
    null_mean=_c3["null"]["mean"], null_sd=_c3["null"]["sd"],
    null_median=_c3["null"]["median"], null_q95=_c3["null"]["q95"],
    null_max=_c3["null"]["max"], n_sim=_c3["null"]["n_sim"],
    p=_c3["p_one_sided"], strata=_c3["strata"],
)


def _fraction_by(key):
    agg = {}
    for s in C3["strata"]:
        k = s[key]
        hit, tot = agg.get(k, (0, 0))
        tot += 1
        hit += 1 if s["sign_positive"] else 0
        agg[k] = (hit, tot)
    return agg


BY_MODEL = _fraction_by("model")      # e.g. {'gemma': (10, 11), ...}
BY_BENCH = _fraction_by("bench")      # e.g. {'truthfulqa': (9, 11), ...}


def strata_grid():
    """66 usable strata as (model, bench, regime) -> sign_positive."""
    return {(s["model"], s["bench"], s["regime"]): s["sign_positive"] for s in C3["strata"]}


# attempted-but-excluded count (paper.md Sec. generalization): 73 attempted =
# 72 (6x6x2 grid) + 1 qwen/bbh anchor; 7 excluded = 6 below min-question-count
# + qwen/bbh. Transcribed from paper.md Sec. 6.5 (generalization), itself
# checked against c3_global_directional_test.json's `skipped` list length (7).
N_ATTEMPTED = 73
N_EXCLUDED = 7
N_USABLE = 66

# ------------------------------------------------------- F2 / noise canyon
_churn = _load("tier1_dY_decision_churn_aggregate.json")
CHURN_ROWS = {(r["regime"], r["tie_convention"]): r for r in _churn["rows"]}
CHURN_STRICT_EASY = CHURN_ROWS[("easy", "strict")]
CHURN_STRICT_HARD = CHURN_ROWS[("hard", "strict")]
CHURN_TIEBREAK_EASY = CHURN_ROWS[("easy", "tiebreak")]
CHURN_TIEBREAK_HARD = CHURN_ROWS[("hard", "tiebreak")]

# full per-cell table (66 cells, strict convention) for S1/S5/S6 atlases and
# F4/F6 per-cell composition
_cells_raw = _load("tier1_dY_decision_churn.json")
CELLS = _cells_raw["cells"]  # list of 66 dicts with ['strict'] and ['tiebreak']


def cell_rows(convention="strict"):
    out = []
    for c in CELLS:
        row = dict(model=c["model"], benchmark=c["benchmark"], regime=c["regime"],
                    n_pooled_questions=c["n_pooled_questions"], **c[convention])
        out.append(row)
    return out


# -------------------------------------------------- F3 / F8 matched-dependence
_t3 = _load("tier3_matched_d.json")
_t3_audit = _load("tier3_matched_d_churn_audit.json")
_audit_by_key = {(p["model"], p["bench"]): p for p in _t3_audit["pairs"]}
_t3_doubly = _load("tier3_doubly_adjusted.json")
_doubly_by_key = {(p["model"], p["bench"]): p for p in _t3_doubly["pairs"]}

PAIRS = []
for p in _t3["pairs"]:
    c05 = p["caliper"]["eps_0.05"]
    c02 = p["caliper"]["eps_0.02"]
    c10 = p["caliper"]["eps_0.10"] if "eps_0.10" in p["caliper"] else p["caliper"]["eps_0.1"]
    nn = p["nn_matching"]
    nm = p["nested_models"]
    audit = _audit_by_key[(p["model"], p["bench"])]
    doubly = _doubly_by_key[(p["model"], p["bench"])]
    PAIRS.append(dict(
        model=p["model"], bench=p["bench"],
        n_shared=p["n_shared"], n_copy=p["n_copy"],
        D_range_shared=tuple(p["D_range_shared"]), D_range_copy=tuple(p["D_range_copy"]),
        dY_gap=c05["dY"]["gap"], dY_ci=tuple(c05["dY"]["ci"]), dY_n=c05["dY"]["n"],
        dA_gap=c05["dA"]["gap"], dA_ci=tuple(c05["dA"]["ci"]),
        p_break_gap=c05["p_break"]["gap"], p_rescue_gap=c05["p_rescue"]["gap"],
        coverage_eps05=c05["dY"]["n"] / p["n_copy"],
        coverage_eps02=c02["dY"]["n"] / p["n_copy"],
        coverage_eps10=c10["dY"]["n"] / p["n_copy"],
        gap_eps02=c02["dY"]["gap"], gap_eps10=c10["dY"]["gap"],
        nn_att=nn["att_copy_minus_shared"], nn_ci=tuple(nn["ci"]),
        mech_coef=nm["mech_coef_given_D"],
        adj_caliper_gap=audit["caliper_churn_adjusted"]["eps_0.05"]["gap"],
        adj_nn_att=audit["nn_churn_adjusted"]["att_copy_minus_shared"],
        adj_mech_coef=audit["nested_models_adjusted_mech_coef"],
        C_within_shared_only=audit["mean_C_within_shared_only"],
        # four-reading sensitivity decomposition (raw/shared-only/copy-only/both),
        # tier3_doubly_adjusted.py -- "shared-only" here duplicates adj_caliper_gap
        # above (same quantity, same source audit value) and is kept for the
        # four-reading panels so all four x-positions read from one table.
        caliper_copy_only=doubly["caliper_gap_copy_only"],
        caliper_both=doubly["caliper_gap_doubly_adj"],
        nn_copy_only=doubly["nn_att_copy_only"],
        nn_both=doubly["nn_att_doubly_adj"],
        mech_copy_only=doubly["mech_coef_copy_only"],
        mech_both=doubly["mech_coef_doubly_adj"],
    ))

MACRO = dict(
    caliper_gap=sum(p["dY_gap"] for p in PAIRS) / 9,
    nn_att=sum(p["nn_att"] for p in PAIRS) / 9,
    mech_coef=sum(p["mech_coef"] for p in PAIRS) / 9,
    adj_caliper_gap=sum(p["adj_caliper_gap"] for p in PAIRS) / 9,
    adj_nn_att=sum(p["adj_nn_att"] for p in PAIRS) / 9,
    adj_mech_coef=sum(p["adj_mech_coef"] for p in PAIRS) / 9,
    p_break_gap=sum(p["p_break_gap"] for p in PAIRS) / 9,
    p_rescue_gap=sum(p["p_rescue_gap"] for p in PAIRS) / 9,
    caliper_copy_only=_t3_doubly["macro"]["caliper_copy_only"],
    caliper_both=_t3_doubly["macro"]["caliper_doubly_adj"],
    nn_copy_only=_t3_doubly["macro"]["nn_att_copy_only"],
    nn_both=_t3_doubly["macro"]["nn_att_doubly_adj"],
    mech_copy_only=_t3_doubly["macro"]["mech_coef_copy_only"],
    mech_both=_t3_doubly["macro"]["mech_coef_doubly_adj"],
)
# caliper CI and nested-model CI as reported in paper.md Sec. 6.2 (macro,
# bootstrap over the 9 pairs -- not recomputable from the per-pair file alone
# since the macro bootstrap resamples pairs, not cells; transcribed from the
# audited manuscript text)
MACRO_CI = dict(
    caliper_gap=(0.072, 0.172),
    nn_att=(0.013, 0.073),
    mech_coef=(0.062, 0.169),
)

# -------------------------------------------------- regime battery (H1-H4)
DA_REGIME = _load("tier1_dA_regime_test.json")

# -------------------------------------------------------- F5 / boundary gate
# Macro-averaged margin-susceptibility slope by regime (paper.md Sec. 6.3),
# cross-checked in the manuscript audit pass against the frozen breadth
# substrate; no single JSON row stores the macro slope directly (it is a
# stratum-resampled bootstrap over the per-stratum regressions).
SUSCEPT = dict(
    low_slope=-0.063, low_ci=(-0.112, -0.012),
    high_slope=-0.137, high_ci=(-0.243, -0.050),
    high_degenerate=(15, 31),  # recomputed from CELLS: both arms at ceiling (dA=dY=0); was wrongly 17 before audit
    perm_audit_hard=dict(bootstrap_sig=5, perm_sig=4, overlap=3, n=35),
    perm_audit_easy=dict(bootstrap_sig=1, perm_sig=6, overlap=1, n=31),
)
# finite-size sweep, single model (Qwen2.5-7B), paper.md Sec. Limitations
FINITE_N = dict(N=[5, 11, 21], slope=[-0.091, -0.125, -0.104])
# per-model fraction of high-competence strata that are exactly degenerate,
# recomputed directly from CELLS (A_shared==1.0 and A_indep==1.0 in the easy
# regime) -- the previous hardcoded values did not sum to (15, 31) and were wrong
DEGENERATE_BY_MODEL = dict(
    deepseek=(5, 6), gemma=(3, 6), llama=(2, 4),
    mistral=(0, 5), phi4=(3, 4), qwen=(2, 6),
)

# -------------------------------------------------------------- scale notes
MODEL_PARAMS_B = dict(
    qwen=7.0, llama=8.0, gemma=9.0, mistral=7.0, phi4=3.8, deepseek=7.0,
)
