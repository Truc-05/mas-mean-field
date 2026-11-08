"""CPU-only tests for MMRT (mechanism-matched repair test) and SMB (susceptibility vs.
benefit), per MMRT_SMB_PREREG.md. All fixtures are hand-constructed; none of this touches the
real frozen Tier-3/SGIR data files.
"""
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "exp6_phase_transition_llm"
sys.path.insert(0, str(EXPERIMENTS))

from mmrt_matched_repair import nested_models_metric, nn_matching_metric  # noqa: E402
from smb_susceptibility_vs_benefit import macro_paired, ols_slope, per_pair_slopes  # noqa: E402


# ---- MMRT: nested_models_metric reads the GIVEN metric, not a hard-coded "dY" ---------------
def _row(mech, D, A0, margin, dY, H):
    return {"mech": mech, "D": D, "A0": A0, "margin": margin, "dY": dY, "H": H}


def test_nested_models_metric_uses_the_given_metric_column():
    sh = [_row("shared", 0.3, 0.5, 0.4, dY=0.9, H=0.0) for _ in range(10)]
    cp = [_row("copy", 0.3, 0.5, 0.4, dY=0.1, H=1.0) for _ in range(10)]
    res_dy = nested_models_metric(sh, cp, "dY")
    res_h = nested_models_metric(sh, cp, "H")
    # dY is identical within each mechanism but differs sharply ACROSS mechanisms in both
    # columns here by construction -- check the two metric columns give DIFFERENT coefficients
    # (proves the function actually reads the metric argument, not a hard-coded "dY").
    assert res_dy["mech_coef_given_D"] != res_h["mech_coef_given_D"]
    assert abs(res_dy["mech_coef_given_D"] - (0.1 - 0.9)) < 1e-9
    assert abs(res_h["mech_coef_given_D"] - (1.0 - 0.0)) < 1e-9


def test_nested_models_metric_recovers_known_mechanism_effect_given_D():
    rng = np.random.default_rng(0)
    n = 200
    D = rng.uniform(0, 1, n)
    mech = (rng.random(n) < 0.5).astype(float)
    # H = 2*D + 0.5*mech + noise  -> mech_coef_given_D should recover ~0.5
    noise = rng.normal(0, 0.01, n)
    H = 2.0 * D + 0.5 * mech + noise
    sh = [_row("shared", D[i], 0.5, 0.4, dY=0.0, H=H[i]) for i in range(n) if mech[i] == 0]
    cp = [_row("copy", D[i], 0.5, 0.4, dY=0.0, H=H[i]) for i in range(n) if mech[i] == 1]
    res = nested_models_metric(sh, cp, "H")
    assert abs(res["mech_coef_given_D"] - 0.5) < 0.05


def test_nested_models_metric_drops_non_finite_metric_rows():
    sh = [_row("shared", 0.3, 0.5, 0.4, dY=0.0, H=float("nan"))] + \
         [_row("shared", 0.3, 0.5, 0.4, dY=0.0, H=0.2) for _ in range(5)]
    cp = [_row("copy", 0.3, 0.5, 0.4, dY=0.0, H=0.4) for _ in range(5)]
    res = nested_models_metric(sh, cp, "H")
    assert res["n"] == 10          # the one NaN-H shared row is excluded


# ---- MMRT: nn_matching_metric matches on (D, A0, margin) and reads the given metric --------
def test_nn_matching_metric_recovers_a_known_att():
    sh = [_row("shared", d, 0.5, 0.5, dY=0.0, H=0.0) for d in [0.1, 0.5, 0.9]]
    cp = [_row("copy", d, 0.5, 0.5, dY=0.0, H=1.0) for d in [0.1, 0.5, 0.9]]
    res = nn_matching_metric(sh, cp, "H", n_boot=500)
    assert abs(res["att_copy_minus_shared"] - 1.0) < 1e-9
    assert res["n_copy"] == 3 and res["n_shared"] == 3


def test_nn_matching_metric_excludes_non_finite_metric_rows():
    sh = [_row("shared", 0.1, 0.5, 0.5, dY=0.0, H=0.0)]
    cp = [_row("copy", 0.1, 0.5, 0.5, dY=0.0, H=float("nan")),
          _row("copy", 0.1, 0.5, 0.5, dY=0.0, H=1.0)]
    res = nn_matching_metric(sh, cp, "H", n_boot=200)
    assert res["n_copy"] == 1          # the NaN-H copy row is excluded
    assert abs(res["att_copy_minus_shared"] - 1.0) < 1e-9


# ---- SMB: ols_slope --------------------------------------------------------------------------
def test_ols_slope_recovers_known_linear_relationship():
    x = np.linspace(0, 1, 50)
    y = 3.0 + 2.0 * x
    assert abs(ols_slope(x, y) - 2.0) < 1e-9


# ---- SMB: per_pair_slopes separates S from B when margin predicts ONE but not the other ----
def test_per_pair_slopes_separates_susceptibility_from_benefit():
    # Reduced binary-correctness world: keep_label/indep_label in {0=correct, 1=incorrect}, so
    # S = 1[labels differ] = 1[correctness flips], and B = indep_correct - keep_correct is the
    # SIGNED version of the same flip. keep_correct is a fair coin, INDEPENDENT of margin; the
    # flip itself happens with probability (1-margin) (susceptibility, margin-dependent). By
    # symmetry (keep_correct unbiased), a flip is equally likely to break or rescue, so E[B] is
    # exactly 0 at every margin even though P(flip)=S is strongly margin-dependent -- a clean
    # ground truth where margin predicts S but not B.
    rng = np.random.default_rng(1)
    n = 2000
    margin = rng.uniform(0, 1, n)
    keep_correct = rng.random(n) < 0.5                     # fair coin, margin-independent
    flip = rng.random(n) < (1 - margin)                    # susceptibility: low margin -> flips
    indep_correct = np.where(flip, ~keep_correct, keep_correct)
    keep_label = (~keep_correct).astype(int)                # 0=correct, 1=incorrect
    indep_label = (~indep_correct).astype(int)
    rows = [{"model": "m", "bench": "b", "margin": float(margin[i]),
             "keep_label": int(keep_label[i]), "indep_label": int(indep_label[i]),
             "keep_correct": bool(keep_correct[i]), "indep_correct": bool(indep_correct[i])}
            for i in range(n)]
    per_pair = per_pair_slopes(rows)
    v = per_pair[("m", "b")]
    assert v["slope_S"] < -0.5            # susceptibility strongly, negatively tied to margin
    assert abs(v["slope_B"]) < 0.08        # net benefit is (by construction) margin-independent
    assert abs(v["slope_S"]) > 5 * abs(v["slope_B"])


def test_per_pair_slopes_respects_eligibility():
    rows = [{"model": "m", "bench": "b", "margin": None, "keep_label": 0, "indep_label": 1,
             "keep_correct": True, "indep_correct": False}] * 3 + \
           [{"model": "m", "bench": "b", "margin": 0.5, "keep_label": 0, "indep_label": 0,
             "keep_correct": True, "indep_correct": True}] * 5
    per_pair = per_pair_slopes(rows)
    assert per_pair[("m", "b")]["n"] == 5     # the 3 margin=None rows are ineligible, excluded


# ---- SMB: macro_paired is a genuinely PAIRED bootstrap over the same 9 pairs -----------------
def test_macro_paired_is_deterministic_given_seed():
    per_pair = {("a", "x"): {"slope_S": 1.0, "slope_B": 0.1},
                ("b", "y"): {"slope_S": -0.8, "slope_B": -0.5},
                ("c", "z"): {"slope_S": 0.3, "slope_B": 0.4}}
    m1 = macro_paired(per_pair, np.random.default_rng(5))
    m2 = macro_paired(per_pair, np.random.default_rng(5))
    assert m1 == m2


def test_macro_paired_diff_matches_hand_computation_when_no_noise():
    # 3 pairs, deterministic -- the point estimate must be the exact mean of per-pair |S|-|B|.
    per_pair = {("a", "x"): {"slope_S": 1.0, "slope_B": 0.2},
                ("b", "y"): {"slope_S": -0.6, "slope_B": -0.1},
                ("c", "z"): {"slope_S": 0.4, "slope_B": 0.4}}
    expected = np.mean([abs(1.0) - abs(0.2), abs(-0.6) - abs(-0.1), abs(0.4) - abs(0.4)])
    m = macro_paired(per_pair, np.random.default_rng(0))
    assert abs(m["paired_diff_absS_minus_absB"]["point"] - expected) < 1e-12
