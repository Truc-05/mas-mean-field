"""Deterministic check of the within-arm churn baseline added to tier1_dY_decision_churn.py
(2026-10-12). Hand-constructed replicate decisions so ΔY_cross, C_shared, C_indep, C_within,
and dC can all be verified by hand arithmetic, independent of any real sweep data.
"""
import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments" / "exp6_phase_transition_llm"
sys.path.insert(0, str(EXPERIMENTS))

from tier1_dY_decision_churn import NONE, SHARED, INDEP, _within_churn, per_question  # noqa: E402


def _mat(votes: list[int]) -> list[list[float]]:
    """One-hot (N_agents, 2) rows from a per-agent vote list; no flat/tied rows."""
    return [[1.0, 0.0] if v == 0 else [0.0, 1.0] for v in votes]


def test_within_churn_hand_values():
    # [0, 0, 1, 1] over 6 unordered pairs: (0,1)same (0,2)diff (0,3)diff (1,2)diff (1,3)diff (2,3)same
    # -> 4 disagreements / 6 pairs
    assert abs(_within_churn([0, 0, 1, 1]) - 4 / 6) < 1e-12
    assert _within_churn([0, 0, 0, 0]) == 0.0
    assert _within_churn([1, 1, 1, 1]) == 0.0
    import math
    assert math.isnan(_within_churn([0]))  # n<2: undefined, not zero


def test_per_question_matches_hand_computation():
    # Decisions across 4 held-out replicates (3-agent plurality vote each, no ties):
    #   Y_shared = [0, 0, 1, 1]      Y_indep = [0, 1, 1, 0]      Y_none = [0, 0, 0, 0]
    shared_votes = [[0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 0]]   # plurality -> 0,0,1,1
    indep_votes = [[0, 0, 1], [1, 1, 0], [1, 1, 0], [0, 0, 1]]    # plurality -> 0,1,1,0
    none_votes = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]     # plurality -> 0,0,0,0

    rec = {
        "qi": 0,
        "x": {
            NONE: [_mat(v) for v in none_votes],
            SHARED: [_mat(v) for v in shared_votes],
            INDEP: [_mat(v) for v in indep_votes],
        },
    }
    res = per_question(rec, eval_reps=range(4), strict=True)

    # ΔY_cross at matched replicate: diffs at r=1 and r=3 only -> 2/4 = 0.5
    assert abs(res["dY"] - 0.5) < 1e-12
    # Both arms individually: 4 disagreeing pairs out of 6 (worked by hand above)
    assert abs(res["C_shared_within"] - 4 / 6) < 1e-12
    assert abs(res["C_indep_within"] - 4 / 6) < 1e-12
    assert abs(res["C_within"] - 4 / 6) < 1e-12
    # dC = ΔY_cross - C_within = 0.5 - 4/6 = -1/6
    assert abs(res["dC"] - (0.5 - 4 / 6)) < 1e-12


def test_identical_arms_have_zero_cross_churn_but_nonzero_baseline():
    # SHARED and INDEP decisions are identical to each other but both churn across replicates
    # within themselves -> dY_cross = 0 even though C_within > 0, so dC = -C_within exactly.
    votes = [[0, 0, 1], [1, 1, 0], [0, 0, 1], [1, 1, 0]]          # plurality -> 0,1,0,1
    none_votes = [[0, 0, 0]] * 4
    rec = {
        "qi": 0,
        "x": {
            NONE: [_mat(v) for v in none_votes],
            SHARED: [_mat(v) for v in votes],
            INDEP: [_mat(v) for v in votes],
        },
    }
    res = per_question(rec, eval_reps=range(4), strict=True)
    assert res["dY"] == 0.0
    # [0,1,0,1]: pairs (0,1)d (0,2)s (0,3)d (1,2)d (1,3)s (2,3)d -> 4/6
    assert abs(res["C_shared_within"] - 4 / 6) < 1e-12
    assert abs(res["C_within"] - 4 / 6) < 1e-12
    assert abs(res["dC"] - (-4 / 6)) < 1e-12
