"""Derived aggregates for F4/F5, kept separate from data.py (which is shared
with other figure scripts) to avoid clobbering concurrent edits there.
Everything here is a plain aggregation of data.CELLS -- no new source files,
no invented numbers.
"""
import numpy as np
import data as dt


def _regime_rows(regime, convention="strict"):
    return [c[convention] for c in dt.CELLS if c["regime"] == regime]


def directional_breakdown(regime, convention="strict"):
    """Macro-average (equal weight per stratum) p_break, p_rescue, p_churn,
    dY, dA for one regime. p_churn is recovered from the identity
    dY = p_break + p_rescue + p_churn (Proposition: Directional partition) --
    it is not stored as its own field in the frozen table, but the identity
    is exact by construction (checked offline when the table was produced),
    so this recovery introduces no new assumption.
    """
    rows = _regime_rows(regime, convention)
    pb = float(np.mean([r["p_break"] for r in rows]))
    pr = float(np.mean([r["p_rescue"] for r in rows]))
    dy = float(np.mean([r["dY"] for r in rows]))
    da = float(np.mean([r["dA"] for r in rows]))
    pc = dy - pb - pr
    return dict(n=len(rows), p_break=pb, p_rescue=pr, p_churn=pc, dY=dy, dA=da)


BREAKDOWN_EASY = directional_breakdown("easy")
BREAKDOWN_HARD = directional_breakdown("hard")

# mechanism-gap directional breakdown (copying minus shared), matched-D
# substrate, eps=0.05 primary caliper -- from data.MACRO (already verified
# against paper.md: p_break_gap=0.026, p_rescue_gap=0.033) plus the churn-gap
# recovered the same way, by the same identity, from the already-reported
# dY_gap=0.119 total.
MECH_GAP_BREAKDOWN = dict(
    dY_gap=dt.MACRO["caliper_gap"],
    p_break_gap=dt.MACRO["p_break_gap"],
    p_rescue_gap=dt.MACRO["p_rescue_gap"],
    p_churn_gap=dt.MACRO["caliper_gap"] - dt.MACRO["p_break_gap"] - dt.MACRO["p_rescue_gap"],
)
