"""Appendix: every headline interval estimate in the paper, grouped by
section. Each group keeps its own x-scale.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_ci_atlas")

fig, axes = st.grid(2, 3, figsize=(13.5, 8.2))
st.suptitle(fig, "CI atlas: every headline interval")

def forest(ax, letter, title, rows, extra_vlines=None, xlim=None, fmt="{:.3f}"):
    st.style_axes(ax)
    st.panel_title(ax, letter, title)
    n = len(rows)
    los = [r[2] for r in rows]; his = [r[3] for r in rows]
    if xlim is None:
        span = max(his) - min(los)
        xlim = (min(los) - span * 0.10, max(his) + span * 0.55)
    for i, (label, mean, lo, hi, color) in enumerate(rows):
        y = n - i
        ax.plot([lo, hi], [y, y], color=color, lw=2.4, solid_capstyle="round")
        ax.scatter([mean], [y], s=42, color=color, zorder=5, edgecolor="white", lw=0.8)
        ax.text(hi + (xlim[1] - xlim[0]) * 0.03, y, fmt.format(mean), fontsize=7.8,
                color=st.INK, va="center", fontweight="bold")
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.2)
    ax.set_ylim(0.3, n + 0.9)
    ax.set_xlim(*xlim)
    ax.axvline(0, color=st.BASELINE, lw=0.9)
    for v in (extra_vlines or []):
        ax.axvline(v, color=st.CRITICAL, lw=0.8, ls=(0, (2, 2)))

rows1 = [
    ("hard", dt.CHURN_STRICT_HARD["dC_mean"], *dt.CHURN_STRICT_HARD["dC_ci"], st.REGIME_HARD),
    ("easy", dt.CHURN_STRICT_EASY["dC_mean"], *dt.CHURN_STRICT_EASY["dC_ci"], st.REGIME_EASY),
]
forest(axes[0, 0], "a", "Excess churn $\\delta C$", rows1)

H2 = dt.DA_REGIME["H2"]
rows2 = [
    ("hard", H2["hard"]["mean"], H2["hard"]["lo90"], H2["hard"]["hi90"], st.REGIME_HARD),
    ("easy", H2["easy"]["mean"], H2["easy"]["lo90"], H2["easy"]["hi90"], st.REGIME_EASY),
    ("pooled", H2["all"]["mean"], H2["all"]["lo90"], H2["all"]["hi90"], st.GOOD),
]
forest(axes[0, 1], "b", "Accuracy TOST ($E[\\Delta A]$, 90% CI)", rows2, extra_vlines=[0.05, -0.05], xlim=(-0.065, 0.065))

H3 = dt.DA_REGIME["H3"]
rows3 = [("easy x hard", H3["beta1"], H3["lo"], H3["hi"], st.INK_SECONDARY)]
forest(axes[0, 2], "c", "Regime interaction coefficient", rows3)

rows4 = [
    ("caliper", dt.MACRO["caliper_gap"], *dt.MACRO_CI["caliper_gap"], st.COPYING),
    ("NN-ATT", dt.MACRO["nn_att"], *dt.MACRO_CI["nn_att"], st.COPYING),
    ("nested-LPM", dt.MACRO["mech_coef"], *dt.MACRO_CI["mech_coef"], st.COPYING),
]
forest(axes[1, 0], "d", "Matched-dependence mechanism gap", rows4, xlim=(-0.02, 0.22))

rows5 = [
    ("low-competence", dt.SUSCEPT["low_slope"], *dt.SUSCEPT["low_ci"], st.REGIME_HARD),
    ("high-competence", dt.SUSCEPT["high_slope"], *dt.SUSCEPT["high_ci"], st.REGIME_EASY),
]
forest(axes[1, 1], "e", "Margin-susceptibility slope", rows5, xlim=(-0.30, 0.06))

ax = axes[1, 2]
st.style_axes(ax, spine="none")
st.panel_title(ax, "f", "Not shown as a CI")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.text(0.3, 8.0, "Instrument sign test", fontsize=9.5, fontweight="bold", color=st.INK)
ax.text(0.3, 6.9, "63/66 strata, $p<0.0005$ -- a count\nagainst a simulated null, not an interval",
        fontsize=8.3, color=st.INK_SECONDARY, va="top")
ax.text(0.3, 4.3, "Finite-size slopes", fontsize=9.5, fontweight="bold", color=st.INK)
ax.text(0.3, 3.2, "N=5,11,21 (Qwen2.5-7B only) --\nno per-point CI reported in the manuscript",
        fontsize=8.3, color=st.INK_SECONDARY, va="top")
ax.set_xticks([]); ax.set_yticks([])

st.savefig(fig, OUT)
print("wrote", OUT)
