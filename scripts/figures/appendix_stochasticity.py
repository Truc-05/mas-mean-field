"""Appendix: per-cell stochasticity -- raw decision-change rate against its
own within-arm noise floor, all 66 cells, plus tie-convention sensitivity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_stochasticity")

rows = dt.cell_rows("strict")
df = pd.DataFrame(rows)

fig, axes = st.grid(1, 2, figsize=(11.5, 5.6))
st.suptitle(fig, "Stochasticity atlas")

ax = axes[0]
st.style_axes(ax)
st.panel_title(ax, "a", "Per-cell raw rate vs. its own noise floor")
palette = {"hard": st.REGIME_HARD, "easy": st.REGIME_EASY}
sns.scatterplot(data=df, x="C_within", y="dY", hue="regime", palette=palette, ax=ax,
                 s=46, alpha=0.85, edgecolor="white", linewidth=0.4)
lim = max(df["C_within"].max(), df["dY"].max()) * 1.08
ax.plot([0, lim], [0, lim], color=st.BASELINE, lw=1.1, zorder=1)
ax.text(lim * 0.60, lim * 0.68, "$\\Delta Y = C_{within}$\n($\\delta C=0$)", fontsize=7.8,
        color=st.INK_MUTED, rotation=36, ha="center")
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("within-arm noise floor $C_{within}$")
ax.set_ylabel("raw decision-change rate $\\Delta Y$")
ax.legend(title="")

ax = axes[1]
st.style_axes(ax)
st.panel_title(ax, "b", "Tie-convention sensitivity (strict vs. tiebreak)")
conv = [
    ("hard,\nstrict", dt.CHURN_STRICT_HARD["dC_mean"], dt.CHURN_STRICT_HARD["dC_ci"], st.REGIME_HARD, 1.0),
    ("hard,\ntiebreak", dt.CHURN_TIEBREAK_HARD["dC_mean"], dt.CHURN_TIEBREAK_HARD["dC_ci"], st.REGIME_HARD, 0.5),
    ("easy,\nstrict", dt.CHURN_STRICT_EASY["dC_mean"], dt.CHURN_STRICT_EASY["dC_ci"], st.REGIME_EASY, 1.0),
    ("easy,\ntiebreak", dt.CHURN_TIEBREAK_EASY["dC_mean"], dt.CHURN_TIEBREAK_EASY["dC_ci"], st.REGIME_EASY, 0.5),
]
y = np.arange(len(conv))[::-1]
for i, (lab, mean, ci, color, alpha) in enumerate(conv):
    ax.plot(ci, [y[i]] * 2, color=color, alpha=alpha, lw=2.6, solid_capstyle="round")
    ax.scatter([mean], [y[i]], color=color, alpha=alpha, s=60, zorder=5, edgecolor="white", lw=0.8)
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(y)
ax.set_yticklabels([c[0] for c in conv])
ax.set_xlabel("excess churn $\\delta C$")

st.savefig(fig, OUT)
print("wrote", OUT)
