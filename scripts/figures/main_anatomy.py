"""Main figure: anatomy of the decision-change gap -- harmful, beneficial,
and churn components, for both the matched-dependence and breadth-grid
substrates (kept separate, never merged into one estimand).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt
import data_extra as de

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_anatomy")

fig, axes = st.grid(2, 2, figsize=(11.0, 9.0))
st.suptitle(fig, "Anatomy of decision change")

# --------------------------------------------------- a: matched-D stacked bar
ax = axes[0, 0]
st.style_axes(ax)
st.panel_title(ax, "a", "Matched-$D$: copying $-$ shared ($\\Delta Y$=0.118)")
comp = de.MECH_GAP_BREAKDOWN
labels = ["harmful\n(break)", "beneficial\n(rescue)", "churn\n(wrong$\\to$wrong)"]
vals = [comp["p_break_gap"], comp["p_rescue_gap"], comp["p_churn_gap"]]
colors = [st.CRITICAL, st.GOOD, st.CHURN]
ax.bar(labels, vals, color=colors, width=0.55)
for i, v in enumerate(vals):
    ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9.5, color=colors[i], fontweight="bold")
ax.set_ylabel("$\\Delta Y$ gap component")
ax.set_ylim(0, 0.072)

# --------------------------------------------------- b: breadth grid by regime
ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "Breadth grid: shared $-$ independent, by regime")
rows_b = []
for reg, bd in [("hard", de.BREAKDOWN_HARD), ("easy", de.BREAKDOWN_EASY)]:
    rows_b += [
        {"regime": reg, "component": "harmful", "value": bd["p_break"]},
        {"regime": reg, "component": "beneficial", "value": bd["p_rescue"]},
        {"regime": reg, "component": "churn", "value": bd["dY"] - bd["p_break"] - bd["p_rescue"]},
    ]
bdf = pd.DataFrame(rows_b)
comp_colors = {"harmful": st.CRITICAL, "beneficial": st.GOOD, "churn": st.CHURN}
sns.barplot(data=bdf, x="regime", y="value", hue="component", ax=ax,
            palette=comp_colors, order=["hard", "easy"])
ax.set_ylabel("mean per-cell rate")
ax.set_xlabel("")
ax.legend(title="", loc="upper right", fontsize=8)

# --------------------------------------------------- c: net accuracy check
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "Net accuracy stays near zero despite the gap")
pdf = pd.DataFrame(dt.PAIRS)
y = np.arange(len(pdf))
pdf = pdf.sort_values("dA_gap").reset_index(drop=True)
for i, row in pdf.iterrows():
    color = st.CRITICAL if row["dA_ci"][0] > 0 or row["dA_ci"][1] < 0 else st.INK_MUTED
    ax.plot(row["dA_ci"], [i, i], color=color, lw=2.2, solid_capstyle="round")
    ax.scatter([row["dA_gap"]], [i], color=color, s=45, zorder=5, edgecolor="white", lw=0.6)
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(y)
ax.set_yticklabels([f"{st.MODEL_LABEL[r.model].split('-')[0]} / {st.BENCH_LABEL[r.bench]}" for r in pdf.itertuples()],
                    fontsize=7.6)
ax.set_xlabel("$\\Delta A$ gap: copying $-$ shared (caliper-matched)")

# --------------------------------------------------- d: cross-substrate comparison
ax = axes[1, 1]
st.style_axes(ax, spine="none")
st.panel_title(ax, "d", "Two different estimands, not one")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.text(5, 9.0, "the mechanism gap and the breadth-grid excess measure", ha="center", fontsize=8.6,
        color=st.INK_SECONDARY)
ax.text(5, 8.3, "different things and are never pooled into one number", ha="center", fontsize=8.6,
        color=st.INK_SECONDARY)
rows_d = [
    ("matched-$D$ mechanism gap", "copying vs. shared, equal $D$", comp["dY_gap"], st.COPYING),
    ("breadth excess, hard", "shared vs. independent, noise-netted", dt.CHURN_STRICT_HARD["dC_mean"], st.REGIME_HARD),
    ("breadth excess, easy", "shared vs. independent, noise-netted", dt.CHURN_STRICT_EASY["dC_mean"], st.REGIME_EASY),
]
y0 = 6.6
for i, (lab, sub, val, color) in enumerate(rows_d):
    y = y0 - i * 2.2
    ax.text(0.2, y + 0.55, lab, fontsize=9.2, color=st.INK, fontweight="bold", va="bottom")
    ax.text(0.2, y - 0.15, sub, fontsize=7.6, color=st.INK_MUTED, va="bottom")
    ax.barh([y - 0.9], [val * 20], left=0, height=0.4, color=color, alpha=0.85)
    ax.text(val * 20 + 0.3, y - 0.9, f"{val:.3f}", fontsize=8.3, color=color, va="center", fontweight="bold")
ax.set_xticks([])
ax.set_yticks([])

st.savefig(fig, OUT)
print("wrote", OUT)
