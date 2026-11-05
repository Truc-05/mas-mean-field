"""Main figure: robustness checks -- caliper width, estimator agreement,
noise-adjustment direction, tie-convention sensitivity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_robustness")

pdf = pd.DataFrame(dt.PAIRS)
fig, axes = st.grid(2, 2, figsize=(11.0, 9.0))
st.suptitle(fig, "Robustness checks")

# ------------------------------------------------------ a: caliper width
ax = axes[0, 0]
st.style_axes(ax)
st.panel_title(ax, "a", "Sign is stable across caliper width")
long = pdf.melt(id_vars=["model", "bench"], value_vars=["gap_eps02", "dY_gap", "gap_eps10"],
                 var_name="eps", value_name="gap")
eps_lab = {"gap_eps02": "0.02", "dY_gap": "0.05", "gap_eps10": "0.10"}
long["eps"] = long["eps"].map(eps_lab)
long["pair"] = long["model"] + " / " + long["bench"]
sns.lineplot(data=long, x="eps", y="gap", units="pair", estimator=None, ax=ax,
             color=st.COPYING, alpha=0.55, marker="o", markersize=5, sort=False)
macro_by_eps = [pdf["gap_eps02"].mean(), pdf["dY_gap"].mean(), pdf["gap_eps10"].mean()]
ax.plot(["0.02", "0.05", "0.10"], macro_by_eps, color=st.INK, lw=2.8, marker="D", markersize=9, zorder=6)
ax.axhline(0, color=st.BASELINE, lw=1)
ax.set_xlabel("caliper $\\varepsilon$")
ax.set_ylabel("$\\Delta Y$ gap")

# ------------------------------------------------------ b: estimator agreement
ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "Matching method does not change the sign")
ests = ["caliper", "NN-ATT", "nested-LPM"]
vals = [dt.MACRO["caliper_gap"], dt.MACRO["nn_att"], dt.MACRO["mech_coef"]]
cis = [dt.MACRO_CI["caliper_gap"], dt.MACRO_CI["nn_att"], dt.MACRO_CI["mech_coef"]]
y = np.arange(3)
for i in range(3):
    ax.plot(cis[i], [y[i]] * 2, color=st.COPYING, lw=2.8, solid_capstyle="round")
    ax.scatter([vals[i]], [y[i]], color=st.COPYING, s=70, zorder=5, edgecolor="white", lw=1)
    ax.text(cis[i][1] + 0.008, y[i], f"{vals[i]:.3f}", va="center", fontsize=9, fontweight="bold", color=st.INK)
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(y)
ax.set_yticklabels(ests)
ax.set_xlabel("mechanism-gap estimate")
ax.set_xlim(-0.02, 0.22)

# ------------------------------------------------------ c: noise-adjustment direction
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "Copy-side correction alone is the weak reading")
xs = [0, 1, 2, 3]
cols = ["dY_gap", "adj_caliper_gap", "caliper_copy_only", "caliper_both"]
for i, row in pdf.iterrows():
    ys = [row[c] for c in cols]
    ax.plot(xs, ys, color=st.INK_MUTED, lw=1.2, alpha=0.55, marker="o", markersize=4)
macro_vals = [dt.MACRO["caliper_gap"], dt.MACRO["adj_caliper_gap"],
              dt.MACRO["caliper_copy_only"], dt.MACRO["caliper_both"]]
ax.plot(xs, macro_vals, color=st.CRITICAL, lw=3, marker="D", markersize=9, zorder=6)
ax.axhline(0, color=st.BASELINE, lw=1)
ax.set_xticks(xs)
ax.set_xticklabels(["raw", "shared-side\nonly", "copy-side\nonly", "both\n(combined)"], fontsize=8.2)
ax.set_xlim(-0.3, 3.3)
ax.set_ylabel("$\\Delta Y$ gap (caliper)")

# ------------------------------------------------------ d: tie-convention sensitivity
ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "d", "Tie-breaking convention does not change the picture")
rows_d = [
    ("hard, strict", dt.CHURN_STRICT_HARD["dC_mean"], dt.CHURN_STRICT_HARD["dC_ci"], st.REGIME_HARD, 1.0),
    ("hard, tiebreak", dt.CHURN_TIEBREAK_HARD["dC_mean"], dt.CHURN_TIEBREAK_HARD["dC_ci"], st.REGIME_HARD, 0.5),
    ("easy, strict", dt.CHURN_STRICT_EASY["dC_mean"], dt.CHURN_STRICT_EASY["dC_ci"], st.REGIME_EASY, 1.0),
    ("easy, tiebreak", dt.CHURN_TIEBREAK_EASY["dC_mean"], dt.CHURN_TIEBREAK_EASY["dC_ci"], st.REGIME_EASY, 0.5),
]
y = np.arange(len(rows_d))[::-1]
for i, (lab, mean, ci, color, alpha) in enumerate(rows_d):
    ax.plot(ci, [y[i]] * 2, color=color, lw=2.6, alpha=alpha, solid_capstyle="round")
    ax.scatter([mean], [y[i]], color=color, alpha=alpha, s=60, zorder=5, edgecolor="white", lw=0.8)
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows_d])
ax.set_xlabel("excess churn $\\delta C$")

st.savefig(fig, OUT)
print("wrote", OUT)
