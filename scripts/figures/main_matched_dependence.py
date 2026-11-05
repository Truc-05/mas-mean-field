"""Main figure: copying vs. shared evidence at matched dependence, across
nine (model, benchmark) pairs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_matched_dependence")

pdf = pd.DataFrame(dt.PAIRS)
pdf["pair"] = [f"{st.MODEL_LABEL[r.model]}\n{st.BENCH_LABEL[r.bench]}" for r in pdf.itertuples()]
pdf = pdf.sort_values("dY_gap").reset_index(drop=True)

fig, axes = st.grid(2, 2, figsize=(11.5, 9.5))
st.suptitle(fig, "Matched-dependence mechanism gap")

# -------------------------------------------------------- a: per-pair forest
ax = axes[0, 0]
st.style_axes(ax)
st.panel_title(ax, "a", "$\\Delta Y$ gap, caliper-matched ($\\varepsilon$=0.05)")
y = np.arange(len(pdf))
for i, row in pdf.iterrows():
    ax.plot(row["dY_ci"], [i, i], color=st.COPYING, lw=2.4, solid_capstyle="round")
    ax.scatter([row["dY_gap"]], [i], color=st.COPYING, s=55, zorder=5, edgecolor="white", lw=0.8)
ax.axvline(0, color=st.BASELINE, lw=1)
macro_ci = dt.MACRO_CI["caliper_gap"]
ax.plot(macro_ci, [-1.3, -1.3], color=st.INK, lw=3.0, solid_capstyle="round")
ax.scatter([dt.MACRO["caliper_gap"]], [-1.3], color=st.INK, marker="D", s=70, zorder=6)
ax.set_yticks(list(y) + [-1.3])
ax.set_yticklabels(list(pdf["pair"]) + ["macro (all 9)"], fontsize=7.8)
ax.set_ylim(-2.0, len(pdf) + 0.2)
ax.set_xlabel("$\\Delta Y$ gap: copying $-$ shared")

# --------------------------------------------------------- b: estimator triangulation
ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "Three estimators, same direction")
long = pdf.melt(id_vars=["pair"], value_vars=["dY_gap", "nn_att", "mech_coef"],
                 var_name="estimator", value_name="value")
est_lab = {"dY_gap": "caliper", "nn_att": "NN-ATT", "mech_coef": "nested-LPM"}
long["estimator"] = long["estimator"].map(est_lab)
sns.stripplot(data=long, x="estimator", y="value", order=["caliper", "NN-ATT", "nested-LPM"],
              ax=ax, color=st.COPYING, alpha=0.55, jitter=0.12, size=6, zorder=2)
macro_pts = pd.DataFrame({
    "estimator": ["caliper", "NN-ATT", "nested-LPM"],
    "value": [dt.MACRO["caliper_gap"], dt.MACRO["nn_att"], dt.MACRO["mech_coef"]],
})
ax.scatter(macro_pts["estimator"], macro_pts["value"], color=st.INK, marker="D", s=90, zorder=6, label="macro")
ax.axhline(0, color=st.BASELINE, lw=1)
ax.set_xlabel("")
ax.set_ylabel("estimate")
ax.legend(loc="upper right")

# --------------------------------------------------------- c: common support
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "Common-support coverage by caliper")
cov = pdf.melt(id_vars=["pair"], value_vars=["coverage_eps02", "coverage_eps05", "coverage_eps10"],
               var_name="eps", value_name="coverage")
eps_lab = {"coverage_eps02": "0.02", "coverage_eps05": "0.05", "coverage_eps10": "0.10"}
cov["eps"] = cov["eps"].map(eps_lab)
sns.lineplot(data=cov, x="eps", y="coverage", units="pair", estimator=None, ax=ax,
             color=st.INK_MUTED, alpha=0.5, lw=1.1, marker="o", markersize=5,
             sort=False)
flagged = pdf[(pdf["model"] == "llama") & (pdf["bench"] == "aqua")].iloc[0]
flagged_y = [flagged["coverage_eps02"], flagged["coverage_eps05"], flagged["coverage_eps10"]]
ax.plot(["0.02", "0.05", "0.10"], flagged_y, color=st.CRITICAL, lw=2.4, marker="o", markersize=7, zorder=6)
ax.annotate("Llama x AQuA: 30% at $\\varepsilon$=0.02", xy=(0, flagged_y[0]), xytext=(0.15, 0.45),
            fontsize=8.6, color=st.CRITICAL, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=st.CRITICAL, lw=1))
ax.set_xlabel("caliper $\\varepsilon$")
ax.set_ylabel("coverage (fraction matched)")
ax.set_ylim(0, 1.1)

# --------------------------------------------------------- d: four-reading sensitivity
ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "d", "Sensitivity: all four noise-correction readings")
xs = [0, 1, 2, 3]
cols = ["dY_gap", "adj_caliper_gap", "caliper_copy_only", "caliper_both"]
for i, row in pdf.iterrows():
    ys = [row[c] for c in cols]
    ax.plot(xs, ys, color=st.INK_MUTED, lw=1.1, alpha=0.55, zorder=2, marker="o", markersize=3.5)
macro_vals = [dt.MACRO["caliper_gap"], dt.MACRO["adj_caliper_gap"],
              dt.MACRO["caliper_copy_only"], dt.MACRO["caliper_both"]]
ax.plot(xs, macro_vals, color=st.CRITICAL, lw=2.6, zorder=5)
ax.scatter(xs, macro_vals, color=st.CRITICAL, s=90, marker="D", zorder=6)
ax.axhline(0, color=st.BASELINE, lw=1)
ax.text(3.1, macro_vals[-1], f"macro both: {macro_vals[0]:.3f}$\\to${macro_vals[-1]:.3f}", fontsize=8.0,
        color=st.CRITICAL, va="center", fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(["raw", "shared-side\nonly", "copy-side\nonly", "both\n(combined)"], fontsize=8.2)
ax.set_xlim(-0.3, 4.3)
ax.set_ylabel("$\\Delta Y$ gap (caliper)")

st.savefig(fig, OUT)
print("wrote", OUT)
