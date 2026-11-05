"""Main figure: does the instrument raise measured dependence? Heatmap of
all 66 usable strata plus per-model / per-benchmark sign fractions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_dependence_instrument")

fig, axes = st.grid(2, 2, figsize=(11.5, 9.5), gridspec_kw={"height_ratios": [1.0, 0.8]})
st.suptitle(fig, "Does the instrument raise measured dependence?")

# ----------------------------------------------------------- a: strata heatmap
ax = axes[0, 0]
grid_map = dt.strata_grid()
rows_lab = [st.MODEL_LABEL[m] for m in st.MODEL_ORDER]
cols = [(b, r) for b in st.BENCH_ORDER for r in ("easy", "hard")]
col_labels = [f"{st.BENCH_LABEL[b]} ({'e' if r == 'easy' else 'h'})" for b, r in cols]
mat = np.full((6, 12), np.nan)
for i, m in enumerate(st.MODEL_ORDER):
    for j, (b, r) in enumerate(cols):
        key = (m, b, r)
        if key in grid_map:
            mat[i, j] = 1.0 if grid_map[key] else 0.0
from matplotlib.colors import ListedColormap
masked = np.ma.masked_invalid(mat)
cmap_a = ListedColormap([st.CRITICAL, st.SHARED])
cmap_a.set_bad(st.GRID)
sns.heatmap(masked, ax=ax, cmap=cmap_a, cbar=False,
            linewidths=1.2, linecolor="white", vmin=0, vmax=1,
            xticklabels=col_labels, yticklabels=rows_lab)
ax.set_yticklabels(rows_lab, rotation=0)
ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=7.4)
for j in range(0, 13, 2):
    ax.axvline(j, color="white", lw=2.2)
st.panel_title(ax, "a", "Every usable stratum (grey = below minimum $n_q$)")
ax.set_xlabel("")
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=st.SHARED, label="sign positive (63)"),
           mpatches.Patch(color=st.CRITICAL, label="sign negative (3)"),
           mpatches.Patch(color=st.GRID, label="below minimum $n_q$ (6)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False, fontsize=7.8)

# -------------------------------------------------------------- b: null density
ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "Observed count vs. simulated null")
null_mean, null_sd = dt.C3["null_mean"], dt.C3["null_sd"]
x = np.linspace(max(0, null_mean - 4.5 * null_sd), 66, 400)
y = np.exp(-0.5 * ((x - null_mean) / null_sd) ** 2)
ax.fill_between(x, 0, y, color=st.CHURN, alpha=0.3)
ax.plot(x, y, color=st.CHURN, lw=1.3)
ax.axvline(dt.C3["S_obs"], color=st.SHARED, lw=2.4, ymax=0.78)
ax.scatter([dt.C3["S_obs"]], [0.03], color=st.SHARED, marker="D", s=90, zorder=5)
ax.annotate(f"observed = {dt.C3['S_obs']}", xy=(dt.C3["S_obs"], 0.80), xytext=(dt.C3["S_obs"] - 20, 0.92),
            fontsize=10, color=st.SHARED, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=st.SHARED, lw=1))
ax.text(null_mean, 1.10, f"null mean {null_mean:.1f}  (sd {null_sd:.2f})", ha="center", fontsize=8.6,
        color=st.INK_SECONDARY)
ax.set_xlabel("strata with shared $>$ independent dependence (2000 simulated draws)")
ax.set_ylabel("density (normalized)")
ax.set_ylim(0, 1.25)
ax.set_yticks([])

# --------------------------------------------------------- c: per-model fraction
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "By model family")
mdf = pd.DataFrame([
    {"model": st.MODEL_LABEL[m], "fraction": dt.BY_MODEL[m][0] / dt.BY_MODEL[m][1],
     "label": f"{dt.BY_MODEL[m][0]}/{dt.BY_MODEL[m][1]}"}
    for m in st.MODEL_ORDER
])
sns.barplot(data=mdf, y="model", x="fraction", ax=ax, color=st.SHARED, orient="h")
for i, row in mdf.iterrows():
    ax.text(row["fraction"] + 0.02, i, row["label"], va="center", fontsize=8.3, color=st.INK)
ax.set_xlim(0, 1.18)
ax.set_xlabel("fraction of strata sign-positive")
ax.set_ylabel("")

# ------------------------------------------------------- d: per-benchmark fraction
ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "d", "By benchmark")
bdf = pd.DataFrame([
    {"bench": st.BENCH_LABEL[b], "fraction": dt.BY_BENCH[b][0] / dt.BY_BENCH[b][1],
     "label": f"{dt.BY_BENCH[b][0]}/{dt.BY_BENCH[b][1]}"}
    for b in st.BENCH_ORDER
])
sns.barplot(data=bdf, y="bench", x="fraction", ax=ax, color=st.SUSCEPT_C, orient="h")
for i, row in bdf.iterrows():
    ax.text(row["fraction"] + 0.02, i, row["label"], va="center", fontsize=8.3, color=st.INK)
ax.set_xlim(0, 1.18)
ax.set_xlabel("fraction of strata sign-positive")
ax.set_ylabel("")

st.savefig(fig, OUT)
print("wrote", OUT)
