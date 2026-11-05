"""Appendix: full 66/72-cell strata atlas, every (model, benchmark, regime)
combination with its sign outcome and question count.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import style as st
import data as dt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_strata_atlas")

grid_map = dt.strata_grid()
nq = {(s["model"], s["bench"], s["regime"]): s["n_q"] for s in dt.C3["strata"]}

fig, axes = st.grid(1, 1, figsize=(12.5, 6.2))
ax = axes
st.suptitle(fig, "Strata atlas: all 72 grid cells")

cols = [(b, r) for b in st.BENCH_ORDER for r in ("easy", "hard")]
col_labels = [f"{st.BENCH_LABEL[b]} ({'e' if r == 'easy' else 'h'})" for b, r in cols]
mat = np.full((6, 12), np.nan)
annot = np.empty((6, 12), dtype=object)
for i, m in enumerate(st.MODEL_ORDER):
    for j, (b, r) in enumerate(cols):
        key = (m, b, r)
        if key in grid_map:
            mat[i, j] = 1.0 if grid_map[key] else 0.0
            annot[i, j] = str(nq[key])
        else:
            annot[i, j] = ""
masked = np.ma.masked_invalid(mat)
cmap_a = ListedColormap([st.CRITICAL, st.SHARED])
cmap_a.set_bad(st.GRID)
sns.heatmap(masked, ax=ax, cmap=cmap_a, cbar=False, linewidths=1.4, linecolor="white",
            vmin=0, vmax=1, annot=annot, fmt="", annot_kws={"fontsize": 8, "color": "white"},
            xticklabels=col_labels, yticklabels=[st.MODEL_LABEL[m] for m in st.MODEL_ORDER])
ax.set_xticklabels(col_labels, rotation=50, ha="right", fontsize=8.6)
ax.set_yticklabels([st.MODEL_LABEL[m] for m in st.MODEL_ORDER], rotation=0, fontsize=9.2)
for j in range(0, 13, 2):
    ax.axvline(j, color="white", lw=2.6)
handles = [mpatches.Patch(color=st.SHARED, label="sign positive (63)"),
           mpatches.Patch(color=st.CRITICAL, label="sign negative (3)"),
           mpatches.Patch(color=st.GRID, label="below minimum $n_q$ (6)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)

st.savefig(fig, OUT)
print("wrote", OUT)
