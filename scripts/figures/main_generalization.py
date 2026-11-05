"""Main figure: does the picture generalize across the full model x
benchmark grid?
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_generalization")

rows = dt.cell_rows("strict")
df = pd.DataFrame(rows)

fig, axes = st.grid(2, 2, figsize=(11.5, 9.5))
st.suptitle(fig, "Does the picture generalize?")

# --------------------------------------------------- a: excess churn heatmap
ax = axes[0, 0]
pivot = df[df.regime == "hard"].pivot_table(index="model", columns="benchmark", values="dC")
pivot = pivot.reindex(index=st.MODEL_ORDER, columns=st.BENCH_ORDER)
pivot.index = [st.MODEL_LABEL[m] for m in pivot.index]
pivot.columns = [st.BENCH_LABEL[b] for b in pivot.columns]
sns.heatmap(pivot, ax=ax, cmap=st.div_cmap(neg_i=3, pos_i=1), center=0, linewidths=1.2, linecolor="white",
            cbar_kws={"label": "$\\delta C$", "shrink": 0.75}, annot=True, fmt=".3f",
            annot_kws={"fontsize": 7})
st.panel_title(ax, "a", "Excess churn $\\delta C$, hard regime")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# --------------------------------------------------- b: accuracy heatmap
ax = axes[0, 1]
pivot2 = df[df.regime == "hard"].pivot_table(index="model", columns="benchmark", values="dA")
pivot2 = pivot2.reindex(index=st.MODEL_ORDER, columns=st.BENCH_ORDER)
pivot2.index = [st.MODEL_LABEL[m] for m in pivot2.index]
pivot2.columns = [st.BENCH_LABEL[b] for b in pivot2.columns]
lim = np.nanmax(np.abs(pivot2.values))
sns.heatmap(pivot2, ax=ax, cmap=st.div_cmap(neg_i=3, pos_i=2), center=0, vmin=-lim, vmax=lim, linewidths=1.2, linecolor="white",
            cbar_kws={"label": "$\\Delta A$", "shrink": 0.75}, annot=True, fmt=".3f",
            annot_kws={"fontsize": 7})
st.panel_title(ax, "b", "Net accuracy $\\Delta A$, hard regime")
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=38, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# --------------------------------------------------- c: per-model mean churn
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "Per-model mean $\\delta C$ (hard regime)")
mdf = pd.DataFrame([
    {"model": st.MODEL_LABEL[m], "dC": df[(df.model == m) & (df.regime == "hard")]["dC"].mean()}
    for m in st.MODEL_ORDER
]).sort_values("dC")
sns.barplot(data=mdf, y="model", x="dC", ax=ax, color=st.SHARED, orient="h")
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_xlabel("mean per-cell excess churn $\\delta C$")
ax.set_ylabel("")

# --------------------------------------------------- d: per-benchmark mean churn
ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "d", "Per-benchmark mean $\\delta C$ (hard regime)")
bdf = pd.DataFrame([
    {"bench": st.BENCH_LABEL[b], "dC": df[(df.benchmark == b) & (df.regime == "hard")]["dC"].mean()}
    for b in st.BENCH_ORDER
]).sort_values("dC")
sns.barplot(data=bdf, y="bench", x="dC", ax=ax, color=st.SUSCEPT_C, orient="h")
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_xlabel("mean per-cell excess churn $\\delta C$")
ax.set_ylabel("")

st.savefig(fig, OUT)
print("wrote", OUT)
