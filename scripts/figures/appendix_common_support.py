"""Appendix: full caliper-coverage table for all 9 pairs at all 3 calipers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_common_support")

pdf = pd.DataFrame(dt.PAIRS)
pdf["pair"] = [f"{st.MODEL_LABEL[r.model]} x {st.BENCH_LABEL[r.bench]}" for r in pdf.itertuples()]
mat = pdf[["coverage_eps02", "coverage_eps05", "coverage_eps10"]].values
mat_pct = mat * 100

fig, axes = st.grid(1, 1, figsize=(8.6, 6.0))
ax = axes
st.suptitle(fig, "Common-support atlas")
sns.heatmap(mat_pct, ax=ax, cmap=st.seq_cmap(1), vmin=0, vmax=100, annot=True, fmt=".0f",
            linewidths=1.4, linecolor="white", cbar_kws={"label": "coverage %", "shrink": 0.8},
            yticklabels=pdf["pair"], xticklabels=["$\\varepsilon$=0.02", "$\\varepsilon$=0.05", "$\\varepsilon$=0.10"],
            annot_kws={"fontsize": 9.5})
st.panel_title(ax, "", "Coverage (% of copying observations matched), all 9 pairs")
ax.set_yticklabels(pdf["pair"], rotation=0, fontsize=9.5)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

st.savefig(fig, OUT)
print("wrote", OUT)
