"""Appendix: per-model report card -- excess churn by regime, net accuracy
by regime, instrument sign fraction, for all six model families.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_per_model")

rows = dt.cell_rows("strict")
df = pd.DataFrame(rows)

fig, axes = st.grid(2, 3, figsize=(13.0, 7.6))
st.suptitle(fig, "Per-model atlas")

for idx, m in enumerate(st.MODEL_ORDER):
    ax = axes[idx // 3, idx % 3]
    st.style_axes(ax)
    letter = "abcdef"[idx]
    hit, tot = dt.BY_MODEL[m]
    st.panel_title(ax, letter, f"{st.MODEL_LABEL[m]}  ({hit}/{tot} sign+)")
    sub = df[df.model == m]
    hard = sub[sub.regime == "hard"]["dC"]
    easy = sub[sub.regime == "easy"]["dC"]
    da_h = sub[sub.regime == "hard"]["dA"].mean()
    da_e = sub[sub.regime == "easy"]["dA"].mean()
    ax.bar([0], [hard.mean()], width=0.55, color=st.REGIME_HARD)
    ax.bar([1], [easy.mean()], width=0.55, color=st.REGIME_EASY)
    ax.axhline(0, color=st.BASELINE, lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"hard\n$\\Delta A$={da_h:+.3f}", f"easy\n$\\Delta A$={da_e:+.3f}"], fontsize=7.8)
    if idx % 3 == 0:
        ax.set_ylabel("mean $\\delta C$")
    ax.set_ylim(-0.04, 0.045)

st.savefig(fig, OUT)
print("wrote", OUT)
