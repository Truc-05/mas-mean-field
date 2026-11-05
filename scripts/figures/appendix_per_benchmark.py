"""Appendix: per-benchmark report card -- excess churn by regime, net
accuracy by regime, instrument sign fraction, for all six benchmarks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_per_benchmark")

rows = dt.cell_rows("strict")
df = pd.DataFrame(rows)

fig, axes = st.grid(2, 3, figsize=(13.0, 7.6))
st.suptitle(fig, "Per-benchmark atlas")

for idx, b in enumerate(st.BENCH_ORDER):
    ax = axes[idx // 3, idx % 3]
    st.style_axes(ax)
    letter = "abcdef"[idx]
    hit, tot = dt.BY_BENCH[b]
    st.panel_title(ax, letter, f"{st.BENCH_LABEL[b]}  ({hit}/{tot} sign+)")
    sub = df[df.benchmark == b]
    hard = sub[sub.regime == "hard"]["dC"]
    easy = sub[sub.regime == "easy"]["dC"]
    da_h = hard.index.size and sub[sub.regime == "hard"]["dA"].mean()
    da_e = easy.index.size and sub[sub.regime == "easy"]["dA"].mean()
    ax.bar([0], [hard.mean() if len(hard) else 0], width=0.55, color=st.SUSCEPT_C)
    ax.bar([1], [easy.mean() if len(easy) else 0], width=0.55, color=st.REGIME_EASY)
    ax.axhline(0, color=st.BASELINE, lw=0.9)
    ax.set_xticks([0, 1])
    lab_h = f"hard\n$\\Delta A$={da_h:+.3f}" if len(hard) else "hard\nn/a"
    lab_e = f"easy\n$\\Delta A$={da_e:+.3f}" if len(easy) else "easy\nn/a"
    ax.set_xticklabels([lab_h, lab_e], fontsize=7.8)
    if idx % 3 == 0:
        ax.set_ylabel("mean $\\delta C$")
    ax.set_ylim(-0.025, 0.035)

st.savefig(fig, OUT)
print("wrote", OUT)
