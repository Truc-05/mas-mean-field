"""Appendix: per-pair (not just macro) comparison of the three
matched-dependence estimators.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_estimator_comparison")

pdf = pd.DataFrame(dt.PAIRS)
pdf["pair"] = [f"{st.MODEL_LABEL[r.model]} x {st.BENCH_LABEL[r.bench]}" for r in pdf.itertuples()]

fig, axes = st.grid(1, 1, figsize=(9.5, 6.4))
ax = axes
st.suptitle(fig, "Estimator comparison atlas")
st.style_axes(ax)
st.panel_title(ax, "", "Caliper, NN-ATT, and nested-LPM estimates, per pair")

n = len(pdf)
y = np.arange(n)
markers = {"dY_gap": ("o", "caliper"), "nn_att": ("s", "NN-ATT"), "mech_coef": ("D", "nested-LPM")}
for i, row in pdf.iterrows():
    vals = [row["dY_gap"], row["nn_att"], row["mech_coef"]]
    ax.plot([min(vals), max(vals)], [i, i], color=st.COPYING, lw=1.0, alpha=0.4, zorder=2)
    for key, (mk, lab) in markers.items():
        ax.scatter([row[key]], [i], marker=mk, color=st.COPYING, s=48, zorder=4,
                   edgecolor="white", lw=0.6, label=lab if i == 0 else "_nolegend_")
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(y)
ax.set_yticklabels(pdf["pair"], fontsize=9.2)
ax.set_xlabel("mechanism-gap estimate")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False)

st.savefig(fig, OUT)
print("wrote", OUT)
