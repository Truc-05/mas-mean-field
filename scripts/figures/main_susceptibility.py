"""Main figure: decision susceptibility to the margin gate -- slope by
regime, degeneracy, finite-size sweep, per-model saturation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_susceptibility")

fig, axes = st.grid(2, 2, figsize=(11.0, 9.0))
st.suptitle(fig, "The decision-boundary gate")

# ---------------------------------------------------------- a: regime slope forest
ax = axes[0, 0]
st.style_axes(ax)
st.panel_title(ax, "a", "Margin-susceptibility slope, by regime")
n_deg, n_tot = dt.SUSCEPT["high_degenerate"]
cats = ["low-competence", f"high-competence\n({n_deg}/{n_tot} degenerate)"]
means = [dt.SUSCEPT["low_slope"], dt.SUSCEPT["high_slope"]]
cis = [dt.SUSCEPT["low_ci"], dt.SUSCEPT["high_ci"]]
colors = [st.REGIME_HARD, st.REGIME_EASY]
y = np.arange(2)
for i in range(2):
    ax.plot(cis[i], [y[i], y[i]], color=colors[i], lw=2.6, solid_capstyle="round")
    ax.scatter([means[i]], [y[i]], color=colors[i], s=70, zorder=5, edgecolor="white", lw=1)
    ax.text(cis[i][1] + 0.006, y[i], f"{means[i]:.3f}", va="center", fontsize=9, color=colors[i], fontweight="bold")
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_yticks(y)
ax.set_yticklabels(cats)
ax.set_xlabel("slope of $|\\Delta A|$ on margin $M$")
ax.set_xlim(-0.30, 0.08)

# --------------------------------------------------------- b: degenerate fraction
ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "High-competence saturation, by model")
mdf = pd.DataFrame([
    {"model": st.MODEL_LABEL[m], "deg": dt.DEGENERATE_BY_MODEL[m][0], "n": dt.DEGENERATE_BY_MODEL[m][1]}
    for m in st.MODEL_ORDER
])
mdf["frac"] = mdf["deg"] / mdf["n"]
mdf = mdf.sort_values("frac")
sns.barplot(data=mdf, y="model", x="frac", ax=ax, color=st.REGIME_EASY, orient="h")
for i, row in enumerate(mdf.itertuples()):
    ax.text(row.frac + 0.02, i, f"{row.deg}/{row.n}", va="center", fontsize=8.3, color=st.INK)
ax.set_xlim(0, 1.15)
ax.set_xlabel("fraction of high-competence strata exactly degenerate")
ax.set_ylabel("")

# --------------------------------------------------------- c: finite-size sweep
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "Finite-size sweep (Qwen2.5-7B only)")
ax.plot(dt.FINITE_N["N"], dt.FINITE_N["slope"], color=st.SUSCEPT_C, lw=2, marker="o", markersize=9, zorder=5)
for n, s in zip(dt.FINITE_N["N"], dt.FINITE_N["slope"]):
    ax.text(n, s - 0.006, f"{s:.3f}", ha="center", va="top", fontsize=8.6, color=st.SUSCEPT_C, fontweight="bold")
ax.axhline(0, color=st.BASELINE, lw=1)
ax.set_xticks(dt.FINITE_N["N"])
ax.set_xlabel("population size $N$")
ax.set_ylabel("slope of $|\\Delta A|$ on margin")
ax.set_ylim(-0.16, 0.02)

# --------------------------------------------------------- d: significance audit
ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "d", "Per-stratum significance flags disagree")
audit = pd.DataFrame([
    {"regime": "hard", "method": "bootstrap", "frac": dt.SUSCEPT["perm_audit_hard"]["bootstrap_sig"] / dt.SUSCEPT["perm_audit_hard"]["n"]},
    {"regime": "hard", "method": "permutation", "frac": dt.SUSCEPT["perm_audit_hard"]["perm_sig"] / dt.SUSCEPT["perm_audit_hard"]["n"]},
    {"regime": "easy", "method": "bootstrap", "frac": dt.SUSCEPT["perm_audit_easy"]["bootstrap_sig"] / dt.SUSCEPT["perm_audit_easy"]["n"]},
    {"regime": "easy", "method": "permutation", "frac": dt.SUSCEPT["perm_audit_easy"]["perm_sig"] / dt.SUSCEPT["perm_audit_easy"]["n"]},
])
sns.barplot(data=audit, x="regime", y="frac", hue="method", ax=ax,
            palette=[st.INK, st.INK_MUTED])
ax.set_ylabel("fraction individually significant")
ax.set_xlabel("")
ax.set_ylim(0, 0.235)
ax.legend(title="", loc="upper left")
ax.text(0.98, 0.95, "macro slope does not\ndepend on these flags", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.6, color=st.INK_MUTED, style="italic")

st.savefig(fig, OUT)
print("wrote", OUT)
