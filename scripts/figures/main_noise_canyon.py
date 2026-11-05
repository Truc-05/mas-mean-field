"""Main figure: how much of the raw decision-change rate is resampling
noise? Per-regime breakdown, per-cell distribution, significance count,
tie-convention sensitivity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_noise_canyon")

rows = dt.cell_rows("strict")
df = pd.DataFrame(rows)
df["regime_lab"] = df["regime"].map({"hard": "hard / low-comp.", "easy": "easy / high-comp."})

fig, axes = st.grid(2, 2, figsize=(11.0, 9.0))
st.suptitle(fig, "The noise canyon")

# ------------------------------------------------------- a: raw vs noise vs excess
ax = axes[0, 0]
st.style_axes(ax)
st.panel_title(ax, "a", "Raw rate vs. within-arm noise floor")
cats = ["hard / low-comp.", "easy / high-comp."]
raw = [dt.CHURN_STRICT_HARD["dY_cross_mean"], dt.CHURN_STRICT_EASY["dY_cross_mean"]]
noise = [dt.CHURN_STRICT_HARD["C_within_mean"], dt.CHURN_STRICT_EASY["C_within_mean"]]
dc = [dt.CHURN_STRICT_HARD["dC_mean"], dt.CHURN_STRICT_EASY["dC_mean"]]
dc_ci = [dt.CHURN_STRICT_HARD["dC_ci"], dt.CHURN_STRICT_EASY["dC_ci"]]
x = np.arange(2)
w = 0.35
ax.bar(x - w / 2, raw, width=w, color=st.CHURN, alpha=0.5, label="raw $\\Delta Y$")
ax.bar(x + w / 2, noise, width=w, color=st.INK_MUTED, alpha=0.9, label="noise floor $C_{within}$")
for i in range(2):
    ax.annotate(f"$\\delta C$ = {dc[i]:+.3f}\n[{dc_ci[i][0]:.3f}, {dc_ci[i][1]:.3f}]",
                 xy=(x[i], max(raw[i], noise[i])), xytext=(x[i], max(raw[i], noise[i]) + 0.045),
                 ha="center", fontsize=8.3, color=st.SHARED, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("decision-change rate")
ax.set_ylim(0, 0.37)
ax.legend(loc="upper right")

# ------------------------------------------------------- b: per-cell dY vs C_within
ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "Per-cell: raw rate vs. its own noise floor")
palette = {"hard / low-comp.": st.REGIME_HARD, "easy / high-comp.": st.REGIME_EASY}
sns.scatterplot(data=df, x="C_within", y="dY", hue="regime_lab", palette=palette,
                 ax=ax, s=42, alpha=0.8, edgecolor="white", linewidth=0.4, legend="brief")
lim = max(df["C_within"].max(), df["dY"].max()) * 1.08
ax.plot([0, lim], [0, lim], color=st.BASELINE, lw=1.1, zorder=1)
ax.text(lim * 0.62, lim * 0.70, "$\\Delta Y = C_{within}$\n($\\delta C=0$)", fontsize=7.6,
        color=st.INK_MUTED, rotation=36, ha="center")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("within-arm noise floor $C_{within}$")
ax.set_ylabel("raw decision-change rate $\\Delta Y$")
ax.legend(title="", loc="upper left", fontsize=8)

# --------------------------------------------------------- c: per-cell dC strip
ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "Per-cell excess churn (not pooled across regimes)")
sns.stripplot(data=df, x="regime_lab", y="dC", order=cats, ax=ax, palette=palette,
              hue="regime_lab", legend=False, jitter=0.22, alpha=0.65, size=5)
sns.boxplot(data=df, x="regime_lab", y="dC", order=cats, ax=ax, showcaps=True,
            boxprops=dict(facecolor="none", edgecolor=st.INK, linewidth=1.1),
            whiskerprops=dict(color=st.INK, linewidth=1.1), medianprops=dict(color=st.INK, linewidth=1.6),
            showfliers=False, width=0.35)
ax.axhline(0, color=st.BASELINE, lw=1)
ax.set_xlabel("")
ax.set_ylabel("per-cell excess churn $\\delta C$")

# --------------------------------------------------- d: significance + tie sensitivity
ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "d", "Individually significant in a minority of strata")
sig_df = pd.DataFrame([
    {"regime": "hard", "sig": dt.CHURN_STRICT_HARD["n_strata_dC_sig_individually"], "n": dt.CHURN_STRICT_HARD["n_strata"]},
    {"regime": "easy", "sig": dt.CHURN_STRICT_EASY["n_strata_dC_sig_individually"], "n": dt.CHURN_STRICT_EASY["n_strata"]},
])
sig_df["frac"] = sig_df["sig"] / sig_df["n"]
colors = [st.REGIME_HARD, st.REGIME_EASY]
bars = ax.bar(sig_df["regime"], sig_df["frac"], color=colors, width=0.5)
for i, row in sig_df.iterrows():
    ax.text(i, row["frac"] + 0.01, f"{row['sig']}/{row['n']}", ha="center", fontsize=9, color=st.INK, fontweight="bold")
ax.set_ylabel("fraction individually significant")
ax.set_ylim(0, 0.3)
ax.set_xlabel("")

st.savefig(fig, OUT)
print("wrote", OUT)
