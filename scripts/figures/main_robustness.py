"""Main figure: robustness checks -- caliper width, estimator agreement,
noise-adjustment direction, tie-convention sensitivity.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.patheffects as pe
import style as st
import data as dt

# ---- figure-5-only override: big, black, Times New Roman-compatible type,
# analyst-style polish (gridlines, zebra bands, halo-highlighted summary
# lines), single horizontal row of 4 panels, no shared figure caption ----
_SERIF = "Nimbus Roman"  # metrically-compatible Times New Roman clone
_BLACK = "#000000"
_GRID = "#d9d9d9"
_ZEBRA = "#f2f2f2"
HALO = [pe.withStroke(linewidth=6.5, foreground="white")]
st.FONT = _SERIF
st.INK = _BLACK
st.INK_SECONDARY = _BLACK
st.AXIS_EDGE = _BLACK
matplotlib.rcParams.update({
    "font.family": _SERIF,
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.titlesize": 24,
    "text.color": _BLACK,
    "axes.labelcolor": _BLACK,
    "xtick.color": _BLACK,
    "ytick.color": _BLACK,
    "axes.labelpad": 10,
})
sns.set_theme(style="white", font=_SERIF)

_orig_style_axes = st.style_axes


def panel_title(ax, letter, title):
    """'a)  Title' in bold black, no mathtext (keeps the serif face)."""
    ax.set_title(f"{letter})  {title}", loc="left", fontsize=22,
                  fontweight="bold", color=_BLACK, pad=14)


def style_axes(ax, spine="left_bottom", grid_on=False):
    _orig_style_axes(ax, spine=spine, grid_on=grid_on)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BLACK)
        ax.spines[side].set_linewidth(1.3)
    ax.tick_params(colors=_BLACK, labelsize=18, width=1.3, length=6, pad=7)


def zebra_rows(ax, y_positions, half_height=0.42):
    """Light alternating background bands behind forest-plot rows, for
    professional at-a-glance scanability."""
    for j, yv in enumerate(sorted(y_positions)):
        if j % 2 == 0:
            ax.axhspan(yv - half_height, yv + half_height, color=_ZEBRA, zorder=0, lw=0)


st.panel_title = panel_title
st.style_axes = style_axes

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_robustness")

pdf = pd.DataFrame(dt.PAIRS)
fig, axes = st.grid(1, 4, figsize=(34.0, 8.2), title_band=0.0)

# ------------------------------------------------------ a: caliper width
ax = axes[0]
st.style_axes(ax)
st.panel_title(ax, "a", "Sign is stable across caliper width")
ax.yaxis.grid(True, color=_GRID, lw=1.0, zorder=0)
ax.set_axisbelow(True)
long = pdf.melt(id_vars=["model", "bench"], value_vars=["gap_eps02", "dY_gap", "gap_eps10"],
                 var_name="eps", value_name="gap")
eps_lab = {"gap_eps02": "0.02", "dY_gap": "0.05", "gap_eps10": "0.10"}
long["eps"] = long["eps"].map(eps_lab)
long["pair"] = long["model"] + " / " + long["bench"]
sns.lineplot(data=long, x="eps", y="gap", units="pair", estimator=None, ax=ax,
             color=st.COPYING, alpha=0.55, lw=1.8, marker="o", markersize=7,
             markeredgecolor="white", markeredgewidth=0.8, sort=False, zorder=3)
macro_by_eps = [pdf["gap_eps02"].mean(), pdf["dY_gap"].mean(), pdf["gap_eps10"].mean()]
ax.plot(["0.02", "0.05", "0.10"], macro_by_eps, color=_BLACK, lw=4.2,
         marker="D", markersize=13, zorder=6, path_effects=HALO,
         markeredgecolor="white", markeredgewidth=1.3)
ax.axhline(0, color=st.BASELINE, lw=1.3, zorder=1)
ax.set_xlabel("caliper $\\varepsilon$")
ax.set_ylabel("$\\Delta Y$ gap")
ax.margins(x=0.08)

# ------------------------------------------------------ b: estimator agreement
ax = axes[1]
st.style_axes(ax)
st.panel_title(ax, "b", "Matching method does not change the sign")
ests = ["caliper", "NN-ATT", "nested-LPM"]
vals = [dt.MACRO["caliper_gap"], dt.MACRO["nn_att"], dt.MACRO["mech_coef"]]
cis = [dt.MACRO_CI["caliper_gap"], dt.MACRO_CI["nn_att"], dt.MACRO_CI["mech_coef"]]
y = np.arange(3)
zebra_rows(ax, y)
ax.xaxis.grid(True, color=_GRID, lw=1.0, zorder=0)
ax.set_axisbelow(True)
for i in range(3):
    ax.plot(cis[i], [y[i]] * 2, color=st.COPYING, lw=4.0, solid_capstyle="round", zorder=4)
    ax.scatter([vals[i]], [y[i]], color=st.COPYING, s=190, zorder=5, edgecolor="white", lw=1.8)
    ax.text(cis[i][1] + 0.010, y[i], f"{vals[i]:.3f}", va="center", fontsize=18,
             fontweight="bold", color=_BLACK)
ax.axvline(0, color=st.BASELINE, lw=1.3, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels(ests)
ax.set_ylim(-0.6, 2.6)
ax.set_xlabel("mechanism-gap estimate")
ax.set_xlim(-0.02, 0.23)

# ------------------------------------------------------ c: noise-adjustment direction
ax = axes[2]
st.style_axes(ax)
st.panel_title(ax, "c", "Copy-side correction alone is the weak reading")
ax.yaxis.grid(True, color=_GRID, lw=1.0, zorder=0)
ax.set_axisbelow(True)
xs = [0, 1, 2, 3]
cols = ["dY_gap", "adj_caliper_gap", "caliper_copy_only", "caliper_both"]
for i, row in pdf.iterrows():
    ys = [row[c] for c in cols]
    ax.plot(xs, ys, color=st.INK_MUTED, lw=1.5, alpha=0.5, marker="o", markersize=5,
             markeredgecolor="white", markeredgewidth=0.6, zorder=3)
macro_vals = [dt.MACRO["caliper_gap"], dt.MACRO["adj_caliper_gap"],
              dt.MACRO["caliper_copy_only"], dt.MACRO["caliper_both"]]
ax.plot(xs, macro_vals, color=st.CRITICAL, lw=4.4, marker="D", markersize=13, zorder=6,
         path_effects=HALO, markeredgecolor="white", markeredgewidth=1.3)
ax.axhline(0, color=st.BASELINE, lw=1.3, zorder=1)
ax.set_xticks(xs)
ax.set_xticklabels(["raw", "shared-side\nonly", "copy-side\nonly", "both\n(combined)"], fontsize=18)
ax.set_xlim(-0.3, 3.3)
ax.set_ylabel("$\\Delta Y$ gap (caliper)")

# ------------------------------------------------------ d: tie-convention sensitivity
ax = axes[3]
st.style_axes(ax)
st.panel_title(ax, "d", "Tie-breaking convention does not change the picture")
rows_d = [
    ("hard, strict", dt.CHURN_STRICT_HARD["dC_mean"], dt.CHURN_STRICT_HARD["dC_ci"], st.REGIME_HARD, 1.0),
    ("hard, tiebreak", dt.CHURN_TIEBREAK_HARD["dC_mean"], dt.CHURN_TIEBREAK_HARD["dC_ci"], st.REGIME_HARD, 0.6),
    ("easy, strict", dt.CHURN_STRICT_EASY["dC_mean"], dt.CHURN_STRICT_EASY["dC_ci"], st.REGIME_EASY, 1.0),
    ("easy, tiebreak", dt.CHURN_TIEBREAK_EASY["dC_mean"], dt.CHURN_TIEBREAK_EASY["dC_ci"], st.REGIME_EASY, 0.6),
]
y = np.arange(len(rows_d))[::-1]
zebra_rows(ax, y)
ax.xaxis.grid(True, color=_GRID, lw=1.0, zorder=0)
ax.set_axisbelow(True)
for i, (lab, mean, ci, color, alpha) in enumerate(rows_d):
    ax.plot(ci, [y[i]] * 2, color=color, alpha=alpha, lw=3.8, solid_capstyle="round", zorder=4)
    ax.scatter([mean], [y[i]], color=color, alpha=alpha, s=150, zorder=5, edgecolor="white", lw=1.5)
    ax.text(ci[1] + 0.0006, y[i], f"{mean:.3f}", va="center", fontsize=16,
             fontweight="bold", color=_BLACK, alpha=max(alpha, 0.85))
ax.axvline(0, color=st.BASELINE, lw=1.3, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows_d])
ax.set_ylim(-0.7, 3.7)
ax.set_xlim(0, 0.028)
ax.set_xlabel("excess churn $\\delta C$")

st.savefig(fig, OUT)
print("wrote", OUT)
