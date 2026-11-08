"""Main figure: confirmatory margin-gated intervention (SGIR) -- does the
margin gate from Figure 5 actually pay off against matched-budget
alternatives, and how uniform is that gain across model/benchmark pairs.

House style matches the Figure-5 redesign: big Times New Roman-compatible
serif type, pure-black text, light reference gridlines, zebra-striped forest
rows, halo-highlighted emphasis, bold value callouts. No shared figure-level
caption (that lives in pp.tex's own \\caption{}).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
import matplotlib.patheffects as pe
import style as st
import data as dt

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

_orig_style_axes = st.style_axes


def panel_title(ax, letter, title):
    ax.set_title(f"{letter})  {title}", loc="left", fontsize=22,
                  fontweight="bold", color=_BLACK, pad=14)


def style_axes(ax, spine="left_bottom", grid_on=False):
    _orig_style_axes(ax, spine=spine, grid_on=grid_on)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_BLACK)
        ax.spines[side].set_linewidth(1.3)
    ax.tick_params(colors=_BLACK, labelsize=18, width=1.3, length=6, pad=7)


def zebra_rows(ax, y_positions, half_height=0.42):
    for j, yv in enumerate(sorted(y_positions)):
        if j % 2 == 0:
            ax.axhspan(yv - half_height, yv + half_height, color=_ZEBRA, zorder=0, lw=0)


st.panel_title = panel_title
st.style_axes = style_axes

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "main_sgir_policy")

_BOTTOM_MARGIN = 0.065
fig, axes = st.grid(1, 2, figsize=(18.0, 8.9), title_band=0.0)
fig.get_layout_engine().set(rect=(0.0, _BOTTOM_MARGIN, 1.0, 1.0 - _BOTTOM_MARGIN))

PP = 100.0  # fraction -> percentage points

# ------------------------------------------------------ a: pooled policy gain
ax = axes[0]
st.style_axes(ax)
st.panel_title(ax, "a", "The gate beats every matched-budget comparator")

rows_a = dt.SGIR_DELTAS  # already ordered RANDOM / KEEP / RERUN-ALL
y = np.arange(len(rows_a))[::-1]
zebra_rows(ax, y)
ax.xaxis.grid(True, color=_GRID, lw=1.0, zorder=0)
ax.set_axisbelow(True)

thr = dt.SGIR["practical_threshold_pp"]
ax.axvline(thr, color=st.INK_MUTED, lw=2.0, ls=(0, (6, 4)), zorder=2)
ax.text(thr - 0.12, y.max() + 0.55, f"{thr:.0f} pp practical threshold", ha="right",
         va="bottom", fontsize=16, color=st.INK_MUTED, fontstyle="italic")

for i, row in enumerate(rows_a):
    pt = row["point"] * PP
    lo, hi = row["ci"][0] * PP, row["ci"][1] * PP
    ax.plot([lo, hi], [y[i]] * 2, color=st.COPYING, lw=4.4, solid_capstyle="round",
             zorder=4, path_effects=HALO)
    ax.scatter([pt], [y[i]], color=st.COPYING, s=220, zorder=5, edgecolor="white", lw=1.8)
    ax.text(hi + 0.10, y[i], f"+{pt:.2f} pp", va="center", fontsize=18,
             fontweight="bold", color=_BLACK)

ax.axvline(0, color=_BLACK, lw=1.4, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([r["label"] for r in rows_a])
ax.set_ylim(-0.7, 3.1)
ax.set_xlim(-0.3, 6.6)
ax.set_xlabel("accuracy difference (pp)")

# ------------------------------------------------- b: heterogeneity by pair
ax = axes[1]
st.style_axes(ax)
st.panel_title(ax, "b", "Gains concentrate in Qwen and Llama, not Gemma")

order = []
for m in ("qwen", "llama", "gemma"):
    grp = [p for p in dt.SGIR_PAIRS if p["model"] == m]
    grp.sort(key=lambda p: -p["delta_gate_random"])
    order.extend(grp)

y2 = np.arange(len(order))[::-1]
zebra_rows(ax, y2)
ax.xaxis.grid(True, color=_GRID, lw=1.0, zorder=0)
ax.set_axisbelow(True)

for i, p in enumerate(order):
    pt = p["delta_gate_random"] * PP
    lo, hi = p["delta_gate_random_ci"][0] * PP, p["delta_gate_random_ci"][1] * PP
    sig = lo > 0 or hi < 0
    color = st.COPYING if sig else st.INK_MUTED
    alpha = 1.0 if sig else 0.55
    ax.plot([lo, hi], [y2[i]] * 2, color=color, alpha=alpha, lw=3.4, solid_capstyle="round", zorder=4)
    ax.scatter([pt], [y2[i]], color=color, alpha=alpha, s=140, zorder=5, edgecolor="white", lw=1.4)
    sign = "+" if pt >= 0 else "−"
    ax.text(max(hi, 0) + 0.18, y2[i], f"{sign}{abs(pt):.2f}", va="center", fontsize=15,
             fontweight="bold" if sig else "normal", color=_BLACK if sig else st.INK_MUTED)

ax.axvline(0, color=_BLACK, lw=1.4, zorder=3)
ax.set_yticks(y2)
ax.set_yticklabels([f"{st.MODEL_LABEL[p['model']]} · {st.BENCH_LABEL[p['bench']]}" for p in order],
                     fontsize=15.5)
ax.set_ylim(-0.7, 8.7)
ax.set_xlim(-2.6, 5.4)
ax.set_xlabel("Gate $-$ RANDOM (pp)")

# matches pp.tex Sec. 5.4 prose exactly: 675/2880 test episodes (not the
# eligible-only denominator behind the JSON's intervention_fraction field)
intervened_frac = dt.SGIR["k_test"] / dt.SGIR["n_test_episodes"]
footer = (f"{intervened_frac*100:.1f}% episodes intervened"
          f"   ·   {dt.SGIR_CALLS_SAVED_FRAC*100:.1f}% fewer extra calls than RERUN-ALL")
fig.text(0.5, 0.012, footer, ha="center", va="bottom", fontsize=19, fontweight="bold", color=_BLACK)

st.savefig(fig, OUT)
print("wrote", OUT)
