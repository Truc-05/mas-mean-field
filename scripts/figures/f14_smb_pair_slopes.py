"""
Figure 5 -- Susceptibility-benefit map.

Purpose
-------
Show that the absolute margin-susceptibility association is larger than the
absolute margin-signed-benefit association in every one of the nine
model x benchmark pairs.

Design
------
Each model x benchmark pair is a single point in the plane

    x = |beta_B|   (signed benefit)
    y = |beta_S|   (susceptibility)

The line y = x marks where the two associations would be equal. The claim
of the section -- susceptibility tracks margin more than benefit does -- is
exactly the geometric statement that every point lies above that line. The
region above y = x is tinted faintly to make that reading immediate; no bars,
no per-pair dumbbells, no repeated per-facet axes.

Encoding
--------
    shape  = model       (survives grayscale/printing)
    color  = benchmark

9/9 points fall above the diagonal; the macro paired contrast (equal-pair
mean of |beta_S| - |beta_B|) is reported once, as a compact top-left callout.

Input
-----
results/tables/smb_susceptibility_vs_benefit.json

Output
------
figures/f14_smb_pair_slopes.pdf
figures/f14_smb_pair_slopes.png
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
import style as st


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABLE = os.path.join(ROOT, "results", "tables", "smb_susceptibility_vs_benefit.json")
OUT = os.path.join(ROOT, "figures", "f14_smb_pair_slopes")


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

with open(TABLE, "r", encoding="utf-8") as f:
    data = json.load(f)

per_pair = data["per_pair"]
macro = data["macro"]["paired_diff_absS_minus_absB"]

diff_pt = float(macro["point"])
diff_lo, diff_hi = map(float, macro["ci"])

MODELS = ["qwen", "llama", "gemma"]
MODEL_LABELS = {"qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B", "gemma": "Gemma-2-9B"}
MODEL_SHORT = {"qwen": "Q", "llama": "L", "gemma": "Gem"}
MODEL_MARKER = {"qwen": "o", "llama": "^", "gemma": "s"}

BENCHMARKS = [("truthfulqa", "TruthfulQA"), ("strategyqa", "StrategyQA"), ("aqua", "AQuA")]
BENCH_SHORT = {"truthfulqa": "TQA", "strategyqa": "SQA", "aqua": "AQuA"}

# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

expected = {f"{model}|{bench}" for model in MODELS for bench, _ in BENCHMARKS}
missing = expected.difference(per_pair)
if missing:
    raise KeyError("Missing SMB model x benchmark pairs: " + ", ".join(sorted(missing)))

points = []  # (model, bench, benefit, susceptibility)
for model in MODELS:
    for bench, _ in BENCHMARKS:
        row = per_pair[f"{model}|{bench}"]
        benefit = abs(float(row["slope_B"]))
        susceptibility = abs(float(row["slope_S"]))
        if not (0 <= benefit <= 1 and 0 <= susceptibility <= 1):
            raise ValueError(f"Unexpected slope magnitude for {model}|{bench}")
        points.append((model, bench, benefit, susceptibility))

above = sum(s > b for _, _, b, s in points)
if above != 9:
    raise AssertionError(f"Figure claim violated: only {above}/9 pairs have |beta_S| > |beta_B|.")


# ---------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------

available_fonts = {font.name for font in fm.fontManager.ttflist}
for candidate in ("Times New Roman", "Nimbus Roman", "Liberation Serif", "Times"):
    if candidate in available_fonts:
        SERIF = candidate
        break
else:
    SERIF = "serif"

plt.rcParams.update({
    "font.family": SERIF,
    "font.size": 9.3,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------

INK = st.INK
MUTED = st.INK_MUTED
SHADE_C = st.SUSCEPT_C
DIAG_C = "#B6B9BD"

BENCH_COLOR = {
    "truthfulqa": st.edge(3),   # coral
    "strategyqa": "#4C72B0",    # muted steel blue (softer than the raw palette royal blue)
    "aqua": st.edge(2),         # sage
}


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

X_MAX, Y_MAX = 0.335, 0.80

fig, ax = plt.subplots(figsize=(4.55, 4.95))
fig.subplots_adjust(left=0.135, right=0.97, top=0.90, bottom=0.21)

ax.set_xlim(0, X_MAX)
ax.set_ylim(0, Y_MAX)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# -------------------------------------------------- susceptibility-dominant tint
diag_x = np.linspace(0, X_MAX, 2)
ax.fill_between(diag_x, diag_x, Y_MAX, color=SHADE_C, alpha=0.065, zorder=0, linewidth=0)

# ------------------------------------------------------------------ y = x line
ax.plot([0, X_MAX], [0, X_MAX], color=DIAG_C, lw=1.0, ls=(0, (4, 2.5)), zorder=1)
ax.text(X_MAX * 0.97, X_MAX * 0.97, r"$y=x$", fontsize=8.0, color=MUTED,
        ha="right", va="bottom", style="italic")

# ---------------------------------------------------------------------- points
rng_offsets = {
    ("qwen", "strategyqa"): (0.010, 0.016),
    ("gemma", "aqua"): (0.010, -0.034),
}
for model, bench, b_val, s_val in points:
    ax.scatter(
        b_val, s_val, s=72, marker=MODEL_MARKER[model],
        facecolor=BENCH_COLOR[bench], edgecolor="white", linewidth=0.8,
        zorder=4,
    )
    dx, dy = rng_offsets.get((model, bench), (0.010, 0.016))
    ax.text(
        b_val + dx, s_val + dy, f"{MODEL_SHORT[model]}-{BENCH_SHORT[bench]}",
        fontsize=7.3, color=MUTED, ha="left", va="center", zorder=5,
    )

ax.set_xlabel(r"signed benefit  $|\beta_B|$", fontsize=10.0, color=INK)
ax.set_ylabel(r"susceptibility  $|\beta_S|$", fontsize=10.0, color=INK)
ax.set_xticks([0.0, 0.1, 0.2, 0.3])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.tick_params(colors=MUTED, labelsize=8.4, length=3.0)

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#9a9a9a")
    ax.spines[side].set_linewidth(0.8)

# ---------------------------------------------------------------------- legend
model_handles = [
    Line2D([0], [0], marker=MODEL_MARKER[m], linestyle="none", markerfacecolor="#6b6b6b",
           markeredgecolor="white", markeredgewidth=0.6, markersize=6.6, label=MODEL_LABELS[m])
    for m in MODELS
]
bench_handles = [
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=BENCH_COLOR[k],
           markeredgecolor="white", markeredgewidth=0.6, markersize=6.6, label=lbl)
    for k, lbl in BENCHMARKS
]

leg1 = ax.legend(handles=model_handles, title="model", loc="lower right",
                  bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=7.4,
                  title_fontsize=7.6, handletextpad=0.4, borderaxespad=0.3,
                  labelspacing=0.35)
ax.add_artist(leg1)
ax.legend(handles=bench_handles, title="benchmark", loc="lower right",
          bbox_to_anchor=(1.0, 0.185), frameon=False, fontsize=7.4,
          title_fontsize=7.6, handletextpad=0.4, borderaxespad=0.3,
          labelspacing=0.35)

# --------------------------------------------------------------- macro callout
ax.text(0.025, 0.975, "9 / 9 above equality", transform=ax.transAxes,
        fontsize=9.0, color=INK, ha="left", va="top")
ax.text(0.025, 0.915, "mean vertical excess", transform=ax.transAxes,
        fontsize=7.8, color=MUTED, ha="left", va="top")
ax.text(0.025, 0.845, f"+{diff_pt:.3f}", transform=ax.transAxes,
        fontsize=16.5, fontweight="bold", color=SHADE_C, ha="left", va="top")
ax.text(0.025, 0.775, f"95% CI [{diff_lo:.3f}, {diff_hi:.3f}]", transform=ax.transAxes,
        fontsize=7.6, color=MUTED, ha="left", va="top")

# ---------------------------------------------------------------------- export
fig.savefig(f"{OUT}.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig(f"{OUT}.png", dpi=500, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

print(f"wrote {OUT}.pdf")
print(f"wrote {OUT}.png")
print(f"verified: {above}/9 pairs satisfy |beta_S| > |beta_B|")
