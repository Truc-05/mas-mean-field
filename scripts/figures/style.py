"""Shared visual system for the paper's figure suite (figures/main_*, figures/appendix_*).

Design language: clean, modern seaborn-native statistical graphics in
EQUAL-SIZED panel grids (plt.subplots + constrained_layout, never hand-placed
axes rectangles -- that was the previous approach and it kept producing
overlap bugs). Panel tags are tight "a  Title" strings via ax.set_title with
mathtext bold on the letter only, so there is never a manual-offset
collision between the letter and its title.

Color system (fixed roles, never reassigned per-figure) -- built from the
user-supplied 9 (edge, face) pair palette below:
  - Mechanism identity (categorical, 2 values): shared evidence = pair(1)
    blue, copying = pair(8) orange.
  - Directional polarity: break = harmful = pair(3) coral; rescue =
    beneficial = pair(2) sage. Reserved; never reused as a generic series color.
  - Churn (wrong-to-wrong motion, invisible to accuracy): neutral grey.
  - Regime: easy/high-competence = pair(0) purple; hard/low-competence =
    pair(6) royal blue.
  - Susceptibility = pair(5) indigo. Repairability = pair(7) teal.
  - Model and benchmark identity: encoded by POSITION (facet rows/cols) and
    direct labels, never by hue -- six models and six benchmarks would blow
    past the categorical token ceiling.
  - Dependence magnitude D (continuous): sequential ramp built from the
    mechanism blue, used only for heatmap/gradient fills.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------- typography
for _name in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
    if any(_name == f.name for f in fm.fontManager.ttflist):
        FONT = _name
        break
else:
    FONT = "DejaVu Sans"

sns.set_theme(style="white", font=FONT)

matplotlib.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 500,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "font.family": FONT,
    "font.size": 9.6,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 9.6,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 8.6,
    "legend.frameon": False,
    "figure.titlesize": 13,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.unicode_minus": True,
})

# ------------------------------------------------- user palette: 9 (edge, face)
PAIRS = [
    ("#5C2E91", "#A77BC9"),  # 0 royal purple / lavender
    ("#1264C7", "#73C5F2"),  # 1 deep blue / cyan-light
    ("#5B9C90", "#A7CCC4"),  # 2 sage / light sage
    ("#E07A60", "#F4B698"),  # 3 coral / peach
    ("#E07060", "#F49E8E"),  # 4 salmon / light salmon
    ("#2A2870", "#D2C9E4"),  # 5 indigo / pale lavender
    ("#1F5BD0", "#9CD2EE"),  # 6 royal blue / sky
    ("#2A7682", "#B0DDE6"),  # 7 teal / pale aqua
    ("#E66D14", "#F4CFAE"),  # 8 orange / peach cream
]
EDGES = [e for e, _ in PAIRS]
FACES = [f for _, f in PAIRS]


def pair(i):
    return PAIRS[i % len(PAIRS)]


def edge(i):
    return EDGES[i % len(EDGES)]


def face(i):
    return FACES[i % len(FACES)]


GREY = "#7F7F7F"
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=EDGES)

# -------------------------------------------------------------------- roles
SURFACE = "#ffffff"
INK = "#1a1a1a"
INK_SECONDARY = "#4a4a4a"
INK_MUTED = "#7F7F7F"
GRID = "#e6e6e6"
BASELINE = "#a0a0a0"    # subtle in-chart zero/reference lines
AXIS_EDGE = "#333333"   # spines

SHARED = edge(1)        # mechanism: upstream shared evidence (deep blue)
SHARED_FACE = face(1)
COPYING = edge(8)       # mechanism: downstream copying (orange)
COPYING_FACE = face(8)

GOOD = edge(2)          # status: beneficial / rescue (sage)
GOOD_FACE = face(2)
CRITICAL = edge(3)      # status: harmful / break (coral)
CRITICAL_FACE = face(3)
CHURN = GREY            # neutral: wrong-to-wrong motion, invisible to accuracy

REGIME_EASY = edge(0)       # easy / high-competence (royal purple)
REGIME_EASY_FACE = face(0)
REGIME_HARD = edge(6)       # hard / low-competence (royal blue)
REGIME_HARD_FACE = face(6)

SUSCEPT_C = edge(5)     # indigo
REPAIR_C = edge(7)      # teal

DEP_RAMP = ["#eaf2fd", "#d6e8fb", face(1), "#4f9bdc", SHARED, "#0d4e94", "#082f58"]

MODEL_ORDER = ["qwen", "llama", "gemma", "mistral", "phi4", "deepseek"]
MODEL_LABEL = {
    "qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B", "gemma": "Gemma-2-9B",
    "mistral": "Mistral-7B", "phi4": "Phi-4-mini", "deepseek": "DeepSeek-R1-7B",
}
BENCH_ORDER = ["truthfulqa", "aqua", "arc", "commonsenseqa", "mmlu", "strategyqa"]
BENCH_LABEL = {
    "truthfulqa": "TruthfulQA", "aqua": "AQuA", "arc": "ARC-Challenge",
    "commonsenseqa": "CommonsenseQA", "mmlu": "MMLU", "strategyqa": "StrategyQA",
}


def dep_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("dep_seq", DEP_RAMP)


def seq_cmap(i=1):
    """Sequential colormap built from one palette pair: face(i) -> edge(i)."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f"seq{i}", ["#ffffff", face(i), edge(i)])


def div_cmap(neg_i=3, pos_i=1):
    """Diverging colormap built from two palette pairs (both edges), through
    white at the midpoint -- never matplotlib's default RdBu."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        f"div{neg_i}{pos_i}", [edge(neg_i), face(neg_i), "#ffffff", face(pos_i), edge(pos_i)])


# ---------------------------------------------------------------- layout ---
def grid(nrows, ncols, figsize, title_band=0.055, **kw):
    """Equal-sized panel grid, the default composition for every figure in
    the suite. constrained_layout prevents label/title/panel collisions
    automatically instead of hand-tuned rects. `title_band` reserves a thin
    strip at the top exclusively for suptitle()'s own text (title only, no
    subtitle now) so it can never collide with a panel title, while staying
    tight -- no wasted whitespace."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              constrained_layout=True, **kw)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 1.0 - title_band))
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.015, hspace=0.02)
    return fig, axes


def style_axes(ax, spine="left_bottom", grid_on=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_EDGE)
        ax.spines[side].set_linewidth(0.8)
    if spine == "none":
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.tick_params(colors=INK_SECONDARY, labelsize=8.6)
    if grid_on:
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)


def panel_title(ax, letter, title):
    """Tight 'a  Title' panel tag: bold letter, normal-weight title,
    immediately adjacent, rendered as ONE mathtext string via ax.set_title so
    there is no manual-offset collision risk."""
    ax.set_title(r"$\mathbf{" + letter + r"}$  " + title, loc="left",
                  fontsize=10.3, color=INK, pad=6)


def suptitle(fig, text, subtitle=None, title_band=0.055):
    """Short bold figure name only, inside the reserved top strip (see
    grid()'s title_band) -- no descriptive subtitle/caption sentence baked
    into the image; that belongs in the paper's own \\caption{}."""
    fig.text(0.008, 1.0 - title_band * 0.15, text, fontsize=15, fontweight="bold",
              color=INK, ha="left", va="top")


def savefig(fig, path_noext, dpi=500):
    fig.savefig(f"{path_noext}.pdf")
    fig.savefig(f"{path_noext}.png", dpi=dpi)
    plt.close(fig)
