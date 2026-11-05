"""Appendix: full distributions behind the easy/hard regime split -- why the
two are never pooled.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import style as st
import data as dt

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures", "appendix_easy_hard")

rows = dt.cell_rows("strict")
hard = [r for r in rows if r["regime"] == "hard"]
easy = [r for r in rows if r["regime"] == "easy"]

fig, axes = st.grid(2, 2, figsize=(11.0, 8.6))
st.suptitle(fig, "Easy / hard separation atlas")

ax = axes[0, 0]
st.style_axes(ax)
st.panel_title(ax, "a", "$\\delta C$ distribution by regime")
bins = np.linspace(-0.12, 0.09, 24)
ax.hist([r["dC"] for r in hard], bins=bins, color=st.REGIME_HARD, alpha=0.85, label="hard")
ax.hist([r["dC"] for r in easy], bins=bins, color=st.REGIME_EASY, alpha=0.85, label="easy")
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_xlabel("per-cell excess churn $\\delta C$")
ax.set_ylabel("cells")
ax.legend()

ax = axes[0, 1]
st.style_axes(ax)
st.panel_title(ax, "b", "$\\Delta A$ distribution by regime")
bins2 = np.linspace(-0.4, 0.1, 24)
ax.hist([r["dA"] for r in hard], bins=bins2, color=st.REGIME_HARD, alpha=0.85, label="hard")
ax.hist([r["dA"] for r in easy], bins=bins2, color=st.REGIME_EASY, alpha=0.85, label="easy")
ax.axvline(0, color=st.BASELINE, lw=1)
ax.set_xlabel("net accuracy effect $\\Delta A$")
ax.legend()

ax = axes[1, 0]
st.style_axes(ax)
st.panel_title(ax, "c", "High-competence degeneracy")
n_deg, n_tot = dt.SUSCEPT["high_degenerate"]
ax.bar([0], [n_deg], width=0.5, color=st.REGIME_EASY)
ax.bar([0], [n_tot - n_deg], width=0.5, bottom=[n_deg], color=st.GRID)
ax.text(0.4, n_deg / 2, f"{n_deg} degenerate\n($|\\Delta A|\\equiv0$)", fontsize=8.6, va="center", color=st.INK)
ax.text(0.4, n_deg + (n_tot - n_deg) / 2, f"{n_tot - n_deg} informative", fontsize=8.6, va="center", color=st.INK_SECONDARY)
ax.set_xlim(-0.8, 2.6)
ax.set_xticks([])
ax.set_ylabel("high-competence strata")

ax = axes[1, 1]
st.style_axes(ax)
st.panel_title(ax, "d", "Significance-flag disagreement")
groups = [("hard", dt.SUSCEPT["perm_audit_hard"]), ("easy", dt.SUSCEPT["perm_audit_easy"])]
x = np.arange(2)
boot = [g["bootstrap_sig"] / g["n"] for _, g in groups]
perm = [g["perm_sig"] / g["n"] for _, g in groups]
ax.bar(x - 0.18, boot, width=0.32, color=st.INK, label="bootstrap-sig.")
ax.bar(x + 0.18, perm, width=0.32, color=st.INK_MUTED, label="permutation-sig.")
ax.set_xticks(x)
ax.set_xticklabels(["hard (n=35)", "easy (n=31)"])
ax.set_ylabel("fraction individually significant")
ax.legend()

st.savefig(fig, OUT)
print("wrote", OUT)
