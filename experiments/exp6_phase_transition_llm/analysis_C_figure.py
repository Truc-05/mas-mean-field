"""F2 (main figure) — mechanism-magnitude-consequence: ΔY = f(D, mechanism).

If dependence magnitude D were a sufficient statistic, upstream shared-evidence and
downstream copying would lie on ONE curve ΔY=f(D). They do not: two separated response
curves ⇒ ΔY=f(D, M). This is the visual companion to analysis_C (matched-D falsification:
copy−shared ΔY gap +0.08 survives ε=.02 caliper; nested-F for mechanism|D p=.007).

Reuses the row extractors from analysis_C so the figure and the stats share one substrate.
Offline, NO LLM. Writes results/figures/exp6c_mechanism_consequence_F2.png.

CVD-safe Okabe-Ito categorical pair (blue=shared, vermilion=copy) + secondary encoding
(marker shape o/s, line solid/dashed) so identity is never color-alone.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_C_matched_D_falsification import shared_rows, copy_rows

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "results" / "figures"

BLUE, VERM = "#0072B2", "#D55E00"          # Okabe-Ito, CVD-safe
EDGES = np.array([0.5, 0.65, 0.75, 0.85, 0.92, 0.97, 1.0001])


def binned(rows, y="dY"):
    D = np.array([r["D"] for r in rows]); Y = np.array([r[y] for r in rows])
    ok = np.isfinite(D) & np.isfinite(Y); D, Y = D[ok], Y[ok]
    xc, ym, ylo, yhi, ns = [], [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (D >= lo) & (D < hi)
        if sel.sum() < 5:
            continue
        v = Y[sel]; m = v.mean(); se = v.std(ddof=1) / np.sqrt(sel.sum())
        xc.append(D[sel].mean()); ym.append(m)
        ylo.append(m - 1.96 * se); yhi.append(m + 1.96 * se); ns.append(int(sel.sum()))
    return map(np.array, (xc, ym, ylo, yhi)), ns


def main():
    sh, cp = shared_rows(), copy_rows()
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)

    for rows, color, mk, ls, lab, dx in [
        (sh, BLUE, "o", "-",  "shared evidence (upstream)", -0.004),
        (cp, VERM, "s", "--", "social copying (downstream)", +0.004),
    ]:
        # raw scatter (recessive), then binned mean curve with 95% CI band
        Dr = np.array([r["D"] for r in rows]); Yr = np.array([r["dY"] for r in rows])
        ax.scatter(Dr, Yr, s=14, color=color, alpha=0.12, edgecolor="none", zorder=1)
        (xc, ym, ylo, yhi), ns = binned(rows)
        ax.fill_between(xc, ylo, yhi, color=color, alpha=0.18, zorder=2)
        ax.plot(xc, ym, ls, color=color, lw=2.0, zorder=3)
        ax.scatter(xc, ym, marker=mk, s=70, facecolor=color, edgecolor="k",
                   linewidth=0.8, zorder=4, label=lab)

    ax.annotate("same dependence D → different consequence\n"
                "(mechanism gap survives matched-D caliper ε=.02; F-test p=.007)",
                xy=(0.90, 0.16), xytext=(0.55, 0.42), fontsize=8.5, color="#333",
                ha="left", arrowprops=dict(arrowstyle="->", color="#888", lw=1))

    ax.set_xlabel("inter-agent dependence  D  (mean pairwise vote-agreement)")
    ax.set_ylabel(r"decision change  $\Delta Y = P(Y_{\mathrm{corr}} \neq Y_{\mathrm{indep}})$")
    ax.set_title("Mechanism, not magnitude, governs collective consequence", fontsize=12)
    ax.set_xlim(0.48, 1.02); ax.set_ylim(-0.02, None)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ccc", lw=0.5, alpha=0.5)
    fig.tight_layout()
    out = FIGS / "exp6c_mechanism_consequence_F2.png"
    fig.savefig(out, dpi=150); print(f"wrote {out}")


if __name__ == "__main__":
    main()
