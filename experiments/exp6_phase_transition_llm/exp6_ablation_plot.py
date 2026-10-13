"""Exp6 ablation: social copying ≠ common evidence — the headline figure.

Two mechanisms both drive measured vote-correlation ρ toward 1, yet produce OPPOSITE
collective dynamics:

  • plurality copying (run.py):  agents see the running vote and defer → the group
    locks onto the round-0 aggregate → lift → 0 no matter how high ρ climbs.
  • common evidence (exp6b):     agents each get a shared noisy evidence shock and
    reason independently → lift → p_single − P_maj(0), the theory's ρ→1 limit, with
    sign set by the crowd's competence (sign of ½ − p).

Same ρ, different causal structure, different outcome ⇒ (p, ρ) alone does NOT
characterise a collective. The correlation *coefficient* is not a sufficient
statistic; the mechanism generating the correlation is.

Run:  python experiments/exp6_phase_transition_llm/exp6_ablation_plot.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"


def _read(name: str) -> list[dict]:
    p = TABLES / name
    if not p.exists():
        print(f"[missing] {name}")
        return []
    with open(p) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _series(rows):
    return ([r["rho_measured"] for r in rows], [r["lift"] for r in rows])


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    # (label, csv, colour, marker, linestyle, filled)
    series = [
        ("common-shock  HARD (p≈0.22)", "exp6b_common_shock_hard_N11.csv",
         "#1f77b4", "o", "-", True),
        ("common-shock  EASY (p≈0.48)", "exp6b_common_shock_easy_N11.csv",
         "#d62728", "s", "-", True),
        ("copying  HARD (p≈0.19)", "exp6_phase_transition_llm_truthfulqa_N11_hard.csv",
         "#1f77b4", "o", "--", False),
        ("copying  EASY (p≈0.63)", "exp6_phase_transition_llm_truthfulqa_N11_easy.csv",
         "#d62728", "s", "--", False),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.axhline(0, color="k", lw=0.9, alpha=0.7)
    for label, name, color, mk, ls, filled in series:
        rows = _read(name)
        if not rows:
            continue
        x, y = _series(rows)
        order = np.argsort(x)
        x, y = np.asarray(x)[order], np.asarray(y)[order]
        ax.plot(x, y, ls, color=color, lw=1.8, alpha=0.85, zorder=3)
        ax.scatter(x, y, marker=mk, s=80, zorder=4, edgecolor="k", linewidth=0.8,
                   facecolor=color if filled else "white",
                   color=color if filled else color, label=label)

    ax.annotate("copying → lift collapses to 0\n(locked to round-0 aggregate)",
                xy=(0.97, 0.0), xytext=(0.60, 0.055), fontsize=8.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    ax.annotate("common shock → lift grows,\nsign = sign(½ − p)",
                xy=(0.71, -0.067), xytext=(0.40, -0.05), fontsize=8.5, color="#444",
                arrowprops=dict(arrowstyle="->", color="#888", lw=1))
    ax.set_xlabel("measured vote-correlation  ρ  (Fleiss ICC)")
    ax.set_ylabel("lift = collective_acc − independent_acc")
    ax.set_title("Exp6 ablation: social copying ≠ common evidence\n"
                 "same ρ, different causal structure ⇒ different collective outcome",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.93, ncol=1)
    fig.tight_layout()
    out = FIGS / "exp6_ablation_copying_vs_shock_N11.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
