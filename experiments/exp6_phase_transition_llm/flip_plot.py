"""Exp6 FLIP-law plot: empirical lift vs MEASURED vote-correlation ρ, per realized
regime, overlaid on the Condorcet-with-correlation prediction P_maj(N, p, ρ).

The claim under test is stronger than "a phase transition at some g*":

    sign(collective effect)  ≈  sign(½ − p)   as error correlation ρ increases

i.e. correlation is not intrinsically bad. For competent agents (p>½) herding
destroys the wisdom-of-crowds majority boost (lift<0); for systematically-wrong
agents (p<½) herding *helps* by dismantling the majority's amplification of a shared
error (lift>0). The two regimes should straddle the p=½ crossing and each sit on its
own theoretical curve evaluated at the subset's REALIZED single-agent accuracy p.

x-axis = rho_measured (Fleiss ICC of final-round correctness), NOT g and NOT collapse.
y-axis = lift = debate_acc − indep_acc, with the PAIRED bootstrap CI as error bars.

Run:  python experiments/exp6_phase_transition_llm/flip_plot.py --N 11
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory_overlay import p_maj  # same-directory import when run as a script

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"


def _read(name: str) -> list[dict]:
    path = TABLES / name
    if not path.exists():
        return []
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _theory_curve(N: int, p: float, rhos: np.ndarray) -> np.ndarray:
    base = p_maj(N, p, 0.0)
    return np.array([p_maj(N, p, float(r)) - base for r in rhos])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=11)
    ap.add_argument("--benchmark", default="truthfulqa")
    args = ap.parse_args(argv)
    FIGS.mkdir(parents=True, exist_ok=True)

    subsets = [
        ("hard", "HARD  (realized p<½)", "#1f77b4", "o"),
        ("easy", "EASY  (realized p>½)", "#d62728", "s"),
    ]
    rhos = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)

    for tag, label, color, mk in subsets:
        rows = _read(f"exp6_phase_transition_llm_{args.benchmark}_N{args.N}_{tag}.csv")
        if not rows:
            print(f"[skip] no CSV for {tag}")
            continue
        p = float(np.mean([r["p_single"] for r in rows]))       # realized marginal p
        ax.plot(rhos, _theory_curve(args.N, p, rhos), "-", color=color, lw=2, alpha=0.55,
                label=f"theory P_maj(N={args.N}, p={p:.2f})")

        x = np.array([r["rho_measured"] for r in rows])
        y = np.array([r["lift"] for r in rows])
        lo = np.array([r["lift"] - r["lift_ci_low"] for r in rows])
        hi = np.array([r["lift_ci_high"] - r["lift"] for r in rows])
        ax.errorbar(x, y, yerr=[lo, hi], fmt=mk, color=color, ms=9, capsize=4,
                    lw=1.4, mec="k", mew=0.8, zorder=5,
                    label=f"{label}  (empirical, p={p:.2f})")
        for r in rows:                                          # annotate each g
            ax.annotate(f"g={r['coupling']:.1f}", (r["rho_measured"], r["lift"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7, color=color)

    ax.set_xlabel("measured vote-correlation  ρ  (Fleiss ICC, final round)")
    ax.set_ylabel("lift = debate_acc − indep_acc")
    ax.set_title("Exp6 FLIP law: sign(collective effect) ≈ sign(½ − p) as ρ ↑\n"
                 "hard subset should rise (>0), easy subset should fall (<0)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    fig.tight_layout()
    out = FIGS / f"exp6_flip_lift_vs_rho_N{args.N}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
