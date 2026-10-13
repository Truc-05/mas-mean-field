"""Exp 6 — theory overlay: the Condorcet-with-correlation predictor for `lift`.

The linear averaging coupling in orchestrator.py PRESERVES the population mean
(x̄* = s̄ for every g), so a *linear* readout can never erode truth — which is
exactly why the measured lift sits at ~0. Truth erosion lives in the NONLINEAR
plurality readout under coupling-induced vote correlation ρ(g). This module
implements that closed form and overlays it on the empirical exp6 curves.

Model (equicorrelated Condorcet jury; Boland 1989, Ladha 1992):
  agent i correct  iff  z_i = √ρ·Z0 + √(1-ρ)·ε_i ≤ τ,   τ = Φ⁻¹(p)
  q(Z0) = Φ((τ - √ρ·Z0)/√(1-ρ))                      # correct-prob given the shared shock
  P_maj(N,p,ρ) = E_{Z0}[ Pr(Bin(N, q(Z0)) > N/2) ]   # majority correct
  lift(ρ)      = P_maj(N,p,ρ) − P_maj(N,p,0)

Limits: ρ→0 → independent Condorcet boost; ρ→1 → P_maj → p (one effective vote).
Prediction: lift flips negative iff p > ½ (herding destroys the majority boost);
for p ≤ ½ it stays ≥ 0. p ≈ ½ (the regime exp6 ran in) → lift ≈ 0 → null result.

Run:  python experiments/exp6_phase_transition_llm/theory_overlay.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm, binom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

# Gauss–Hermite nodes for E_{Z0~N(0,1)}[f] = (1/√π) Σ w_i f(√2 x_i)
_GH_X, _GH_W = hermgauss(64)


def p_maj(N: int, p: float, rho: float) -> float:
    """Majority-correct probability of N equicorrelated jurors (marginal p, corr ρ)."""
    k0 = N // 2                                   # strict majority: need > N/2 correct
    if rho <= 0.0:
        return float(binom.sf(k0, N, p))          # independent Condorcet
    if rho >= 1.0:
        return float(p)                           # one effective vote
    tau = norm.ppf(p)
    z = np.sqrt(2.0) * _GH_X
    q = norm.cdf((tau - np.sqrt(rho) * z) / np.sqrt(1.0 - rho))
    integ = binom.sf(k0, N, q)                     # Pr(Bin(N,q) > N/2) per node
    return float((_GH_W * integ).sum() / np.sqrt(np.pi))


def lift_curve(N: int, p: float, rhos: np.ndarray) -> np.ndarray:
    base = p_maj(N, p, 0.0)
    return np.array([p_maj(N, p, r) - base for r in rhos])


def _read_csv(name: str) -> list[dict]:
    path = TABLES / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    rhos = np.linspace(0, 1, 101)
    ps = [0.35, 0.45, 0.55, 0.65, 0.75]
    Ns = [5, 11, 21]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(ps)))

    emp = {
        "TruthfulQA": (_read_csv("exp6_phase_transition_llm_truthfulqa.csv"), "o"),
        "AQuA": (_read_csv("exp6_phase_transition_llm_aqua.csv"), "s"),
    }
    # the p>½ flip run (N=11, easy subset) — plotted only on the N=11 panel
    easy11 = _read_csv("exp6_phase_transition_llm_truthfulqa_N11_easy0.6.csv")

    fig, axes = plt.subplots(1, len(Ns), figsize=(5 * len(Ns), 4.4), sharey=True)
    for ax, N in zip(axes, Ns):
        for p, c in zip(ps, colors):
            ls = "--" if p <= 0.5 else "-"
            ax.plot(rhos, lift_curve(N, p, rhos), ls, color=c, lw=2,
                    label=f"p={p:.2f}")
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)
        ax.set_title(f"N = {N} agents")
        ax.set_xlabel("vote correlation  ρ(g)")
        ax.grid(alpha=0.25)
        # overlay empirical points on every panel (their N=5) using collapse as ρ proxy
        for name, (rows, mk) in emp.items():
            if not rows:
                continue
            x = [max(0.0, float(r["mean_collapse"])) for r in rows]
            y = [float(r["lift"]) for r in rows]
            ax.scatter(x, y, marker=mk, s=70, edgecolor="k", facecolor="none",
                       linewidth=1.4, zorder=5,
                       label=f"{name} (exp6, N=5)" if N == Ns[0] else None)
        if N == 11 and easy11:
            x = [max(0.0, float(r["mean_collapse"])) for r in easy11]
            y = [float(r["lift"]) for r in easy11]
            ax.scatter(x, y, marker="*", s=240, color="crimson", edgecolor="k",
                       linewidth=1.0, zorder=6,
                       label="TruthfulQA easy p>½ (exp6, N=11)")
    axes[0].set_ylabel("lift = debate_acc − indep_acc")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.9)
    fig.suptitle("Exp6 phase transition: lift(ρ) flips negative iff p > ½  "
                 "(solid = p>½ herding hurts, dashed = p≤½ herding harmless)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGS / "exp6_theory_overlay.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # numeric summary: where does the flip land, and how deep, per N at p=0.65?
    print("\npredicted lift at ρ=1 (full herding), by (N, p):")
    print("  p   " + "  ".join(f"N={n:>2}" for n in Ns))
    for p in ps:
        cells = "  ".join(f"{p_maj(n,p,1.0)-p_maj(n,p,0.0):+5.2f}" for n in Ns)
        print(f"  {p:.2f} {cells}")


if __name__ == "__main__":
    main()
