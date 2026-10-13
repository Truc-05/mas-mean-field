"""Exp6B — controlled common-shock on REAL LLM beliefs (offline, no new LLM calls).

The g-sweep (run.py) falsified an implicit assumption: prompt-induced *plurality
copying* is NOT an empirical implementation of the equicorrelated latent-shock model.
Copying locks the collective onto the round-0 aggregate, so even as measured ICC→1 the
lift →0 — the theory's ρ→1 limit (lift → p − P_maj(N,p,0)) is never reached.

This experiment keeps the theory fixed and swaps in the RIGHT generative mechanism.
Each agent's real round-0 belief s_i (from qwen2.5:7b, canonical coords, truth at 0)
is perturbed by a shock whose cross-agent correlation is set experimentally:

    s̃_i = s_i + λ·( √ρ · z_shared  +  √(1-ρ) · z_i ),   z_shared common to all N agents.

Votes = argmax s̃_i, plurality over them. Correlation now arises because agents share
one evidence source z_shared — the FLIP theory's actual assumption — NOT because agent
i copies agent j. Private beliefs stay real; only the dependence structure is
manipulated. No answer label is ever injected.

Prediction (competence–correlation transition): as measured ρ rises, plurality
accuracy → p_single (one effective vote), so
    lift = P_maj(N,p,ρ) − P_maj(N,p,0)   flips sign at p = ½.
Empirical (ρ, plurality_acc) points should lie ON the analytic P_maj(N, p, ·) curve —
the copying run's points do not. That contrast is the headline: same ρ, different
causal structure ⇒ different collective outcome, so (p,ρ) alone is not sufficient.

Exp6A (--sanity) first checks the simulator against the closed form on synthetic
Bernoulli jurors. Run:
    python experiments/exp6_phase_transition_llm/exp6b_common_shock.py --sanity
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from theory_overlay import p_maj

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

# subset → (selection-cache json, #candidates the cache was built from, plot colour/marker)
SUBSETS = {
    "hard": ("exp6_easy_truthfulqa_p0.0-0.3_k5_c400.json", 400, "#1f77b4", "o"),
    "easy": ("exp6_easy_truthfulqa_p0.6-0.95_k5_c250.json", 250, "#d62728", "s"),
}
RHO_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


# ── Exp6A: does the equicorrelated simulator reproduce the closed form? ──────────
def _sim_pmaj(N: int, p: float, rho: float, n_mc: int, rng) -> float:
    """Empirical majority-correct of N equicorrelated Bernoulli(p, ρ) jurors, via the
    1-D Gaussian latent z_i = √ρ·Z0 + √(1-ρ)·ε_i, correct iff z_i ≤ Φ⁻¹(p)."""
    tau = norm.ppf(p)
    Z0 = rng.standard_normal(n_mc)
    eps = rng.standard_normal((n_mc, N))
    z = np.sqrt(rho) * Z0[:, None] + np.sqrt(1.0 - rho) * eps
    correct = z <= tau                                   # (n_mc, N)
    maj = correct.sum(axis=1) > (N / 2.0)
    return float(maj.mean())


def sanity(N: int = 11, n_mc: int = 200_000) -> None:
    rng = np.random.default_rng(0)
    print("Exp6A synthetic sanity — |empirical − analytic| P_maj (N=%d):" % N)
    worst = 0.0
    for p in (0.19, 0.35, 0.5, 0.63, 0.8):
        cells = []
        for rho in (0.0, 0.25, 0.75, 0.99):
            emp = _sim_pmaj(N, p, rho, n_mc, rng)
            ana = p_maj(N, p, rho)
            worst = max(worst, abs(emp - ana))
            cells.append(f"{abs(emp-ana):.3f}")
        print(f"  p={p:.2f}  " + "  ".join(f"ρ={r}:{c}" for r, c in
                                           zip((0.0, 0.25, 0.75, 0.99), cells)))
    print(f"  worst mismatch = {worst:.3f}  → simulator {'OK' if worst < 0.01 else 'CHECK'}\n")


# ── belief loading ──────────────────────────────────────────────────────────────
def load_beliefs(subset: str, N: int) -> np.ndarray:
    """Real round-0 belief matrices (Q, N, K) for a subset, from saved g=0.00 states
    (round 0 is coupling-independent). Canonical coords: truth at column 0."""
    cache, cand, *_ = SUBSETS[subset]
    from mfg_llm.llm_debate.benchmarks import load_truthfulqa_mc1
    pool = load_truthfulqa_mc1(ROOT / "data" / "raw" / "benchmarks" / "truthfulqa.csv",
                               max_questions=cand)
    sel = json.loads((PROC / cache).read_text())["selected"]
    out = []
    for qi, idx in enumerate(sel):
        q = pool[idx][0]
        qh = hashlib.md5(q.encode("utf-8")).hexdigest()[:8]
        npz = PROC / f"exp6_truthfulqa_N{N}_g0.00_q{qi}_{qh}.npz"
        if npz.exists():
            out.append(np.load(npz, allow_pickle=True)["states"][0])
    return np.stack(out, axis=0)


# ── measured ρ (same Fleiss ICC estimator as run.py) ────────────────────────────
def fleiss_icc(counts: list[tuple[int, int]]) -> float:
    counts = [(m, c) for m, c in counts if m >= 1]
    Q = len(counts)
    Ntot = sum(m for m, _ in counts)
    if Q < 2 or Ntot <= Q:
        return 0.0
    ybar = sum(c for _, c in counts) / Ntot
    ssw = sum(c * (1.0 - c / m) for m, c in counts)
    ssb = sum(m * (c / m - ybar) ** 2 for m, c in counts)
    msw = ssw / (Ntot - Q)
    msb = ssb / (Q - 1)
    m0 = (Ntot - sum(m * m for m, _ in counts) / Ntot) / (Q - 1)
    denom = msb + (m0 - 1.0) * msw
    return float((msb - msw) / denom) if denom > 0 else 0.0


# ── Exp6B core: sweep ρ_target, Monte-Carlo the shock, measure realized (p, ρ) ──
def run_shock(beliefs: np.ndarray, lam: float, rho_grid, n_mc: int,
              seed: int) -> list[dict]:
    Q, N, K = beliefs.shape
    rows = []
    base_plur = None
    for rho in rho_grid:
        rng = np.random.default_rng([seed, int(rho * 1000)])
        sqrt_r, sqrt_1r = np.sqrt(rho), np.sqrt(1.0 - rho)
        plur, single_c, single_n = [], 0, 0
        counts: list[tuple[int, int]] = []
        for q in range(Q):
            s = beliefs[q]                                       # (N,K)
            for _ in range(n_mc):
                z0 = rng.standard_normal(K)                      # shared evidence
                Z = rng.standard_normal((N, K))                  # private evidence
                votes = np.argmax(s + lam * (sqrt_r * z0 + sqrt_1r * Z), axis=1)
                c = int((votes == 0).sum())
                single_c += c
                single_n += N
                counts.append((N, c))
                bc = np.bincount(votes, minlength=K)
                w = np.flatnonzero(bc == bc.max())
                plur.append(int((rng.choice(w) if len(w) > 1 else w[0]) == 0))
        plur = np.asarray(plur, dtype=np.float64)
        acc = float(plur.mean())
        if base_plur is None:
            base_plur = acc
        # bootstrap CI on plurality accuracy
        brng = np.random.default_rng([seed, 7, int(rho * 1000)])
        boots = np.array([plur[brng.integers(0, plur.size, plur.size)].mean()
                          for _ in range(2000)])
        rows.append({
            "rho_target": rho, "lam": lam,
            "p_single": single_c / single_n,
            "plurality_acc": acc,
            "acc_ci_low": float(np.quantile(boots, 0.025)),
            "acc_ci_high": float(np.quantile(boots, 0.975)),
            "lift": acc - base_plur,
            "rho_measured": fleiss_icc(counts),
            "n_trials": plur.size,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="run Exp6A synthetic check first")
    ap.add_argument("--N", type=int, default=11)
    ap.add_argument("--lam-frac", type=float, default=1.0,
                    help="shock scale λ = lam_frac × median per-question score std")
    ap.add_argument("--n-mc", type=int, default=200, help="Monte-Carlo shocks per question")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=float, nargs="+", default=RHO_GRID)
    args = ap.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    if args.sanity:
        sanity(args.N)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    rhos = np.linspace(0, 1, 101)
    summary = {}
    for subset, (_, _, color, mk) in SUBSETS.items():
        beliefs = load_beliefs(subset, args.N)
        scale = float(np.median([s.std() for s in beliefs]))
        lam = args.lam_frac * scale
        rows = run_shock(beliefs, lam, args.grid, args.n_mc, args.seed)
        out = TABLES / f"exp6b_common_shock_{subset}_N{args.N}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        p_mean = float(np.mean([r["p_single"] for r in rows]))
        summary[subset] = (rows, p_mean, lam, scale)
        print(f"\n[{subset}] Q={len(beliefs)} λ={lam:.2f} (scale={scale:.2f})  "
              f"p_single≈{p_mean:.3f}")
        for r in rows:
            print(f"  ρ_t={r['rho_target']:.2f} | ρ_meas={r['rho_measured']:.2f} | "
                  f"plur_acc={r['plurality_acc']:.3f} "
                  f"[{r['acc_ci_low']:.3f},{r['acc_ci_high']:.3f}] | "
                  f"lift={r['lift']:+.3f}")
        print(f"  wrote {out}")

        # analytic curve at this subset's realized p, plus empirical points
        ax.plot(rhos, [p_maj(args.N, p_mean, float(r)) for r in rhos], "-",
                color=color, lw=2, alpha=0.55,
                label=f"theory P_maj(N={args.N}, p={p_mean:.2f})")
        x = [r["rho_measured"] for r in rows]
        y = [r["plurality_acc"] for r in rows]
        yl = [r["plurality_acc"] - r["acc_ci_low"] for r in rows]
        yh = [r["acc_ci_high"] - r["plurality_acc"] for r in rows]
        ax.errorbar(x, y, yerr=[yl, yh], fmt=mk, color=color, ms=8, capsize=3,
                    lw=1.2, mec="k", mew=0.7, zorder=5,
                    label=f"{subset} (real beliefs + common shock, p={p_mean:.2f})")
        ax.axhline(p_mean, color=color, lw=0.8, ls=":", alpha=0.6)

    ax.set_xlabel("measured vote-correlation  ρ  (Fleiss ICC)")
    ax.set_ylabel("plurality accuracy   P_maj")
    ax.set_title("Exp6B: real qwen beliefs + controlled common shock trace the\n"
                 "analytic P_maj(N,p,ρ) — accuracy → p_single as ρ→1 (dotted)", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best", framealpha=0.92)
    fig.tight_layout()
    figout = FIGS / f"exp6b_common_shock_N{args.N}.png"
    fig.savefig(figout, dpi=150)
    print(f"\nwrote {figout}")


if __name__ == "__main__":
    main()
