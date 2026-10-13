"""Exp 3: herding vs truth-convergence phase transition (Theorems 1-2).

Sweep conformity strength b with fixed private signal s. For b < 1 the consensus
operator M = A - bI is stable and the population mean converges to the truth
direction; past the spectral threshold the mean field amplifies initial noise
and the crowd herds onto a signal-independent answer. Reports accuracy
mean +/- std over seeds. Writes results/tables/exp3_phase_transition.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mfg_llm.dynamics.particle_system import ParticleSystem
from mfg_llm.metrics.stats import run_replicates

K = 3
A = np.eye(K)
S_MAG = 0.3
TRUTH = np.array([1.0, -0.5, -0.5])
S = S_MAG * TRUTH / np.linalg.norm(TRUTH)
SIGMA, ETA, T, N = 0.05, 0.05, 120, 200
B_SWEEP = [0.0, 0.5, 0.9, 1.2, 2.0, 4.0, 8.0]
N_SEEDS = 10

RESULTS = Path(__file__).resolve().parents[2] / "results" / "tables"


def final_accuracy(b: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    B = np.eye(K) * b
    x0 = rng.normal(size=(N, K))
    ps = ParticleSystem(x0, A, B, S, SIGMA, ETA, rng=rng)
    ps.run(T)
    return {"accuracy": ps.accuracy(a_star_idx=0)}


def main() -> None:
    rows = []
    for b in B_SWEEP:
        stable = 1.0 - b > 0
        stats = run_replicates(lambda seed: final_accuracy(b, seed), n_seeds=N_SEEDS, base_seed=0)
        st = stats["accuracy"]
        rows.append({"b": b, "stable_M": stable, "acc_mean": st.mean, "acc_std": st.std,
                     "acc_ci_low": st.ci_low, "acc_ci_high": st.ci_high, "n_seeds": st.n})
        print(f"b={b:4.1f}  stable_M={str(stable):5s}  accuracy = {st}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp3_phase_transition.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
