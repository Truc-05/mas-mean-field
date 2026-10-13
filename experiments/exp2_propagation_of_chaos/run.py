"""Exp 2: propagation of chaos.

W2(mu_T^N, mu_T) vs N with mean +/- std over seeds; expect slope ~ -1/2 in log-log.
Pure synthetic dynamics (no LLM API). Writes results/tables/exp2_propagation_of_chaos.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mfg_llm.dynamics.fpk_gaussian import simulate_gaussian_fpk
from mfg_llm.dynamics.particle_system import ParticleSystem
from mfg_llm.mean_field import MeanField
from mfg_llm.metrics.stats import run_replicates, summarize
from mfg_llm.metrics.wasserstein import wasserstein2_gaussian

K = 3
A = np.array([[1.0, 0.1, 0.0], [0.0, 0.8, 0.1], [0.1, 0.0, 0.9]])
B = np.eye(K) * 0.3
s = np.array([0.5, -0.2, 0.1])
SIGMA, ETA, T = 0.05, 0.1, 50
NS = [10, 30, 100, 300, 1000]
N_SEEDS = 20

RESULTS = Path(__file__).resolve().parents[2] / "results" / "tables"


def w2_to_limit(N: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(N, K))
    ps = ParticleSystem(x0, A, B, s, SIGMA, ETA, rng=rng)
    ps.run(T)
    mf = MeanField(ps.x)
    m_emp, cov_emp = mf.mean(), mf.covariance()

    m_traj, cov_traj = simulate_gaussian_fpk(
        m0=np.zeros(K), cov0=np.eye(K), A=A, B=B, s=s, sigma=SIGMA, dt=ETA, n_steps=T,
    )
    w2 = wasserstein2_gaussian(m_emp, cov_emp, m_traj[-1], cov_traj[-1])
    return {"W2": w2}


def main() -> None:
    rows = []
    for N in NS:
        stats = run_replicates(lambda seed: w2_to_limit(N, seed), n_seeds=N_SEEDS, base_seed=0)
        st = stats["W2"]
        rows.append({"N": N, "W2_mean": st.mean, "W2_std": st.std,
                     "W2_ci_low": st.ci_low, "W2_ci_high": st.ci_high, "n_seeds": st.n})
        print(f"N={N:5d}  W2 = {st}")

    logN = np.log(np.array([r["N"] for r in rows], dtype=np.float64))
    logW = np.log(np.array([r["W2_mean"] for r in rows]))
    slope = float(np.polyfit(logN, logW, 1)[0])
    print(f"\nfitted log-log slope = {slope:.3f}  (theory: -0.5)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp2_propagation_of_chaos.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
