"""Exp 4: optimal control comparison in an unstable (herding-prone) regime.

Baselines: no control, spectral heuristic (rescale listen weight so
lambda_max(w*B) <= margin * lambda_min(A)), LQR feedback+feedforward on the
population mean, and mean-field RL. Reports final accuracy and time-to-truth
mean +/- std over seeds. Writes results/tables/exp4_optimal_control.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from mfg_llm.control.hjb_solver import solve_lqr_hjb, optimal_feedback, steady_state_control
from mfg_llm.control.mf_rl.env import MeanFieldDebateEnv
from mfg_llm.control.mf_rl.train import train_replicates
from mfg_llm.dynamics.particle_system import ParticleSystem
from mfg_llm.mean_field import MeanField
from mfg_llm.metrics.accuracy import accuracy_trajectory, time_to_accuracy
from mfg_llm.metrics.stats import run_replicates
from mfg_llm.opinion_state import one_hot_truth

K = 3
A = np.eye(K) * 1.0
B_CONF = np.eye(K) * 1.4          # lambda_max(B) > lambda_min(A): herding regime
A_STAR = one_hot_truth("A", ["A", "B", "C"], margin=3.0)
S = 0.15 * A_STAR / np.linalg.norm(A_STAR)
SIGMA, ETA, T, N = 0.05, 0.05, 200, 200
N_SEEDS = 10
ACC_THRESHOLD = 0.9

RESULTS = Path(__file__).resolve().parents[2] / "results" / "tables"


def run_baseline(kind: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    B = B_CONF.copy()
    policy = None

    if kind == "spectral_heuristic":
        lam_max_B = float(np.max(np.linalg.eigvals(B).real))
        lam_min_A = float(np.min(np.linalg.eigvals(A).real))
        w = min(1.0, 0.5 * lam_min_A / lam_max_B)
        B = w * B
    elif kind == "lqr":
        P, K_gain = solve_lqr_hjb(A, B, S)
        u_ss = steady_state_control(A, B, S, A_STAR)

        def policy(x, mu_bar):
            return optimal_feedback(mu_bar, A_STAR, K_gain, u_ss)

    x0 = rng.normal(size=(N, K))
    ps = ParticleSystem(x0, A, B, S, SIGMA, ETA, rng=rng)
    traj = ps.run(T, policy=policy)

    states = [traj[t] for t in range(T + 1)]
    acc_traj = accuracy_trajectory(states, a_star_idx=0)
    t_hit = time_to_accuracy(states, a_star_idx=0, threshold=ACC_THRESHOLD)
    return {
        "final_accuracy": float(acc_traj[-1]),
        "time_to_truth": float(t_hit if t_hit >= 0 else T),
        "final_trace_cov": MeanField(ps.x).trace_cov(),
    }


def main() -> None:
    rows = []
    for kind in ["no_control", "spectral_heuristic", "lqr"]:
        stats = run_replicates(lambda seed: run_baseline(kind, seed), n_seeds=N_SEEDS, base_seed=0)
        rows.append({"baseline": kind,
                     "acc_mean": stats["final_accuracy"].mean,
                     "acc_std": stats["final_accuracy"].std,
                     "t_truth_mean": stats["time_to_truth"].mean,
                     "t_truth_std": stats["time_to_truth"].std,
                     "n_seeds": N_SEEDS})
        print(f"{kind:18s}  accuracy = {stats['final_accuracy']}")
        print(f"{'':18s}  time-to-truth = {stats['time_to_truth']}")

    def env_factory(seed: int) -> MeanFieldDebateEnv:
        return MeanFieldDebateEnv(N=N, K=K, A=A, B=B_CONF, s=S, sigma=SIGMA, a_star_idx=0,
                                  eta=ETA, max_steps=60, token_cost=0.0,
                                  rng=np.random.default_rng(seed))

    rl = train_replicates(env_factory, n_seeds=5, n_episodes=150)
    rows.append({"baseline": "rl", "acc_mean": rl["final_accuracy"].mean,
                 "acc_std": rl["final_accuracy"].std,
                 "t_truth_mean": float("nan"), "t_truth_std": float("nan"), "n_seeds": 5})
    print(f"{'rl':18s}  accuracy = {rl['final_accuracy']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "exp4_optimal_control.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
