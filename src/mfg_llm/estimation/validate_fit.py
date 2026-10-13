from __future__ import annotations

import numpy as np

from mfg_llm.estimation.regression import DebateLog, EstimatedParams
from mfg_llm.dynamics.fpk_gaussian import simulate_gaussian_fpk
from mfg_llm.mean_field import MeanField


def validate_prediction(log: DebateLog, params: EstimatedParams) -> dict:
    x0 = log.states_by_round[0]
    mf0 = MeanField(x0)
    m0, cov0 = mf0.mean(), mf0.covariance()
    T = log.n_rounds() - 1

    m_traj, cov_traj = simulate_gaussian_fpk(
        m0=m0, cov0=cov0, A=params.A, B=params.B, s=params.s,
        sigma=params.sigma, dt=log.eta, n_steps=T,
    )

    empirical_means = np.stack([MeanField(x).mean() for x in log.states_by_round], axis=0)
    empirical_traces = np.array([MeanField(x).trace_cov() for x in log.states_by_round])
    predicted_traces = np.array([float(np.trace(c)) for c in cov_traj])

    mean_err = float(np.linalg.norm(empirical_means - m_traj) / (np.linalg.norm(empirical_means) + 1e-9))
    trace_err = float(np.linalg.norm(empirical_traces - predicted_traces) / (np.linalg.norm(empirical_traces) + 1e-9))

    return {
        "mean_relative_error": mean_err,
        "trace_relative_error": trace_err,
        "predicted_mean_traj": m_traj,
        "predicted_trace_traj": predicted_traces,
        "empirical_mean_traj": empirical_means,
        "empirical_trace_traj": empirical_traces,
    }
