from __future__ import annotations

import numpy as np

from mfg_llm.mean_field import MeanField


def diversity_measures(x_t: np.ndarray) -> dict:
    mf = MeanField(x_t)
    cov = mf.covariance()
    eig = np.linalg.eigvalsh(cov)
    return {
        "trace": float(np.trace(cov)),
        "det": float(np.linalg.det(cov)),
        "rank": int(np.linalg.matrix_rank(cov, tol=1e-8)),
        "min_eig": float(np.min(eig)),
        "max_eig": float(np.max(eig)),
    }


def diversity_trajectory(states_by_round: list[np.ndarray]) -> list[dict]:
    return [diversity_measures(x) for x in states_by_round]


def has_collapsed(x_t: np.ndarray, tol: float = 1e-3) -> bool:
    return diversity_measures(x_t)["trace"] < tol
