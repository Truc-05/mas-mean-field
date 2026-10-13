from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import sqrtm


def wasserstein2_gaussian(m1: np.ndarray, cov1: np.ndarray, m2: np.ndarray, cov2: np.ndarray) -> float:
    cov1 = 0.5 * (np.asarray(cov1, dtype=np.float64) + np.asarray(cov1, dtype=np.float64).T)
    cov2 = 0.5 * (np.asarray(cov2, dtype=np.float64) + np.asarray(cov2, dtype=np.float64).T)
    mean_term = float(np.sum((m1 - m2) ** 2))
    cov1_sqrt = np.real(sqrtm(cov1))
    inner = cov1_sqrt @ cov2 @ cov1_sqrt
    inner_sqrt = np.real(sqrtm(inner))
    trace_term = float(np.trace(cov1 + cov2 - 2 * inner_sqrt))
    return float(np.sqrt(max(mean_term + trace_term, 0.0)))


def wasserstein2_empirical(x: np.ndarray, y: np.ndarray, max_points: int = 2000,
                            rng: np.random.Generator | None = None) -> float:
    """Exact W2 between equal-size empirical measures via optimal assignment.

    Subsampling to max_points floors the measurable W2 at the max_points sampling
    fluctuation (~max_points^{-1/2}); for propagation-of-chaos curves keep
    max_points above the largest N being compared, or use moment-based
    wasserstein2_gaussian which has no such floor.
    """
    rng = rng or np.random.default_rng()
    if x.shape[0] > max_points:
        x = x[rng.choice(x.shape[0], max_points, replace=False)]
    if y.shape[0] > max_points:
        y = y[rng.choice(y.shape[0], max_points, replace=False)]
    n, m = x.shape[0], y.shape[0]
    if n != m:
        k = min(n, m)
        x = x[rng.choice(n, k, replace=False)]
        y = y[rng.choice(m, k, replace=False)]
    cost = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(np.sqrt(cost[row_ind, col_ind].mean()))


def propagation_of_chaos_curve(states_by_N: dict[int, np.ndarray], limit_measure: np.ndarray) -> dict:
    Ns = sorted(states_by_N.keys())
    w2 = [wasserstein2_empirical(states_by_N[n], limit_measure) for n in Ns]
    return {"N": np.array(Ns), "W2": np.array(w2)}
