from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, solve


def _drift_mean(m: np.ndarray, A: np.ndarray, B: np.ndarray, s: np.ndarray) -> np.ndarray:
    return -(A - B) @ m + s


def _drift_cov(Sigma: np.ndarray, A: np.ndarray, sigma: float) -> np.ndarray:
    K = A.shape[0]
    return -A @ Sigma - Sigma @ A.T + 2 * sigma * np.eye(K)


def simulate_gaussian_fpk(m0: np.ndarray, cov0: np.ndarray, A: np.ndarray, B: np.ndarray,
                           s: np.ndarray, sigma: float, dt: float, n_steps: int
                           ) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(m0, dtype=np.float64).copy()
    Sigma = np.asarray(cov0, dtype=np.float64).copy()
    m_traj = [m.copy()]
    cov_traj = [Sigma.copy()]
    for _ in range(n_steps):
        m = m + dt * _drift_mean(m, A, B, s)
        Sigma = Sigma + dt * _drift_cov(Sigma, A, sigma)
        Sigma = 0.5 * (Sigma + Sigma.T)
        m_traj.append(m.copy())
        cov_traj.append(Sigma.copy())
    return np.stack(m_traj, axis=0), np.stack(cov_traj, axis=0)


def fixed_point(A: np.ndarray, B: np.ndarray, s: np.ndarray, sigma: float
                 ) -> tuple[np.ndarray, np.ndarray, bool]:
    M = A - B
    eig_M = np.linalg.eigvals(M)
    stable = bool(np.min(eig_M.real) > 0)
    m_inf = solve(M, s) if stable else np.full_like(s, np.nan)
    try:
        Sigma_inf = solve_continuous_lyapunov(A, 2 * sigma * np.eye(A.shape[0]))
    except Exception:
        Sigma_inf = np.full_like(A, np.nan)
    return m_inf, Sigma_inf, stable


def simulate_gaussian_fpk_exact(m0: np.ndarray, cov0: np.ndarray, A: np.ndarray,
                                 B: np.ndarray, s: np.ndarray, sigma: float, dt: float,
                                 n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form solution of the affine FPK moment ODEs on a uniform time grid.

    Both ODEs are affine, so the augmented-matrix exponential gives the exact
    flow map for one step of size dt; iterating it is exact at every grid point
    (no Euler error, stable for stiff A). Handles singular/unstable M = A - B.
    """
    K = A.shape[0]
    M = A - B

    G_m = np.zeros((K + 1, K + 1))
    G_m[:K, :K] = -M
    G_m[:K, K] = s
    P_m = expm(G_m * dt)

    lyap = -(np.kron(np.eye(K), A) + np.kron(A, np.eye(K)))
    G_c = np.zeros((K * K + 1, K * K + 1))
    G_c[:K * K, :K * K] = lyap
    G_c[:K * K, K * K] = 2.0 * sigma * np.eye(K).ravel()
    P_c = expm(G_c * dt)

    z_m = np.append(np.asarray(m0, dtype=np.float64), 1.0)
    z_c = np.append(np.asarray(cov0, dtype=np.float64).ravel(), 1.0)
    m_traj, cov_traj = [z_m[:K].copy()], [z_c[:K * K].reshape(K, K).copy()]
    for _ in range(n_steps):
        z_m = P_m @ z_m
        z_c = P_c @ z_c
        m_traj.append(z_m[:K].copy())
        Sigma = z_c[:K * K].reshape(K, K)
        cov_traj.append(0.5 * (Sigma + Sigma.T))
    return np.stack(m_traj, axis=0), np.stack(cov_traj, axis=0)


def consensus_collapses(A: np.ndarray, sigma: float, tol: float = 0.1) -> bool:
    """Opinion diversity collapses iff A is stable and tr(Sigma_inf) < tol.

    Sigma_inf solves A Sigma + Sigma A^T = 2 sigma I (= sigma * A^{-1} for
    symmetric A), so the criterion compares like with like: a stationary
    variance against a variance threshold. Unstable A means diversity grows
    without bound, so no collapse.
    """
    eig_A = np.linalg.eigvals(A)
    if np.min(eig_A.real) <= 0:
        return False
    Sigma_inf = solve_continuous_lyapunov(A, 2 * sigma * np.eye(A.shape[0]))
    return bool(np.trace(Sigma_inf) < tol)
