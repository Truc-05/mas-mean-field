from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are


def solve_lqr_hjb(A: np.ndarray, B: np.ndarray, s: np.ndarray, Q: np.ndarray | None = None,
                   R: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    K = A.shape[0]
    A_lqr = -(A - B)
    B_lqr = np.eye(K)
    Q = Q if Q is not None else np.eye(K)
    R = R if R is not None else np.eye(K)
    P = solve_continuous_are(A_lqr, B_lqr, Q, R)
    K_gain = np.linalg.solve(R, B_lqr.T @ P)
    return P, K_gain


def steady_state_control(A: np.ndarray, B: np.ndarray, s: np.ndarray,
                          x_target: np.ndarray) -> np.ndarray:
    """Feedforward making x_target an equilibrium: 0 = -(A-B) x* + s + u_ss."""
    return (A - B) @ x_target - s


def optimal_feedback(x: np.ndarray, x_target: np.ndarray, K_gain: np.ndarray,
                      u_ss: np.ndarray | None = None) -> np.ndarray:
    u = -K_gain @ (x - x_target)
    return u if u_ss is None else u + u_ss


def value_function(x: np.ndarray, x_target: np.ndarray, P: np.ndarray) -> float:
    d = x - x_target
    return float(d @ P @ d)


def time_to_truth(A: np.ndarray, B: np.ndarray, s: np.ndarray, x0: np.ndarray, a_star: np.ndarray,
                   K_gain: np.ndarray, eta: float = 0.05, tol: float = 0.5, max_steps: int = 2000) -> int:
    x = x0.copy()
    u_ss = steady_state_control(A, B, s, a_star)
    for t in range(max_steps):
        u = optimal_feedback(x, a_star, K_gain, u_ss)
        F = -A @ x + B @ x + s + u
        x = x + eta * F
        if np.linalg.norm(x - a_star) < tol:
            return t
    return max_steps
