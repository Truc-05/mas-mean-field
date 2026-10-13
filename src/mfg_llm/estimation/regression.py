from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import null_space

from mfg_llm.mean_field import MeanField


@dataclass
class DebateLog:
    states_by_round: list[np.ndarray] = field(default_factory=list)
    eta: float = 1.0

    def n_rounds(self) -> int:
        return len(self.states_by_round)

    def K(self) -> int:
        return self.states_by_round[0].shape[1]

    def N(self) -> int:
        return self.states_by_round[0].shape[0]


@dataclass
class EstimatedParams:
    A: np.ndarray
    B: np.ndarray
    s: np.ndarray
    sigma: float
    residual_norm: float


def _as_log_list(logs: DebateLog | list[DebateLog]) -> list[DebateLog]:
    return list(logs) if isinstance(logs, (list, tuple)) else [logs]


def _build_design(logs: list[DebateLog]) -> tuple[np.ndarray, np.ndarray]:
    xs, dxs = [], []
    for log in logs:
        for t in range(log.n_rounds() - 1):
            x_t = log.states_by_round[t]
            x_tp1 = log.states_by_round[t + 1]
            mu_t = MeanField(x_t).mean()
            dx = (x_tp1 - x_t) / log.eta
            n = x_t.shape[0]
            design = np.concatenate([x_t, np.tile(mu_t, (n, 1)), np.ones((n, 1))], axis=1)
            xs.append(design)
            dxs.append(dx)
    return np.concatenate(xs, axis=0), np.concatenate(dxs, axis=0)


def estimate_matrices(logs: DebateLog | list[DebateLog], ridge: float = 1e-3) -> EstimatedParams:
    logs = _as_log_list(logs)
    K = logs[0].K()
    eta = logs[0].eta
    X, Y = _build_design(logs)
    XtX = X.T @ X + ridge * np.eye(X.shape[1])
    XtY = X.T @ Y
    theta = np.linalg.solve(XtX, XtY)

    neg_A = theta[:K, :]
    B = theta[K:2 * K, :]
    s = theta[2 * K, :]

    A = -neg_A.T
    B = B.T
    s = s

    resid = Y - X @ theta
    residual_norm = float(np.linalg.norm(resid))
    # dx = F + sqrt(2*sigma/eta)*xi  =>  Var(resid) = 2*sigma/eta
    sigma = float(np.var(resid) * eta / 2.0)

    return EstimatedParams(A=A, B=B, s=s, sigma=sigma, residual_norm=residual_norm)


def estimate_matrices_two_stage(logs: DebateLog | list[DebateLog],
                                 ridge: float = 1e-3) -> EstimatedParams:
    """Two-stage estimator with better identification of B.

    In the joint regression, B*mu_bar and s are constant within a round, so B is
    identified only from cross-round variation of mu_bar (near-collinear with the
    intercept once the mean converges). Decomposing x = mu_bar + dev separates:
      mean:      d(mu_bar)/dt = -(A - B) mu_bar + s   -> identifies M = A - B, s
      deviation: d(dev)/dt    = -A dev               -> identifies A from N*T samples
    then B = A - M.

    A single mean trajectory is a smooth curve with rank-deficient excitation, so
    M stays weakly identified from one debate no matter the estimator — pass a
    list of logs with diverse initial means (different questions / initial
    prompts) to identify M and hence B.
    """
    logs = _as_log_list(logs)
    K = logs[0].K()
    eta = logs[0].eta

    mu_designs, dmus, devs, ddevs = [], [], [], []
    for log in logs:
        mus = np.stack([MeanField(x).mean() for x in log.states_by_round], axis=0)
        T = mus.shape[0] - 1
        mu_designs.append(np.concatenate([mus[:-1], np.ones((T, 1))], axis=1))
        dmus.append((mus[1:] - mus[:-1]) / log.eta)
        for t in range(T):
            dev_t = log.states_by_round[t] - mus[t]
            dev_tp1 = log.states_by_round[t + 1] - mus[t + 1]
            devs.append(dev_t)
            ddevs.append((dev_tp1 - dev_t) / log.eta)

    D = np.concatenate(mu_designs, axis=0)
    dmu = np.concatenate(dmus, axis=0)
    G = D.T @ D + ridge * np.eye(K + 1)
    theta_mu = np.linalg.solve(G, D.T @ dmu)
    M = -theta_mu[:K, :].T
    s = theta_mu[K, :]

    X = np.concatenate(devs, axis=0)
    Y = np.concatenate(ddevs, axis=0)
    H = X.T @ X + ridge * np.eye(K)
    A = -np.linalg.solve(H, X.T @ Y).T
    B = A - M

    resid = Y - X @ (-A.T)
    residual_norm = float(np.linalg.norm(resid))
    sigma = float(np.var(resid) * eta / 2.0)
    return EstimatedParams(A=A, B=B, s=s, sigma=sigma, residual_norm=residual_norm)


def spectrum_report(params: EstimatedParams) -> dict:
    """Full-space and gauge-corrected spectra.

    Centered (gauge-fixed) states make every estimated matrix have row sums ~0,
    i.e. a spurious ~0 eigenvalue along 1. The physically meaningful spectra are
    the restrictions to the hyperplane H = {1^T x = 0} (keys with suffix _H);
    use those for stability/herding diagnoses.
    """
    K = params.A.shape[0]
    M = params.A - params.B
    U = null_space(np.ones((1, K)))
    eig_A = np.linalg.eigvals(params.A)
    eig_B = np.linalg.eigvals(params.B)
    eig_M = np.linalg.eigvals(M)
    eig_A_H = np.linalg.eigvals(U.T @ params.A @ U)
    eig_B_H = np.linalg.eigvals(U.T @ params.B @ U)
    eig_M_H = np.linalg.eigvals(U.T @ M @ U)
    return {
        "lambda_min_A": float(np.min(eig_A.real)),
        "lambda_max_B": float(np.max(eig_B.real)),
        "min_re_lambda_M": float(np.min(eig_M.real)),
        "stable_fixed_point": bool(np.min(eig_M.real) > 0),
        "lambda_min_A_H": float(np.min(eig_A_H.real)),
        "lambda_max_B_H": float(np.max(eig_B_H.real)),
        "min_re_lambda_M_H": float(np.min(eig_M_H.real)),
        "stable_fixed_point_H": bool(np.min(eig_M_H.real) > 0),
        "eig_A": eig_A,
        "eig_B": eig_B,
        "eig_M": eig_M,
        "eig_A_H": eig_A_H,
        "eig_B_H": eig_B_H,
        "eig_M_H": eig_M_H,
    }
