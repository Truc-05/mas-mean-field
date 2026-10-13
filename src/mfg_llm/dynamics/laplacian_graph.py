from __future__ import annotations

import numpy as np
from scipy.linalg import circulant


def row_stochastic(W: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def build_laplacian(W: np.ndarray) -> np.ndarray:
    W = row_stochastic(W)
    D = np.diag(W.sum(axis=1))
    return D - W


def spectral_gap(L: np.ndarray) -> float:
    eig = np.sort(np.linalg.eigvalsh(0.5 * (L + L.T)))
    if len(eig) < 2:
        return 0.0
    return float(eig[1])


def complete_graph(n: int) -> np.ndarray:
    W = np.ones((n, n)) - np.eye(n)
    return W


def erdos_renyi(n: int, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    W = (rng.random((n, n)) < p).astype(np.float64)
    W = np.triu(W, 1)
    W = W + W.T
    return W


def ring_graph(n: int, k: int = 2) -> np.ndarray:
    c = np.zeros(n)
    for j in range(1, k // 2 + 1):
        c[j % n] = 1.0
        c[-j % n] = 1.0
    c[0] = 0.0
    return circulant(c)


def mixing_time_bound(L: np.ndarray, eps: float = 1e-3) -> float:
    gap = spectral_gap(L)
    if gap <= 0:
        return float("inf")
    return float(np.log(1.0 / eps) / gap)
