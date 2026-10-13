from __future__ import annotations

import numpy as np


class MeanField:
    def __init__(self, states: np.ndarray):
        self.states = np.asarray(states, dtype=np.float64)
        if self.states.ndim != 2:
            raise ValueError("states must be shape (N, K)")

    @property
    def N(self) -> int:
        return self.states.shape[0]

    @property
    def K(self) -> int:
        return self.states.shape[1]

    def mean(self) -> np.ndarray:
        return np.mean(self.states, axis=0)

    def covariance(self) -> np.ndarray:
        if self.N < 2:
            return np.zeros((self.K, self.K))
        return np.cov(self.states, rowvar=False, bias=True).reshape(self.K, self.K)

    def trace_cov(self) -> float:
        return float(np.trace(self.covariance()))

    def det_cov(self) -> float:
        return float(np.linalg.det(self.covariance()))

    def rank_cov(self, tol: float = 1e-8) -> int:
        return int(np.linalg.matrix_rank(self.covariance(), tol=tol))

    def empirical_measure_moments(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean(), self.covariance()

    def resample(self, n: int, rng: np.random.Generator | None = None) -> "MeanField":
        rng = rng or np.random.default_rng()
        idx = rng.choice(self.N, size=n, replace=n > self.N)
        return MeanField(self.states[idx])

    @staticmethod
    def from_gaussian(m: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator | None = None) -> "MeanField":
        rng = rng or np.random.default_rng()
        return MeanField(rng.multivariate_normal(m, cov, size=n))
