from __future__ import annotations

import numpy as np

from mfg_llm.mean_field import MeanField


class ParticleSystem:
    def __init__(self, x0: np.ndarray, A: np.ndarray, B: np.ndarray, s: np.ndarray,
                 sigma: float, eta: float = 0.1, rng: np.random.Generator | None = None):
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.s = np.asarray(s, dtype=np.float64)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.rng = rng or np.random.default_rng()
        self.history = [self.x.copy()]

    def drift(self, x: np.ndarray, mu_bar: np.ndarray) -> np.ndarray:
        return -x @ self.A.T + mu_bar @ self.B.T + self.s

    def step(self, u: np.ndarray | None = None) -> np.ndarray:
        mu_bar = MeanField(self.x).mean()
        F = self.drift(self.x, mu_bar)
        if u is not None:
            F = F + u
        noise = self.rng.normal(size=self.x.shape) * np.sqrt(2 * self.sigma * self.eta)
        self.x = self.x + self.eta * F + noise
        self.history.append(self.x.copy())
        return self.x

    def run(self, n_steps: int, policy=None) -> np.ndarray:
        for _ in range(n_steps):
            u = policy(self.x, MeanField(self.x).mean()) if policy is not None else None
            self.step(u)
        return np.stack(self.history, axis=0)

    def accuracy(self, a_star_idx: int) -> float:
        preds = np.argmax(self.x, axis=1)
        return float(np.mean(preds == a_star_idx))
