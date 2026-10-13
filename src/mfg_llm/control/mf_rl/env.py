from __future__ import annotations

import numpy as np

from mfg_llm.mean_field import MeanField
from mfg_llm.control.mf_rl.reward import compute_reward


class MeanFieldDebateEnv:
    def __init__(self, N: int, K: int, A: np.ndarray, B: np.ndarray, s: np.ndarray,
                 sigma: float, a_star_idx: int, eta: float = 0.1, max_steps: int = 50,
                 token_cost: float = 0.01, rng: np.random.Generator | None = None):
        self.N = N
        self.K = K
        self.A = A
        self.B = B
        self.s = s
        self.sigma = sigma
        self.a_star_idx = a_star_idx
        self.eta = eta
        self.max_steps = max_steps
        self.token_cost = token_cost
        self.rng = rng or np.random.default_rng()
        self.x = None
        self.t = 0

    def _observation(self) -> np.ndarray:
        mf = MeanField(self.x)
        # mean field summary: mu_bar + diag(Sigma); the diagonal carries the
        # diversity/herding signal (doc section 7: state = (x, mu))
        return np.concatenate([mf.mean(), np.diag(mf.covariance())])

    @property
    def obs_dim(self) -> int:
        return 2 * self.K

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self.x = self.rng.normal(scale=1.0, size=(self.N, self.K))
        self.t = 0
        return self.x.copy(), self._observation()

    def step(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        temperature_scale, listen_weight = u[0], u[1]
        mu_bar = MeanField(self.x).mean()
        F = -self.x @ self.A.T + mu_bar @ (listen_weight * self.B).T + self.s
        noise_scale = np.clip(temperature_scale, 0.01, 5.0)
        noise = self.rng.normal(size=self.x.shape) * np.sqrt(2 * self.sigma * noise_scale * self.eta)
        self.x = self.x + self.eta * F + noise
        self.t += 1

        mf = MeanField(self.x)
        preds = np.argmax(self.x, axis=1)
        acc = float(np.mean(preds == self.a_star_idx))
        reward = compute_reward(accuracy=acc, token_cost=self.token_cost, n_tokens=self.N)
        done = self.t >= self.max_steps
        info = {"accuracy": acc, "trace_cov": mf.trace_cov()}
        return self.x.copy(), self._observation(), reward, done, info
