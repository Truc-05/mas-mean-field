from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class SummaryStats:
    mean: float
    std: float
    stderr: float
    ci_low: float
    ci_high: float
    n: int

    def __str__(self) -> str:
        return (f"{self.mean:.4g} ± {self.std:.4g} "
                f"(95% CI [{self.ci_low:.4g}, {self.ci_high:.4g}], n={self.n})")


def summarize(samples, confidence: float = 0.95) -> SummaryStats:
    x = np.asarray(samples, dtype=np.float64).ravel()
    n = x.size
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    stderr = std / np.sqrt(n) if n > 1 else 0.0
    if n > 1 and std > 0:
        t = float(scipy_stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
        half = t * stderr
    else:
        half = 0.0
    return SummaryStats(mean=mean, std=std, stderr=stderr,
                        ci_low=mean - half, ci_high=mean + half, n=n)


def run_replicates(run_fn: Callable[[int], dict], n_seeds: int, base_seed: int = 0,
                   confidence: float = 0.95) -> dict[str, SummaryStats]:
    records = [run_fn(base_seed + i) for i in range(n_seeds)]
    return {k: summarize([r[k] for r in records], confidence) for k in records[0]}


def summarize_trajectories(trajs: list[np.ndarray]) -> dict[str, np.ndarray]:
    arr = np.stack([np.asarray(t, dtype=np.float64) for t in trajs], axis=0)
    n = arr.shape[0]
    std = arr.std(axis=0, ddof=1) if n > 1 else np.zeros(arr.shape[1:])
    return {"mean": arr.mean(axis=0), "std": std,
            "stderr": std / np.sqrt(n), "n": np.array(n)}


def bootstrap_ci(samples, stat_fn: Callable = np.mean, n_boot: int = 10000,
                 confidence: float = 0.95,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    rng = rng or np.random.default_rng()
    x = np.asarray(samples, dtype=np.float64).ravel()
    boots = np.array([float(stat_fn(rng.choice(x, size=x.size, replace=True)))
                      for _ in range(n_boot)])
    lo = (1.0 - confidence) / 2.0
    return float(np.quantile(boots, lo)), float(np.quantile(boots, 1.0 - lo))
