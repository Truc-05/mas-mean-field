from __future__ import annotations

import numpy as np


def accuracy_at_t(x_t: np.ndarray, a_star_idx: int) -> float:
    preds = np.argmax(x_t, axis=1)
    return float(np.mean(preds == a_star_idx))


def accuracy_trajectory(states_by_round: list[np.ndarray], a_star_idx: int) -> np.ndarray:
    return np.array([accuracy_at_t(x, a_star_idx) for x in states_by_round])


def time_to_accuracy(states_by_round: list[np.ndarray], a_star_idx: int, threshold: float = 0.9) -> int:
    traj = accuracy_trajectory(states_by_round, a_star_idx)
    hits = np.where(traj >= threshold)[0]
    return int(hits[0]) if len(hits) > 0 else -1
