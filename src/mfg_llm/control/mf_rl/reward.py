from __future__ import annotations


def compute_reward(accuracy: float, token_cost: float, n_tokens: int) -> float:
    return accuracy - token_cost * n_tokens


def terminal_bonus(accuracy: float, threshold: float = 0.9, bonus: float = 5.0) -> float:
    return bonus if accuracy >= threshold else 0.0
