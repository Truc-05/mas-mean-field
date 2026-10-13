from __future__ import annotations

import numpy as np


def logits_to_belief(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def belief_to_logits(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    x = np.log(p)
    return x - np.mean(x, axis=-1, keepdims=True)


def extract_opinion_state(logprob_by_answer: dict[str, float], answer_space: list[str]) -> np.ndarray:
    x = np.array([logprob_by_answer.get(a, -50.0) for a in answer_space], dtype=np.float64)
    return x - np.mean(x)


def batch_extract(records: list[dict[str, float]], answer_space: list[str]) -> np.ndarray:
    return np.stack([extract_opinion_state(r, answer_space) for r in records], axis=0)


def one_hot_truth(a_star: str, answer_space: list[str], margin: float = 10.0) -> np.ndarray:
    k = len(answer_space)
    idx = answer_space.index(a_star)
    x = np.full(k, -margin / (k - 1) if k > 1 else 0.0)
    x[idx] = margin
    return x - np.mean(x)


def argmax_answer(x: np.ndarray, answer_space: list[str]) -> str:
    return answer_space[int(np.argmax(x))]
