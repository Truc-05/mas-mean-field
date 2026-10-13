from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter
from mfg_llm.llm_debate.orchestrator import DebateOrchestrator

# A benchmark item is (question, options) with the TRUE answer at index 0
# (canonical coordinates); display order is shuffled at debate time.
Question = tuple[str, list[str]]

_AQUA_OPT = re.compile(r"^([A-E])\)\s*(.*)$", re.DOTALL)


def load_aqua(path: str | Path, max_questions: int | None = None) -> list[Question]:
    """AQuA-RAT (math MC, K=5): jsonl with fields question/options/correct."""
    items: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            parsed = [_AQUA_OPT.match(o) for o in obj["options"]]
            if any(m is None for m in parsed):
                continue
            by_letter = {m.group(1): m.group(2).strip() for m in parsed}
            correct = obj["correct"].strip()
            if correct not in by_letter:
                continue
            options = [by_letter[correct]] + [v for k, v in by_letter.items() if k != correct]
            items.append((obj["question"].strip(), options))
            if max_questions is not None and len(items) >= max_questions:
                break
    return items


def load_mc_jsonl(path: str | Path, max_questions: int | None = None) -> list[Question]:
    """Generic MC loader for pre-normalized truth@0 JSONL: {"question": str,
    "options": [correct, distractor, ...]}. Produced by fetch_benchmarks.py, so no
    benchmark-specific parsing lives here — options[0] is already the truth."""
    items: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q, opts = obj.get("question"), obj.get("options")
            if not q or not opts or len(opts) < 2 or len(set(opts)) != len(opts):
                continue
            items.append((q.strip(), [str(o).strip() for o in opts]))
            if max_questions is not None and len(items) >= max_questions:
                break
    return items


def load_truthfulqa_mc1(path: str | Path, k: int = 4, seed: int = 0,
                         max_questions: int | None = None) -> list[Question]:
    """TruthfulQA MC1 (K=k): best answer + (k-1) sampled popular misconceptions."""
    rng = np.random.default_rng(seed)
    items: list[Question] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            best = row["Best Answer"].strip().rstrip(".")
            wrong = [w.strip().rstrip(".") for w in row["Incorrect Answers"].split(";") if w.strip()]
            wrong = [w for w in wrong if w and w != best]
            if not best or len(wrong) < k - 1:
                continue
            distractors = [wrong[i] for i in rng.choice(len(wrong), size=k - 1, replace=False)]
            items.append((row["Question"].strip(), [best] + distractors))
            if max_questions is not None and len(items) >= max_questions:
                break
    return items


def run_debate_canonical(question: str, options: list[str], cfg: dict,
                          rng: np.random.Generator, n_agents: int,
                          n_rounds: int) -> list[np.ndarray]:
    """Run one debate with shuffled display order; return per-round states with
    columns permuted back to canonical order (truth at index 0)."""
    shuffled = list(options)
    rng.shuffle(shuffled)
    perm = [shuffled.index(opt) for opt in options]

    limiter = get_rate_limiter(cfg["rpm"]) if cfg.get("rpm") else None
    agents = [
        DebateAgent(i, model=cfg["model"], backend=cfg.get("backend", "groq"),
                    base_url=cfg.get("base_url"),
                    temperature=cfg.get("temperature", 1.0), rate_limiter=limiter,
                    timeout=cfg.get("timeout", 60.0))
        for i in range(n_agents)
    ]
    orch = DebateOrchestrator(agents, question, shuffled,
                              n_rounds=n_rounds, n_workers=cfg.get("n_workers", 4),
                              coupling=cfg.get("coupling", 1.0), rng=rng)
    states = orch.run()
    return [x[:, perm] for x in states]


def plurality_prediction(x_t: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """Plurality vote over *informative* agents' argmax; -1 means no decision.

    An agent whose opinion row is flat (all option scores equal) carries no vote
    and is dropped. This matters because scoring is done in canonical coordinates
    with the truth pinned at index 0: an unparseable LLM reply collapses to the
    zero state, whose argmax is index 0, so the old code silently scored every
    parse failure as *correct*. On AQuA that inflated debate accuracy from a true
    ~0.55 to a spurious 1.00 (7/20 questions had every agent flat).

    With the flat rows removed we take the plurality argmax. A genuine tie (or no
    informative agent at all) returns -1 — a non-answer, scored incorrect by the
    caller's `== 0` check rather than leaked onto the truth index. Pass `rng` to
    instead break genuine ties uniformly at random (unbiased plurality estimate)."""
    informative = x_t.max(axis=1) > x_t.min(axis=1)
    if not informative.any():
        return -1
    votes = np.argmax(x_t[informative], axis=1)
    counts = np.bincount(votes, minlength=x_t.shape[1])
    winners = np.flatnonzero(counts == counts.max())
    if len(winners) == 1:
        return int(winners[0])
    return int(rng.choice(winners)) if rng is not None else -1
