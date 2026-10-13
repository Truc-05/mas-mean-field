"""Single source of truth for which benchmarks the evidence intervention is VALID on.

Created 2026-10-03. Before this, the bbh exclusion lived in `tier1_breadth.BENCHMARKS` (the
sweep driver) and `tier1_breadth_aggregate.EXCLUDED_BENCHMARKS` (one analysis), so the churn
analysis silently kept bbh and reported `qwen/bbh/easy ΔY = 0.125*`. An exclusion that is
re-declared per consumer drifts; this module is imported instead.

The rule is about the STIMULUS, so it binds every downstream quantity derived from the
intervention -- ΔD (C3), the susceptibility slope (C2), and decision churn (C1/C5) alike. If a
manipulation is invalid, the decision changes it produces are not mechanism effects, and
measuring churn under it would launder an invalid manipulation into evidence.
"""
from __future__ import annotations

ALL_BENCHMARKS = ("truthfulqa", "aqua", "arc", "commonsenseqa", "mmlu", "strategyqa", "bbh")

# benchmark -> why the evidence intervention cannot be interpreted there
INVALID_FOR_INTERVENTION = {
    "bbh": ("logical_deduction: all 80 selected items are closed-premise constraint puzzles "
            "whose answer is fully determined by premises already in the prompt, so no external "
            "evidence can be informative. The clue generator can only emit plausible world "
            "knowledge, producing content orthogonal to the constraints; injecting it moved the "
            "model off deduction onto plausibility-weighing and LOWERED accuracy "
            "(A_none 0.634 -> A_shared 0.422 easy; 0.438 -> 0.388 hard) -- the only benchmark "
            "where correct-pointing evidence hurts. Not repairable by regenerating clues: the "
            "task class admits no valid stimulus."),
}

VALID_BENCHMARKS = tuple(b for b in ALL_BENCHMARKS if b not in INVALID_FOR_INTERVENTION)


def is_valid(benchmark: str) -> bool:
    return benchmark not in INVALID_FOR_INTERVENTION


def exclusion_note() -> str:
    return "; ".join(f"{b}: {why}" for b, why in INVALID_FOR_INTERVENTION.items()) or "none"
