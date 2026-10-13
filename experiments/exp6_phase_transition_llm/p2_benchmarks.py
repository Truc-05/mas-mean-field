"""P2b — benchmark adapter registry + benchmark-aware selection (ADAPTER-ONLY).

Locked principles:
  • A loader ONLY normalizes a dataset into the truth@0 `Question = (question, options)`
    format. NO benchmark-specific prompt engineering — every downstream prompt (answer,
    clue, evidence framing) is the SAME frozen text as TruthfulQA.
  • The p-probe (p2_probe.py) is used ONLY to stratify easy (p̂≥0.80) / hard (p̂≤0.30).
    Thresholds are benchmark-independent and frozen; a benchmark with only one populated
    bin contributes that bin.
  • `truthfulqa` dispatches to the ORIGINAL frozen selection (byte-identical), so the
    existing qwen results/caches are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from mfg_llm.llm_debate.benchmarks import load_aqua, load_mc_jsonl, load_truthfulqa_mc1

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "raw" / "benchmarks"
PROC = ROOT / "data" / "processed"

# frozen, benchmark-INDEPENDENT stratification (same bins as the TruthfulQA run)
PRED = {"easy": (lambda p: p >= 0.80), "hard": (lambda p: p <= 0.30)}
PROBE_POOL = 400                       # candidates probed per benchmark

# name -> {loader(path, max_questions) -> list[Question truth@0], path, K}
BENCHMARKS: dict[str, dict] = {
    "truthfulqa": {"loader": lambda p, m: load_truthfulqa_mc1(p, k=4, max_questions=m),
                   "path": BENCH_DIR / "truthfulqa.csv", "K": 4},
    "aqua": {"loader": load_aqua, "path": BENCH_DIR / "aqua_test.json", "K": 5},
    # ── scale-up suite (fetch_benchmarks.py writes truth@0 JSONL; generic loader) ──
    "arc":        {"loader": load_mc_jsonl, "path": BENCH_DIR / "arc_challenge.jsonl", "K": 4},
    "commonsenseqa": {"loader": load_mc_jsonl, "path": BENCH_DIR / "commonsenseqa.jsonl", "K": 5},
    "mmlu":       {"loader": load_mc_jsonl, "path": BENCH_DIR / "mmlu.jsonl", "K": 4},
    "strategyqa": {"loader": load_mc_jsonl, "path": BENCH_DIR / "strategyqa.jsonl", "K": 2},
    "bbh":        {"loader": load_mc_jsonl, "path": BENCH_DIR / "bbh.jsonl", "K": 6},
}


def probe_cache_path(benchmark: str) -> Path:
    return PROC / f"exp6_select_{benchmark}.json"


def load_pool(benchmark: str) -> list:
    spec = BENCHMARKS.get(benchmark)
    if spec is None:
        raise SystemExit(f"unknown benchmark '{benchmark}'; known: {sorted(BENCHMARKS)}")
    if not Path(spec["path"]).exists():
        raise SystemExit(f"benchmark '{benchmark}' data missing at {spec['path']} — "
                         f"download it into data/raw/benchmarks/ first (adapter-only, no code change)")
    return spec["loader"](str(spec["path"]), PROBE_POOL)


def select_questions(regime: str, n: int, benchmark: str = "truthfulqa") -> list:
    """Benchmark-aware selection. All seven benchmarks now share ONE probe path.

    truthfulqa previously took a legacy branch into `exp6c_shared_context.select_questions`,
    which read the `exp6_easy_truthfulqa_p*.json` caches produced by the g-sweep driver's
    own pre-probe. That branch was removed when the stimulus was rebuilt in English
    (2026-10-01), for two reasons. (1) It is threshold-identical: the legacy predicates are
    `p>=0.80` / `p<=0.30`, exactly `PRED` here, so unifying changes no selection rule, only
    which cache holds the p-hat and how large the candidate pool is (400 for every benchmark
    now, instead of 400 for hard and 250 for easy). (2) It made the Tier-1 question set
    depend on a cache that only the g-sweep driver regenerates, so an English rebuild of
    Tier-1 could not refresh truthfulqa's selection without also running the copying
    experiment. The legacy caches are retained under the archive directory for provenance.
    """
    pool = load_pool(benchmark)
    cache = probe_cache_path(benchmark)
    if not cache.exists():
        raise SystemExit(f"probe cache {cache.name} missing — run "
                         f"`python p2_probe.py --benchmark {benchmark} --config <cfg>` first")
    phat = {int(k): v for k, v in json.loads(cache.read_text())["phat"].items()}
    pred = PRED[regime]
    idx = [i for i in sorted(phat) if pred(phat[i])][:n]
    return [pool[i] for i in idx]
