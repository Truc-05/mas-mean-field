"""P2b probe — single-agent p̂ per candidate question, to stratify easy/hard for a benchmark.

ADAPTER-ONLY: measures competence with the SAME frozen no-evidence answer prompt as every
run (no benchmark-specific prompting). Writes data/processed/exp6_select_{benchmark}.json
= {"phat": {qi: p̂}}; select_questions() then applies the frozen easy≥0.80 / hard≤0.30 bins.
Needs an Ollama server (LLM calls). Not needed for truthfulqa (it has frozen caches already).

  python p2_probe.py --benchmark aqua --config src/config/debate_gpu1.yaml \
      --n-agents 11 --replicates 4 --max-questions 400
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_benchmarks as p2b                                        # noqa: E402
from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter  # noqa: E402


def _correct(row: np.ndarray) -> bool:
    row = np.asarray(row, float)
    return bool(row.max() > row.min() and int(np.argmax(row)) == 0)  # informative & truth@0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-agents", type=int, default=11)
    ap.add_argument("--replicates", type=int, default=4)
    ap.add_argument("--max-questions", type=int, default=p2b.PROBE_POOL)
    ap.add_argument("--n-workers", type=int, default=8)
    args = ap.parse_args(argv)
    # A truthfulqa guard used to sit here ("already has frozen selection caches; probe not
    # needed"). It was removed with the English rebuild (2026-10-01). It was the third of
    # three special cases for the same thing, alongside `p2_benchmarks.select_questions`
    # and `tier1_breadth.probe_count`; together they made truthfulqa's question selection
    # depend on caches that only the g-sweep driver regenerates, so Tier-1 could not be
    # rebuilt on its own. All seven benchmarks are now probed here, with identical
    # thresholds (`p2_benchmarks.PRED`) and an identical candidate pool.

    cfg = yaml.safe_load(Path(args.config).read_text())
    pool = p2b.load_pool(args.benchmark)[:args.max_questions]
    N, R = args.n_agents, args.replicates
    print(f"probe {args.benchmark}: {len(pool)} candidates, N={N} R={R}, model={cfg['model']}")

    limiter = get_rate_limiter(cfg["rpm"]) if cfg.get("rpm") else None
    agents = [DebateAgent(i, model=cfg["model"], backend=cfg.get("backend", "openai"),
                          base_url=cfg.get("base_url"), temperature=cfg.get("temperature", 1.0),
                          rate_limiter=limiter, timeout=cfg.get("timeout", 300))
              for i in range(N)]

    phat: dict[int, float] = {}
    for qi, (q, opts) in enumerate(pool):
        shuffled = list(opts)
        np.random.default_rng([12345, qi]).shuffle(shuffled)       # same frozen display layout
        perm = [shuffled.index(o) for o in opts]
        correct: list[bool] = []
        for _ in range(R):
            def _one(i):
                try:
                    xs = agents[i].elicit_belief(q, shuffled, None, None)
                    return _correct(np.asarray(xs, float)[perm])
                except Exception:
                    return False
            with ThreadPoolExecutor(max_workers=min(args.n_workers, N)) as ex:
                correct.extend(ex.map(_one, range(N)))
            for a in agents:
                a.transcript.clear()
        phat[qi] = float(np.mean(correct)) if correct else 0.0
        print(f"  q{qi:03d} p̂={phat[qi]:.2f}", flush=True)

    out = p2b.probe_cache_path(args.benchmark)
    out.write_text(json.dumps({"benchmark": args.benchmark, "model": cfg["model"],
                               "n_agents": N, "replicates": R, "phat": phat}, ensure_ascii=False))
    vals = np.array(list(phat.values()))
    print(f"\nwrote {out}  ({len(phat)} q; easy≥.80={int((vals >= 0.8).sum())} "
          f"hard≤.30={int((vals <= 0.3).sum())})")


if __name__ == "__main__":
    main()
