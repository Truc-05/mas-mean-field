"""Exp 5: benchmark accuracy — single agent vs voting vs debate.

Three arms per question (canonical coordinates, truth at index 0):
  single: 1 agent, 1 round
  voting: N agents, 1 round, no interaction, plurality vote
  debate: N agents, n_rounds with mean-field summary, plurality at final round
The voting-vs-debate contrast is the empirical face of the martingale claim
(doc section 5, Theorem 1). Debate-arm states are saved for matrix estimation.

Usage:
  export GROQ_API_KEY=gsk_...
  python experiments/exp5_benchmarks/run.py --benchmark aqua --n-questions 20
  python experiments/exp5_benchmarks/run.py --benchmark truthfulqa --n-questions 20
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from mfg_llm.llm_debate.benchmarks import (
    load_aqua, load_truthfulqa_mc1, run_debate_canonical, plurality_prediction,
)
from mfg_llm.metrics.stats import summarize

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "raw" / "benchmarks"

LOADERS = {
    "aqua": lambda n: load_aqua(BENCH_DIR / "aqua_test.json", max_questions=n),
    "truthfulqa": lambda n: load_truthfulqa_mc1(BENCH_DIR / "truthfulqa.csv", max_questions=n),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(LOADERS), required=True)
    parser.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_N5.yaml"))
    parser.add_argument("--n-questions", type=int, default=20)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("backend", "groq") == "groq" and not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY chưa được set. Chạy: export GROQ_API_KEY=gsk_...")

    questions = LOADERS[args.benchmark](args.n_questions)
    if not questions:
        sys.exit(f"không load được câu hỏi nào từ {BENCH_DIR} — kiểm tra file benchmark")

    N, R = cfg["n_agents"], cfg["n_rounds"]
    calls_per_q = 1 + N + N * R
    total_calls = calls_per_q * len(questions)
    rpm = cfg.get("rpm")
    eta_note = f" ≈ {total_calls / rpm:.0f} phút với rpm={rpm}" if rpm else ""
    print(f"benchmark={args.benchmark}  câu hỏi={len(questions)}  "
          f"~{total_calls} API calls (single 1 + voting {N} + debate {N}x{R}){eta_note}\n")

    rng = np.random.default_rng(cfg.get("seed", 0))
    states_dir = ROOT / "data" / "processed"
    tables_dir = ROOT / "results" / "tables"
    states_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    def flat_frac(states: list[np.ndarray]) -> float:
        """Fraction of agent rows that are flat (uninformative / parse-failed)."""
        x = np.stack(states, axis=0)
        return float(np.mean(x.max(axis=-1) - x.min(axis=-1) < 1e-9))

    rows = []
    degen = []  # per-question flatness of the debate arm — data-quality guard
    for qi, (question, options) in enumerate(questions):
        single = run_debate_canonical(question, options, cfg, rng, n_agents=1, n_rounds=1)
        voting = run_debate_canonical(question, options, cfg, rng, n_agents=N, n_rounds=1)
        debate = run_debate_canonical(question, options, cfg, rng, n_agents=N, n_rounds=R)

        row = {
            "question": question[:80],
            "single": int(plurality_prediction(single[-1]) == 0),
            "voting": int(plurality_prediction(voting[-1]) == 0),
            "debate": int(plurality_prediction(debate[-1]) == 0),
        }
        rows.append(row)
        degen.append(flat_frac(debate))
        np.savez(states_dir / f"exp5_{args.benchmark}_q{qi}.npz",
                 states=np.stack(debate, axis=0), question=question,
                 answer_space=np.array(options))
        print(f"[{qi + 1}/{len(questions)}] single={row['single']} voting={row['voting']} "
              f"debate={row['debate']}  | {question[:60]}")

    print(f"\n=== {args.benchmark} (n={len(rows)} câu, N={N}, R={R}) ===")
    for arm in ["single", "voting", "debate"]:
        print(f"{arm:8s} accuracy = {summarize([r[arm] for r in rows])}")

    # Data-quality guard: flat (parse-failed) agents carry no opinion; with the
    # truth pinned at index 0 they used to inflate accuracy, and they make the
    # estimated A/B/s unreliable. A healthy debate run keeps this low.
    degen = np.array(degen)
    print(f"\n[data quality] debate arm: mean flat-agent fraction = {degen.mean():.2f}  "
          f"| questions with >50% flat = {int((degen > 0.5).sum())}/{len(degen)}")
    if degen.mean() > 0.2:
        print("  ⚠ high degeneracy — accuracy/estimation suspect; check parsing or use a stronger model")

    out = tables_dir / f"exp5_{args.benchmark}.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")
    print(f"wrote {states_dir}/exp5_{args.benchmark}_q*.npz  (debate states — dùng được cho ước lượng A,B,s)")


if __name__ == "__main__":
    main()
