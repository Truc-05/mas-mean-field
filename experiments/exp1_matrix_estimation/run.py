"""Exp 1: estimate A, B, s, sigma from real LLM debate logs.

Runs debates over several questions — diverse questions are REQUIRED to identify
B (one trajectory leaves M = A - B rank-deficiently excited) — then pools the
logs in canonical coordinates (true answer permuted to index 0) and fits the
two-stage estimator. Option display order is shuffled per debate to decorrelate
LLM position bias from the truth signal.

Usage:
  export GROQ_API_KEY=gsk_...
  python experiments/exp1_matrix_estimation/run.py --config src/config/debate_N5.yaml
  python experiments/exp1_matrix_estimation/run.py --config src/config/debate_N5.yaml --n-questions 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from mfg_llm.estimation.regression import DebateLog, estimate_matrices, estimate_matrices_two_stage, spectrum_report
from mfg_llm.estimation.validate_fit import validate_prediction
from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter
from mfg_llm.llm_debate.orchestrator import DebateOrchestrator
from mfg_llm.metrics.accuracy import accuracy_at_t
from mfg_llm.metrics.stats import summarize

ROOT = Path(__file__).resolve().parents[2]

# K = 4 for every question; the first option listed is the true answer a*.
QUESTIONS = [
    ("17 x 23 = ?", ["391", "381", "401", "371"]),
    ("Thủ đô của Úc là gì?", ["Canberra", "Sydney", "Melbourne", "Perth"]),
    ("Số nguyên tố nhỏ nhất lớn hơn 30 là số nào?", ["31", "33", "37", "39"]),
    ("Nước nào có diện tích lớn nhất thế giới?", ["Nga", "Canada", "Trung Quốc", "Mỹ"]),
    ("Căn bậc hai của 289 là bao nhiêu?", ["17", "19", "13", "23"]),
    ("Nguyên tố hóa học có ký hiệu Fe là gì?", ["Sắt", "Đồng", "Kẽm", "Chì"]),
    ("2^10 bằng bao nhiêu?", ["1024", "512", "2048", "1000"]),
    ("Hành tinh lớn nhất Hệ Mặt Trời là gì?", ["Mộc tinh", "Thổ tinh", "Trái Đất", "Hỏa tinh"]),
]


def run_one_debate(question: str, options: list[str], cfg: dict,
                   rng: np.random.Generator) -> tuple[DebateLog, float, str]:
    """Run one debate with shuffled display order; return the log in canonical
    coordinates (truth at column 0), final-round accuracy, and final answer."""
    shuffled = list(options)
    rng.shuffle(shuffled)
    perm = [shuffled.index(opt) for opt in options]

    limiter = get_rate_limiter(cfg["rpm"]) if cfg.get("rpm") else None
    agents = [
        DebateAgent(i, model=cfg["model"], backend=cfg.get("backend", "groq"),
                    temperature=cfg.get("temperature", 1.0), rate_limiter=limiter)
        for i in range(cfg["n_agents"])
    ]
    orch = DebateOrchestrator(agents, question, shuffled,
                              n_rounds=cfg["n_rounds"], n_workers=cfg.get("n_workers", 4))
    states = orch.run()

    states_canon = [x[:, perm] for x in states]
    log = DebateLog(states_by_round=states_canon, eta=cfg.get("eta", 1.0))
    acc = accuracy_at_t(states_canon[-1], a_star_idx=0)
    return log, acc, orch.final_answer()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_N5.yaml"))
    parser.add_argument("--n-questions", type=int, default=len(QUESTIONS))
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("backend", "groq") == "groq" and not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY chưa được set. Chạy: export GROQ_API_KEY=gsk_...  (key free tại console.groq.com)")

    rng = np.random.default_rng(cfg.get("seed", 0))
    questions = QUESTIONS[: args.n_questions]

    raw_dir = ROOT / "data" / "raw"
    params_dir = ROOT / "data" / "estimated_params"
    tables_dir = ROOT / "results" / "tables"
    for d in (raw_dir, params_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    logs, rows = [], []
    for qi, (question, options) in enumerate(questions):
        print(f"[{qi + 1}/{len(questions)}] {question}")
        log, acc, answer = run_one_debate(question, options, cfg, rng)
        logs.append(log)
        correct = answer == options[0]
        print(f"    đáp án cuối: {answer!r} ({'ĐÚNG' if correct else 'SAI'}), accuracy vòng cuối = {acc:.2f}")

        np.savez(raw_dir / f"exp1_states_q{qi}.npz",
                 states=np.stack(log.states_by_round, axis=0),
                 question=question, answer_space=np.array(options))
        rows.append({"question": question, "final_answer": answer, "correct": correct,
                     "final_accuracy": acc})

    params = estimate_matrices_two_stage(logs)
    joint = estimate_matrices(logs)
    report = spectrum_report(params)

    fits = [validate_prediction(log, params) for log in logs]
    mean_errs = [f["mean_relative_error"] for f in fits]
    trace_errs = [f["trace_relative_error"] for f in fits]

    acc_stats = summarize([r["final_accuracy"] for r in rows])
    print("\n=== Ước lượng (two-stage, pooled qua các câu hỏi) ===")
    with np.printoptions(precision=3, suppress=True):
        print(f"A =\n{params.A}")
        print(f"B =\n{params.B}")
        print(f"s = {params.s}   sigma = {params.sigma:.4f}")
    print(f"phổ trên H (đã khử gauge): lambda_min(A|H) = {report['lambda_min_A_H']:.3f}   "
          f"lambda_max(B|H) = {report['lambda_max_B_H']:.3f}   "
          f"min Re lambda(M|H) = {report['min_re_lambda_M_H']:.3f}")
    print(f"fixed point ổn định trên H: {report['stable_fixed_point_H']}")
    print(f"accuracy vòng cuối (qua {len(rows)} câu): {acc_stats}")
    print(f"validate fit — mean err: {summarize(mean_errs)}   trace err: {summarize(trace_errs)}")

    out_json = params_dir / "exp1_params.json"
    out_json.write_text(json.dumps({
        "config": cfg,
        "n_questions": len(questions),
        "two_stage": {"A": params.A.tolist(), "B": params.B.tolist(),
                      "s": params.s.tolist(), "sigma": params.sigma,
                      "residual_norm": params.residual_norm},
        "joint": {"A": joint.A.tolist(), "B": joint.B.tolist(),
                  "s": joint.s.tolist(), "sigma": joint.sigma},
        "spectrum": {k: v for k, v in report.items() if isinstance(v, (int, float, bool))},
        "final_accuracy": {"mean": acc_stats.mean, "std": acc_stats.std,
                           "ci": [acc_stats.ci_low, acc_stats.ci_high], "n": acc_stats.n},
        "validate_fit": {"mean_rel_err": mean_errs, "trace_rel_err": trace_errs},
    }, ensure_ascii=False, indent=2))

    out_csv = tables_dir / "exp1_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out_json}")
    print(f"wrote {out_csv}")
    print(f"wrote {raw_dir}/exp1_states_q*.npz")


if __name__ == "__main__":
    main()
