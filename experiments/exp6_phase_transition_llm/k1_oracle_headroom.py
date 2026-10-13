"""K1 — oracle action-selection headroom gate (go/no-go before building CDCR).

Motivation (post-Exp6C). Exp6C established the MECHANISM: shared evidence causally raises
inter-agent error dependence (ΔD ≈ 0.26–0.50, 4/4 H1 significant) — yet the *accuracy*
consequence is small and mostly NOT significant (ShareEffect ≈ −0.01…−0.04). Dependence
strength ≠ collective-accuracy change: the outcome also depends on the mechanism, baseline
competence, answer structure, evidence quality, and saturation. So before building a
Context-Dependent Collective-Reasoning (CDCR) controller we must first check whether there
is any accuracy headroom to control at all — that is the K1 gate.

K1 measures that headroom on a COMMON matched question set with the full action set
    A = {single, indep_vote, debate, diversity, shared_ev, stop}.
For each question q and action a we estimate  y_qa = P(a answers q correctly)  over R
stochastic replicates, then (in k1_analyze.py, no LLM calls):
    A_best_fixed = max_a  mean_q y_qa                (best single FIXED policy)
    A_oracle     = mean_q max_a y_qa                 (per-question best action; upper bound)
    H            = A_oracle − A_best_fixed            (headroom a perfect per-q router buys)
    H_M          = A_oracle(full) − A_oracle(no-mech) (no-mech = {single,indep_vote,stop})
                   — do debate/diversity/shared_ev open headroom the base actions cannot?

This RUNNER produces per-question, per-replicate correctness for ONE regime; the pooled
H / H_M / bootstrap-CI / oracle-frequency / action-complementarity / cost-Pareto / gate
verdict live in k1_analyze.py (which pools both regimes).

ACTIONS (each answered independently per replicate; truth pinned at canonical index 0):
  single      1 agent, no evidence/personas/peers → its own correctness. Estimated as the
              mean individual correctness of the N independent agents (a low-variance
              estimate of single-agent accuracy on q); a flat/parse-fail row counts wrong.
  indep_vote  N agents, independent, no evidence → plurality vote.
  diversity   N agents each with a DISTINCT persona, independent → plurality
              (error-decorrelation mechanism; NO peer signal).
  shared_ev   N agents, SAME shared correct-leaning STRONG evidence broadcast → plurality
              (exp6C shared_correct_strong: the realistic "good retriever" realization).
  debate      N agents, multi-round DebateOrchestrator (coupling from cfg) → final plurality.
  stop        abstain — scored incorrect (y=0), added by k1_analyze. Never wins the accuracy
              oracle by construction; it is an action for the COST-Pareto / K2 budget only.

DESIGN NOTES.
  • shared_ev uses the CORRECT-leaning realization (a controller injects retrieved evidence
    hoping it helps); this is the optimistic case, so its headroom is an UPPER BOUND on what
    a real, quality-uncertain retriever delivers — flagged again in the analysis.
  • The plug-in oracle max_a ŷ_qa has a winner's-curse upward bias at finite R; per-replicate
    outcomes are stored so k1_analyze can also report a cross-fit (honest) oracle.
  • Display order is seeded by QUESTION only → identical across actions & replicates, so the
    six actions are compared on the SAME shuffled layout (paired, low nuisance variance).

Run (per regime; NOTE: hits the LLM server — run yourself):
  python .../k1_oracle_headroom.py --regime easy --config src/config/debate_gpu1.yaml \
      --n-questions 40 --n-agents 11 --replicates 6 --debate-rounds 3
  python .../k1_oracle_headroom.py --regime hard --config src/config/debate_gpu0.yaml \
      --n-questions 40 --n-agents 11 --replicates 6 --debate-rounds 3
Then:
  python .../k1_analyze.py            # pools both regimes, prints the gate verdict
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                         # sibling import of exp6c helpers

from exp6c_shared_context import (                     # noqa: E402  (reuse validated machinery)
    CONDITIONS, answer_question, build_clues, evidence_for, select_questions,
)
from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter  # noqa: E402
from mfg_llm.llm_debate.benchmarks import plurality_prediction       # noqa: E402
from mfg_llm.llm_debate.orchestrator import DebateOrchestrator        # noqa: E402

ROOT = _HERE.parents[1]
TABLES = ROOT / "results" / "tables"

# distinct reasoning stances for the diversity action (cycle-padded to N). Same-model,
# same-prompt agents make near-identical CORRELATED errors; a persona decorrelates them —
# this is the error-diversity lever, kept independent (no peer signal) so it is not
# confounded with debate's social coupling.
PERSONAS = [
    "Bạn là người hoài nghi, luôn kiểm tra lại những giả định phổ biến trước khi tin.",
    "Bạn là nhà khoa học thực nghiệm, chỉ tin vào bằng chứng kiểm chứng được.",
    "Bạn là luật sư phản biện, chủ động tìm lỗ hổng trong mọi lập luận.",
    "Bạn là người thực dụng, ưu tiên lời giải đơn giản và trực tiếp nhất.",
    "Bạn là sử gia, cân nhắc nguồn gốc và độ tin cậy của từng thông tin.",
    "Bạn là nhà thống kê, cảnh giác với ngộ nhận và thiên kiến phổ biến.",
    "Bạn là kỹ sư, suy luận từ các nguyên lý cơ bản (first principles).",
    "Bạn là nhà báo điều tra, luôn hỏi 'điều này có thật sự đúng không?'.",
    "Bạn là triết gia phân tích, tách bạch định nghĩa khỏi bằng chứng.",
    "Bạn là giáo viên cẩn trọng, giải thích từng bước trước khi kết luận.",
    "Bạn là người theo trực giác, chú ý tới bối cảnh và sắc thái tinh tế.",
]

# action → the per-question replicate list this runner emits (stop is added in analysis)
ACTIONS = ["single", "indep_vote", "diversity", "shared_ev", "debate"]
SEV_COND = "shared_correct_strong"                     # the shared_ev realization (see notes)


def _correct(row: np.ndarray) -> bool:
    """One agent's answer is correct iff its opinion row is informative (not flat) and its
    argmax lands on the canonical truth index 0. A flat row = parse/format failure → wrong
    (never leaked onto the truth index), matching plurality_prediction's non-answer rule."""
    return bool(row.max() > row.min() and int(np.argmax(row)) == 0)


def _plur_correct(X: np.ndarray, rng: np.random.Generator) -> int:
    """Plurality of (N,K) canonical states scored against truth@0; ties broken at random."""
    return int(plurality_prediction(X, rng) == 0)


def eval_question(qi: int, question: str, opts: list[str], clues_q: dict,
                  plain_agents: list[DebateAgent], persona_agents: list[DebateAgent],
                  R: int, debate_rounds: int, n_workers: int, coupling: float) -> dict:
    """All five active actions on ONE question, R replicates → per-replicate correctness.

    Returns {action: [r0, r1, ...]}. `single` entries are the per-replicate MEAN individual
    correctness of the N independent agents (a fraction in [0,1]); the four voting actions
    are 0/1 plurality-correct indicators. Display order is seeded by qi only (paired layout);
    the shared-evidence realization is reseeded per replicate so it genuinely varies."""
    N = len(plain_agents)
    reps: dict[str, list[float]] = {a: [] for a in ACTIONS}
    none_ev = [None] * N
    sev_ci = CONDITIONS.index(SEV_COND)                # mirror exp6c's seeding of this cell
    for r in range(R):
        disp = lambda: np.random.default_rng([12345, qi])          # noqa: E731 fixed layout

        # (1) independent no-evidence pass → serves BOTH single and indep_vote (shared calls)
        X_ind = answer_question(question, opts, none_ev, plain_agents, disp())
        reps["single"].append(float(np.mean([_correct(row) for row in X_ind])))
        reps["indep_vote"].append(_plur_correct(X_ind, np.random.default_rng([7, qi, r, 1])))

        # (2) diversity: independent personas (decorrelation, no peer signal)
        X_div = answer_question(question, opts, none_ev, persona_agents, disp())
        reps["diversity"].append(_plur_correct(X_div, np.random.default_rng([7, qi, r, 2])))

        # (3) shared_ev: one common correct-leaning strong evidence broadcast to all N
        ev = evidence_for(SEV_COND, clues_q, len(opts), N,
                          np.random.default_rng([qi, r, sev_ci]))
        X_sev = answer_question(question, opts, ev, plain_agents, disp())
        reps["shared_ev"].append(_plur_correct(X_sev, np.random.default_rng([7, qi, r, 3])))

        # (4) debate: multi-round social coupling → final-round plurality
        shuffled = list(opts)
        np.random.default_rng([12345, qi]).shuffle(shuffled)       # same layout as above
        perm = [shuffled.index(o) for o in opts]                   # shuffled → canonical
        orch = DebateOrchestrator(plain_agents, question, shuffled, n_rounds=debate_rounds,
                                  n_workers=n_workers, coupling=coupling,
                                  rng=np.random.default_rng([777, qi, r]))
        final = orch.run()[-1][:, perm]
        reps["debate"].append(_plur_correct(final, np.random.default_rng([7, qi, r, 4])))
        for a in plain_agents:                                     # keep memory bounded
            a.transcript.clear()
    return reps


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["easy", "hard"], required=True)
    ap.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_gpu0.yaml"))
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--n-agents", type=int, default=11)
    ap.add_argument("--replicates", type=int, default=6,
                    help="stochastic replicates per question×action (identifies y_qa)")
    ap.add_argument("--debate-rounds", type=int, default=3)
    ap.add_argument("--tag", default="v1", help="output-filename tag")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    N, R = args.n_agents, args.replicates
    coupling = float(cfg.get("coupling", 1.0))
    n_workers = int(cfg.get("n_workers", 4))
    TABLES.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"K1 ORACLE HEADROOM — regime={args.regime}  N={N}  R={R}  "
          f"debate_rounds={args.debate_rounds}  model={cfg['model']}")
    print(f"actions={ACTIONS}+stop   coupling={coupling}   shared_ev={SEV_COND}")
    print("emits per-question, per-replicate correctness; H/H_M/gate computed by k1_analyze.py")
    print("=" * 72)

    questions = select_questions(args.regime, args.n_questions)
    print(f"selected {len(questions)} {args.regime} questions")
    clues = build_clues(args.regime, questions, cfg, N, fresh=False)   # cached (no LLM)

    limiter = get_rate_limiter(cfg["rpm"]) if cfg.get("rpm") else None

    def mk(persona: str | None, base: int) -> list[DebateAgent]:
        return [DebateAgent(base + i, model=cfg["model"], backend=cfg.get("backend", "openai"),
                            base_url=cfg.get("base_url"),
                            temperature=cfg.get("temperature", 1.0), rate_limiter=limiter,
                            timeout=cfg.get("timeout", 300),
                            persona=(PERSONAS[i % len(PERSONAS)] if persona == "div" else None))
                for i in range(N)]

    plain_agents = mk(None, 0)
    persona_agents = mk("div", 1000)

    out_json = TABLES / f"k1_yqa_{args.tag}_{args.regime}_N{N}.json"
    out_csv = TABLES / f"k1_yqa_{args.tag}_{args.regime}_N{N}.csv"
    meta = {"regime": args.regime, "N": N, "R": R, "debate_rounds": args.debate_rounds,
            "model": cfg["model"], "coupling": coupling, "actions": ACTIONS,
            "shared_ev_cond": SEV_COND}
    q_records = []
    for qi, (q, opts) in enumerate(questions):
        reps = eval_question(qi, q, opts, clues[qi], plain_agents, persona_agents,
                             R, args.debate_rounds, n_workers, coupling)
        q_records.append({"qid": qi, "K": len(opts), "reps": reps})
        means = {a: float(np.mean(reps[a])) for a in ACTIONS}
        # incremental write so a killed run keeps what it has
        out_json.write_text(json.dumps({"meta": meta, "questions": q_records},
                                       ensure_ascii=False))
        print(f"[q{qi:02d} K={len(opts)}] " +
              "  ".join(f"{a}={means[a]:.2f}" for a in ACTIONS))

    # flat CSV of per-question means (for a quick glance; JSON is the source of truth)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "qid", "K"] + [f"y_{a}" for a in ACTIONS])
        for rec in q_records:
            w.writerow([args.regime, rec["qid"], rec["K"]] +
                       [f"{np.mean(rec['reps'][a]):.4f}" for a in ACTIONS])

    # quick single-regime plug-in headroom (sanity only; authoritative analysis pools regimes)
    Y = {a: np.array([np.mean(rec["reps"][a]) for rec in q_records]) for a in ACTIONS}
    Y["stop"] = np.zeros(len(q_records))
    acts = ACTIONS + ["stop"]
    Ymat = np.stack([Y[a] for a in acts], axis=1)               # (Q, |A|)
    a_fixed = Ymat.mean(0)
    best_i = int(np.argmax(a_fixed))
    A_best, A_oracle = float(a_fixed[best_i]), float(Ymat.max(1).mean())
    print("\n" + "-" * 72)
    print("per-action mean accuracy: " +
          "  ".join(f"{a}={a_fixed[i]:.3f}" for i, a in enumerate(acts)))
    print(f"[{args.regime}] A_best_fixed={A_best:.3f} ({acts[best_i]})  "
          f"A_oracle={A_oracle:.3f}  H(plug-in, single-regime)={A_oracle - A_best:+.3f}")
    print(f"\nwrote {out_json}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
