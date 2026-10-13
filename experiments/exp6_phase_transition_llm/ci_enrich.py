"""Counterfactual Isolation (a_CI) — copying-targeted intervention runner (HARD-only).

FROZEN 2026-08-21 (Design A′). Implements CI_PROTOCOL_FROZEN.md. This is a DAMAGE–REPAIR
experiment: induce copying on every question (coupling=1.0), then test whether a_CI repairs the
copying-induced loss. It is a MECHANICAL extension of action_enrich.py — same estimator, same
questions, same clue pool — with ONE new operator, a_CI:

    ACTIONS = {single, indep_vote, diversity, shared_ev, debate, a_copy, a_common, a_dissent} + {a_CI}

a_CI (Counterfactual Isolation): after the coupled debate produces the contaminated consensus X_c
(its plurality label), strip peer context but keep each agent's OWN post-debate reasoning, then have
each agent INDEPENDENTLY build the strongest case AGAINST X_c (allowed to keep X_c if it survives),
commit, and re-vote → plurality. = a_copy's post-contamination context + a_dissent's counterfactual
move, aimed at the consensus label specifically. Prompt frozen in CI_PROTOCOL_FROZEN.md §1.

X_c is read from the SAME debate rollout used by the `debate`/`a_copy` actions (no extra debate).
own-reasoning is SNAPSHOT once right after the debate, and BOTH a_copy and a_CI re-elicit from that
snapshot — so their transcript appends never contaminate each other (order-independent, paired).

Emits the same JSON schema as action_enrich.py (meta + questions[{qid,K,reps{action:[...]}}]); the
gate is computed by ci_enrich_analyze.py (Δ_CI recovery, Recovery Ratio, C_q-interaction).

Run (HARD only; hits the LLM). Shardable across 2 GPUs by question range:
  python ci_enrich.py --config src/config/debate_qwen_gpu1.yaml --n-agents 11 \
      --replicates 8 --debate-rounds 3 --q-start 0  --q-count 20 --tag s0
  python ci_enrich.py --config src/config/debate_qwen_gpu0.yaml --n-agents 11 \
      --replicates 8 --debate-rounds 3 --q-start 20 --q-count 20 --tag s1
Then:
  python ci_enrich_analyze.py            # merges shards, prints the CI damage–repair gate
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from exp6c_shared_context import (                     # noqa: E402  (validated machinery)
    CONDITIONS, answer_question, build_clues, evidence_for, select_questions,
)
# reuse ALL of action_enrich's validated helpers/constants — a_CI is the only new thing
from action_enrich import (                            # noqa: E402
    A0, NEW, SEV_COND, IND_COND, DISSENT_PERSONA, _LABELS,
    _correct, _plur_correct, _own_reasoning, _custom_elicit, _break_copy_prompt,
)
from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter    # noqa: E402
from mfg_llm.llm_debate.benchmarks import plurality_prediction         # noqa: E402
from mfg_llm.llm_debate.orchestrator import DebateOrchestrator          # noqa: E402
from k1_oracle_headroom import PERSONAS                                 # noqa: E402

ROOT = _HERE.parents[1]
TABLES = ROOT / "results" / "tables"

ACTIONS = A0 + NEW + ["a_CI"]      # 8 existing + Counterfactual Isolation


def _ci_prompt(agent: DebateAgent, question: str, answer_space: list[str],
               own_reasoning: str, xc_label: str, xc_text: str) -> str:
    """FROZEN CI prompt (CI_PROTOCOL_FROZEN.md §1). Mirrors _break_copy_prompt's format so the
    parse path is byte-identical, but NAMES the group consensus X_c and forces a counter-case
    against it (agent may keep X_c if it survives the challenge — a decorrelator, not a blind flip)."""
    labels = [_LABELS[i] for i in range(len(answer_space))]
    options = "\n".join(f"{lab}. {a}" for lab, a in zip(labels, answer_space))
    role = f"{agent.persona}\n\n" if agent.persona else ""
    own = (f"\n\nYour OWN EARLIER reasoning (after the group discussion):\n{own_reasoning}"
           if own_reasoning else "")
    return (
        f"{role}Question: {question}\n\nOptions:\n{options}{own}\n\n"
        f"The group is currently leaning toward the answer: {xc_label}. {xc_text}\n\n"
        "DETACH YOURSELF from the group's consensus pressure. Your task: construct the "
        f"STRONGEST possible case AGAINST the consensus answer \"{xc_label}\", and take "
        "SERIOUSLY the possibility that a DIFFERENT option is the correct one. Once you have "
        "argued against it, DECIDE INDEPENDENTLY on your own judgement (you ARE PERMITTED to "
        f"keep \"{xc_label}\" if, after the challenge, you find it genuinely correct). Then "
        f"rate how strongly you believe EACH label ({', '.join(labels)}) is correct, on a "
        "scale from -10 to 10, where 10 means certainly correct. Reply with JSON ONLY, in "
        'the form {"scores": {"A": <number>, "B": <number>, ...}, "reasoning": "<brief>"}.'
    )


def _isolate_rollout(question: str, shuffled: list[str], perm: list[int],
                     agents: list[DebateAgent], own: list[str], prompt_fn,
                     n_workers: int) -> np.ndarray:
    """Generic post-debate re-elicit: strip peers, feed each agent its OWN (snapshotted) reasoning
    via prompt_fn, re-score independently → (N,K) canonical (truth@0). Shared by a_copy and a_CI;
    `own` is passed in (snapshotted before either rollout) so the two never see each other's appends."""
    N = len(agents)
    res: list = [None] * N

    def _call(i: int) -> np.ndarray:
        p = prompt_fn(agents[i], question, shuffled, own[i])
        return _custom_elicit(agents[i], p, shuffled)

    with ThreadPoolExecutor(max_workers=min(n_workers, N)) as ex:
        futs = {ex.submit(_call, i): i for i in range(N)}
        for f in futs:
            i = futs[f]
            try:
                res[i] = f.result()
            except Exception:
                res[i] = np.zeros(len(shuffled))
    return np.stack(res, axis=0)[:, perm]


def eval_question(qi: int, question: str, opts: list[str], clues_q: dict,
                  plain_agents: list[DebateAgent], persona_agents: list[DebateAgent],
                  dissent_agents: list[DebateAgent], R: int, debate_rounds: int,
                  n_workers: int, coupling: float) -> dict:
    """ACTIONS on ONE question, R replicates. Identical to action_enrich.eval_question for the 8
    existing actions (same seeds ⇒ paired), plus a_CI on the post-debate contaminated state."""
    N = len(plain_agents)
    reps: dict[str, list[float]] = {a: [] for a in ACTIONS}
    none_ev = [None] * N
    sev_ci = CONDITIONS.index(SEV_COND)
    ind_ci = CONDITIONS.index(IND_COND)
    for r in range(R):
        disp = lambda: np.random.default_rng([12345, qi])              # noqa: E731 fixed layout

        # (1) independent no-evidence pass → single + indep_vote  (== A_indep)
        X_ind = answer_question(question, opts, none_ev, plain_agents, disp())
        reps["single"].append(float(np.mean([_correct(row) for row in X_ind])))
        reps["indep_vote"].append(_plur_correct(X_ind, np.random.default_rng([7, qi, r, 1])))

        # (2) diversity: independent personas
        X_div = answer_question(question, opts, none_ev, persona_agents, disp())
        reps["diversity"].append(_plur_correct(X_div, np.random.default_rng([7, qi, r, 2])))

        # (3) shared_ev: one COMMON correct-leaning strong clue broadcast to all N
        ev_sev = evidence_for(SEV_COND, clues_q, len(opts), N,
                              np.random.default_rng([qi, r, sev_ci]))
        X_sev = answer_question(question, opts, ev_sev, plain_agents, disp())
        reps["shared_ev"].append(_plur_correct(X_sev, np.random.default_rng([7, qi, r, 3])))

        # (a_common) each agent its OWN independently-resampled strong correct-leaning clue
        ev_ind = evidence_for(IND_COND, clues_q, len(opts), N,
                              np.random.default_rng([qi, r, ind_ci]))
        X_com = answer_question(question, opts, ev_ind, plain_agents, disp())
        reps["a_common"].append(_plur_correct(X_com, np.random.default_rng([7, qi, r, 6])))

        # (a_dissent) forced counterfactual personas, independent (no peers, no evidence)
        X_dis = answer_question(question, opts, none_ev, dissent_agents, disp())
        reps["a_dissent"].append(_plur_correct(X_dis, np.random.default_rng([7, qi, r, 7])))

        # (4) debate: coupled multi-round → contaminated final plurality  (== A_copy). SHARED below.
        shuffled = list(opts)
        np.random.default_rng([12345, qi]).shuffle(shuffled)           # same layout
        perm = [shuffled.index(o) for o in opts]
        orch = DebateOrchestrator(plain_agents, question, shuffled, n_rounds=debate_rounds,
                                  n_workers=n_workers, coupling=coupling,
                                  rng=np.random.default_rng([777, qi, r]))
        final = orch.run()[-1][:, perm]                                # canonical, truth@0
        reps["debate"].append(_plur_correct(final, np.random.default_rng([7, qi, r, 4])))

        # consensus X_c to argue against = debate plurality (fallback: soft argmax on a tie)
        xc = plurality_prediction(final, np.random.default_rng([9, qi, r]))
        if xc < 0:
            xc = int(np.argmax(final.mean(0)))
        xc_pos = shuffled.index(opts[xc])                              # display position of X_c
        xc_label, xc_text = _LABELS[xc_pos], shuffled[xc_pos]

        # snapshot each agent's post-debate reasoning ONCE (before either re-elicit)
        own = [_own_reasoning(a) for a in plain_agents]

        # (a_copy) passive break-copy (the dead contrast) — from the snapshot
        X_copy = _isolate_rollout(question, shuffled, perm, plain_agents, own,
                                  _break_copy_prompt, n_workers)
        reps["a_copy"].append(_plur_correct(X_copy, np.random.default_rng([7, qi, r, 5])))

        # (a_CI) Counterfactual Isolation — same snapshot, argue against X_c
        ci_prompt_fn = lambda ag, q_, sp, ow: _ci_prompt(ag, q_, sp, ow, xc_label, xc_text)  # noqa: E731
        X_ci = _isolate_rollout(question, shuffled, perm, plain_agents, own,
                                ci_prompt_fn, n_workers)
        reps["a_CI"].append(_plur_correct(X_ci, np.random.default_rng([7, qi, r, 8])))

        for a in (*plain_agents, *persona_agents, *dissent_agents):    # bound memory
            a.transcript.clear()
    return reps


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_qwen_gpu1.yaml"))
    ap.add_argument("--regime", default="hard", choices=["hard", "easy"],
                    help="FROZEN: HARD is the sharpest falsifier")
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--q-start", type=int, default=0)
    ap.add_argument("--q-count", type=int, default=None)
    ap.add_argument("--n-agents", type=int, default=11)
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--debate-rounds", type=int, default=3)
    ap.add_argument("--benchmark", default="truthfulqa",
                    help="P2b benchmark adapter; selection + clue cache follow it")
    ap.add_argument("--tag", default="s0")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    N, R = args.n_agents, args.replicates
    coupling = float(cfg.get("coupling", 1.0))
    n_workers = int(cfg.get("n_workers", 4))
    TABLES.mkdir(parents=True, exist_ok=True)

    # See the matching note in action_enrich.py: `--benchmark` was absent, so selection and
    # the clue cache silently defaulted to truthfulqa.
    questions = select_questions(args.regime, args.n_questions, args.benchmark)
    clues = build_clues(args.regime, questions, cfg, N, False, args.benchmark)  # cached (no LLM)
    q0 = args.q_start
    q1 = len(questions) if args.q_count is None else min(len(questions), q0 + args.q_count)

    print("=" * 78)
    print(f"CI ENRICHMENT (a_CI) — regime={args.regime}  N={N}  R={R}  "
          f"debate_rounds={args.debate_rounds}  coupling={coupling}  model={cfg['model']}")
    print(f"ACTIONS={ACTIONS}")
    print(f"questions [global {q0}:{q1}] of {len(questions)}   tag={args.tag}   "
          f"base_url={cfg.get('base_url')}")
    print("=" * 78)

    limiter = get_rate_limiter(cfg["rpm"]) if cfg.get("rpm") else None

    def mk(kind: str | None, base: int) -> list[DebateAgent]:
        return [DebateAgent(base + i, model=cfg["model"], backend=cfg.get("backend", "openai"),
                            base_url=cfg.get("base_url"),
                            temperature=cfg.get("temperature", 1.0), rate_limiter=limiter,
                            timeout=cfg.get("timeout", 300),
                            persona=(PERSONAS[i % len(PERSONAS)] if kind == "div"
                                     else DISSENT_PERSONA if kind == "dis" else None))
                for i in range(N)]

    plain_agents = mk(None, 0)
    persona_agents = mk("div", 1000)
    dissent_agents = mk("dis", 2000)

    out_json = TABLES / f"ci_enrich_{args.tag}_{args.regime}_N{N}.json"
    meta = {"regime": args.regime, "N": N, "R": R, "debate_rounds": args.debate_rounds,
            "model": cfg["model"], "coupling": coupling, "A0": A0, "NEW": NEW + ["a_CI"],
            "actions": ACTIONS, "shared_ev_cond": SEV_COND, "indep_cond": IND_COND,
            "q_start": q0, "q_end": q1, "design": "A_prime_damage_repair"}
    q_records = []
    for qi in range(q0, q1):
        q, opts = questions[qi]
        reps = eval_question(qi, q, opts, clues[qi], plain_agents, persona_agents,
                             dissent_agents, R, args.debate_rounds, n_workers, coupling)
        q_records.append({"qid": qi, "K": len(opts), "reps": reps})
        out_json.write_text(json.dumps({"meta": meta, "questions": q_records},
                                       ensure_ascii=False))                # incremental save
        means = {a: float(np.mean(reps[a])) for a in ACTIONS}
        cq = means["indep_vote"] - means["debate"]
        print(f"[q{qi:02d} K={len(opts)}] indep={means['indep_vote']:.2f} copy={means['debate']:.2f} "
              f"a_CI={means['a_CI']:.2f} a_common={means['a_common']:.2f} a_copy={means['a_copy']:.2f}"
              f"  C_q={cq:+.2f}", flush=True)

    print(f"\nwrote {out_json}  ({len(q_records)} questions)")


if __name__ == "__main__":
    main()
