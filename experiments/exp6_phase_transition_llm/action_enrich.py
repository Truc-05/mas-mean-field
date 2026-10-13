"""Action-enrichment falsification (HARD-only) — the CDCR go/no-go on ACTIONABILITY.

FROZEN 2026-08-18 (before any A1 number is seen). Executes the pre-registered gate in
README §5 "NEXT GATE — Action-enrichment falsification". K2 / K2.5-pre established that no
router on the A0 action set beats static shared_ev (pooled H^CV +0.037, HARD H^CV ≈ 0.003):
the bottleneck is ACTIONABILITY, not feature quality. So the question here is NOT a new
classifier — it is whether THREE mechanism-derived INTERVENTIONS open routing headroom that
A0 cannot. Falsification, not "find a winner".

    A0 = {single, indep_vote, diversity, shared_ev, debate, stop}      (the K1 set)
    A1 = A0 ∪ {a_copy, a_common, a_dissent}                            (the 3 new operators)

THE THREE OPERATORS (truth-independent: no correct answer revealed, no oracle evidence).
  a_copy  (Break-Copy, downstream social copying) — FROZEN design (user-confirmed):
          run `debate_rounds` COUPLED rounds to INDUCE copying, then STRIP the peer context
          but keep each agent's OWN last reasoning, and re-elicit under an explicit
          "ignore the others, decide independently" directive → plurality. Distinct from
          indep_vote (agents went through debate first) and from debate (vote is post-reset).
  a_common (Break-Common, upstream common evidence) — each agent gets its OWN independently
          resampled STRONG correct-leaning clue (exp6c indep_correct_strong construction),
          one independent pass → plurality. Distinct from shared_ev (a COMMON clue) and from
          indep_vote (no evidence). Caveat, symmetric with how A0 treats shared_ev: the clue
          is correct-leaning ⇒ this is an UPPER BOUND on a real quality-uncertain retriever.
  a_dissent (Counterfactual Dissent, consensus lock-in) — each agent, independently and with
          NO peer signal, is instructed to first build the strongest case for an ALTERNATIVE
          option (steelman its own counter-hypothesis) before scoring → plurality. Pure
          forced-decorrelation; reuses the validated persona mechanism (cf. `diversity`).

WHAT IS EMITTED. Per HARD question q, per replicate r, the correctness of every A1 action —
same JSON schema as k1_oracle_headroom.py (meta + questions[{qid,K,reps{action:[...]}}]), so
action_enrich_analyze.py reuses the k1 crossfit/oracle machinery. `single` is the mean
individual correctness of the N agents; the voting actions are 0/1 plurality-correct.

FROZEN REPORT (all THREE — H alone hides the uniform-lift failure mode), computed paired on
THIS run (A0 recomputed here, so nothing depends on matching K1's exact prior numbers):
    Δ_fixed  = A_bestfixed(A1) − A_bestfixed(A0)        # uniform lift? (good intervention, no CDCR)
    Δ_oracle = A_oracle(A1)    − A_oracle(A0)
    H^CV_new = A_oracle^cf(A1) − A_bestfixed(A1)        # routing headroom — the CDCR-relevant one
LOCKED GATE on H^CV_hard,new:  <0.02 KILL · 0.02–0.05 conditional · ≥0.05 restored · 0.08–0.10+ strong.
SECONDARY WIN (independent algorithm contribution, no router needed): any single new operator
with Δ_fixed > 0 that also beats the best A0 baseline — validate cross-model as follow-up.

FROZEN mechanism-matching (pre-registered, weighs more than raw accuracy). Per-question
mechanism label from THIS run's own A0 means (the two mechanisms, measured as their effect on
the collective vote): copying-heavy = |y_debate − y_indep_vote| large (downstream copying moved
the vote); common-heavy = |y_shared_ev − y_indep_vote| large (upstream common evidence moved it).
Prediction: copying-heavy q ⇒ a_copy gains most; common-heavy q ⇒ a_common gains most. A
`targeted-action × mechanism` interaction (not just marginal accuracy) is the publishable claim.

KILL CRITERION (locked): exactly ONE round, these THREE operators, HARD-only. No prompt-shopping,
no trying 15 and keeping the best 3. If H^CV_hard,new < 0.02 AND no operator gives Δ_fixed>0 →
CDCR path stays killed; the K2/K2.5 negative chain is retained as the motivating result.

Run (HARD only; hits the LLM — run yourself). Shardable across 2 GPUs by question range:
  python action_enrich.py --config src/config/debate_qwen_gpu1.yaml --n-agents 11 \
      --replicates 6 --debate-rounds 3 --q-start 0  --q-count 20 --tag s0
  python action_enrich.py --config src/config/debate_qwen_gpu0.yaml --n-agents 11 \
      --replicates 6 --debate-rounds 3 --q-start 20 --q-count 20 --tag s1
Then:
  python action_enrich_analyze.py            # merges shards, prints the gate verdict
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
from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter    # noqa: E402
from mfg_llm.llm_debate.benchmarks import plurality_prediction         # noqa: E402
from mfg_llm.llm_debate.orchestrator import DebateOrchestrator          # noqa: E402
from mfg_llm.opinion_state import extract_opinion_state                 # noqa: E402

# reuse K1's persona set (identical → `diversity` stays comparable to the K1 run)
from k1_oracle_headroom import PERSONAS                                 # noqa: E402

ROOT = _HERE.parents[1]
TABLES = ROOT / "results" / "tables"
_LABELS = "ABCDEFGHIJ"

A0 = ["single", "indep_vote", "diversity", "shared_ev", "debate"]      # +stop in analysis
NEW = ["a_copy", "a_common", "a_dissent"]
ACTIONS = A0 + NEW
SEV_COND = "shared_correct_strong"                                      # a0 shared_ev realization
IND_COND = "indep_correct_strong"                                       # a_common realization

# a_dissent stance: forced counterfactual — build the strongest case for an ALTERNATIVE first.
DISSENT_PERSONA = ("Before you answer, ARGUE AGAINST YOURSELF: construct the STRONGEST "
                   "possible case for an option OTHER than your initial intuition, take "
                   "seriously the possibility that this other option is correct, and only "
                   "THEN decide.")


def _correct(row: np.ndarray) -> bool:
    return bool(row.max() > row.min() and int(np.argmax(row)) == 0)


def _plur_correct(X: np.ndarray, rng: np.random.Generator) -> int:
    return int(plurality_prediction(X, rng) == 0)


def _own_reasoning(agent: DebateAgent) -> str:
    """Agent's own last-turn reasoning text (post-debate), for the Break-Copy re-elicit.
    Prefer the parsed `reasoning` field; fall back to the raw reply. Truncated to bound the
    prompt. Returns '' if the agent has no transcript."""
    if not agent.transcript:
        return ""
    text = agent.transcript[-1].get("response") or ""
    try:
        import re
        m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text, re.DOTALL)
        if m:
            return m.group(1)[:600]
    except Exception:
        pass
    return str(text)[:600]


def _custom_elicit(agent: DebateAgent, prompt: str, answer_space: list[str]) -> np.ndarray:
    """elicit_belief's flat-retry loop, but on an ARBITRARY prompt (reuses the agent's own
    validated chat+parse path, so scoring is identical to every other action)."""
    x = np.zeros(len(answer_space))
    for _ in range(agent.max_flat_retries + 1):
        text = agent._chat_with_retries(prompt)
        scores = agent._parse_scores(text, answer_space)
        agent.transcript.append({"question": prompt[:80], "response": text, "scores": scores})
        x = extract_opinion_state(scores, answer_space)
        if x.max() > x.min():
            return x
    agent.parse_failures += 1
    return x


def _break_copy_prompt(agent: DebateAgent, question: str, answer_space: list[str],
                       own_reasoning: str) -> str:
    """Rebuild the standard scoring prompt with NO peer context and NO evidence, but prepend
    the agent's own prior reasoning + an explicit ignore-the-group / decide-independently
    directive. Mirrors DebateAgent._build_prompt's format so parsing is unchanged."""
    labels = [_LABELS[i] for i in range(len(answer_space))]
    options = "\n".join(f"{lab}. {a}" for lab, a in zip(labels, answer_space))
    role = f"{agent.persona}\n\n" if agent.persona else ""
    own = (f"\n\nYour OWN EARLIER reasoning:\n{own_reasoning}" if own_reasoning else "")
    return (
        f"{role}Question: {question}\n\nOptions:\n{options}{own}\n\n"
        "COMPLETELY IGNORE the views and choices of the other agents in the group. Relying "
        "ONLY on your own INDEPENDENT judgement, reconsider the question from scratch, then "
        "rate how strongly you believe EACH label "
        f"({', '.join(labels)}) is correct, on a scale from -10 to 10, where 10 means "
        "certainly correct. Reply with JSON ONLY, in the form "
        '{"scores": {"A": <number>, "B": <number>, ...}, "reasoning": "<brief>"}.'
    )


def _break_copy_rollout(question: str, shuffled: list[str], perm: list[int],
                        plain_agents: list[DebateAgent], n_workers: int) -> np.ndarray:
    """AFTER a coupled debate has run (plain_agents' transcripts hold their peer-influenced
    last reasoning): strip peers, keep own reasoning, re-elicit independently → (N,K) canonical."""
    own = [_own_reasoning(a) for a in plain_agents]                    # capture BEFORE re-elicit
    N = len(plain_agents)
    res: list = [None] * N

    def _call(i: int) -> np.ndarray:
        p = _break_copy_prompt(plain_agents[i], question, shuffled, own[i])
        return _custom_elicit(plain_agents[i], p, shuffled)

    with ThreadPoolExecutor(max_workers=min(n_workers, N)) as ex:
        futs = {ex.submit(_call, i): i for i in range(N)}
        for f in futs:
            i = futs[f]
            try:
                res[i] = f.result()
            except Exception:
                res[i] = np.zeros(len(shuffled))
    return np.stack(res, axis=0)[:, perm]                              # canonical, truth@0


def eval_question(qi: int, question: str, opts: list[str], clues_q: dict,
                  plain_agents: list[DebateAgent], persona_agents: list[DebateAgent],
                  dissent_agents: list[DebateAgent], R: int, debate_rounds: int,
                  n_workers: int, coupling: float) -> dict:
    """A1 = A0 + {a_copy, a_common, a_dissent} on ONE question, R replicates. Each action uses
    its OWN freshly-seeded rng (display layout seeded by qi only ⇒ paired across actions), so
    the order of passes never couples results. The debate rollout is SHARED between the
    `debate` action and `a_copy` (one coupled debate, then the copy-break re-elicit)."""
    N = len(plain_agents)
    reps: dict[str, list[float]] = {a: [] for a in ACTIONS}
    none_ev = [None] * N
    sev_ci = CONDITIONS.index(SEV_COND)
    ind_ci = CONDITIONS.index(IND_COND)
    for r in range(R):
        disp = lambda: np.random.default_rng([12345, qi])              # noqa: E731 fixed layout

        # (1) independent no-evidence pass → single + indep_vote
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

        # (NEW) a_common: each agent its OWN independently-resampled strong correct-leaning clue
        ev_ind = evidence_for(IND_COND, clues_q, len(opts), N,
                              np.random.default_rng([qi, r, ind_ci]))
        X_com = answer_question(question, opts, ev_ind, plain_agents, disp())
        reps["a_common"].append(_plur_correct(X_com, np.random.default_rng([7, qi, r, 6])))

        # (NEW) a_dissent: forced counterfactual personas, independent (no peers, no evidence)
        X_dis = answer_question(question, opts, none_ev, dissent_agents, disp())
        reps["a_dissent"].append(_plur_correct(X_dis, np.random.default_rng([7, qi, r, 7])))

        # (4) debate: coupled multi-round → final plurality; SHARED with a_copy below
        shuffled = list(opts)
        np.random.default_rng([12345, qi]).shuffle(shuffled)           # same layout
        perm = [shuffled.index(o) for o in opts]
        orch = DebateOrchestrator(plain_agents, question, shuffled, n_rounds=debate_rounds,
                                  n_workers=n_workers, coupling=coupling,
                                  rng=np.random.default_rng([777, qi, r]))
        final = orch.run()[-1][:, perm]
        reps["debate"].append(_plur_correct(final, np.random.default_rng([7, qi, r, 4])))

        # (NEW) a_copy: break the copy — strip peers, keep own reasoning, re-decide independently
        X_copy = _break_copy_rollout(question, shuffled, perm, plain_agents, n_workers)
        reps["a_copy"].append(_plur_correct(X_copy, np.random.default_rng([7, qi, r, 5])))

        for a in (*plain_agents, *persona_agents, *dissent_agents):    # bound memory
            a.transcript.clear()
    return reps


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_qwen_gpu1.yaml"))
    ap.add_argument("--regime", default="hard", choices=["hard", "easy"],
                    help="FROZEN: HARD is the sharpest falsifier (H^CV_hard≈0.003)")
    ap.add_argument("--n-questions", type=int, default=40, help="size of the stratified set")
    ap.add_argument("--q-start", type=int, default=0, help="global index to start at (sharding)")
    ap.add_argument("--q-count", type=int, default=None, help="how many from q-start (default: rest)")
    ap.add_argument("--n-agents", type=int, default=11)
    ap.add_argument("--replicates", type=int, default=6)
    ap.add_argument("--debate-rounds", type=int, default=3)
    ap.add_argument("--benchmark", default="truthfulqa",
                    help="P2b benchmark adapter; selection + clue cache follow it")
    ap.add_argument("--tag", default="s0", help="shard/output tag")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    N, R = args.n_agents, args.replicates
    coupling = float(cfg.get("coupling", 1.0))
    n_workers = int(cfg.get("n_workers", 4))
    TABLES.mkdir(parents=True, exist_ok=True)

    # `--benchmark` was previously absent here, so both the selection and the clue cache
    # defaulted to truthfulqa and this runner could not reach any other benchmark. That was
    # the blocker behind the dropped Tier-2 depth tier; threading it costs nothing and
    # removes a latent trap, even though C4 is reported on truthfulqa/hard only.
    questions = select_questions(args.regime, args.n_questions, args.benchmark)
    clues = build_clues(args.regime, questions, cfg, N, False, args.benchmark)  # cached (no LLM)
    q0 = args.q_start
    q1 = len(questions) if args.q_count is None else min(len(questions), q0 + args.q_count)

    print("=" * 76)
    print(f"ACTION-ENRICHMENT (A1) — regime={args.regime}  N={N}  R={R}  "
          f"debate_rounds={args.debate_rounds}  model={cfg['model']}")
    print(f"A0={A0}+stop   NEW={NEW}   coupling={coupling}")
    print(f"questions [global {q0}:{q1}] of {len(questions)}   tag={args.tag}   "
          f"base_url={cfg.get('base_url')}")
    print("emits per-q per-replicate A1 correctness; gate computed by action_enrich_analyze.py")
    print("=" * 76)

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

    out_json = TABLES / f"actionenrich_{args.tag}_{args.regime}_N{N}.json"
    meta = {"regime": args.regime, "N": N, "R": R, "debate_rounds": args.debate_rounds,
            "model": cfg["model"], "coupling": coupling, "A0": A0, "NEW": NEW,
            "actions": ACTIONS, "shared_ev_cond": SEV_COND, "indep_cond": IND_COND,
            "q_start": q0, "q_end": q1}
    q_records = []
    for qi in range(q0, q1):
        q, opts = questions[qi]
        reps = eval_question(qi, q, opts, clues[qi], plain_agents, persona_agents,
                             dissent_agents, R, args.debate_rounds, n_workers, coupling)
        q_records.append({"qid": qi, "K": len(opts), "reps": reps})
        out_json.write_text(json.dumps({"meta": meta, "questions": q_records},
                                       ensure_ascii=False))                # incremental save
        means = {a: float(np.mean(reps[a])) for a in ACTIONS}
        print(f"[q{qi:02d} K={len(opts)}] " +
              "  ".join(f"{a}={means[a]:.2f}" for a in ACTIONS), flush=True)

    print(f"\nwrote {out_json}  ({len(q_records)} questions)")


if __name__ == "__main__":
    main()
