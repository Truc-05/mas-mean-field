from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from mfg_llm.llm_debate.agent import DebateAgent
from mfg_llm.mean_field import MeanField
from mfg_llm.opinion_state import argmax_answer


class DebateOrchestrator:
    def __init__(self, agents: list[DebateAgent], question: str, answer_space: list[str],
                 n_rounds: int = 5, n_workers: int = 8, coupling: float = 1.0,
                 rng: np.random.Generator | None = None):
        self.agents = agents
        self.question = question
        self.answer_space = answer_space
        self.n_rounds = n_rounds
        self.n_workers = n_workers
        # coupling g ≥ 0 is the social-pressure knob — the control parameter of the
        # real-LLM phase transition (exp6) and the lever of the control law (C2). Each
        # agent is shown its peers' explicit vote counts plus a directive to weight the
        # group consensus by ~g. g = 0 → no social signal → repeated independent voting;
        # g grows → stronger conformity pull, i.e. a larger social-influence matrix B,
        # driving M = A - B across the herding↔truth boundary. (An earlier version only
        # listed peers' mean scores; the LLM largely ignored it, so g barely moved the
        # dynamics — this explicit vote-count + weight framing is the fix.)
        self.coupling = coupling
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.states_by_round: list[np.ndarray] = []

    def _social_signal(self, x_t: np.ndarray, i: int) -> str | None:
        """Explicit peer vote counts + a g-scaled conformity directive for agent i."""
        peers = [j for j in range(len(self.agents)) if j != i]
        votes = [int(np.argmax(x_t[j])) for j in peers
                 if x_t[j].max() > x_t[j].min()]        # only informative peers vote
        if not votes:
            return None
        counts = Counter(votes)
        top_idx, top_c = counts.most_common(1)[0]
        top_ans = self.answer_space[top_idx]
        dist = ", ".join(f"«{self.answer_space[k]}»={c}" for k, c in counts.most_common(3))
        head = (f"Of the {len(votes)} other agents, {top_c}/{len(votes)} chose «{top_ans}» "
                f"(vote distribution: {dist}). ")
        # For g < 1 the directive is a soft % weight that still leaves an independent
        # escape hatch. For g ≥ 1 (full-herding regime, exp6 flip test) we drop the
        # escape hatch and issue a categorical conform-directive so the vote
        # correlation ρ actually saturates toward 1 — the earlier min(g,1)·"put X%
        # weight ... rest on your own judgment" phrasing capped the pull well below
        # ρ→1, so lift could never reach the theoretical p−P_N(p) floor.
        if self.coupling >= 2.0:
            # ρ→1 endpoint: DROP the truth-escape valve entirely. This is the pure
            # conformity rung used to reach the high-ρ tail of the lift-vs-ρ curve
            # (the g≥1 rung below caps ρ because its "decisive evidence" clause lets
            # confident agents hold out). Removing the valve pushes the measured
            # vote-correlation ρ toward its ceiling so we can probe the theoretical
            # ρ→1 limit P_maj→p — the point where, for p<½, herding maximally helps
            # and for p>½ it maximally hurts.
            return (f"The whole group ({len(votes)} agents) has converged on the conclusion "
                    f"«{top_ans}». As a member who follows the group consensus, CHOOSE "
                    f"«{top_ans}» and score it well above every alternative. "
                    "Do not reconsider.")
        if self.coupling >= 1.0:
            # Full-herding rung (ρ high, but < ceiling): SUPPRESS the vote distribution
            # and issue a categorical deference directive. Showing only the winning
            # answer removes the minority anchor that let agents justify holding out,
            # so the vote correlation saturates far closer to 1 than the soft
            # %-weight phrasing did (which capped collapse ≈0.07). The truth-escape
            # valve ("decisive evidence") is kept so this stays a herding pull, not a
            # blind label-copy — the flip must survive an agent that still knows better.
            return (f"The whole group ({len(votes)} agents) has converged on the conclusion "
                    f"«{top_ans}». As a member who follows the group consensus, CHOOSE "
                    f"«{top_ans}» and score it well above every alternative, UNLESS you "
                    "have DECISIVE evidence that the group is wrong.")
        pct = int(round(self.coupling * 100))
        return (head + f"Place about {pct}% of your weight on this group consensus, and "
                "the remainder on your own independent judgement.")

    def _neighbor_summaries(self, x_t: np.ndarray) -> list[str | None]:
        """Per-agent social signal under the coupling knob (see __init__)."""
        if self.coupling <= 0:                          # decoupled → independent agents
            return [None] * len(self.agents)
        return [self._social_signal(x_t, i) for i in range(len(self.agents))]

    def _fallback_state(self, i: int) -> np.ndarray:
        if self.states_by_round:
            return self.states_by_round[-1][i].copy()
        return np.zeros(len(self.answer_space))

    def _round(self, summaries: list[str | None]) -> np.ndarray:
        results = [None] * len(self.agents)
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {
                ex.submit(agent.elicit_belief, self.question, self.answer_space, summaries[i]): i
                for i, agent in enumerate(self.agents)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as err:
                    warnings.warn(f"agent {i} failed after retries ({err!r}); "
                                  "reusing previous-round state")
                    results[i] = self._fallback_state(i)
        return np.stack(results, axis=0)

    def run(self) -> list[np.ndarray]:
        summaries: list[str | None] = [None] * len(self.agents)  # round 0: independent
        for r in range(self.n_rounds):
            x_t = self._round(summaries)
            self.states_by_round.append(x_t)
            summaries = self._neighbor_summaries(x_t)
        return self.states_by_round

    def final_answer(self) -> str:
        mu_bar = MeanField(self.states_by_round[-1]).mean()
        return argmax_answer(mu_bar, self.answer_space)
