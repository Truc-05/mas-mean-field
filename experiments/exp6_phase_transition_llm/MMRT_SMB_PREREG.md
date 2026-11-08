# MMRT + SMB — pre-registration of two secondary analyses on frozen Tier-3 data

> **STATUS: FROZEN 2026-11-08, BEFORE either analysis has been run.** No new data, no new LLM
> calls, no dev/test split: both reuse the already-frozen Tier-3 matched-D rows
> (`tier3_matched_d.shared_rows_tier1`/`copy_rows_tier3`) and the already-frozen SGIR episode
> table (`sgir_common.build_episodes`). The risk these two analyses carry is not overfitting a
> fitted rule to held-out data (there is no rule being fit) — it is **metric/test fishing**:
> many plausible calipers, outcome definitions, and correlation statistics exist, and trying
> several until one "works" would be post-hoc. This document commits the primary metric, the
> primary caliper/test, and the aggregation method for each analysis before either is computed.

## MMRT — Mechanism-Matched Repair Test

**Question.** The paper's thesis is "same observed dependence does not imply same mechanism."
This asks the next link: does it also not imply the same *intervention response*? Match
shared-evidence and copying rows on observed dependence `D` (same caliper machinery already
frozen for C3), then ask whether the **repair gain from the same independence-restoring
re-reason** — `H = Acc(INDEP) - Acc(mechanism state)`, already computed per row by both
`shared_rows_tier1` and `copy_rows_tier3` with an identical sign convention (verified: both are
`P(mechanism-wrong, INDEP-correct) - P(mechanism-correct, INDEP-wrong)`) — differs between the
two mechanisms at matched `D`.

**Population.** The same frozen 9 (model, benchmark) pairs — Qwen/Llama/Gemma $\times$
TruthfulQA/StrategyQA/AQuA — already used for every other Tier-3 matched-D result in the paper
(`tier3_matched_d.MODELS`/`BENCHMARKS`). No new pair, no new caliper width beyond the three
already frozen for C3 (`{0.02, 0.05, 0.10}`).

**Primary metric and test (frozen before computing):**
- Primary metric: `H` (repair gain), using `caliper_contrast(sh, cp, eps, "H", n_boot=2000)` —
  the exact same, already-frozen estimator used for `dY`/`dA`/`p_break`/`p_rescue`, applied
  unmodified to a new metric. No new matching logic is written for this test.
- Primary caliper: `eps=0.05` — the paper's own stated primary caliper everywhere else
  (Table~tab:matched-pair-results). `0.02`/`0.10` are reported as sensitivity, never primary.
- Secondary: a nested linear-probability model `H ~ D`, `H ~ D + mechanism`,
  `H ~ D + mechanism + D*mechanism` (nested F-test on the mechanism coefficient given `D`) and
  nearest-neighbour matching on `(D, A0, margin)` with `H` as the outcome — both generalizing
  `nested_models`/`nn_matching` (currently hard-coded to `dY`) to an explicit `metric` argument,
  reusing their `_ols`/`NearestNeighbors` internals unchanged.
- Aggregation: macro-average across the 9 pairs, equal weight, resample-**pairs** (not rows)
  bootstrap, `n_boot=20000` — identical convention to every other Tier-3 macro number already in
  the paper (`tier3_matched_d.macro_bootstrap`).

**What would count as a result, either way.** This is an estimand of *difference in causal
response under matched observed dependence* — it does not need the intervention to beat any
baseline to be informative. A caliper-matched `H` gap whose CI excludes zero (either sign) is a
positive finding for the thesis ("matched dependence does not imply matched intervention
response"); a CI including zero is also reportable (repair response does not detectably differ
once `D` is matched) and must be reported as such, not re-run with a different caliper to find
significance.

## SMB — Susceptibility vs. Benefit

**Question.** Does vote margin predict *whether* the intervention changes the answer
(susceptibility, `S`) more strongly than it predicts *whether that change helps*
(net benefit, `B`)? This gives a direct empirical separation of the two constructs the paper's
theory already separates algebraically (Theorems 5/6), rather than relying only on the small
SGIR accuracy gain as indirect evidence.

**Population.** All `5760` SGIR episodes (`sgir_common.build_episodes()`, full population — not
restricted to the SGIR test split, since nothing is fit or threshold-selected here; there is no
overfitting risk from using the full frozen population for a correlational characterization).

**Per-episode outcomes (frozen definitions):**
- `S`-episode `= 1[keep_label != indep_label]` (did the answer change at all).
- `B`-episode `= indep_correct - keep_correct ∈ {-1, 0, +1}` (signed net change; its stratum mean
  is `p_rescue - p_break` exactly, the same quantity as the SGIR decomposition).

**Primary test (frozen before computing):** per (model, benchmark) pair, an OLS linear-
probability-style regression of each outcome on the realized KEEP margin `M(q,r)` (the exact
margin definition used throughout SGIR — the realized vote margin of the dependent collective
answer, zero marginal cost, reads no truth label), giving `slope_S` and `slope_B` per pair.
Primary comparison: `|slope_S| - |slope_B|`, **paired at the pair level** (both slopes come from
the same 9 pairs' same episodes), macro-averaged with equal weight and a resample-pairs
bootstrap (`n_boot=20000`), matching the paper's own "mean stratum slope" convention
(Section~sec:main-susceptibility) and its caveat that a margin regression is a diagnostic, not a
direct estimate of a theoretical quantity — carried over here unchanged.

**What would count as a result, either way.** A macro `|slope_S| - |slope_B|` CI excluding zero
in the positive direction is direct empirical support for "margin predicts susceptibility more
than benefit." A CI including zero, or a negative point estimate, is also reportable as-is — not
grounds to try Spearman/Kendall correlation, re-bin margin, or restrict to a subset of pairs
until a cleaner separation appears.

## Stop rule

Both are run once, as specified above. Primary results (MMRT's caliper-matched `H` gap at
`eps=0.05`; SMB's macro `|slope_S|-|slope_B|`) are reported as-is, whatever sign or significance
they come out with. Per the user's explicit instruction, no further experiment is planned after
these two; any later work is a new, separately-scoped request.
