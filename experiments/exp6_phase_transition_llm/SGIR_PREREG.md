# SGIR_PREREG — Confirmatory margin-gated intervention test

> **STATUS: FROZEN 2026-11-08, BEFORE the TEST split is read for outcomes.**
> This document, `sgir_split_manifest.json` (+ its hash), and `tests/test_sgir_confirmatory.py`
> must all be committed/shown and signed off **before** `sgir_confirmatory.py` is run in
> `--evaluate-test` mode. Amendments after that point append with a date and a reason; they do
> not overwrite. If the confirmatory run is null/negative, §8 (stop rule) governs — no new
> threshold family, feature, split, or learner may be tried on this TEST set afterward.

## 0. Why this exists, and why it is confirmatory, not exploratory

The question — *can a pre-intervention-observable rule decide, per question, whether to pay for
an independence-restoring re-reason, and do better than paying for the same number of
re-reasons at random?* — has already been asked three times in this project, on three different
slices of data, and killed each time:

1. `a_copy` / `a_CI` (Counterfactual Isolation): a stronger, prompted version of "re-reason
   independently," gated on a detected copying-contaminated state. Pre-registered GO/KILL gate,
   **KILL, 2026-08-25** (`CI_PROTOCOL_FROZEN.md`, `theory/06_mechanism_chain_MDCA.md:128`).
2. `k2_router.py`: a learned (RidgeCV, leave-one-question-out cross-fit) router over a *richer*
   action space (6 actions) and a *richer* feature set (margin + entropy + agreement + ...).
   Pre-registered primary estimator: pooled gain `+0.000`, CI `[0,0]` — qwen-only pilot,
   now-retired easy/hard probe labels, 2026-09-08.
3. `sgir_copying_oracle.py` / `sgir_boundary_oracle.py`: the same margin-only gate proposed here,
   run non-confirmatory (in-sample threshold, no dev/test split, "random" approximated by the
   unconditional mean, no cross-model population lock). Copying axis: KILL (qwen only). Shared-
   evidence axis, run live on 2026-11-08 across the full available pool (6 models, 7 benchmarks,
   n=3358 rows): overall gain `+0.012 [+0.007,+0.017]`, low-margin-third gain `+0.029
   [+0.019,+0.040]` — still below the project's own `+0.05` actionability bar, but margin visibly
   *concentrates* the gain (~2.4×) rather than doing nothing, unlike the copying axis. That is new
   and is the reason this confirmatory test exists instead of just citing the KILL precedent.

And, orthogonal to all three empirical attempts, **Theorem 6** (`theory/05_theorems/
thm6_repair_headroom.md`) proves — as an assumption-free algebraic identity, not an empirical
finding — that *any* independence-restoring, information-conserving (IRIC) intervention,
applied to a subset of questions chosen from pre-intervention information, has its *realized*
accuracy on that subset ceilinged by the subset's own `A_0` (independent accuracy), and the net
recoverable headroom equals `p_+- − p_-+` (break minus rescue), which is bounded by the
decision-change rate `ε`. The margin gate proposed here is IRIC by construction (it only decides
*which* questions get a fresh independent redraw; it injects no new evidence). So this experiment
cannot show that decorrelation manufactures information the independent ensemble never had; it
can only show whether margin **targets** the already-bounded headroom better than chance.

**What is actually open**, and what this confirmatory test is pre-registered to answer, is two
separate questions that prior runs conflated by using a single `+0.05` PASS/KILL bar:

- **Q1 (targeting):** does the margin gate beat a random-intervention baseline matched on the
  *same compute budget*, on the exact frozen Tier-3 population, with a dev-fit/test-frozen
  threshold and honest clustered uncertainty? (statistical question — is the CI excluding 0?)
- **Q2 (magnitude):** if Q1 is yes, is the beaten margin large enough to matter practically?
  (an *equivalence-style* question, reported separately, never substituted for Q1.)

A positive Q1 with small-magnitude Q2 is itself a confirmatory result for the paper's central
non-implication chain `D ⇏ ΔY ⇏ H_repair`: margin would be shown to carry real targeting
information about *susceptibility*, while *repairability* stays small — i.e. **susceptibility ≠
repairability**, demonstrated constructively rather than only by the Theorem-6 ceiling argument.

## 1. Population (frozen)

- **Models:** `qwen, llama, gemma` — exactly `tier3_matched_d.MODELS`. No mistral/phi4/deepseek
  (dropped at Tier-3 promotion; see memory `tier3-promoted-tier2-dropped`).
- **Benchmarks:** `truthfulqa, strategyqa, aqua` — exactly `tier3_matched_d.BENCHMARKS`. This is
  already disjoint from `bbh` (invalid for the evidence intervention per
  `intervention_validity.py`) and from the broader Tier-1 breadth benchmarks (`mmlu, arc,
  commonsenseqa`) that are outside the Tier-3 grid. `sgir_common.py` asserts
  `all(intervention_validity.is_valid(b) for b in BENCHMARKS)` at import time as a guard against
  silent population drift, not as a filter that does anything on this frozen list.
- **Regimes:** `easy, hard`, both pooled as plain data partitions (same convention as
  `tier3_matched_d.shared_rows_tier1`). The probe-based easy/hard *labels* are retired
  (`PREREG_dA_regime.md`) and are **not used as a scientific stratifier or claim** anywhere in
  this experiment — they only appear in the source filenames and in descriptive breakdowns.
- **Source files:** `data/processed/exp6c_raw_{easy,hard}_scaleup_{model}_{bench}_N11.json`,
  `raw["shared_correct_strong"]` and `raw["indep_correct_strong"]`, via `MODEL_TAG` exactly as
  `tier3_matched_d.py` resolves it. Verified present for all 3×3×2 = 18 cells, 40 questions each,
  `N=11` agents, `R=8` replicates per question (checked live 2026-11-08; see shell log).
- **Episode unit:** `(model, bench, regime, qi, r)` — one replicate of one question. 720
  questions × 8 replicates = **5760 episodes**.
- **Cluster key (for all uncertainty below):** `(model, bench, regime, qi)` — 720 clusters of 8
  episodes each. Replicates of the same question are not independent draws of question
  difficulty and must never be treated as 8 independent observations.

## 2. Actions (exactly two — no router, no extra features)

- **KEEP:** the realized outcome of `shared_correct_strong[q][r]` (the already-paid,
  dependence-inducing collective process). Zero marginal cost.
- **INDEP:** the realized outcome of `indep_correct_strong[q][r]` (the matched-evidence,
  independent re-reason). Marginal cost = `N` additional agent calls per gated question.
- **No `diversity`/`debate`/`stop`/learned-router action, no entropy/agreement/stability
  feature.** The gate is a single scalar threshold on `M`.

**Gate signal `M(q,r)`:** the vote margin of the **KEEP** realization itself —
`M(q,r) = margin(shared_correct_strong[q][r])`, using `margin()` imported verbatim from
`analysis_C_matched_D_falsification.py` (`(n_(1) - n_(2)) / n_informative`; reads vote counts
only, never the truth index). This is the thing already in hand before any intervention
decision — zero extra cost, and it matches the paper's own Lemma 1 / Theorem 5 notion of
"margin of the baseline vote," not a new definition invented for this test.

**Explicitly rejected alternatives, and why:**
- Margin of `indep_correct_strong` (used by `tier3_matched_d.shared_rows_tier1` for a different
  purpose, matching covariates): **circular here** — computing it requires already paying for
  the independent re-reason on every question, defeating the budget argument.
- Margin of the `none` (no-evidence) condition, as `k2_router.py` used: not reliably present at
  per-replicate granularity in these particular raw files in a cost-free way for this exact
  grid, and introduces its own extra-call cost. Left as a labeled *exploratory* alternative gate
  for a future, separately pre-registered test — not used here.
- Any entropy/agreement/stability feature, or a fitted/learned combination: explicitly excluded
  by the user's scope lock (point 2 of the frozen instructions). `k2_router.py` already tested
  richer features with a learned router and found null/negative gain; this test asks the
  narrower question of whether margin *alone*, as a simple threshold, targets benefit.

**Eligibility.** An episode is *eligible for gating* iff `M(q,r)` is finite (≥1 informative vote
in the KEEP realization). Ineligible episodes always get KEEP under every gate policy (there is
nothing to threshold) but are **included** in all accuracy denominators for every policy,
including as candidates for INDEP under RERUN-ALL (which needs no margin). `RANDOM@budget` draws
its intervened subset only from the eligible pool, matching exactly the population the margin
gate could have chosen from.

### 2.1 Realization-level gating, question-level split, and why pooled accuracy is unambiguous

**The gate decision is made PER REALIZATION, not per question.** `M(q,r)` and the intervene
decision are both functions of the specific replicate `r`'s own realized KEEP vote; a given
question's 8 replicates may receive *different* intervene decisions depending on each
replicate's own margin. This is the correct simulation of "deploy this pipeline repeatedly;
each run reacts to whatever margin that run's own collective process happened to produce."

**The dev/test SPLIT acts per question** (§3): all 8 replicates of a question move together.
A question is never gated differently in dev vs. test by construction, but *within* whichever
split it lands in, its 8 replicates are still gated independently of each other.

**Aggregation.** Every reported accuracy pools flatly over `(question, replicate)` episodes.
Because every question-cluster in this frozen grid contributes exactly `R = 8` episodes (no
cluster is larger or smaller — verified live 2026-11-08, see §1), this flat pool is numerically
**identical** to first computing each question's own accuracy (its mean over 8 replicates) and
then averaging those per-question numbers across questions with equal weight. There is therefore
no separate "realization-weighted vs. question-weighted" choice to make for the *point estimate*
in this grid — but the two would diverge if cluster sizes were ever unequal, so
`sgir_common.assert_balanced_clusters` is called on every episode array before any fit or
evaluation and raises loudly rather than silently reporting a number under a changed
weighting convention. The **uncertainty**, independently of this equivalence, is always computed
by resampling whole question-clusters (§5), never individual replicates, because replicates of
the same question are not independent draws of question difficulty.

## 3. Dev/test split (deterministic, frozen before any outcome is read)

- Stratify by `(model, bench)` — 9 strata, 80 pooled-regime questions each (40 easy + 40 hard).
- Within each stratum, the 80 questions are listed in the canonical order
  `[(regime, qi) for regime in ("easy","hard") for qi in range(40)]`, then permuted by
  `numpy.random.default_rng(seed=[20261108, hash_of(model), hash_of(bench)])` (a
  `SeedSequence`-style per-stratum independent seed derived deterministically from the frozen
  date and the stratum name — **not** `Date.now()`/global RNG state) and split 50/50: the first
  40 permuted questions → **dev**, the remaining 40 → **test**.
- Result: 360 dev questions (2880 episodes) / 360 test questions (2880 episodes), balanced
  across all 9 strata.
- Written to `results/tables/sgir_split_manifest.json`: the full explicit
  `(model,bench,regime,qi) -> "dev"|"test"` assignment, the seed, and a `sha256` hash of the
  canonical (sorted-key, fixed-separator) JSON serialization of that assignment. The hash is
  what "frozen" cashes out to operationally: if the manifest file's hash ever changes, the
  protocol has been violated and the confirmatory run is invalid.
- **Test-split outcomes (`shared_correct_strong`/`indep_correct_strong` correctness) are not
  read by any code path until `sgir_confirmatory.py --evaluate-test` — a single, explicit,
  one-time step gated behind that flag.** Building the manifest touches only question identity
  (model, bench, regime, qi) and the split assignment RNG — never margin, never correctness.

## 4. Threshold selection on DEV only

- **Candidate thresholds — frozen, data-independent grid (`sgir_common.TAU_GRID`).** `margin()`
  always returns `(c₁−c₂)/m` for integers `0 ≤ c₂ ≤ c₁`, `c₁+c₂ ≤ m`, with `m` = the number of
  informative votes, `1 ≤ m ≤ N_AGENTS = 11`. Every achievable margin is therefore of the form
  `d/m` with `0 ≤ d ≤ m ≤ 11` — a finite set determined **purely by `N_AGENTS`**, enumerable
  before any margin value, dev or test, is ever read:
  `TAU_GRID = sorted({d/m : 1≤m≤11, 0≤d≤m})` (43 distinct values, `0.0` to `1.0`,
  `sgir_common.frozen_tau_grid`). **The grid is not derived from observed dev margins** — it is
  a superset fixed by the voting setup alone, so the search space cannot be (even
  unintentionally) adapted to whatever dev happens to contain.
- **Objective (computed on DEV only), stratum-matched:**
  `obj(τ) = Acc(π_τ; DEV) − E[Acc(RANDOM@budget; DEV)]`, where the random comparator is matched
  to the gate's intervention count **separately within every `(model, benchmark)` stratum** —
  `k_s(τ)` = number of DEV episodes in stratum `s` with `M(q,r) ≤ τ` among that stratum's
  eligible pool — **never a single pooled `k` across the whole DEV set.** This is required
  because the gate can intervene unevenly across strata (e.g. more on `qwen` than `gemma`); a
  pooled-budget random comparator would then differ from the gate in model/benchmark
  composition, and a composition shift could masquerade as a targeting effect. `E[Acc(RANDOM@
  budget)]` on DEV is the **closed form**: within stratum `s`, for a uniformly random size-`k_s`
  subset of that stratum's eligible pool of size `n_s`, `E[Acc_s] = k_s·mean(INDEP|elig,s) +
  (n_s−k_s)·mean(KEEP|elig,s)` plus the ineligible-episode KEEP contribution, summed over strata
  and divided by the total `n` (exact by linearity of expectation; no simulation needed at fit
  time). Implemented in `sgir_common.random_budget_closed_form` /
  `sgir_common.select_threshold`.
- **Selection rule:** `τ̂ = argmax_τ obj(τ)` over `TAU_GRID`. **Tie-break** (in order): (a)
  prefer the `τ` with the *smaller total* `k(τ) = Σ_s k_s(τ)` (fewer interventions); (b) if
  still tied, prefer the smaller `τ` value.
- `τ̂`, `k(τ̂)` (total and per-stratum), and the full `obj(τ)` curve over all 43 grid points are
  written to the frozen-run output so the selection is auditable. **`τ̂` is then fixed and never
  re-fit after TEST is read.**

## 5. Single frozen TEST evaluation

Run exactly once, after this document + the split manifest + the tests below are frozen.

**Policies evaluated on TEST (2880 episodes, 720 clusters):**
- `KEEP` — shared outcome on every episode.
- `RERUN-ALL` / `INDEP` — independent outcome on every episode.
- `RANDOM@budget` — `k` is **recomputed on TEST**, *per `(model, benchmark)` stratum*, as
  "however many TEST episodes in that stratum satisfy `M(q,r) ≤ τ̂`" (the frozen threshold is
  applied to TEST exactly like the gate is; `RANDOM@budget` is matched to the gate's *actual*
  TEST intervention count **within every stratum separately**, `k_s`, never a single pooled
  count — same reasoning as §4). Within each stratum, a uniformly random subset of size `k_s`
  from that stratum's eligible TEST pool gets INDEP, the rest KEEP; pooled across strata for the
  headline number, with the per-stratum `k_s` always reported alongside
  (`k_by_stratum_test`). Estimated both by **closed form** (diagnostic) and by **explicit Monte
  Carlo** resampling (≥10,000 independent random subset draws, still per-stratum, on the real
  TEST pool) per the user's requirement to see the random-assignment-induced spread, not just
  its mean.
- `π_τ̂` — the frozen margin gate: INDEP if `M(q,r) ≤ τ̂` and eligible, else KEEP.

**Primary estimand:**
```
Δ_gate-random = Acc(π_τ̂; TEST) − E[Acc(RANDOM@budget); TEST]
```
**Secondary:** `Δ_gate-keep = Acc(π_τ̂) − Acc(KEEP)`, `Δ_gate-rerunall = Acc(π_τ̂) − Acc(RERUN-ALL)`,
intervention fraction `k/n_eligible`, mean extra calls/question (`= intervention_fraction × N`),
and the absolute accuracy of all four policies, each with a CI.

**Uncertainty — combined cluster + randomization bootstrap, nested, `n_boot = 20000`:**
For each bootstrap replicate: (1) resample the 720 question-clusters with replacement (cluster
bootstrap — standard data uncertainty; a cluster contributes all 8 of its episodes together or
not at all); (2) *within that same replicate*, re-derive each stratum's `k_s` from the gate
applied to the resampled data, then draw **one** fresh random subset-of-size-`k_s` *per stratum*
from the resampled eligible pool for the `RANDOM@budget` arm (never a single pooled-`k` draw).
This nests randomization noise inside the data-resampling loop so the reported CI reflects both
"which questions we happened to have" and "which subset randomization happened to pick," as the
user asked, while keeping RANDOM's model/benchmark composition matched to the gate's in every
single resample. `KEEP`/`RERUN-ALL`/`π_τ̂` are deterministic given the resampled data, so only
step (1) matters for their CIs.
Pooled test-set result **and** a `model × benchmark` breakdown (9 cells) are both reported;
breakdown cells use the same nested bootstrap restricted to that cell's clusters.

## 6. Break/rescue/churn decomposition (mandatory, TEST, gated-in episodes only)

Restricted to the episodes where `π_τ̂` actually assigns INDEP (`M(q,r) ≤ τ̂`, eligible). Using
the full state arrays (not just aggregate correctness), compute the realized plurality label
under KEEP and under INDEP for each such episode and partition exhaustively:
```
p_+- = P(KEEP correct, INDEP wrong)              — break
p_-+ = P(KEEP wrong,    INDEP correct)            — rescue
p_churn_diff = P(KEEP wrong, INDEP wrong, different wrong label)
p_churn_same = P(KEEP wrong, INDEP wrong, same label)
p_stay_correct = P(KEEP correct, INDEP correct)
```
(these five sum to 1 on the gated-in subset). Report all five, plus
`Δ_acc_gated = p_-+ − p_+-` (must equal `Acc(INDEP) − Acc(KEEP)` on the gated-in subset exactly,
by Prop 6.1 — asserted as an internal consistency check, not just reported) and `ΔY_gated =
P(KEEP label ≠ INDEP label)`. Same nested cluster bootstrap as §5, restricted to clusters with
≥1 gated-in episode (bootstrap draws contributing zero gated-in episodes are recorded as
`nan` for this statistic and excluded from the CI quantiles, with the dropped-draw count
reported, not silently hidden).

## 7. Reporting policy — two questions, not one bar

- Report `Δ_gate-random` point estimate + 95% CI **first**, with no significance threshold
  applied to decide whether to report it.
- **Q1 (targeting):** CI excludes 0 and is positive → margin carries real targeting information
  beyond chance at matched budget. CI straddles 0 → inconclusive/null. CI entirely negative →
  margin gate is worse than random at matched budget.
- **Q2 (magnitude):** separately, compare the point estimate (and CI) against a **labeled
  practical-effect / equivalence threshold of ±5 percentage points** (carried over from the
  prior pilots' actionability bar, now explicitly relabeled as a practical-effect threshold, not
  a significance criterion). This is reported alongside, never substituted for the CI-vs-0
  judgment, and never used to decide whether `Δ_gate-random` is "significant."
- A result such as `+2.9pp, CI [+1.1, +4.7]` is reported as **positive evidence for Q1, small
  for Q2** — itself a confirmatory finding for the paper's `D ⇏ ΔY ⇏ H_repair` chain, not a null
  result and not grounds to keep optimizing.

## 8. Stop rule

This is a single confirmatory evaluation. After TEST is read:
- **Any outcome** (positive-targeting-small-magnitude, null, or negative) is final for this
  pre-registration. Do not change the threshold family, add a feature, change the split, retry
  with a different learner, or re-run with a different τ-selection objective on this TEST set.
- Any such follow-up must be explicitly labeled **exploratory**, use fresh held-out data or the
  DEV split only, and may not reuse this TEST manifest.
- `K1`/oracle-headroom numbers (if reported alongside) are descriptive sanity context only —
  never a comparator for `Δ_gate-random` and never used to tune `τ̂`.

## 9. Frozen artifacts and amendment policy

Frozen by this document: population (§1), action/gate definitions (§2), split procedure + seed
+ manifest hash (§3), threshold grid/objective/selection/tie-break (§4), baselines and the
nested bootstrap specification (§5), decomposition formulas (§6), and the reporting template
(§7–8). Any deviation discovered necessary after this point is appended below with a date and a
reason — it does not silently overwrite the text above.

**Amendment 2026-11-08 (before `--fit-dev` was ever run; no outcome data had been read at the
time of this amendment):** three gaps found on review, fixed here rather than silently in code:
1. Added §2.1 making explicit that the gate acts per *realization*, the split acts per
   *question*, and that flat pooled accuracy equals the per-question-then-averaged quantity
   only because every cluster has `R=8` episodes — now asserted in code
   (`assert_balanced_clusters`), not merely assumed.
2. §4/§5's `RANDOM@budget` comparator was pooled (one global `k` drawn from the whole DEV/TEST
   set). Rewritten to match the gate's intervention count **within every `(model,benchmark)`
   stratum separately**, both in the closed form and the Monte Carlo/nested-bootstrap
   implementations, so a composition shift between strata cannot be mistaken for a targeting
   effect. `sgir_common.random_budget_closed_form`/`random_budget_monte_carlo` now take
   `strata` + the gate's mask instead of a scalar `k`.
3. §4's candidate threshold grid was `np.unique(observed DEV margins)` — adaptable, in
   principle, to whatever DEV happened to contain. Replaced with `sgir_common.TAU_GRID`, a
   43-point grid enumerated from `N_AGENTS=11` alone via `frozen_tau_grid`, fixed before any
   margin value (dev or test) is read.

All three were verified against new unit tests in `tests/test_sgir_confirmatory.py` (stratified
budget matching, cluster-balance guard, frozen/data-independent grid) before `--fit-dev` was
run. No dev or test outcome was read before or during this amendment.
