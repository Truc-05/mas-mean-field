# Pre-registration — regime dependence of the accuracy contrast `dA`

**Frozen 2026-10-03, at 17 of 72 cells complete (qwen done, llama partial, gemma/mistral/phi4/deepseek not started).**
Nothing below may be edited after the first of the four remaining models reports. Amendments
append with a date and a reason; they do not overwrite.

Implemented by `tier1_dA_regime_test.py`. Run it only when the sweep is complete; it refuses
to emit a verdict before `--min-cells` strata are available.

---

## 0. Why this exists

`dA = A_shared − A_indep` is currently small and mostly negative, with two strata at
−0.07…−0.08. The project has been describing `dA ≈ 0` as "the Thm-4 signature". Reading
`theory/05_theorems/thm4_mechanism_non_equivalence.md` §3 and §5 shows that attribution is
**wrong at the mechanism level**, and the wording is corrected here before more data arrives.

Theorem 4 indexes three dependence-generating operators and gives three different `ρ→1` limits:

| operator | arm in exp6c | predicted `Δ` for `p > ½` |
|---|---|---|
| `M₁` common latent evidence (upstream) | **`shared_correct_strong`** | **`Δ → p − P_maj(N,p,0) < 0`** |
| `M₂` synchronous consensus copying (downstream) | *(no copying arm at breadth)* | `Δ = 0` exactly |
| `M₃` sequential cascade | not tested | `Δ ≤ 0` |

`Δ = 0` is the prediction for **`M₂`**, and the breadth tier has **no copying arm**
(declared limitation, `paper.md` §7.1). The shared-evidence arm is `M₁`, for which the theorem
predicts a *strictly negative* limit. So:

* **Do not write** "`dA ≈ 0` is the Thm-4 signature". It transfers `M₂`'s prediction onto
  `M₁`'s arm.
* The defensible statements are (a) **Corollary 4.1** — `A` does not factor through `(p, D, N)`,
  which is what large `ΔD` with small `|dA|` illustrates — and (b) Theorem 4 part (1), which
  predicts `dA < 0` for `M₁`, attenuated at the finite dependence we actually reach.
* Where a limiting statement is meant, write "the predicted limiting cancellation" and name the
  operator; never attach `dA ≈ 0` to Theorem 4 unqualified.

This also means the two strata flagged as "Thm-4 violations" in earlier notes
(llama/aqua/easy `−0.078`, qwen/mmlu/easy `−0.069`) are **not violations**. They sit on the
`−0.07` that §5 of the theorem file already records as the observed `M₁` lift. They are the
closest thing in the dataset to a point confirmation.

---

## 1. Substrate (frozen)

* **Strata:** the **recovered split-sample strata** from `tier1_split_sample_strata.py`
  (bin on `none` replicates 0–3, evaluate on held-out 4–7). The probe-based `easy`/`hard`
  labels are **retired** — `p2_probe.py:56` and `exp6c_boundary.py:239` disagree on the
  display-shuffle `qi` convention, so they do not reproduce. Ratified 2026-09-30.
* **Excluded:** `bbh` (all 80 items are the single `logical_deduction` subtask; closed-premise
  questions admit no informative evidence, and correct-pointing clues *lower* accuracy by
  0.21 — see `tier1_breadth.BENCHMARKS`).
* **Primary `ΔD`:** pooled ratio `Σnum / Σden`, with the degenerate-bootstrap guard
  (`tier1_split_sample_strata.boot_pooled_ratio`) and an `n_eff = (Σw)²/Σw² ≥ 15` gate.
  Mean-of-ratios is reported beside it, never instead of it.
* A stratum enters the `dA` models below on its **`dA` alone**; `dA` is defined for every
  question, so it is *not* subject to the `n_eff` gate that governs `ΔD`. The gate is reported
  per stratum so readers can see which strata have an interpretable `ΔD` alongside.

## 2. Equivalence band (frozen)

$$\epsilon = 0.05$$

Not chosen here: this is the band the paper already uses for exactly this quantity —
`|dA| ≤ 0.05` at `paper.md:157` and `paper.md:212`, and in the C1-at-breadth decision at
`README.md:268` (`|dA| ≤ 0.05` in 42/42 cells). Adopting the existing band is what makes this
a pre-specification rather than a choice made after seeing the sweep.

## 3. Hypotheses (frozen, in reporting order)

**H1 — directional, Theorem 4 part (1), SIGN-MATCHED.** One-sided, `α = 0.05`:

$$H_1:\ \operatorname{sign}(dA_{s}) = \operatorname{sign}\!\big(p_s - P_{\mathrm{maj}}(N, p_s, 0)\big) = \operatorname{sign}(1/2 - p_s)$$

tested as a sign test on the agreement indicator across strata `s`, with `p_s = A_{\text{none},s}`.

> **Specification note (2026-10-03, before any of the four remaining models reported).** An
> earlier draft of this document wrote `H1: E[dA] < 0` globally. That is **mis-specified for this
> substrate.** Theorem 4 gives `sign(Δ_M₁) = sign(½ − p)`, so the prediction *flips at `p = ½`*:
> negative only for competent crowds. The recovered strata sit at the two extremes —
> `easy` has `A_none ≈ 1.00`, `hard` has `A_none ≈ 0.01–0.14` — so almost no stratum has
> `p > ½`, and in `hard` the theorem actually predicts `dA > 0`. A blanket "dA < 0" would have
> tested a hypothesis the theory does not make here and scored the result against the wrong
> direction. Corrected to the sign-matched form above. Recorded rather than silently replaced.

**Declared design limitation, consequence of the same observation.** Theorem 4's quantitative
content lives near `p` slightly above `½`, where `p − P_maj(N,p,0)` is large and negative. This
design never samples that region: the `easy`/`hard` split is built from `p ≥ 0.80` / `p ≤ 0.30`
predicates and the recovered strata concentrate at `p ≈ 1` and `p ≈ 0.05`, where the Condorcet
term is degenerate (`p − P_maj → 0` as `p → 1`; it is *positive* for `p < ½`). So this substrate
can test the **sign** of the `M₁` effect but has little power on its **magnitude**, and a null on
H4 is uninformative about the theorem rather than evidence against it. Write it that way.

**H2 — practical equivalence (TOST).** Two one-sided tests at `α = 0.05`, i.e. a 90% CI for
`E[dA]` contained in `(−ε, +ε)`:

$$H_{\mathrm{eq}}:\ |\mathbb{E}[dA]| < \epsilon = 0.05.$$

Reported **separately for `easy` and `hard`**, never pooled into one verdict. "Not significant"
is not evidence of equivalence; only H2 licenses "materially unchanged".

H1 and H2 are **not** mutually exclusive, and the most likely joint outcome is that **both**
hold: a reliably negative `dA` that is nonetheless inside `±0.05`. That is a *stronger* and more
honest result than either alone, and the wording for it is fixed in §5.

**H3 — regime interaction (the quantity of interest).**

$$dA_{mbr} = \beta_0 + \beta_1\,\mathbb{1}[\text{easy}] + u_{\text{model}} + v_{\text{benchmark}} + \varepsilon_{mbr}$$

Random intercepts for model and benchmark, crossed. Report `β̂₁` with a 95% CI. Inference by
**paired stratum bootstrap**: resample (model, benchmark) *pairs*, keeping each pair's easy and
hard strata together, so the easy−hard contrast is within-pair and the CI respects the crossed
design. This avoids fitting REML variance components on ~12 clusters, where they are unstable.

**H4 — the sharp version of Theorem 4.** The theorem does not merely sign `Δ_M₁`, it gives its
limit, `p − P_maj(N, p, 0)`. With `p = A_none` per stratum and `N = 11`, compute
`Δ_pred = A_none − P_maj(11, A_none, 0)` and test

$$\mathrm{corr}\!\left(\Delta_{\mathrm{pred}},\, dA\right) > 0$$

one-sided, with a stratum bootstrap CI. This is the only test here that can *support* Theorem 4
rather than merely fail to contradict it, so it is reported even though it is the least powered.
A null result here is reported as "no support", not as a refutation.

## 4. Decision rules (frozen)

Let `β̂₁` be H3's estimate and let `eq(easy)`, `eq(hard)` be H2's verdicts.

| condition | verdict | wording to use |
|---|---|---|
| `eq(easy)` and `eq(hard)` both hold; `β̂₁` CI covers 0 | **R1 — regime-invariant negligibility** | "Dependence can increase substantially without materially changing accuracy (\|dA\| < 0.05 in both regimes)." Cite Corollary 4.1, not Theorem 4 part 1. |
| `eq(hard)` holds, `eq(easy)` fails with `β̂₁ < 0` (CI excludes 0) | **R2 — regime-dependent cancellation** | "Cancellation is regime-dependent: high dependence does **not** imply `dA ≈ 0`; the accuracy cost appears where the crowd has Condorcet boost to lose." Promote to a finding — it is Theorem 4 part (1) becoming visible as `p` rises. |
| H1 holds broadly and `eq` fails in both regimes | **R3 — systematic accuracy cost** | Drop any "`dA ≈ 0`" claim from the headline. Keep Theorem 4 at its own assumptions (a `ρ→1` limit statement). C1/C2/C3/C5 stand independently; the empirical cancellation claim does not. |
| H1 fails (`dA` not reliably negative) and `eq` holds | **R4 — null-but-bounded** | "`dA` is bounded within ±0.05 with no reliable sign." Weakest reportable outcome; no mechanism claim either way. |

`β̂₁` is reported with its CI **in every case**, including R1, so a reader can see the
interaction was tested rather than assumed away.

## 5. Pre-committed wording for the likely joint outcome

If H1 and H2 both hold — reliably negative, inside `±0.05`:

> Sharing a single evidence realization reliably lowers collective accuracy relative to
> independent realizations of the same evidence (one-sided sign test over `K` strata,
> `p = …`), but the effect is bounded well inside the paper's materiality band
> (90% CI for `E[dA]` ⊂ (−0.05, 0.05)). The direction is what Theorem 4 predicts for the
> common-evidence operator `M₁` at `p > ½`; the magnitude is small because the attained
> dependence is far from the `ρ → 1` limit in which `Δ_M₁ → p − P_maj(N,p,0)`.

## 6. What this does not license

* No claim about `M₂`/copying at breadth — there is no copying arm. Stays qwen-only.
* No `M ↔ D_shared` claim. Dropped 2026-10-03: after the degenerate `±1` ratios are handled,
  `corr(M, D_shared)` is `+0.125 [+0.003, +0.245]`, which will not carry weight.
* No increase of `R` to lift strata over the `n_eff` gate unless a power calculation is run
  first and shows the increase actually moves `n_eff` past it. Saturation is missing
  information, not a tuning parameter; `R = 8 → 16` on a stratum whose crowd never varies buys
  nothing. The limitation is written, not engineered around:

  > Dependence is undefined or weakly identified when agent errors exhibit insufficient
  > replicate-level variation. We therefore aggregate covariance and variance components
  > before forming the dependence ratio, and require a minimum effective sample size for
  > cell-level inference.
