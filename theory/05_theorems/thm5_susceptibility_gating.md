# Theorem 5 — Susceptibility Gating (Collective Decision-Boundary Amplification)

**Informal claim.** The effect on a *collective* decision of a perturbation to the
agents' dependence structure is **concentrated near the aggregation boundary**: on
questions where the pre-perturbation vote margin is large, a bounded-reach dependence
perturbation leaves the majority/plurality outcome unchanged (`ΔA = 0`); only near the
boundary (small margin) can it flip the collective answer. Consequently a large change
in dependence *strength* `ΔD` can coexist with a negligible *average* accuracy change
`|ΔA|` — the accuracy consequence of a mechanism change is expressed only through the
low-margin subset. This is the formal content of contribution **C2** and the link that
turns the isolated Exp6C dissociation (`ΔD ≈ 0.26–0.50` vs `|ΔA| ≤ 0.04`) into a
statement. It is the edge `D ─[χ]→ A` of the chain `M → D →χ→ A` (see
`../06_mechanism_chain_MDCA.md`); Theorem 4 supplies `M → A` limits and the C3 instrument
supplies `M → D`.

---

## 1. Setup

`N` agents answer a `K`-option question; option `0` is the truth. Let the informative
(non-flat) votes have counts `n = (n_0,…,n_{K−1})`, `Σ_j n_j = n_tot ≤ N`. Sorted counts
`n_(1) ≥ n_(2) ≥ …`. The collective answer is the plurality `j* = argmax_j n_j`
(ties broken by a fixed rule); collective correctness `A = 1[j* = 0]`.

**Margin.** The observable order parameter (identical to the empirical boundary metric)

    M = (n_(1) − n_(2)) / N ∈ [0, 1].              (normalized runner-up gap)

`M = 0` ⇔ a tie at the top (the collective is *on* its decision boundary); `M = 1` ⇔
unanimous.

**Dependence perturbation.** A *mechanism* operator `T` (Theorem 4's `M₁` shared-evidence
injection, or an `M₂` copying round, or any intervention that changes the dependence
structure) acts on the vote vector, reassigning some agents' votes: `n ↦ n' = T(n; ω)`,
`ω` the perturbation randomness. Its **reach** is

    κ(T) = #{ agents whose vote T changes } .

`κ` is the number of vote reassignments; each reassignment moves one unit from one option
to another, so the total-variation of counts is `Σ_j |n'_j − n_j| = 2κ`. Write the
perturbed plurality `j*'` and `A' = 1[j*' = 0]`, and the **collective effect**

    ΔA = A' − A ∈ {−1, 0, +1}.

**Susceptibility model.** Agents are moved independently: agent `i` has its vote changed
by `T` with probability `s_i ≤ s` (the per-agent *susceptibility*; `s` its cap). Then
`κ ⪯ Binomial(N, s)` (stochastically dominated), `E[κ] ≤ Ns`. (`s` is exactly what the
mechanism/copy strength controls; `s → 1` is the `ρ → 1` regime of Theorem 4.)

---

## 2. Deterministic gate

> **Lemma 5.1 (margin gate).** If `T` reassigns at most `κ` votes, then the plurality is
> unchanged, `j*' = j*`, whenever `N·M > 2κ`. Hence
>
>     M > 2κ/N   ⟹   ΔA = 0.

**Proof.** For any option `c ≠ j*`, reassigning `κ` votes can lower `n_{j*}` by at most
`κ` and raise `n_c` by at most `κ`, so

    n'_{j*} − n'_c  ≥  (n_{j*} − n_c) − 2κ  ≥  (n_(1) − n_(2)) − 2κ  =  N·M − 2κ.

If `N·M > 2κ` this is `> 0` for every `c`, so `j*` remains the strict plurality and
`A' = A`. ∎

The gate is **sharp** in order: a perturbation that moves `κ` votes *can* flip a question
with `N·M ≤ 2κ` (move the runner-up past the leader), so the bound `2κ/N` is the correct
scale, not a loose one.

---

## 3. Boundary-concentration theorem

> **Theorem 5 (susceptibility gating).** Under the susceptibility model of §1
> (`κ ⪯ Binomial(N, s)`), for every question with margin `M`:
>
> 1. **(Support)** `ΔA = 0` almost surely whenever `M > 2κ_max/N`; in particular the
>    perturbation cannot move an off-boundary collective.
> 2. **(Envelope)** the expected magnitude is bounded by a function that is strictly
>    decreasing in the margin for `M > 2s`:
>
>        E[ |ΔA| ⏐ M ]  ≤  P( κ ≥ N·M/2 )  ≤  exp( −N · KL(M/2 ‖ s) )     (Chernoff),
>
>    where `KL(a‖s)=a·ln(a/s)+(1−a)ln((1−a)/(1−s))` and the exponent is positive and
>    increasing in `M` once `M/2 > s`.
> 3. **(Slope)** hence the population regression of `|ΔA|` on `M` has non-positive slope,
>    `d E[|ΔA|]/dM ≤ 0`, strict on any margin range where the envelope is non-flat; and
>    the mean effect over a question set is a mixture **dominated by its low-margin mass**.

**Proof.** (1) is Lemma 5.1 with `κ ≤ κ_max`. (2) `|ΔA| ≤ 1[j*' ≠ j*] ≤ 1[2κ ≥ N·M]` by
the contrapositive of Lemma 5.1, so `E[|ΔA|⏐M] ≤ P(κ ≥ N·M/2)`; the Chernoff bound for
`κ ⪯ Binomial(N,s)` gives the stated exponential, positive when `M/2 > s`. (3) The upper
envelope `exp(−N·KL(M/2‖s))` is decreasing in `M` on `M > 2s`; averaging over a question
distribution `π(M)`, `E_π|ΔA| ≤ ∫ π(M) e^{−N·KL(M/2‖s)} dM` is controlled by the mass
`π` places near `M ≈ 0`. ∎

**Finite-size sharpening (a testable prediction).** The exponent carries a factor `N`:
the gate steepens as the population grows, so the boundary concentration is *sharper* at
larger `N`. This predicts the empirical `|ΔA|~M` slope becomes **more negative with `N`**
— a clean target for the N-scaling reproduction (P2), and a bridge to the finite-size
"hero figure."

---

## 4. Corollary — why `ΔD ↑↑` need not force `|ΔA| ↑↑` (links C2 to C1)

> **Corollary 5.2 (dissociation).** A dependence-mechanism intervention can raise the
> measured dependence `D` by an arbitrary amount `ΔD` (Exp6C: `ΔD = 0.26–0.50`, 4/4 sig)
> while changing the collective decision on **no** off-boundary question. Its mean
> accuracy footprint is bounded by the boundary mass:
>
>     mean_q |ΔA_q|  ≤  π( M ≲ 2s )  →  the fraction of low-margin questions,
>
> which is *decoupled* from the magnitude of `ΔD`. Therefore `|ΔA|` small **does not**
> falsify a large mechanism effect on dependence; the two live on different axes, joined
> only through the susceptibility gate `χ`.

This is the precise sense in which **susceptibility `χ` is the gate through which an
`M → D` change reaches `A`** — and why the honest headline of Exp6C is the *dissociation*,
not a null. It also explains, without appeal to noise, why K1/K2 found the accuracy
headroom concentrated on the low-margin subset.

---

## 5. Empirical instantiation (observed)

`exp6c_boundary.py`, N=11, R=8, qwen2.5:7b. Primary D-free metric `|ΔA| ~ M`:

| set | slope `|ΔA|~M` | 95% CI | bins mean\|ΔA\| (low→high M) |
|---|---|---|---|
| easy | −0.212 | [−0.425, −0.072] | 0.071 → 0.000 → 0.000 |
| hard | −0.097 | [−0.188, −0.011] | 0.080 → 0.038 → 0.010 |
| **pooled** | **−0.125** | **[−0.198, −0.057]** | **0.088 → 0.014 → 0.000** |

All slopes negative with CI excluding 0; the bin profile is monotone and vanishes off the
boundary — exactly Theorem 5(1)–(3). The `D`-based interaction term is noisy at R=8 and is
corroborating only; the D-free `|ΔA|~M` law is the primary, robust statement.

---

## 6. Positioning (post-audit 2026-08-14) — what makes C2 distinct

The 2026 conformity literature gates behavior by an agent's **own** state; we gate the
**consequence of a dependence intervention** by the **collective aggregation boundary**.

- **Bellina et al. 2026 (`2601.05384`)** — conformity rises at an agent's *individual
  competence boundary*. Ours: `M` is a *collective vote-margin* (a threshold on the tally),
  and the gated object is `ΔA` from a dependence perturbation, not an individual's yield.
- **`2605.27288` (uncertainty-gated conformity)** — *dyadic, individual* yield-to-pushback
  correlated with *self* epistemic uncertainty; no aggregation, no vote margin, no
  population accuracy, no inter-agent dependence intervention.
- **`2607.05545` (speaker-free floor)** — most apparent conformity is spontaneous
  instability (66.5% revision with the speaker removed). This *strengthens* C2: our
  estimand `ΔA = A_shared − A_indep` is a **matched paired difference** (identical frame,
  matched evidence quality), so a common spontaneous-flip floor **cancels** in the
  contrast. (P2 must verify this cancellation empirically across models.)

**Locked claim (C2):** *Collective decision-boundary amplification — the effect of a
causal dependence perturbation on aggregate decisions concentrates near low-margin
collective states, explaining why large shifts in error dependence can coexist with
negligible average accuracy change.* Do **not** state it as "LLMs are more susceptible
near a boundary" (Bellina); the novelty is the **collective margin gating the consequence
of a dependence intervention**, bound to C1.

---

## 7. Status & what is proven vs. open

- **Proven here:** the gate (Lemma 5.1), the boundary-concentration envelope and slope
  sign (Theorem 5), and the dissociation corollary (5.2) — all from the threshold nature
  of plurality aggregation plus a bounded-reach susceptibility model. These are
  model-agnostic (hold for any mechanism `T` with bounded reach).
- **Observed:** the sign and monotone bin profile on qwen2.5:7b (§5).
- **Open (P2):** (i) the `N`-scaling steepening of the slope; (ii) the paired-difference
  cancellation of the spontaneous floor across `≥3` model families; (iii) whether the
  low-margin mass `π(M≲2s)` — hence the size of the mechanism footprint — is itself
  model-/benchmark-stable. These are exactly the generalization targets the literature
  audit flags as the open territory.
