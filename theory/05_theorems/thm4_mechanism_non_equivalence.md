# Theorem 4 — Mechanism Non-Equivalence of Statistical Dependence

**Informal claim.** Two homogeneous LLM populations can reach the *same* terminal
error-correlation ρ→1 (maximal dependence *strength*) and yet converge to *different*
collective accuracies, because the accuracy is governed by the causal *mechanism*
that generated the dependence, not by ρ. In particular: for a competent crowd (p>½),
*downstream* social copying is **less** harmful than *upstream* common evidence — the
reverse of the naïve "herding is bad" intuition.

This turns the empirical ablation (Exp6B vs the coupling sweep) into a statement, and
motivates the state descriptor `S = (p, D, M, N)` — competence, dependence strength,
dependence **mechanism**, population size — replacing the impoverished `(p, ρ)`.

---

## 1. Setup

Binary correctness. `N` agents answer a question with truth `T`. Agent `i` emits
`X_i ∈ {correct, wrong}`; write `C_i = 1[X_i = T]`. Marginal competence
`p = P(C_i = 1)`. The collective decision is the majority `Y = maj(X_1,…,X_N)` and the
collective accuracy is `A = P(Y = T)`. All three mechanisms below start from the SAME
independent baseline and are indexed by a dependence-strength knob whose limit is the
terminal pairwise correctness correlation `ρ → 1`.

**Baseline (independence, ρ=0).** `C_i` iid Bernoulli(p). Condorcet:

    A₀ = P_maj(N, p, 0) = Σ_{k > N/2} C(N,k) p^k (1−p)^{N−k}.

Standard facts (used throughout): for `N ≥ 3`, `A₀ > p` iff `p > ½`, `A₀ < p` iff
`p < ½`, and `A₀ = p` iff `p = ½`. (Condorcet boost above ½; anti-boost below.)

---

## 2. Three dependence-generating operators

### M₁ — Common latent evidence (upstream of aggregation)
A shared evidence variable enters each agent's *private formation*:

    C_i = 1[ √ρ · Z₀ + √(1−ρ) · ε_i ≤ Φ⁻¹(p) ],   Z₀ shared, ε_i ⟂ iid, all N(0,1).

Marginal is preserved (`P(C_i=1)=p ∀ρ`), because the latent
`L_i = √ρ·Z₀ + √(1−ρ)·ε_i` has `Var(L_i) = ρ + (1−ρ) = 1`, so `L_i ~ N(0,1)` for every
`ρ` and `P(C_i=1) = Φ(Φ⁻¹(p)) = p`.

**CORRECTION (2026-10-01).** An earlier version of this line claimed the pairwise
*correctness* correlation is "exactly ρ". That is **wrong**: `ρ` is the correlation of
the **latent** `L_i`, and thresholding attenuates it. The correctness correlation is the
tetrachoric image, with `a = Φ⁻¹(p)`:

    τ_p(ρ) = Corr(C_i, C_j) = [ Φ₂(a, a; ρ) − p² ] / [ p(1−p) ]
           = Σ_{k≥1} w_k ρ^k,     w_k = φ(a)² He_{k−1}(a)² / (k! · p(1−p)) ≥ 0,

by Mehler's expansion of the bivariate normal density (`He_k` = Hermite). Since
`Σ_k w_k = τ_p(1) = 1` and `w_1 = φ(a)²/(p(1−p)) > 0`, the map `τ_p` is a strictly
increasing bijection of `[0,1]` onto itself and **attenuates**: `τ_p(ρ) ≤ ρ`, strictly on
`(0,1)`. Numerically, `p=0.9, ρ=0.5` gives `τ_p = 0.249`, not `0.5`; at `p=½` the map is
`(2/π)·arcsin ρ`. Nothing downstream in this file breaks, because every statement here is
taken in the limit, where `ρ→1 ⟺ τ_p(ρ)→1`. What the correction buys is the reason the
limit is the right place to stand: at any `ρ<1` the measured correlation is strictly below
one, so only at the shared ceiling is a correlation-based measure genuinely unable to
separate the mechanisms. Invertibility of `τ_p` is also what makes matching the two arms
on *measured* dependence well posed. `pp.tex` states and proves this as the
"Dependence calibration" lemma; keep the two files in agreement.

Causal graph `Z₀ → C_i → Y`. Then

    A_M₁(ρ) = P_maj(N, p, ρ)
            = ∫ φ(z) [ Σ_{k>N/2} C(N,k) q(z)^k (1−q(z))^{N−k} ] dz,
      q(z) = Φ( (Φ⁻¹(p) − √ρ · z) / √(1−ρ) ).

### M₂ — Synchronous consensus copying (downstream feedback)
Start from the independent baseline `X_i^{(0)}`, `Y_0 = maj(X^{(0)})`. Update

    X_i^{(t+1)} = Y_t = maj(X^{(t)})        (adopt the current plurality).

Causal graph carries the feedback loop `X^{(t)} → Y_t → X^{(t+1)}`. Terminal agreement
(hence measured correlation) → 1.

### M₃ — Sequential cascade copying (downstream, order-dependent)
Agents act in a fixed order; agent `i` sees predecessors' public actions and, once the
public tally is decisive, follows it regardless of private signal (Bikhchandani–
Hirshleifer–Welch 1992; Banerjee 1992). Terminal agreement → 1.

---

## 3. Theorem

> **Theorem 4 (Mechanism non-equivalence at ρ→1).**
> Fix `N ≥ 3` and `p ∈ (0,1)`, `p ≠ ½`, and the common independent initial population.
> As the terminal correctness-correlation `ρ_final → 1`:
>
> 1. **(M₁ common evidence)** `A_M₁ → p`, hence `Δ_M₁ := A_M₁ − A₀ → p − P_maj(N,p,0)`,
>    which is nonzero with `sign(Δ_M₁) = sign(½ − p)`.
> 2. **(M₂ synchronous copying)** `A_M₂ = P_maj(N,p,0)` for every `t`, hence `Δ_M₂ = 0`,
>    although terminal agreement → 1.
> 3. **(M₃ cascade copying)** `A_M₃ ≤ P_maj(N,p,0)`, with strict inequality on a
>    positive-measure set of signal realizations; hence `Δ_M₃ ≤ 0`.
>
> Consequently the three mechanisms share `ρ_final → 1` but occupy three distinct
> collective limits `{ p, P_maj(N,p,0), ≤P_maj(N,p,0) }`. In particular `Δ_M₁ ≠ Δ_M₂`
> for all `p ≠ ½`, and **for `p > ½` the ranking is `Δ_M₂ = 0 > Δ_M₁ < 0`**: downstream
> consensus freezing preserves the Condorcet boost that upstream common evidence
> destroys.

### Proof.
**(1)** As `ρ→1`, `√(1−ρ)→0`, so `q(z) → 1[z ≤ Φ⁻¹(p)]` pointwise (a.e.). The inner
majority term → `1[z ≤ Φ⁻¹(p)]` (all agents share the single sign of the common draw,
so the majority is correct iff that draw is correct). Dominated convergence gives
`A_M₁ → ∫ φ(z) 1[z ≤ Φ⁻¹(p)] dz = Φ(Φ⁻¹(p)) = p`. Then `Δ_M₁ → p − A₀`, and by the
Condorcet fact `A₀ ⋛ p ⇔ p ⋛ ½`, so `sign(Δ_M₁) = sign(½ − p) ≠ 0`. ∎(1)

**(2)** `X^{(1)} = (Y_0,…,Y_0)` is constant across agents, so `Y_1 = Y_0`; by induction
`Y_t = Y_0 ∀t`. Thus `Y_∞ = Y_0` and `A_M₂ = P(Y_0 = T) = P_maj(N,p,0)`, independent of
how strongly agents have "converged." Hence `Δ_M₂ = 0` while the empirical agreement /
ICC / pair-agreement all → 1. ∎(2)

**(3)** In a BHW cascade the public history can lock the population onto the action
favored by an early minority whose private signals happened to align against `T`. The
locked action equals `T` only when the pre-cascade tally already favored `T`; there is
a positive-probability set of orderings/signals on which the cascade is wrong while
`maj(X^{(0)}) = T`, and none on which the cascade *creates* correctness the majority
lacked (public info is a coarsening of private info). Hence `A_M₃ ≤ A₀` with strict
inequality on that set. ∎(3)

**Non-equivalence.** `Δ_M₁ = p − A₀ ≠ 0 = Δ_M₂` whenever `p ≠ ½`; for `p > ½`, `A₀ > p`
gives `Δ_M₁ < 0 = Δ_M₂`. ∎

---

## 4. Corollary — dependence *strength* is not a sufficient statistic

> **Corollary 4.1.** Let `D` be any functional of the joint answer distribution that is
> determined by the marginal `p` and the pairwise correctness correlation `ρ` alone
> (e.g. Fleiss ICC, exchangeable pair-agreement). Then the map `(p, D) ↦ A_coll` is
> **not well-defined**: M₁ and M₂ can be tuned to a common `(p, ρ_final→1)` yet give
> `A_coll = p` vs `A_coll = P_maj(N,p,0) ≠ p`. Predicting `A_coll` requires the
> mechanism label `M`. Equivalently: `A_coll = F(p, D, M, N)` does not factor through
> `(p, D, N)`.

This is strictly stronger than the (trivial) statistical remark "pairwise correlation
does not determine a multivariate binary law." Here the *dynamical* endpoints differ,
the difference is **sign-definite and p-dependent**, and it separates two mechanisms
that are operationally realizable in LLM collectives.

*Slogan.* Dependence **strength** says *how much* agents agree; dependence
**mechanism** says *what that agreement means*.

---

## 5. Empirical instantiation (already observed)

| Mechanism | ρ_final | predicted Δ (p>½) | observed |
|---|---|---|---|
| M₁ common evidence (Exp6B, easy) | high | `<0` (→ p−A₀) | lift `−0.07` ✓ |
| M₂ sync copying (coupling sweep, easy) | →1 | `0` | lift `≈0.00` ✓ |
| M₃ cascade (native, sequential) | →1 | `≤0` | **not yet tested** — prediction |

The `M₂ = 0 > M₁ < 0` ranking for competent crowds is visible in the current data
(easy: copying lift ≈0, common-shock lift ≈−0.07) and is the counter-intuitive nugget
to lead with.

---

## 6. Honest novelty accounting (for positioning)

- Part (1) is the classical correlated-Condorcet limit (Ladha 1992; Boland–Proschan–
  Tong 1989) — **not new**.
- Part (3) is the information-cascade result (BHW 1992; Banerjee 1992) — **not new**.
- Part (2) is an elementary fixed-point fact — **trivial in isolation**.
- **What is new** is (a) the *juxtaposition* proving three mechanisms share ρ→1 but
  split into three sign-definite limits; (b) Corollary 4.1 stated at the level of
  *collective dynamics* (not joint distributions); (c) the p-dependent ranking; and (d)
  the demonstration that all three are *operationally distinct, realizable regimes of
  real LLM collectives* — which is what makes the `(p,D,M,N)` descriptor actionable and
  sets up the mechanism-aware controller. Do **not** sell (1)/(3) as discoveries.

**Preferred one-sentence claim:** *different causal dependence mechanisms possessing
comparable observable dependence induce distinct limiting collective dynamics.* Do NOT
headline it as "correlation is not a sufficient statistic" — that reads as
statistically obvious; the content is the *dynamics-level, sign-definite, LLM-realizable*
separation.

## 7. What this buys the paper

This theorem is **claim C1 (Mechanism Non-Equivalence)** and the `M → A` edge of the
chain `M → D →χ→ A` (`../06_mechanism_chain_MDCA.md`). It pairs with **C2 = Theorem 5**
(`thm5_susceptibility_gating.md`, the `D ─χ→ A` gate) and **C3** = the matched
shared-vs-independent evidence instrument (Exp6C, the `M → D` edge). Together they answer
*why* / *when* / *why-`ρ`-isn't-enough* — the scientific spine, standing on its own.

*(Claim numbering follows the post-audit README §2b — C1 mechanism / C2 boundary / C3
instrument — not the older §2 deliverables table.)*

**On the controller (parked).** Because `A_coll` depends on `M`, a controller that
*diagnoses M* could in principle dominate any mechanism-blind fixed policy. Whether it
cashes out is the K-gate sequence: K1 found only a thin conditional headroom
(`cross-fit H = +0.037`), and **K2 / K2.5-pre found no observable-feature (even leaky)
router beats static `shared_ev`** — the bottleneck is *actionability*, not observability.
So the controller is parked behind an action-enrichment falsification; **this theorem
stands as the "mechanism sets the collective limit" result regardless of whether the
controller edge ever closes.**
