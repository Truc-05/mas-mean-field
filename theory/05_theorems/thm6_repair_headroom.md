# Theorem 6 — Repair Headroom (Diagnosis ⇏ Actionability)

**Informal claim.** When a dependence mechanism (copying/herding) mostly *amplifies* the
independent aggregate rather than *changing its winner*, the accuracy that any
**independence-restoring, no-new-information** intervention can recover is bounded by the
rate at which the mechanism flips the plurality — and, more sharply, equals the *net* of
its harmful and beneficial flips. Consequently a perfectly diagnosed, arbitrarily strong
dependence (`D → 1`) can be **non-actionable**: there is no accuracy to recover by
decorrelation alone, because the collective decision was already determined by the
independent votes. The **bound is universal**; the **recoverable magnitude scales with the
independent diversity `1 − D_0`** — i.e. with how much genuine diversity copying had to
destroy — so it is near-zero when the independent aggregate is already near-consensus
(TruthfulQA, `D_0=0.83`) and larger when it is diverse (AQuA, `D_0=0.53`). This is the
formal content of the CDCR/K2/`a_copy`/`a_CI`/SGIR negative chain and the theoretical core
of contribution **C5**. It adds the final edge to the chain
`M → D →χ→ A` (see `../06_mechanism_chain_MDCA.md`): a third layer *below* the decision,

    D (dependence)  →  ΔY (decision change)  →  H_repair (actionability),

each strictly weaker than the last. Theorem 5 established `D →χ→ ΔY` (only near-boundary
questions change); Theorem 6 establishes `ΔY → H_repair` (only the *net-directional* part
of a decision change is recoverable), and that the two gaps compound.

---

## 1. Setup

`N` agents answer a `K`-option question; option `0` is the truth `Y*`. Let

    V_0 = independent (round-0, coupling g=0) vote vector,   Y_0 = Plurality(V_0),
    V_g = T_g(V_0; ω) after a copying dynamics of strength g, Y_g = Plurality(V_g),

with `Plurality` the same threshold rule as Theorem 5 (informative votes only; a top-tie
scored wrong). Accuracies over the question × ensemble law:

    A_0 = P(Y_0 = Y*),      A_g = P(Y_g = Y*).

**Intervention class (the scope — load-bearing).** An intervention `R` is
**independence-restoring & information-conserving (IRIC)** if it (i) discards the
correlated votes `V_g`, (ii) re-aggregates a *fresh independent* draw from the **same**
per-agent belief-generating process — **no new evidence, no new query** — and (iii) uses
the same aggregation rule. Its decision `Y_R` then has the **marginal law of `Y_0`**, so

    A_R = P(Y_R = Y*) = A_0.                                            (†)

The failed controller actions are **IRIC or consistent with it**: `a_copy` and SGIR
(susceptibility-gated independent resampling) restore an independent aggregate with no new
information (IRIC by construction); the K2 router *selects among* IRIC actions (so its
achievable set is bounded by them, though the router object itself is not a single
transformation); `a_CI` (counterfactual isolation) uses prompting to re-elicit without new
external evidence — plausibly IRIC, but this needs a per-operator argument (§6) rather than
assertion. `a_common` (shared-evidence injection) is **not** IRIC — it adds information —
and is therefore outside the bound (see §5, where it is the *only* action that helps).
**We therefore say the negative chain is _consistent with_ the bound, not that Theorem 6
mechanically "explains all five."**

**Flip contingency.** Partition the `(Y_0, Y_g)` outcomes by correctness:

    p_+-  = P(Y_0 = Y*, Y_g ≠ Y*)      copying BREAKS a correct plurality  (gross damage)
    p_-+  = P(Y_0 ≠ Y*, Y_g = Y*)      copying RESCUES a wrong plurality
    ε     = P(Y_g ≠ Y_0)               winner-change rate (decision susceptibility ΔY)

Amplification is the empirical regime `ε ≤ ε₀` small (winner preserved w.p. `≥ 1−ε₀`).

---

## 2. Exact decomposition (assumption-free)

> **Proposition 6.1.** For any two decision rules `Y_0, Y_g` and truth `Y*`,
>
>     A_0 − A_g  =  p_+-  −  p_-+ .
>
> Hence for any IRIC intervention, by (†), the **recoverable headroom** is exactly
>
>     H_repair  :=  A_R − A_g  =  A_0 − A_g  =  p_+-  −  p_-+ .

**Proof.** Split each accuracy on the other rule's correctness:
`A_0 = P(Y_0=Y*,Y_g=Y*) + P(Y_0=Y*,Y_g≠Y*)` and
`A_g = P(Y_0=Y*,Y_g=Y*) + P(Y_0≠Y*,Y_g=Y*)`. The shared term
`P(Y_0=Y*,Y_g=Y*)` cancels on subtraction, leaving `p_+- − p_-+`. ∎

No model of `T_g` is used: this is a pure decomposition of the accuracy difference between
the correlated and the independent decision. It is what the data measures directly (§4).

---

## 3. The headroom bound

> **Theorem 6 (repair-headroom bound).** Let `R` be any IRIC intervention. Then
>
> 1. **(Directional identity)** `H_repair = p_+- − p_-+`.
> 2. **(Susceptibility bound)** `|H_repair| ≤ p_+- + p_-+ ≤ ε`.
>    In the amplification regime `ε ≤ ε₀`, therefore `|H_repair| ≤ ε₀`.
> 3. **(Ceiling)** `A_R = A_0`: no IRIC intervention exceeds the independent-aggregate
>    accuracy. The best achievable improvement over the correlated collective is exactly
>    `A_0 − A_g ≤ ε₀`, attained only if every winner-flip is harmful (`p_-+ = 0`).

**Proof.** (1) is Proposition 6.1 with (†). (2) The events `{Y_0=Y*,Y_g≠Y*}` and
`{Y_0≠Y*,Y_g=Y*}` are disjoint, and each implies `Y_g ≠ Y_0` (if `Y_0=Y*≠Y_g` then
`Y_g≠Y_0`; if `Y_g=Y*≠Y_0` then `Y_0≠Y_g`). Hence `p_+- + p_-+ ≤ P(Y_g≠Y_0) = ε`, and
`|p_+- − p_-+| ≤ p_+- + p_-+`. (3) is (†); the extremal case sets `p_-+ = 0`. ∎

**Corollary 6.2 (idealization is an upper bound).** Any *realizable* IRIC intervention has
accuracy `≤ A_0` (it cannot extract more truth than the independent ensemble contains under
the fixed aggregator), with equality in the limit of a perfect fresh resample. Thus (†) is
not optimistic: it is the **ceiling** for the whole class, and Theorem 6(3) bounds the
entire class, not just the oracle.

---

## 4. The three-layer separation — `D ≠ ΔY ≠ H_repair`

> **Corollary 6.3 (non-implication chain).** With `D` the measured error dependence
> (Fleiss ICC of `V_g`), `ΔY = ε` the winner-change rate, and `H_repair` as above:
>
>     H_repair  ≤  p_+-  ≤  ΔY,     and     ΔY  is not determined by  D.
>
> Each inequality can be strict, so **large dependence need not imply a changed decision,
> and a changed decision need not imply recoverable accuracy.** The "wasted" decision
> change `ΔY − H_repair = p_-+ + (ΔY − p_+- − p_-+)` is the beneficial-flip mass plus the
> wrong→other-wrong churn — decision motion that carries no net actionable signal.

This is strictly stronger than "accuracy is unchanged" (`|ΔA|` small, Theorem 5's
dissociation). Theorem 5 says a dependence change reaches accuracy only through the
low-margin gate `χ`; Theorem 6 says that even a *realized* low-margin decision change is
recoverable only in its **net-directional** part. Diagnosing `D` — even diagnosing the flip
`ΔY` — does **not** certify actionability: `H_repair` is a third, strictly smaller quantity.

**Compact statement of C5 (reframed 2026-09-01):** *Diagnosis does not imply actionability.*
For any IRIC intervention the recoverable accuracy is the net directional flip
`p_+- − p_-+ ≤ ΔY` — not the dependence `D` nor the decision change `ΔY`. The bound is
**universal**; its *magnitude* scales with the **independent diversity `1 − D_0`** (the
diversity copying destroys), so a perfectly identified, arbitrarily strong dependence is
**effectively non-actionable when the independent aggregate is near-consensus** (high `D_0`),
and only weakly actionable when it is diverse (low `D_0`) — and even then bounded by `ΔY`.

---

## 5. Empirical instantiation (observed)

`data/processed/exp6_truthfulqa_N11_g*.npz` (qwen2.5:7b, N=11), independent `g=0` vs each
copying strength `g`; every question paired on text. Script:
`experiments/exp6_phase_transition_llm/repair_headroom.py`.

| g (copy strength) | n | A_0 | A_g | **H_repair = A_0−A_g** | p_+- | p_-+ | ε = ΔY | identity | bound |
|---|---|---|---|---|---|---|---|:--:|:--:|
| 0.25 | 46 | 0.717 | 0.717 | **+0.000** | 0.065 | 0.065 | 0.152 | ✓ | ✓ |
| 0.50 | 98 | 0.490 | 0.469 | **+0.020** | 0.041 | 0.020 | 0.122 | ✓ | ✓ |
| 0.75 | 46 | 0.717 | 0.674 | **+0.043** | 0.065 | 0.022 | 0.130 | ✓ | ✓ |
| 1.00 | 98 | 0.490 | 0.490 | **+0.000** | 0.020 | 0.020 | 0.092 | ✓ | ✓ |
| 2.00 | 60 | 0.350 | 0.317 | **+0.033** | 0.033 | 0.000 | 0.117 | ✓ | ✓ |

Reading:
- **Proposition 6.1 holds to machine precision** at every `g` (`A_0−A_g = p_+- − p_-+`).
- **Decision change is 5–15×  the actionable headroom:** the winner changes on `ε = 9–15%`
  of questions, yet `H_repair` is only `0–4.3%` — the flips are near-symmetric
  (`p_+- ≈ p_-+`) except at the pathological `g=2` (no escape-hatch, `p_-+ = 0`), where the
  bound is met with equality.
- At **realistic copying `g=1`**, the amplification is stark: `Y_g = Y_0` on **91%** of
  questions; the 9% that move split `2:2` correct↔wrong, so `H_repair = 0` **exactly** —
  not because a recovery operator is poorly built, but because there is nothing to recover.
- The SGIR margin-gated oracle (`sgir_copying_oracle.py`) confirms the operational reading:
  low-margin recovery gain `+0.000` (g=1) / `+0.037` (g=2), and the margin gate adds
  `≈ 0` over resampling a random subset of equal compute — consistent with `H_repair`
  being small and *not* concentrated beyond what `ε` already implies.

**Cross-benchmark (qwen, AQuA math; `repair_headroom_xbench.py`).** The **theorem
generalizes** — identity and bound hold exactly on AQuA too — but the *magnitude* of
`H_repair` tracks the **independent diversity** `1 − D_0` (mean independent pairwise
vote-agreement `D_0`):

| bench | `D_0` | `D_g` (g=1) | `H_repair` (g=1) | `ε = ΔY` (g=1) | id / bound |
|---|---|---|---|---|---|
| TruthfulQA (n=98) | 0.83 | 0.98 | **+0.000** | 0.092 | ✓ / ✓ |
| AQuA (n=20) | **0.53** | 0.93 | **+0.100** | 0.500 | ✓ / ✓ |

On TruthfulQA the independent crowd already near-agrees (`D_0 = 0.83`), so copying has
little genuine diversity to destroy and `H_repair ≈ 0`. On AQuA the independent crowd is
more diverse (`D_0 = 0.53`); copying destroys more of it (`ε = 0.50`) and the recovery
*ceiling* rises to `+0.10`. So `H_repair ≈ 0` is **not universal** — it is the regime of a
high-`D_0` (near-consensus) independent aggregate. This is exactly the sign/magnitude
hypothesis of Cor 6.3(ii): recoverable headroom `≈` the diversity copying destroys, which
is set by the correctness/agreement structure of the independent aggregate, not by `D` (see
§6 Discussion hook, C5↔C1). **Do not write "decorrelation never helps."**

**The scope condition — suggestive, not proven.** Across the action-enrichment study the
single action that ever raised accuracy is `a_common` (pooled `+0.016`, **0/4 models
significant**) — the only **non-IRIC** action (it injects new evidence, escaping Theorem 6).
Every IRIC (or IRIC-bounded) action (`a_copy`, `a_CI`, SGIR, K2 router) lands within
`[−, +ε]` of the correlated collective and none is a robust win. This is a **suggestive
consistency** with the bound's dividing line — *new information vs. rearrangement* — **not**
empirical proof that "new information solves the problem" (`a_common` is weak and
non-significant).

*Statistical caveat.* One realization per question, `n = 20–98`, so per-`g` `H_repair`
point estimates carry `±0.03` (TQA) to `±0.13` (AQuA n=20 = 2 net questions), and the
non-monotone `g`-profile is within noise. The robust claims are: (a) the identity and bound
are **exact regardless of `n`**; (b) `H_repair` is small **and `≤ ε`** and **decoupled from
`D`** — not a precise `H_repair(g)` curve, and its *magnitude* is benchmark-dependent
(TruthfulQA ≈ 0, AQuA up to +0.10, both fragile).

---

## 6. Positioning — what makes C5 a result, not an apology

The controller path is **not** "we tried five algorithms and they failed." It is one bound
plus a negative chain **consistent with** it (each operator's IRIC status argued, not
assumed — §1):

- **vs. a controller contribution.** We do *not* claim "no intervention can help" — that is
  false and out of scope: `a_common` (new evidence) and any retrieval/aggregation change
  live outside IRIC. We claim the *decorrelation* lever specifically is bounded by
  `p_+- − p_-+ ≤ ε`, and empirically that net is `≈ 0` when the independent aggregate is
  near-consensus (TruthfulQA) and rises to `≈ +0.10` when it is diverse (AQuA, n=20) — a
  benchmark-dependent magnitude, always `≤ ε`.
- **vs. the "correlation hurts crowds" folk claim (dead, Kim 2025 / Condorcet).** Theorem 6
  sharpens *why* the naive fix fails: dependence that amplifies the aggregate leaves the
  decision — hence the recoverable accuracy — pinned to the independent votes.
- **vs. Theorem 5.** Thm 5: a dependence change reaches accuracy only near the boundary
  (`D → ΔA` gated by `χ`). Thm 6: the *realized* boundary decision change is recoverable
  only in its net-directional part (`ΔY → H_repair`). Together: `D ⇏ ΔY ⇏ H_repair`.
- **Discussion hook.** *Decorrelation cannot manufacture information the collective never
  had.* If the error is already in the independent aggregate, making agents less alike does
  not fix a majority that is independently wrong — only new evidence can move `Y_0`.

---

## 7. Status — proven vs. observed

- **Proven here (model-agnostic):** the exact decomposition (Prop 6.1), the headroom bound
  and class ceiling (Thm 6, Cor 6.2), and the three-layer non-implication (Cor 6.3). These
  follow from the threshold nature of plurality plus the IRIC definition; no dynamics model
  for `T_g` is assumed.
- **Observed (qwen2.5:7b):** identity holds to machine precision at 5 coupling strengths
  (TruthfulQA) and 2 (AQuA); `g=1` amplification `P(Y_g=Y_0)=0.91` (TruthfulQA);
  `H_repair`≈0 on TruthfulQA (`D_0=0.83`) but up to `+0.10` on AQuA (`D_0=0.53`, n=20) —
  cross-benchmark, GPU-free; `a_common` (non-IRIC) the sole positive action (weak).
- **Open:** (i) **cross-model** reproduction — the copying (g-sweep) states exist only for
  qwen; gemma/llama/mistral have the *shared-evidence* boundary arms (done, `H_repair`-like
  low-M gains all `< 0.05`) but **no g-sweep** — so the copying-axis identity/bound on ≥3
  model families would need a GPU run (deliberately not run). (ii) a sign/magnitude law —
  Cor 6.3 leaves the *sign* of `p_+- − p_-+` to the correctness structure of the independent
  aggregate; the TruthfulQA↔AQuA `D_0`-gradient is suggestive **data** for it, but it stays
  a **Discussion hypothesis** (not formalized): *whether dependence is net harmful or
  beneficial is set not by its magnitude but by the correctness structure of the independent
  aggregate it amplifies* — the C5↔C1 bridge. (iii) characterize non-IRIC interventions
  (retrieval/new evidence) that can exceed `A_0` — the genuinely open algorithmic direction.
