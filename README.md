# mfg-llm-debate → Mechanism-Aware Collective Reasoning

> **Project identity:** *A mechanism-aware theory and control framework for collective
> intelligence in populations of LLM agents.* (No longer "MFG applied to LLM debate" —
> that framing is retired.)

**Central question.** *How does the causal **origin** of statistical dependence
determine whether a collective of LLM agents produces wisdom or herding — and can that
structure be measured, predicted, and controlled?*

> **STATUS (2026-09-01).** Theory closed (Thm 4/5/6); five contributions C1–C5 on the spine
> `M → D →χ→ ΔY → H_repair` (three-stage insufficiency: `D ⇏ M ⇏ ΔY ⇏ H_repair`). TACL analyses
> A (C2 ≠ agent uncertainty) + B (same-D mechanism split) done. **Now auto-driving a scale-up
> campaign: 6 model families × 7 benchmarks, tiered compute (breadth → depth → C5 → meta), GPU1
> only** (§5.0). Paper skeleton in `paper.md`.
>
> **UPDATE (2026-09-22).** Post-literature-review scope **locked** (no new methods/models/
> benchmarks — run/stop list in §5.0). Two offline falsification analyses **DONE** on the frozen
> 4×2 substrate: (#1) **matched-D falsification of C1** — new `analysis_C_matched_D_falsification.py`
> (copy−shared ΔY gap **+0.08–0.09 survives ε=.02 caliper***, nested-F mechanism|D **p=.007**,
> NN-match on (D,A0,margin) **+0.05***); (#2) **C2 competing-predictor** re-confirmed (margin beats
> agent-confidence, collinear with vote-entropy/disagreement → keep "susceptibility **family**").
> **F2 main figure** built. **Consistency audit** of all load-bearing `paper.md` numbers vs
> `results/tables` = ALL PASS. **Tier-1 6×7 breadth LAUNCHED and running on GPU1** (this machine;
> see §5.0 checklist). Details: `OFFLINE_ANALYSES_2026-09-22.md`.

The project began as a mean-field-game theory of truth-seeking LLM debate (HJB–
Fokker–Planck equilibrium for `N→∞` populations). It has since sharpened into a
**mechanism-aware** account of collective reasoning: the headline is **not** "LLM
debate has a phase transition" (that is classical correlated-Condorcet theory), but

> **Collective accuracy is governed by *why* agent errors are correlated, not merely by
> *how much* they are correlated. Shared evidence and social copying can produce
> identical agreement yet opposite collective outcomes.**

---

## 1. Conceptual framework

We replace the impoverished `(p, ρ)` (competence, correlation) description with a
four-part collective state:

```
S = (p, D, M, N)
    p : single-agent competence           (estimable pre-debate; D1)
    D : dependence STRENGTH               (scalar: ICC / pair-agreement / error-corr)
    M : dependence MECHANISM  ∈ {private, common-evidence, social-copying, shared-model-bias}
    N : population size
```

Collective accuracy is `A_coll = F(p, D, M, N)` and **does not factor through
`(p, D, N)`** — the mechanism label `M` is required (Theorem 4). The three realizable
mechanisms and their `ρ→1` limits:

| Mechanism `M` | causal graph | limit of `A_coll` as `ρ→1` | `Δ = A − A₀` |
|---|---|---|---|
| **common evidence** (upstream) | `Z₀ → Xᵢ → Y` | `p` | `sign(½−p)` |
| **synchronous copying** (downstream) | `X⁽ᵗ⁾→Yₜ→X⁽ᵗ⁺¹⁾` | `P_maj(N,p,0)` | `0` |
| **cascade copying** (downstream, ordered) | BHW 1992 | `≤ P_maj(N,p,0)` | `≤0` |

Counter-intuitive consequence (already in our data): for a **competent** crowd (`p>½`),
downstream copying (`Δ=0`) is **less** harmful than upstream common evidence (`Δ<0`).

### 1.1 Unifying model — `M → D →χ→ ΔY → H_repair`  **[spine; extended 2026-09-01 with C5]**

The contributions are **one causal chain**, not separate findings — and the organizing idea is a
**three-stage insufficiency principle**: dependence magnitude `D` determines *neither* the
mechanism, *nor* the consequence, *nor* the repairability.

```
        M ─────► D ──[χ]──► ΔY ──[directionality]──► H_repair
  M  : causal MECHANISM generating dependence   (shared evidence vs social copying)
  D  : measured error DEPENDENCE                (replicate-level ΔD; identifiable, C3)
  χ  : collective decision SUSCEPTIBILITY       (low-margin / boundary gating; C2)
  ΔY : collective DECISION CHANGE               (P(Y_corr ≠ Y_indep))
  H_repair : recoverable ACCURACY               (= p₊₋ − p₋₊ ≤ ΔY; C5, Theorem 6)

  Stage I   D ⇏ M      magnitude does not reveal the mechanism            (C1, Thm 4)
  Stage II  D ⇏ ΔY     dependence moves the decision only through χ       (C2, Thm 5)
  Stage III ΔY ⇏ H_repair  a changed decision is recoverable only in its net direction (C5, Thm 6)
```

It answers questions that **no single prior result answers together**:
- **Why?**  `M → D` — where the dependence comes from (not merely that it exists).  [C1/C3]
- **When?**  `D ─χ→ ΔY` — when a *change* in dependence moves the collective decision.  [C2]
- **How does it respond to control?** the mechanism sets the *intervention* response.  [C4]
- **Is it fixable?** `ΔY → H_repair` — a diagnosed dependence is repairable (by decorrelation
  alone) only in its net directional flip; can be `≈0` even at `D→1`.  [C5]
- **Why isn't `ρ` enough?**  `A ≠ F(p, D, N)` — the map needs `M` **and** `χ`.

Strictly stronger than the earlier `S=(p,D,M,N)` state description: a faithful predictor is
`A = F(p, D, M, χ, N)`, and the chain makes the *mechanism → consequence → actionability* pathway
explicit. **Formalized** in `theory/06_mechanism_chain_MDCA.md` (overview),
`theory/05_theorems/thm4_…md` (`M→A` limits = C1), `thm5_…md` (`χ`-gate = C2), and
`thm6_repair_headroom.md` (repair bound = C5). [P1 ✓ theory]

---

## 2. Contributions & status

*Deliverables `D1–D5` (experiments/theory built). The **novelty claims** `C1/C2/C3/C4/C5` —
Mechanism Non-Equivalence / boundary amplification / matched instrument / **mechanism-dependent
intervention response** / **diagnosis⇏actionability (repair-headroom bound)** — live in §2b and are
the canonical `C`-numbers everywhere else (theory docs, §2c). `C4` (added 2026-08-20) is the
intervention-side extension of `C1`. `C5` (added 2026-09-01, Theorem 6) turns the CDCR/SGIR negative
chain into a **bound**: for independence-restoring, no-new-information interventions the recoverable
accuracy is exactly the net directional flip `p_+- − p_-+`, so an arbitrarily strong, perfectly
diagnosed dependence can be **non-actionable** by decorrelation.*

| | Deliverable | Status |
|---|---|---|
| **D1** | **Measurement of endogenous dependence** — homogeneous small LLMs make correlated errors; early order-parameter predicts per-question help-vs-hurt | **[DONE — prototype]** |
| **D2** | **Controlled causal instrument** — inject mathematically-controlled correlation onto *real* LLM beliefs (Exp6B), holding private beliefs fixed | **[DONE]** |
| **D3** | **Mechanism-aware theory** — Theorem 4 (`M→A` limits) + Theorem 5 (`χ`-gate) + Theorem 6 (repair-headroom bound) + chain `M→D→χ→A→H_repair` | **[DONE — proofs written; P1 ✓]** |
| **D4** | **Native validation** — LLM-native shared evidence reproduces the dependence transition (Exp6C ✓ H1 4/4 sig); multi-model / multi-benchmark generalization | **[DONE — Exp6C qwen · P2 multi-model DONE 2026-08-18: 4/5 configs pass C1/C2/C3 (3 families + AQuA cross-dataset); mistral7b = boundary case (C3 ✓, C1/C2 ✗)]** |
| **D5** | **CDCR controller → repair-headroom bound (C5)** — the negative chain, now formalized: no independence-restoring / no-new-info intervention beats the independent aggregate | **[RESOLVED as a bound, not parked — K1 thin · K2/K2.5-pre null · action-enrichment (a_copy dead, a_CI KILL 08-25) · SGIR KILL 09-01 · Theorem 6 explains all: IRIC headroom = `p_+- − p_-+ ≤ ΔY ≈ 0`. Only NON-IRIC (new evidence, e.g. `a_common`) can exceed `A_0`. See `theory/05_theorems/thm6_repair_headroom.md`, §5 K3]** |

---

## 2b. Novelty after literature audit (to 2026-08-14)  — narrower, sharper, still alive

Systematic audit of the 2026 LLM-collective / MAD / social-influence landscape, with the
load-bearing papers **independently verified** (arXiv IDs confirmed, §2c). **Verdict: novelty
survives, but is narrower than assumed and two 2026 papers must be dialogued directly.**

| Claim | Post-audit verdict | Overlap |
|---|---|---|
| **Mechanism Non-Equivalence** — causal *origin* of dependence (not just strength) sets the outcome | 🟢 alive — core candidate | 🟡 medium |
| **Collective decision-boundary amplification** — `χ` gates the *consequence* of a dependence perturbation | 🟡🟢 alive, **narrow** claim | 🟠 close |
| **Matched shared-vs-independent evidence instrument** — replicate-level ΔD | 🟢 alive if estimand kept exact | 🟡 medium |
| **Mechanism-dependent intervention response** — the *effect* of a decorrelation intervention depends on which mechanism generates the dependence (C4, new 2026-08-20) | 🟡🟢 alive — interaction robust, marginal-accuracy claim withheld | 🟡 medium |
| **Diagnosis ⇏ actionability** — repair-headroom bound: decorrelation w/o new info cannot beat the independent aggregate (C5, new 2026-09-01, Theorem 6) | 🟢 alive — impossibility-style, proven + observed | 🟢 low (no direct neighbor) |
| "Correlation hurts crowds" | 🔴 **dead** — Kim 2025 + classical Condorcet | 🔴 |
| "LLM agents herd / conform" | 🔴 **dead** — many | 🔴 |
| "Boundary makes agents conform" | 🔴 **near-dead** — Bellina 2026 | 🔴 |
| CDCR passive router | 🔴 not claimable now (K2 / K2.5-pre null) | 🔴 crowded |

**Claim wording — LOCKED (do not drift back to the dead phrasings):**
- **C1** 🟢 — NOT "correlation is insufficient." → *"**Mechanism Non-Equivalence**: statistically
  similar dependence has behaviorally different consequences depending on its causal origin —
  upstream shared evidence and downstream social copying induce distinct collective limits."*
- **C2** 🟡🟢 — NOT "LLMs are susceptible near their boundary" (Bellina too close). → *"**Collective
  decision-boundary amplification**: the effect of a causal dependence perturbation on aggregate
  decisions concentrates near low-margin collective states — explaining why large shifts in ΔD
  coexist with negligible mean-accuracy change."*  (pooled `|ΔA|~M` slope `−0.125` [−0.198,−0.057])
- **C3** 🟢 — NOT "first causal study of LLM collectives." → *"**Matched dependence intervention**:
  a shared-vs-independent evidence manipulation that changes dependence structure while matching
  evidence quality, enabling replicate-level estimation of the causal increase in inter-agent
  error dependence."*
- **C4** 🟡🟢 — NOT "we found an accuracy-improving decorrelation algorithm" (`a_common` marginal lift
  is small and not universally significant — see §4). → *"**Mechanism-dependent intervention response**:
  the effect of a decorrelation intervention on collective accuracy changes systematically with the
  causal *source* of the dependence — there is **no single universal decorrelation policy**. Upstream
  common-evidence states and downstream copying states respond differently to the same intervention
  (cross-model `mechanism × intervention` interaction significant), reinforcing C1 on the intervention
  side: mechanism governs not only the collective's spontaneous dynamics but also its response to
  control."*  **Withheld (do NOT claim):** that the *correct* copying-targeted action has been found —
  `a_copy` is dead, so the copying cell of a matched policy is still empty (§5 K3).
- **C5** 🟢 — NOT "no intervention can improve LLM collectives" (false, out of scope). → *"**Diagnosis
  does not imply actionability** (repair-headroom bound, Theorem 6): for any *independence-restoring,
  information-conserving* intervention, the recoverable accuracy over a correlated collective equals the
  net directional flip `p_+- − p_-+` and is bounded by the winner-change rate `ΔY`. The **bound is
  universal**; the recoverable *magnitude* scales with the **independent diversity `1 − D_0`** (how much
  diversity copying destroyed) — so an arbitrarily strong, perfectly diagnosed dependence is
  **effectively non-actionable by decorrelation when the independent aggregate is near-consensus**
  (TruthfulQA `D_0=0.83`, `H_repair≈0`) and only weakly actionable when it is diverse (AQuA `D_0=0.53`,
  `H_repair≈+0.10`, n=20)."*
  **Scope (state explicitly):** the bound is for the IRIC class only; **new-evidence / retrieval**
  interventions (e.g. `a_common`) escape it — and are the *only* actions that helped (a **suggestive**
  consistency, `a_common` +0.016 / 0/4 sig — NOT proof that "new info solves it"). **Empirical:** identity
  `A_0−A_g = p_+- − p_-+` exact at 5 coupling strengths (TruthfulQA) + 2 (AQuA); `H_repair` benchmark-
  dependent — **≈0 on TruthfulQA** (`D_0=0.83`, near-consensus) but **up to +0.10 on AQuA** (`D_0=0.53`,
  n=20, fragile) — always `≤ ΔY` (`repair_headroom.py`, `repair_headroom_xbench.py`). CDCR/K2/`a_copy`/
  `a_CI`/SGIR are **consistent with** the bound (each operator's IRIC status argued in Thm6 §1, not
  assumed); do **not** write "Theorem 6 explains all five failures" or "decorrelation never helps."

**Positioning sentence (must open related work with this):**
> Prior work establishes that LLM errors are correlated (Kim et al. 2025) and that LLM
> collectives herd / conform (Bellina 2026; SNLA 2026). We ask **why** the dependence arises
> (mechanism) and **when** a causal change in it alters the collective decision (susceptibility)
> — not that dependence or herding exists.

**"To our knowledge, first to…" — permitted ONLY for** (no direct duplicate found in audit):
(i) *causal mechanism generating dependence changes the collective consequence*;
(ii) *collective-vote-boundary gating of a dependence **intervention*** (distinct from
conformity-at-competence-boundary — position carefully vs Bellina / 2605.27288);
(iii) *matched shared-vs-independent evidence dependence instrument with replicate-level ΔD*.

**Still cannot claim:** ❌ "new phase transition" (classical correlated-Condorcet) · ❌ "superior
MA algorithm" (CDCR null; and the `a_common` standalone win did **not** replicate strongly cross-model
— pooled paired lift `+0.016` [−0.003,+0.037], **0/4 models significant**, mistral even negative, §4 —
so `a_common` is a **secondary** result, not an algorithm contribution; the robust claim is the C4
*interaction*, not a marginal-accuracy gain) · ❌ "universal
*scaling law*" — multi-model (4/5 + AQuA) **and** N-scaling are now DONE, but the Theorem-5 monotone
`N↑` prediction did **not** confirm (non-monotone, peaks at `N=11`; §5 ①), so the finite-size claim
must be reframed (interior-`N` peak) rather than stated as a clean law. **Risk is now
evidence-shape / write-up honesty**, not missing generalization (see §5 P2 / N-scaling).

## 2c. Key related work — verified arXiv (audit 2026-08-14)

| Paper | arXiv | Mech? | Causal interv.? | Boundary? | How we differ | Flag |
|---|---|:--:|:--:|:--:|---|---|
| **CAGE-CAL** — Counterfactual Graph for MA-LLM Calibration | `2605.30653` (May'26) | ~ | ✓ comm/no-comm | ✗ | they contrast **communication/topology** for calibration; we contrast two causal **origins** of dependence (shared-ev vs copying) at matched `D` | 🔴 **DANGER** |
| **SNLA** — Social Networks of LLM Agents | `2607.03695` (Jul'26) | ~ influence | ✗ | ✗ | attention-width herding↔wisdom transition; **no** matched same-`D`/different-mechanism comparison | 🟠 OVERLAP |
| **Conformity & Social Impact** (Bellina) | `2601.05384` (Jan'26) | ✗ | ✓ Asch | ✓ | their boundary = **individual competence**→conformity; ours = **collective vote-margin**→consequence of a dependence perturbation | 🟠 OVERLAP (C2) |
| **Not All Flips Are Conformity** | `2606.00820` (Jun'26) | ✓ 3-way | ✓ counterfactual | ✗ | decomposes an **individual** flip's cause (~40% spontaneous); we ask population dependence-origin → collective accuracy | 🟡 cite-strong |
| **Do Latent Channels Actually Communicate?** | `2607.26773` (Jul'26) | ✓ | ✓ msg-replacement | ✗ | "aggregate perf can't identify mechanism" — same philosophy, latent-channel scope | 🟡 cite |
| **Correlated Errors in LLMs** (Kim) | `2506.07962` (ICML'25) | ✗ | ✗ | ✗ | establishes errors correlate across 350 models; **kills "we found correlation"** | 🟡 must-cite |
| **HiddenBench / Hidden Profile** | `2505.11556` (May'25) | ✗ | ~ info-asym | ✗ | distributed-info integration failure; different estimand | ⚪ context |

**Extra 2026 hits the audit missed (from this independent search — read before submission):**
- **`2605.27288`** *It's Not Always Sycophancy: LLM Conformity as a Function of Epistemic Uncertainty* — 🟠 conformity gated by **uncertainty ≈ our `χ`**; **closest C2 neighbor, read closely.**
- **`2607.05545`** *Most LLM Conformity Needs No Speaker: the Speaker-Free Floor* — spontaneous-flip **baseline floor**; bears on interpreting `ΔA` (some flips are instability, not influence).
- **`2608.02758`** *Everyone Conforms, No One Believes: Pluralistic Ignorance in LLM Populations* — consensus-without-belief; herding framing.
- **`2607.18269`** *Wisdom of LLM Crowds: Aggregation and Contamination in LM Ensembles* — ensemble aggregation.

---

## 3. What is done — detailed

### 3.1 Synthetic mean-field core (blueprint exp1–4)  **[DONE]**
- **exp2 propagation of chaos** — `W₂` vs `N`, slope ≈ `−1/2` (matches mean-field rate almost exactly).
- **exp3 phase transition** — synthetic herding↔truth sweep; `signal/conformity > κ*` ⇒ consensus.
- **exp4 optimal control** — no-control / spectral / **LQR** / RL on the linearised dynamics.
- Core dynamics: `dm/dt = −(A−B)m + s(q)`, `dΣ/dt = −AΣ − ΣAᵀ + 2σI`, stability `Re eig(A−B) > 0`.

### 3.2 Real-LLM debate, honestly scored (exp5)  **[DONE]**
- Fixed the **flat-state parse artifact** (AQuA "debate = 1.00" was 63% unparsed rows scored as correct). Label-scoring prompt + tolerant key resolver + retry-on-flat; `plurality_prediction` drops flat agents, ties → `−1` (scored wrong). **38 unit tests pass.**
- Honest re-run (qwen2.5:7b, n=100): **AQuA** single 0.36 / vote 0.31 / debate 0.27; **TruthfulQA** single 0.59 / vote **0.65** / debate 0.61. Debate beats plain voting on neither → herding, on-thesis.
- **Root cause = error correlation**: on wrong questions ~4/5 agents pick the *same* wrong answer (pair-agreement 0.65 AQuA / 0.75 TQA). Five identical qwen agents ≈ five correlated draws.

### 3.3 D1 early-warning predictor  **[DONE — prototype]**
- Order parameters (no truth label) vs final correctness, AUC:
  round-0 independent margin (≈ MFG source term `s`, deployable pre-debate) **0.626 AQuA / 0.804 TQA**; final_margin 0.835 TQA; confidence 0.72–0.76.
- Debate erodes the independent signal (TQA net +2, AQuA net −4 flips). *Caveat: must beat the trivial confidence baseline.*

### 3.4 exp6 — the herding↔truth transition on real LLMs
- **Coupling knob `g`** in `orchestrator.py` (per-agent partial mean field; `g≥1` categorical conformity; `g≥2` no escape-hatch → ρ→1). **[DONE]**
- **`run.py` g-sweep** with realized `p_single`, **paired** lift CI + McNemar, direct ρ (Fleiss ICC): hard (n=40, p≈0.19) lift `+0.025→0`, easy (**n=21**, p≈0.63) lift `−0.05→0`. Sign-correct at low coupling, **collapses to 0 at high ρ** — copying locks onto the round-0 aggregate. **[DONE]**
- **Theory overlay** `theory_overlay.py` — closed-form `P_maj(N,p,ρ)` via Gauss–Hermite; lift flips negative iff `p>½`. **[DONE]**
- **Exp6A** synthetic Bernoulli equicorrelated simulator = analytic (worst |err| 0.026 at ρ=0.99 only). **[DONE]**
- **Exp6B** `exp6b_common_shock.py` — controlled common shock on real round-0 beliefs, **offline**. Reproduces the competence–correlation transition: hard (p≈0.22) lift `0→+0.026 ↑`, easy (p≈0.48) lift `0→−0.070 ↓`, both significant. Secondary finding: in real LLM beliefs, competence & independence are *coupled*. **[DONE]**
- **Ablation** `exp6_ablation_plot.py` — *social copying ≠ common evidence*: same ρ, copying → lift 0, common shock → lift → `p − P_maj`. The headline figure. **[DONE]**
- **Exp6C** `exp6c_shared_context.py` — LLM-**native** shared evidence, each agent reasons independently (never sees votes). 7 conditions = `none` + {correct, misleading} × {shared-weak, shared-strong, indep-strong}; **independent-evidence arm is the causal control**. Cleaned easy set (`p̂≥0.8`). **[DONE — 2026-08-10]**

  **Estimator fix (2026-08-09).** The original primary "ρ = within-question error-correlation" was a **−1/(N−1) artifact** — computed from a *single* realization per question it is algebraically pinned to `−1/(N−1)` (`−0.10` at N=11) for every cell → not identifiable. Replaced by a **replicate-level** estimator: `R` stochastic replicates per question×condition, evidence `C` **resampled each replicate** (a real common shock), so `(E_i, E_j)` error samples exist and dependence is identifiable.

  **Result (2026-08-10, `results/tables/exp6c_full_{easy,hard}_N11.csv`).** H1 confirmed
  **4/4** cells, all significant — shared evidence *causally* raises inter-agent error
  dependence: `ΔD_paired` = easy·correct **+0.260** [0.081,0.486]\*, easy·misleading
  **+0.422** [0.241,0.622]\*, hard·correct **+0.226** [0.096,0.377]\*, hard·misleading
  **+0.498** [0.312,0.717]\* (shared arm `D>0`, indep arm `≈0`). **H2 weak**: `ShareEffect`
  (accuracy) is significant *only* on easy·correct (**−0.022** [−0.047,−0.003]\*); the other
  three cells are directionally negative but CI-crosses-0 (−0.009…−0.037). **Headline = the
  dissociation:** dependence strength ↑↑ (ΔD 0.26–0.50) does **not** force collective
  accuracy to move ↑↑ (|ΔAcc| ≤ 0.04) — the outcome also depends on mechanism, competence,
  answer structure, evidence quality, and saturation. **Exp6C is CLOSED** (no hard-side
  chase, no N-inflation for a p-value); the next number is K1 (§5).

  **Frozen protocol & checklist:**
  - [x] Diagnose artifact (algebra: `ρ_q ≡ −1/(N−1)`), drop `within_q_error_corr`
  - [x] Primary `D(q) = pooled_pair_corr` (Var-of-sum identity; NaN if an arm saturates)
  - [x] **Primary H1** `mean_ΔD_paired = mean_q[D_shared(q) − D_indep(q)] > 0`, paired bootstrap on the shared∩indep mask
  - [x] **Primary H2** `ShareEffect = A_shared − A_indep` (plurality acc); easy `p>½ ⇒ <0`
  - [x] Independent arm = **dedup + without-replacement** draws (kill duplicate-artifact); `_frame` identical across arms → matched control (no strength/content change)
  - [x] `D_pooled` (per-arm NaN mask, descriptive) **≠** `mean_ΔD_paired` estimand — labelled apart so no phantom subtraction (paired component means subtract to ΔD exactly)
  - [x] **ENGLISH REBUILD (2026-10-01).** Three truthfulqa special cases removed so all seven benchmarks share one probe path: the legacy branch in `p2_benchmarks.select_questions`, the `return 40` short-circuit in `tier1_breadth.probe_count`, and a hard `SystemExit` guard at the top of `p2_probe.py`. **Threshold-identical** (legacy predicates were `p>=0.80`/`p<=0.30`, exactly `PRED`); only the cache holding p̂ and the candidate pool changed (400 uniformly, was 400 hard / 250 easy). `exp6c_shared_context.select_questions` now delegates to that path and `REGIMES` is retained as `_LEGACY_REGIMES` for provenance only. Model-facing prompts translated in `agent.py`, `orchestrator.py` (all 3 coupling rungs), `exp6c_shared_context.py`, `action_enrich.py`, `ci_enrich.py`; `agent.py:16 _LABEL_WORDS` deliberately keeps its Vietnamese entries because it is reply-**parsing** tolerance, not a prompt. Rendered prompts verified against `pp.tex` Appendix D.
  - [x] **C1 AT BREADTH — FINAL, ON THE COMPLETE ENGLISH-REBUILD SWEEP (closed 2026-10-12).** English-rebuild reached **72/72 cells** (2026-10-11); all four frozen downstream analyses have now been re-run on the full grid, in order, with no threshold/estimator/exclusion changes made after seeing output: `tier1_split_sample_strata.py` → `tier1_dA_regime_test.py` → `c3_global_directional_test.py` → `tier1_dY_decision_churn.py` (+ `tier1_dY_decision_churn_aggregate.py`). **66/72 strata usable** (31 easy, 35 hard; 6 below the `n≥15` floor, `bbh` excluded — see below).

    **ΔY > 0 raw headline is RETIRED — replaced by `dC` (excess churn net of within-arm noise).** The 42/84 Vietnamese-era-substrate snapshot previously struck through here is gone for good, and so is the English-substrate raw-`ΔY` framing that would have replaced it: raw `ΔY_cross = P(Y_shared≠Y_indep)` at matched replicate is **descriptive only**, because most of it is indistinguishable from the decision churn a single arm already produces across replicates from sampling noise alone (`C_within`, the all-pairs disagreement rate within one arm over the held-out evaluation replicates). The **inferential estimand is `dC = ΔY_cross − C_within`**, audited and frozen 2026-10-12 (`tests/test_tier1_dY_decision_churn.py`, 3/3 pass). Primary aggregation is the **macro-average across model–benchmark strata** (`dC = (1/K) Σ_k dC_k`, each of the 66 usable strata weighted equally regardless of its question count — the question is whether the effect generalizes across configurations, not what it averages to over our particular benchmark-size mix; `strict` tie convention primary). Audited 2026-10-12 against the question-count-weighted alternative (`tier1_dY_decision_churn_aggregate_sensitivity.py`) — **agrees in sign and order of magnitude in every regime/tie-convention combination**, so the conclusion is not a weighting artifact; sensitivity numbers are reported alongside, never promoted to a second headline. Leading with **magnitude, not significance**: after accounting for within-arm stochastic instability, excess cross-arm decision churn is positive but small — **easy `dC = +0.004`** (0.4 points) 95% CI `[+0.001,+0.009]` (raw `ΔY_cross=0.019`, `C_within=0.014`); **hard `dC = +0.014`** (1.4 points) 95% CI `[+0.002,+0.025]` (raw `ΔY_cross=0.270`, `C_within=0.256`). Both aggregate CIs exclude 0, even though only **1/31** (easy) and **7/35** (hard) *individual* strata reach significance alone — those per-stratum counts are supporting heterogeneity detail, not the headline; do not lead with them, and do not write "`dC` is significant in both regimes" as the headline sentence — lead with the magnitude. Full table: `results/tables/tier1_dY_decision_churn.{json,md}` (frozen per-cell), `tier1_dY_decision_churn_aggregate.json` (macro headline), `tier1_dY_decision_churn_aggregate_sensitivity.json` (weighting sensitivity, audit-only).

    **`dA`: PREREG battery run (`PREREG_dA_regime.md`), gate `K=66≥48` OPEN.** Verdict **R1 — regime-invariant negligibility**: TOST equivalence (`|E[dA]|<0.05`) holds in both easy and hard, and the easy/hard interaction (`β̂₁`) CI covers 0. ⇒ cite **Corollary 4.1** (`A` does not factor through `(p,D,N)`), not Theorem 4 part (1). **H1 (sign-matched to Thm-4's `M₁` prediction) is NOT supported** (12/28 testable strata, `p=0.83` — worse than chance), so do not claim the theorem's directional prediction is confirmed either. `results/tables/tier1_dA_regime_test.json`.

    **C3: frozen global directional test ONLY** (`c3_global_directional_test.py`), gate `K=66≥48` OPEN. Statistic is **signs only**, magnitude never used: `S/K = 63/66` strata with `ΔD̂>0`, null simulated at each stratum's own design (mean `13.38/66`), **one-sided `p=0.00050`** ⇒ **C3 supported, directionally**. Per the frozen reporting rule: **no per-stratum `ΔD` point estimate, no bootstrap interval, and no `k/n significant` tally may be quoted for C3** — `tier1_breadth_aggregate.py`'s per-cell `ΔD` tallies are a diagnostic table, never the C3 inference path. `results/tables/c3_global_directional_test.json`.

    ⇒ **`dA ≈ 0` is still NOT a null — never report it without `dC` beside it** (not raw `ΔY`). **Coverage gap, declared not fixed:** Tier-1 has no copying arm, so C1's *matched-`D` mechanism-non-equivalence* stays **qwen-only** (`analysis_B`/`analysis_C`) and is not replicated at breadth.
  - [x] **STRATIFICATION DEFECT + REPAIR (decided 2026-10-01).** The probe's `easy`/`hard` labels do **not** reproduce: `p2_probe.py:56` seeds the option-display shuffle on the **dataset index**, `exp6c_boundary.py:243` on the **subset position**, so selection and measurement show different option permutations. Scope: **labels only** — all within-cell contrasts (`ΔD`, `dA`/`ΔY`, `|ΔA|~M`) are intact, since the three arms share one question set and one display order. Code deliberately **not** re-seeded (would orphan the completed cells). Repair: `tier1_split_sample_strata.py`, offline — bin on `none` replicates 0–3, evaluate on held-out 4–7, frozen thresholds; `per_q` reproduces the runner bit-for-bit (720 cells, 0 mismatches). **The recovered strata are the reported stratification; the probe labels are retired.** Disclosed in-text at `paper.md` §5.1, not in Limitations. **Updated on the complete 72/72-cell English-rebuild grid (2026-10-12):** 37 (model,benchmark) pairs × 2 regimes = 74 candidate strata, minus 2 `bbh` (excluded on stimulus validity) = 72, minus 6 below the `n≥15` floor ⇒ **66/72 strata usable** (31 easy, 35 hard). Cost unchanged: `R=8→4`, `easy` near-saturated ⇒ **C1/C2 quoted off `hard`, C3 off both, never averaged.**

    **AMENDMENT 2026-10-03 — two consequences that were not drawn at the time.**
    (i) *Any* correlation or stratum count computed on the retired probe labels is **SUPERSEDED** and must not appear in the manuscript, including the `corr(%p̂=1.0, n_dD)` and `corr(M, |dA|)` figures produced while diagnosing the saturation. Re-derive every reported easy/hard quantity from `tier1_split_sample_strata.py`'s recovered strata. `exp6c_D_estimators.py` still reads the probe-label dumps **on purpose** — it is an estimator audit, not a regime analysis — and says so in its header.
    (ii) The `R=8→4` cost is larger than the `n_q ≥ 15` floor suggested. The binding constraint is the *effective* sample size of the pooled dependence ratio, `n_eff = (Σw)²/Σw²` with `w = den`: questions carry unequal denominator weight, so `n_q` overstates the information behind the estimate (e.g. `n_q = 41`, `n_eff = 2.4`). Worse, blind simulation at `N=11, R=4` shows the pooled ratio is **upward-biased and its CI under-covers at any `n_eff`**, i.e. finite-`R` bias — not thin strata — may be the limiting problem. Estimability is therefore being calibrated against known-truth simulation (`calibrate_estimability.py`, tolerances frozen in its docstring) before any C3 count is written. **Until that rule is frozen, quote no `k/n` C3 figure**; say only that cell-level inference uses a calibrated estimability criterion. A non-estimable stratum is not a null.
  - [x] **SCOPE AMENDMENT (ratified 2026-09-30).** The two rows above stay in force **for this 4×2 substrate**: primary ΔD here remains `mean_ΔD_paired` (mean-of-ratios), for continuity with the published numbers (qwen ΔD +0.272 etc.). For the **Tier-1 recovered split-sample strata only**, primary ΔD is the **pooled ratio** (Σ num / Σ den over all questions, so a saturated question contributes num=den=0 and drops out arithmetically) — because there mean-of-ratios is computable on as few as **4 of 45** questions and that surviving subset is selected on error variance, the very quantity D measures. Both estimators are reported side by side in `results/tables/tier1_split_sample_strata.md` so the choice stays visible. ⇒ **The paper reports two different primary ΔD estimators on two different substrates and must say so explicitly.** Implemented at `tier1_split_sample_strata.py:364`.
  - [x] Offline synthetic go/no-go: recovers ΔD `+0.48` vs old-formula `−0.10` on the same data
  - [x] Mini-validation (10q, R=8, 5 cond): easy ΔD `+0.28*` (correct) / `+0.67*` (misleading), hard `+0.16*` (correct), hard misleading `+0.18` (n.s., NaN-saturation); `D(none)≈0` easy / `+0.06` hard
  - [x] **Full run** — 40q × 7 cond × **R=8** × N=11, easy→gpu1 / hard→gpu0 → `results/tables/exp6c_full_{regime}_N11.csv` *(done 2026-08-10; H1 4/4 sig, H2 only easy·correct sig)*
  - [x] Read in frozen order; hard beneficial side reported as **null** (H2 n.s.); mediation `bigger ΔD ⇒ bigger destruction` = **descriptive only**
  - [x] **→ K1** oracle-headroom harness **built & no-GPU-validated** (§5); no more exp6 variants

### 3.5 Theorem 4 — Mechanism Non-Equivalence  **[DONE]**
`theory/05_theorems/thm4_mechanism_non_equivalence.md`. Proves the three `ρ→1` limits
above and **Corollary 4.1**: `(p, D) ↦ A_coll` is not well-defined ⇒ dependence
strength is not a sufficient statistic. Includes honest novelty accounting: parts (1)/(3)
are classical (Ladha 1992 / Boland 1989 / BHW 1992), the *juxtaposition + dynamics-level
insufficiency + LLM instantiation* is the new content.

### 3.6 Theorem 6 — Repair Headroom (C5, diagnosis ⇏ actionability)  **[DONE 2026-09-01]**
`theory/05_theorems/thm6_repair_headroom.md`. **Prop 6.1 (exact, assumption-free):** for any
*independence-restoring, information-conserving* (IRIC) intervention, recoverable accuracy
`H_repair = A_0 − A_g = p₊₋ − p₋₊` (net directional decision flip). **Theorem 6 (bound):**
`|H_repair| ≤ ΔY` (decision-change rate); ceiling `= A_0`. **Universal; magnitude ∝ `1 − D₀`**
(the diversity the mechanism destroyed). Verified offline (`repair_headroom.py`): identity + bound
exact at 5 coupling strengths (`H_repair 0–4%` while `ΔY 9–15%`, copying preserves the independent
plurality 91% of the time), and **cross-benchmark** (`repair_headroom_xbench.py`): holds on AQuA,
with `H_repair≈0` on TruthfulQA (`D₀=0.83`) vs `≈+0.10` on AQuA (`D₀=0.53`, n=20 fragile). This
recasts the whole controller path (CDCR/K2/`a_copy`/`a_CI`/SGIR) as a negative chain **consistent
with** one bound — the only action that ever helped is the non-IRIC `a_common` (new evidence,
weak). See §5 for the SGIR gate that motivated it.

### 3.7 Analyses A & B — TACL hardening  **[DONE 2026-09-01, offline]**
Two pre-registered falsifications on cached belief states (`exp6c_raw_*`, g-sweep npz), GPU-free.
- **A — C2 uncertainty controls** (`analysis_A_c2_controls.py`, n=398, 5 configs). Margin's effect
  on `|ΔA|` is **undiminished by controlling for agent-level confidence** (β=−0.0345
  [−0.046,−0.024]*, confidence n.s.) ⇒ **not** individual epistemic uncertainty (rebuts Bellina /
  `2605.27288`); but statistically **inseparable from vote-entropy/disagreement** (collinear).
  ⇒ **C2 reframed** (pre-registered rule): *collective decision-boundary susceptibility **family***,
  robust to agent-confidence controls — margin is **not** sold as unique over entropy.
- **B — same-D, different-mechanism** (`analysis_B_matched_mechanism.py`). At matched dependence,
  downstream **copying is 2–3× more consequential** than upstream shared-evidence (D 0.80–0.90:
  copy ΔY 0.17/H +0.125 vs shared ΔY 0.10/H +0.042; D≈0.95: copy ΔY 0.078 vs shared ≈0). Empirical
  C1 smoking gun (**main-figure candidate**). *Caveat:* cached shared arm uses correct evidence, so
  part of the gap is upstream-converges-to-correct; a fully evidence-matched contrast needs a
  misleading-shared arm (not run).

---

## 4. Key results at a glance

```
                        lift = collective_acc − independent_acc,  as dependence ρ ↑
  regime (real p)   copying (Exp6 g-sweep)     common-shock (Exp6B)
  easy  (p>½)       ≈ 0                         −0.07   (herding destroys the crowd)
  hard  (p<½)       ≈ 0 / +0.02                +0.026  (herding removes anti-Condorcet)
```
Same measured correlation, opposite mechanism, different outcome — the empirical core of Theorem 4.

**Exp6C (LLM-native shared evidence, replicate-level ΔD, N=11, R=8).** Shared evidence
*causally* raises error dependence `ΔD ≈ 0.26–0.50` (**4/4 significant**) — yet the accuracy
effect is small (`ShareEffect −0.01…−0.04`, significant only on easy·correct). **Dependence
strength ↑↑ ⇏ |Δaccuracy| ↑↑** — the sharpest single-experiment statement of Theorem 4's
mechanism-vs-strength split, and the reason K1 (headroom) — not more correlation — is next.

**P2 generalization (multi-model, DONE 2026-08-18, N=11).** The C1/C2/C3 findings are **no
longer single-model / single-dataset**. Across 3 model families + an AQuA cross-dataset cell,
**4/5 configs pass all three claims**:

```
config         model          C1  C2 pooled |ΔA|~M slope     C3 pooled ΔD
gemma9b        gemma2:9b       ✓  −0.088 [−0.193,−0.019]    +0.536 [+0.350,+0.728]
llama8b        llama3.1:8b     ✓  −0.091 [−0.165,−0.019]    +0.148 [+0.104,+0.194]
qwen2.5:7b     qwen2.5:7b      ✓  −0.125 [−0.198,−0.056]    +0.272 [+0.162,+0.393]
qwen7b_aqua    qwen2.5:7b/AQuA ✓  −0.138 [−0.217,−0.058]    +0.159 [+0.101,+0.224]   ← cross-dataset
mistral7b      mistral:7b      ✗  −0.052 [−0.129,+0.014]    +0.249 [+0.145,+0.352]   ← boundary case
```

`qwen7b_aqua` reproducing all three on **AQuA math** (not TruthfulQA) shows C1–C3 are not a
single-benchmark artifact. **Mistral is a kept, reported exception** (C3 ✓; C1/C2 ✗) — a useful
boundary case, not to be hidden or tuned away (see §5 diagnostic). Aggregate:
`results/tables/p2_aggregate_summary.json`.

**C4 — mechanism-dependent intervention response (cross-model, DONE 2026-08-20, HARD, N=11, R=8).**
The action-enrichment gate (§5) added three mechanism-derived interventions to `A₀`. Two things
must be read **separately**:

*(i) Marginal accuracy of the best new action `a_common` — WEAK, secondary.* Paired lift of
`a_common` over each model's own best fixed action, and pooled:

```
model         best A₀    a_common lift   CI95              rank  sig
qwen2.5:7b    shared_ev   +0.037         [−0.009, +0.087]   1    ✗
gemma2:9b     shared_ev   +0.022         [+0.000, +0.050]   1    ✗ (touches 0)
llama3.1:8b   shared_ev   +0.016         [−0.037, +0.066]   1    ✗
mistral:7b    shared_ev   −0.009         [−0.037, +0.019]   3    ✗
POOLED (160q)             +0.016         [−0.003, +0.037]        0/4 sig
```
Direction is consistent (top-ranked in 3/4) but the effect is **small, CI-crosses-0, 0/4 significant,
and mistral is negative**. ⇒ `a_common` is **not** a general accuracy-improving algorithm; we do not
sell it as an algorithm contribution.

*(ii) The `mechanism × intervention` interaction — ROBUST, the C4 headline.* On common-evidence-heavy
questions the common-targeted action beats the copy-targeted one; on copying-heavy questions the
ordering **reverses** (a_copy is actively harmful):

```
model      subset (n)          Δ(common − copy)        CI95              sig
gemma2:9b  common-heavy (39)   +0.202                  [+0.096, +0.324]   ✓
llama3.1:8b common-heavy (40)  +0.288                  [+0.184, +0.397]   ✓
mistral:7b copying-heavy (7)   −0.446  (= Δ(copy−common)) [−0.768,−0.143] ✓  ← ordering flips
```
**The intervention that helps depends on the mechanism generating the dependence** — a *universal*
decorrelation policy is not supported. This reinforces C1 on the intervention side. **Careful wording:**
mistral's copying-heavy result shows a_copy is *worse* than a_common there — it demonstrates the
*response interacts with mechanism*, **not** that the correct copying-targeted action has been found
(a_copy is dead; §5 K3). Gates: `results/tables/actionenrich_gate_{gemma9b,llama8b,mistral7b}.json`;
rollup `actionenrich_xmodel_summary.json`.

**C5 — repair-headroom bound (Theorem 6, DONE 2026-09-01).** Copying *amplifies* the independent
plurality rather than flipping it, so for any IRIC intervention the recoverable accuracy is the net
directional flip and is bounded by the decision-change rate:

```
bench (qwen)   D₀→D_g (g=1)   ΔY (decision change)   H_repair = A_0−A_g   identity/bound
TruthfulQA     0.83 → 0.98    0.09                    +0.000               ✓ / ✓
AQuA           0.53 → 0.93    0.50                    +0.100 (n=20)        ✓ / ✓
```
`H_repair` is **≤ ΔY** and its magnitude scales with independent diversity `1−D₀` — near-0 when the
crowd already agrees, larger when it doesn't. Same-D mechanism split (Analysis B): at matched `D`,
copying is 2–3× more consequential than shared evidence. `repair_headroom.py`,
`repair_headroom_xbench.py`, `analysis_B_matched_mechanism.json`.

**A — C2 is not agent uncertainty (Analysis A).** Margin's effect on `|ΔA|` survives agent-confidence
controls (β=−0.0345*, confidence n.s.) but is inseparable from vote-entropy/disagreement ⇒ C2 =
**collective decision-boundary susceptibility family** (not margin-as-unique). `analysis_A_c2_controls.py`.

---

## 5. Roadmap

### 5.0 CURRENT WORK — TACL scale-up campaign  **[ACTIVE, started 2026-09-01; auto-driving on GPU1]**

After the theory closed (Thm 4/5/6) and A+B hardened the claims, the project scaled from 4 models /
2 benchmarks to **6 model families × 7 benchmarks** with a **tiered compute hierarchy** (breadth
screen, then deep causal analysis only on a representative subset — not full C1–C5 on all cells).

**Model suite (6, all in ollama):** `qwen2.5:7b` (anchor) · `llama3.1:8b` · `gemma2:9b` ·
`mistral:7b` (boundary case, kept) · `phi4-mini` (new family) · `deepseek-r1:7b` (reasoning-tuned
contrast, new). 16 GB VRAM → 7–9 B quantized; scale from *diversity*, not parameter count.

**Benchmark suite (7):** `truthfulqa` · `aqua` (existing) + `arc` · `commonsenseqa` · `mmlu` ·
`strategyqa` · `bbh` (fetched via HF datasets-server → truth@0 JSONL by `fetch_benchmarks.py`,
generic loader `benchmarks.load_mc_jsonl`, registered in `p2_benchmarks`). **GSM8K excluded** —
open-numeric, incompatible with the `(N×K)` belief-vector plurality machinery; AQuA is the math slot.

**Compute hierarchy (NOT 42 full runs):**
```
Tier 1  breadth   6 models × 7 benchmarks   C1/C2/C3 (exp6c_boundary, N=11, R=8)  → generalization heatmap
Tier 2  depth     3 × 3 {qwen,llama,gemma}×{truthfulqa,strategyqa,aqua}  C1–C4 + A-controls
Tier 3  C5        3 × 3  copying g-sweep g∈{0,1,2} → repair_headroom
Meta    offline   H_repair ~ β0 + β1(1−D₀) + u_model + v_task   (test the diversity→repairability law)
```
Fixed protocol, **no per-cell tuning**, frozen estimators; Mistral (and any other) boundary cells
**reported, not hidden** (no all-✓ heatmap). If the meta-regression `β1>0` is robust, "independent
diversity predicts repair opportunity" becomes a contribution (with directionality:
`Repairability = Opportunity × Directionality`).

**Infra:** GPU1-only (GPU0 runs another project — untouched; placement verified). Dedicated server
`:11445` (`CUDA_VISIBLE_DEVICES=1` + `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `np=8`), one model loaded at a
time (configs `src/config/scaleup_<model>_gpu1.yaml`). Orchestrator `tier1_breadth.py` (resumable,
continue-on-error, `--dry-run`).

**Status (2026-09-22): Tier-1 RUNNING on GPU1 inside a persistent `tmux` session.** Probes for
all 5 new benchmarks DONE (`data/processed/exp6_select_<b>.json` present). Tier-1 dry-run = **84
cells, 0 skipped, 0 awaiting-probe**. Realistic wall-clock ≈ **~2 days GPU1**. Then Tier 2 → 3 → meta.

> **Run it in `tmux`, not a session-bound background job.** A first launch (via the agent's
> background-task mechanism) was torn down with the session at cell 1/84 → 0 cells completed.
> Relaunched detached in **`tmux` session `tier1`** so it survives session/agent teardown:
> `tmux new -d -s tier1 "PYTHONPATH=src <venv>/python …/tier1_breadth.py --log $PWD/tier1_breadth.log \| tee tier1_stdout.log"`.
> Attach to watch: `tmux attach -t tier1`. Live progress: `tail -f tier1_breadth.log`. The
> orchestrator is **resumable** (skips cells whose output exists), so a restart loses at most the
> one in-flight cell. Progress: cells done = `ls results/tables/exp6c_boundary_*scaleup*_N11.json`.

**Machine reconciliation (this host, 2026-09-22):** the original `/home/alex/venvs/…` paths are
gone — this repo is the working copy now. Infra was re-pointed to run here, GPU0 left untouched:
- ollama on `:11434` is **already fenced to GPU1** (`CUDA_VISIBLE_DEVICES=1`, `CUDA_DEVICE_ORDER=
  PCI_BUS_ID`, `OLLAMA_MAX_LOADED_MODELS=1`); scale-up configs re-pointed `:11445 → :11434`.
- `tier1_breadth.py` `PY` fixed to `/home/alex/trucn/venv/bin/python`; `httpx` installed into that venv.
- 6-model suite completed (`qwen2.5:7b, mistral:7b, phi4-mini, deepseek-r1:7b` pulled; `gemma2:9b,
  llama3.1:8b` already present).
- **GPU0 isolation verified** before/during the run: GPU0 stays at its baseline (RTX 5090, ~22.6 GB,
  `/home/alex/Hoang/code`, pid unchanged); all inference lands on GPU1's `ollama llama-server`.

### 5.0a Work checklist (2026-09-22)

| # | Task | Status |
|---|------|--------|
| **Theory** | Thm 4 (C1) + Thm 5 (C2) + Thm 6 (C5) proofs | ✅ DONE |
| **Analyses A/B** | C2 controls (A, n=398) + same-D mechanism split (B) | ✅ DONE |
| **Offline #1** | Matched-D falsification C1 — caliper ε∈{.02,.05,.10} + nested LPM + NN-matching (`analysis_C_matched_D_falsification.py`) | ✅ DONE (ΔY gap +0.08–0.09 survives ε=.02*, F-test p=.007, NN +0.05*) |
| **Offline #2** | C2 competing-predictor χ_margin vs entropy/disagreement/confidence (`analysis_A_c2_controls.py`) | ✅ DONE — "susceptibility **family**" wording confirmed |
| **F2 figure** | Mechanism–magnitude–consequence plot ΔY=f(D,M) (`analysis_C_figure.py`) | ✅ DONE (`results/figures/exp6c_mechanism_consequence_F2.png`) |
| **Audit** | `paper.md` load-bearing numbers vs `results/tables` | ✅ ALL PASS (C3 ΔD, C2 slope −0.125, A, B, C5 identity/bound) |
| **Tier-1** | 6 models × 6 benchmarks breadth (N=11, R=8) → C1/C2/C3 | ✅ **ENGLISH-REBUILD COMPLETE 2026-10-11 10:25** (72/72 cells; `bbh` dropped from the breadth grid 2026-10-03 on stimulus validity, kept only in phase-1 probe). All four frozen downstream analyses re-run on the complete grid 2026-10-12, in order, no changes made after seeing output: split-sample strata (66/72 usable) → `dA` regime test (gate OPEN, verdict R1) → C3 global directional test (gate OPEN, `p=0.00050`) → decision-churn audit (`dC` aggregate, raw `ΔY` retired as descriptive-only). See the C1-at-breadth and stratification-defect entries above for the numbers. Vietnamese-era artifacts (43/84 partial, language-contaminated) remain archived at `archive/vietnamese-era-2026-10-01/` and are superseded, not reused. |
| **Tier-2** | 3×3 representative depth → C1–C4 + A-controls | 🔴 **DROPPED 2026-10-01** (decision stands). C4 is reported **truthfulqa/hard only**, a declared scope limit rather than a pending run. **The blocker that justified dropping it is now fixed anyway**: `action_enrich.py` and `ci_enrich.py` had no `--benchmark`, so selection and the clue cache silently defaulted to truthfulqa; the flag was threaded through on 2026-10-01 and both now route through the unified probe path. So reviving Tier-2 is no longer blocked on editing frozen estimator code, only on GPU budget. C4's own cells are being re-run in English via `run_c4_english.sh` (~11h, guarded so it refuses to start while `english-rebuild` is active). |
| **Tier-3** | 3×3 copying g-sweep → **H_repair breadth** (spectrum incl. negatives, NOT re-validate identity) | 🟡 **LAUNCHED 2026-10-12 14:58, SPLIT ACROSS BOTH GPUs 2026-10-12 16:12 (explicit user override of the prior GPU0-off-limits rule).** Exactly 18 cells still, unchanged — split by MODEL, not by adding cells: `tier3-copying` (GPU1) runs qwen's remaining cells; `tier3-copying-gpu0` (new) runs llama+gemma (12 cells) against a second `ollama` instance on port 11435, left at its Vulkan default so it lands on GPU0 (the documented quirk in `ollama-vulkan-ignores-cuda-pin`, used deliberately here instead of worked around). Verified via `nvidia-smi` that both GPUs are independently loaded (GPU0 ~13.6GB/94%, GPU1 ~8.6GB/97%) after the split. `tier3_copying.py` gained `--models`/`--gpu` flags (tooling change, not an experimental-scope change) and new configs `scaleup_{llama,gemma}_gpu0.yaml`. Resumable, 6h per-cell timeout each. ETA improves from ~25.7h sequential to roughly the slower queue's length (GPU0's 12 cells, upper bound ~17h) since both run concurrently. **Risk accepted knowingly**: GPU0 belongs to another project and could be reclaimed without warning; if so, `tier3-copying-gpu0` is the only unit affected and GPU1's qwen queue is unaffected. |
| **Tier-3 consumer** | cross-model matched-`D` analysis over the new copying states | ✅ **`tier3_matched_d.py` written and frozen 2026-10-12, BEFORE any Tier-3 cell produced data** (dry-run confirmed clean "pending" state on all 9 (model,benchmark) pairs, shared-side loader validated against real Tier-1 data). Does **NOT** edit `analysis_B_matched_mechanism.py` / `analysis_C_matched_D_falsification.py` (both stay byte-identical, audited); imports their `plur`/`agreement`/`margin`/`caliper_contrast`/`nested_models`/`nn_matching` verbatim. **Shared-side source changed**: NOT analysis_B/C's original `exp6c_raw_{easy,hard}_N11.json` (dated 2026-09-08, pre-English-fix, never regenerated, and qwen-only) — sourced instead from the audited English-rebuild Tier-1 breadth dumps for the 9 overlapping (model,benchmark) pairs. **Per-mechanism noise correction is ASYMMETRIC by design, not an oversight**: `C_within_shared`/`dC_shared` are computable (Tier-1's R=8 independent replicates); `C_within_copy` is reported as `null` — `run.py`'s `R` is sequential belief-update rounds within ONE draw per (question,g), not independent redraws, so no copy-side noise floor is estimable from this grid. Caliper matching reports `n_matched/n_copy_total` coverage per `eps`, flagged below 50%, so a common-support failure reads as "could not test", not a null. |
| **Meta** | `H_repair ~ β0+β1(1−D₀)+u_model+v_task` — moderator hypothesis only (drop if not robust) | ⬜ offline, after Tier-3 |
| **C2 (susceptibility), recovered strata** | `abs(ΔA) ~ M` slope, macro-average across the 66 Tier-1 strata | 🔴 **RETRACTED AND CORRECTED 2026-10-12.** First pass (`tier1_c2_recovered_strata_aggregate.py`) reported hard `β̄ = −0.0627` CI `[−0.112,−0.012]` — looked like confirmation. **Estimand audit found this is a noise-floor artifact**: regressing raw `abs(ΔA)` (not signed `ΔA`) inflates by Jensen's inequality from `ΔA`'s sampling variance, which itself correlates with margin `M` — the exact same pathology `dC` corrected for `ΔY`, discovered here because the user refused to let the raw aggregate stand as "C2 solved". Audited with `tier1_c2_noise_audit.py` (per-question, pooled) and confirmed at the stratum-macro level with `tier1_c2_noise_adjusted_aggregate.py`: splitting each arm's 4 held-out replicates into the 3 balanced 2+2 halves gives a within-arm noise floor `C2_within`; the corrected outcome `dA_excess = abs(ΔA) − C2_within` regressed on `M` gives hard `β̄ = +0.0192` CI `[−0.027,+0.075]` — **sign flips, CI covers 0, no effect survives**. Easy likewise null after correction (`−0.0111`, CI covers 0). **The raw −0.0627 number must not be quoted in the manuscript.** Old file kept as provenance only. `results/tables/tier1_c2_noise_adjusted_aggregate.json` is the current (null) C2 finding; feeds directly into demoting Theorem 2 part 3 to an empirical hypothesis (see manuscript-audit entry below) since even the hypothesis now lacks empirical support. |
| **Manuscript audit (2026-10-12)** | Full pre-submission pass over `paper.md`: comments, benchmark-count consistency, terminology, algorithm line-refs, theory scope, missing §6.2/6.3/6.5, citations, abstract, compile/package hygiene | 🟡 **RECEIVED, NOT YET APPLIED.** ~20 distinct issues flagged (full list in session transcript, not duplicated here to keep this file from exploding) spanning: internal/dated comments to scrub before blind review; `bbh` must read as never-existed everywhere (6 benchmarks, no exception note); `cell` (model×benchmark, Alg.1) vs `stratum` (×regime) terminology unification; Alg.1/Alg.2 off-by-one line references + moving the display-permutation `π` outside the per-condition loop (currently inside, contradicting the Limitations claim that all three arms share one display order); Table 2 caption fragment; Phi-4-mini param count (3.8B, not 7–9B); App E.1's stale "four-model substrate" line; Thm 2 part 3 (regression slope ≤0 does not follow from a decreasing bound — demote to empirical hypothesis, now doubly justified by the C2 retraction above); Gate/Lemma 2 scope (fixed-profile perturbation ≠ cross-draw stochastic comparison); Thm 1 part 3 needs a `p>½` condition; Thm 3's near-tautological framing; §2's "measured causally" → "observationally"; Intro/Discussion/Conclusion overclaiming relative to §6.4's actual `dC`/dA-directional-not-supported numbers; missing citations (Dietrich & Spiekermann 2013, Xia 2012, Magrino et al. 2011, Milani Fard 2016, Yan et al. 2021, Pearl PNS/PN); Abstract not yet written. **Not experimental — no GPU, no Tier-1 rerun involved.** |
| **Freeze** | STOP: no model #7 / benchmark #8 / new controller / new N / prompt-tuning / significance-hunting | ⬜ |
| **Writing** | Intro / Related Work / Theory assumptions / figures per locked positioning | ⬜ |

> **Framing corrections locked (2026-09-22):** (a) C1 matched-D sold on **decision-change (ΔY)**,
> not net accuracy — at matched D `dA≈0`, which the **correct-evidence caveat** explains. (Do *not* write "consistent with Thm 4" here: the theorem predicts a *nonzero* accuracy gap between `M₂` and `M₁`, `Δ_M₂ = 0 > Δ_M₁ < 0`, so a zero gap is not its prediction. Mechanism correction 2026-10-03.) (b)
> `H_repair ∝ (1−D₀)` is a **moderator hypothesis, not** C5; exact C5 = `H_repair = p₊₋ − p₋₊`,
> `|H_repair| ≤ ΔY`. (c) Mistral kept as a **boundary case** (C3 ✓, C1/C2 ✗), not tuned/hidden.

---

### 5.1 Earlier roadmap  **[PIVOT 2026-08-14 — formalize + generalize FIRST; CDCR-rescue PARKED]**

**Priority order (post literature audit).** The audit shows the field is moving fast (CAGE-CAL,
SNLA both 2026) and the clearest *remaining* gap is a **formal + generalized** mechanism theory,
not the controller. So the immediate plan is:

> **P1 — Formalize `M → D →χ→ A`** (Mechanism Non-Equivalence as the spine; §1.1 + §2b claims).
> **P2 — Reproduce on ≥3 model families × ≥2 benchmarks** (qwen / llama / gemma) — the
> generalization evidence the audit flags as the real open territory.
> **P3 — CDCR path is PARKED** at the action-enrichment gate below — *not abandoned* (the
> K2 / K2.5-pre negative chain is a keeper and motivates the algorithm), but not worth weeks
> ahead of P1–P2.

The gate sequence below records the **honest state of the controller path**; it is no longer the
immediate priority. Each gate remains **frozen before it is run** (thresholds + estimand
pre-registered).

**P2 run status — DONE (completed 2026-08-18 08:01, GPU1 only; `:11443` torn down).** Multi-model
generalization is now established — **4/5 configs pass C1/C2/C3** across 3 families + AQuA
cross-dataset; mistral7b is a kept boundary exception. Full PASS matrix + numbers in §4;
aggregate `results/tables/p2_aggregate_summary.json`. This decides D4 and **changes the
priority**: the immediate work is the two items below (theory-predicted N-scaling — **now DONE, see
①** — then close the spine), **not** more generalization runs and **not** the parked CDCR path.

**IMMEDIATE PLAN (post-P2, priority order) — 2026-08-18:**

**① N-scaling — DONE 2026-08-19 01:41 (`results/tables/p2_aggregate_nscaling.json`); Theorem-5
monotonicity NOT supported — reframe or extend, do NOT overclaim.** Ran `N ∈ {5, 21}` on qwen×TruthfulQA
alongside the existing `N=11` (+ all P2 groups re-aggregated). Generalization holds — **6/7 groups
pass C1/C2/C3** (mistral7b|N11 still the `✗✗✓` boundary). **But the pre-registered direction
(`N↑ ⇒ C2 |ΔA|~M slope steepens`) fails as a monotone law:** qwen HARD slope `N5=−0.032 →
N11=−0.097 → N21=−0.055` (peaks at **N=11**, wrong sign on the `5→11` leg); pooled
`−0.091 / −0.125 / −0.104` is likewise non-monotone; the `easy` regime is near-saturated (most
`M`-bins pile at `1.0`, `|ΔA|=0`) so its slope is uninformative. **All N's CIs overlap ⇒ no
monotone N-trend is detectable at this power.** *Action before writing:* either (a) **reframe
Theorem 5's finite-size claim as a non-monotone / interior-N peak** (gating maximised at
intermediate `N`, not `N→∞`) and state it as such, or (b) **add `N=41` + more questions** to tighten
CIs and settle the shape. **Do not claim "slope → 0 monotonically as `N↑`" — the data contradicts
the `5→11` leg.** Report honestly; this is a caveat, not a spine-breaker (C1/C2/C3 still generalize).

**② Mistral diagnostic — concurrent/after; explain, do NOT tune.** **Hard rule: do not tune
mistral to force a pass** (no action-shopping, no per-model knobs). Instead, ask whether its
`C1/C2` fail is explainable by **pre-specified** quantities already in the pipeline:
- *competence regime* — hard `p_single≈0.473` sits at `p≈½`, where the Condorcet sign is noise
  (this is the exact C1-hard sign fail: `condorcet_gap +0.027` but predicted negative);
- *margin distribution / saturation* — is `M` piled at the extremes so the boundary bin is thin?
- *mechanism contrast* — is the shared−indep separation genuinely null, or just underpowered
  (C2 slopes are **right-signed** but CIs cross 0: pooled `−0.052` [−0.129, +0.014])?
- *architecture-specific behavior* — anything mistral-specific in belief/format handling.

If the fail is accounted for by these, **mistral becomes a documented boundary condition of the
theory** — scientifically *more* valuable than a suspiciously perfect 5/5. Leading hypothesis
from the numbers (still to verify): right-signed-but-underpowered + a hard set landing at `p≈½`,
i.e. a competence/power boundary, not a contradiction.

**③ THEN close the experimental spine.** N-scaling has now run (①); the empirical spine
(`M→D→χ→A`, multi-model, N-scaling, mechanism split) is **effectively closed** — move to **figures +
statistics hardening + paper writing** and deliberately **stop generating new experiments**, with
two bounded loose ends to resolve *in the writing*, not via more sweeps: (i) the Theorem-5 finite-size
claim needs reframing (interior-`N` peak) or one `N=41` point (①), and (ii) the action-enrichment
write-up must lead with the **C4 interaction** (mechanism-dependent intervention response) and demote
`a_common`'s marginal win to secondary — its cross-model replication landed **weak** (pooled `+0.016`,
0/4 sig, §4), so it is not an algorithm claim. CDCR stays **parked** (§K3, now a *two-tier* open
design whose blocker is a missing copying-targeted intervention); the K2 / K2.5-pre negative chain +
the C4 interaction are retained as motivation, not reopened.

```
Exp6C ✓ → K1 (headroom?) ✓ → K2 (observable?) ✗ → K2.5-pre (even leaky-observable?) ✗
        → ACTION-ENRICHMENT (does an effective intervention exist at all?) → [K3 CDCR only if it passes]
```

**Progression established so far (a clean negative chain):**
> passive features fail (K2) → even *leaky* mechanism features fail (K2.5-pre) →
> **the current CDCR bottleneck is ACTIONABILITY, not feature quality.**
> Diagnosis is useless without an intervention that actually recovers accuracy.

### K1 — is there actionable headroom?  **[DONE 2026-08-12 — CONDITIONAL GO]**
`k1_oracle_headroom.py` + `k1_analyze.py`, action set
`A₀ = {single, indep_vote, debate, diversity, shared_ev, stop}`, estimate `y_{q,a}=P(a
correct)` over `R` replicates. **Decision number = cross-fit (leave-one-replicate-out) `H`,
not the plug-in** (the no-GPU synthetic test caught a plug-in `H=+0.09` that was a *mirage*,
honest `cfH=−0.05`). **Result: cross-fit `H = +0.037` pooled → conditional GO** (leans go on
cost-Pareto: the oracle mix routes ~half the calls). Caveats that keep it *conditional*, not
green: `shared_ev` uses correct-leaning evidence ⇒ `H` is an **upper bound** on a real,
quality-uncertain retriever; and the `H_M=+0.125` "mechanism headroom" was **audited down** —
by the exact identity `H_M = H + Δ_fix − R_base`, ~63% of it is fixed-policy (own `shared_ev`),
**not** deployable routing headroom. Honest headline = `cfH=+0.037`.

### K2 — is the headroom *observable/exploitable*?  **[DONE — NULL]**
`k2_router.py`: fit `V̂_a(x)=P(Y_a|x)` on **leakage-free** observable `none`-state features
(permutation-invariant, never touch the truth index), route `a*=argmax V̂_a`, score under
leave-one-question-out cross-fit. **No learned router beats static `shared_ev`:**

```
regime  A_bestfixed  A_oracle_cf  headroom   best router
easy       0.891       0.915       +0.024    knn +0.037  (CI [-0.0003,+0.09], MARGINAL; near-ceiling)
hard       0.362       0.366       +0.003    all ≤ 0     (gboost/knn sig-negative)
pooled     0.603       0.640       +0.037    ridge +0.000 / gboost -0.022 / knn -0.033  (none > 0)
```

The `+0.037` headroom **exists in the oracle but is not realizable** by a policy on observable
features. On **hard the cross-fit oracle headroom itself collapses to `+0.003`** — a ceiling no
feature can beat, because per-question **no action in `A₀` beats `shared_ev`**. Cost-Pareto is
the one consolation: matched accuracy at ~½ the compute (pooled 0.603 @ cost 7.5 vs 14.6).

### K2.5-pre — even a deliberately-LEAKY mechanism feature set?  **[DONE 2026-08-14 — NULL, 0 GPU]**
`k2p5_pre.py`: a necessary-condition screen with features `z = [margin, χ̂, d_shared, d_indep]`
where `d_• = TV(P_collective[shared|indep evidence injection], P_collective[none])` — the
belief **elasticity** a probe would measure, but built from the *same* treatment conditions that
produce the labels ⇒ **leaky ⇒ an optimistic ceiling.** Same estimator/oracle/cross-fit as K2
(`denom_cf` reproduces K2 byte-identically). **Still fails:** pooled best-leaky router `+20%`
captured (non-sig, CI straddles 0), ridge `−13%`, **mechanism-only `d_shared,d_indep` `−8%`**;
hard headroom `≈0`. Even the leaky ceiling cannot capture the headroom on `A₀`.
→ **Stop feature engineering / do not spend GPU on fresh probes.** The wall is the action set.

### Susceptibility-gating (boundary) — supporting, and it rules out one alternative  **[DONE 2026-08-14]**
`exp6c_boundary.py`: primary D-free metric `|ΔA| ~ margin M` has slope **negative & significant
every regime** (easy `−0.212` [−0.425,−0.072], hard `−0.097` [−0.188,−0.011], **pooled `−0.125`
[−0.198,−0.057]**): copying flips answers **only near the decision boundary**. So susceptibility
`χ̂` **is** observable — the K2 null is *not* "susceptibility is hidden"; it is "no action in
`A₀` exploits it." Exactly the actionability diagnosis.

### NEXT GATE — Action-enrichment falsification  **[DONE 2026-08-20 — router gate WEAK/CONDITIONAL (qwen H^CV_new=+0.035, KILL/parked); `a_common` marginal win did NOT replicate strongly (pooled +0.016, 0/4 sig); the surviving result is the C4 `mechanism × intervention` interaction, §4]**
**Question (falsification, not "find a winner"):** *when dependence causes harm, does an
intervention **derived from the causal mechanism** recover collective accuracy?* Three actions,
**frozen before the run** (no researcher DoF — do **not** try 15 and keep the best 3), all
**truth-independent** (no correct answer revealed, no oracle evidence):

| action | mechanism targeted | intervention |
|---|---|---|
| `a_copy` | downstream social copying | reset peer context → agents re-reason **independently** → re-vote |
| `a_common` | upstream common evidence | resample **independent** evidence per agent → re-reason → re-vote |
| `a_dissent` | consensus lock-in | each agent independently constructs the strongest **counter-hypothesis** before re-vote |

**Protocol.** Run **HARD first** (`H^CV_hard=0.003` = the sharpest falsification environment).
Freeze `A₀` (6 actions), add the 3 → `A₁`, recompute the **same** cross-fit estimator. Report
**three** quantities (H alone hides the failure mode where actions lift everything uniformly —
best-fixed rises, routing headroom does not):
```
Δ_fixed  = A_bestfixed(A₁) − A_bestfixed(A₀)      # uniform lift? (good intervention, no CDCR)
Δ_oracle = A_oracle(A₁)   − A_oracle(A₀)
H^CV_new = A_oracle^cf(A₁) − A_bestfixed(A₁)      # routing headroom — the CDCR-relevant one
```
**Locked gate (set before the run):**
```
H^CV_hard,new < 0.02      → KILL CDCR   (phenomenon real, but no intervention leverage)
0.02 ≤ · < 0.05           → weak/conditional (continue only if complementarity is interpretable)
· ≥ 0.05                  → actionability restored → THEN fresh mechanism probes are worthwhile
· ≈ 0.08–0.10+            → strong
```
**Pre-registered mechanism-matching (weighs more than raw accuracy).** Prediction written first:
`copying-heavy q ⇒ a_copy gains most`; `common-evidence-heavy q ⇒ a_common gains most`. If all
three improve but with **no relation to the targeted mechanism** → "better prompting", **not**
mechanism-aware control. A `targeted-action × mechanism` interaction is the publishable story:
*identify causal failure mode → apply matched intervention → recover collective accuracy.*

**Frozen protocol & checklist (action-enrichment):**
- [x] Diagnosis established: K2 null + K2.5-pre null (leaky ceiling fails) ⇒ bottleneck = actionability
- [x] Susceptibility `χ̂` shown observable (boundary `|ΔA|~M` pooled `−0.125*`) ⇒ K2 null is not an observability failure
- [x] **Froze 3 actions BEFORE run (2026-08-18)** — `a_copy` / `a_common` / `a_dissent`, truth-independent. `a_copy` design **user-confirmed**: run `debate_rounds` coupled rounds to *induce* copying → strip peer context, keep each agent's own reasoning, re-elicit "decide independently" → plurality (distinct from `indep_vote`/`debate`). `a_common` = per-agent independently-resampled `indep_correct_strong` clue. `a_dissent` = counterfactual-steelman persona (reuses the diversity persona mechanism). Frozen in `action_enrich.py` header.
- [x] Implemented as new rollout actions on the **same** `Y_qa` estimator + LOO cross-fit as `A₀` — `action_enrich.py` (runner, reuses K1 machinery; `debate` rollout shared with `a_copy`) + `action_enrich_analyze.py` (`crossfit_perq` subset-generalizes `k1_analyze.crossfit_oracle`). **Validated no-GPU** (synthetic 20-q: crossfit subset / 3-quantity report / gate / interaction all correct; H^CV(A₀) went *negative* on noise ⇒ no winner's-curse inflation) **and live-smoke** (N=5/R=1/1q on `:11443`: all 8 actions execute, incl. `a_copy` debate→re-elicit + `a_dissent`).
- [x] **Ran HARD first** (`H^CV_hard=0.003` = sharpest falsifier), N=11, **R=8**, matched to exp6c hard, **2-shard across both GPUs** — **DONE 2026-08-19 05:01** (`action_enrich_run.sh` self-sequenced after N-scaling COMPLETE). Result (`results/tables/actionenrich_gate.json`): `H^CV_hard,new=+0.035` CI[−0.018,+0.094] (**WEAK/CONDITIONAL** band, not restored); **secondary win** — `a_common`=0.400 is the A₁ best-fixed policy, beats best A₀ (`shared_ev`=0.362) by `Δ_fixed=+0.038` CI[0.000,0.088]; `a_copy`=0.200 is **dead** (=`single`). Mechanism-matching pre-reg direction **fails** (copying-heavy only n=2), but the common-heavy interaction `Δ(common−copy)=+0.178` CI[+0.082,+0.289] is significant. Logs: `action_enrich_run.log`, `action_enrich_s{0,1}.log`.
- [x] Reports **all three** — `Δ_fixed`, `Δ_oracle`, `H^CV_new` (not `H` alone) so uniform-lift ≠ routing-headroom is visible (secondary win: a NEW op as A₁ best-fixed with `Δ_fixed>0` = standalone algorithm, no router)
- [x] Applies **locked gate** (`<0.02` kill / `0.02–0.05` conditional / `≥0.05` restored / `0.08–0.10+` strong) — thresholds set above, not after seeing numbers
- [x] **Pre-registered** mechanism-matching: per-q label from A₀ means (copying-heavy=`|debate−indep_vote|`, common-heavy=`|shared_ev−indep_vote|`); `gain_op = y_op − max_{A₀}`; test `copying-heavy⇒a_copy`, `common-heavy⇒a_common` interaction (implemented, weighed above marginal accuracy)
- [x] **Retain the K2/K2.5-pre negative chain** regardless of outcome (motivation for the algorithm; the progression is a result even if action-enrichment also fails)
- [x] **Decision at that time — 2026-08-19, qwen single-model (POINT-IN-TIME, not the final verdict — see the cross-model item below):** router gate landed WEAK (`+0.035`, not `≥0.05`) ⇒ K3 CDCR stays **parked**; the `Δ_fixed>0` standalone win held **on qwen only** ⇒ path taken = **validate `a_common` cross-model** (the pre-registered branch), CDCR-router not resurrected. *(FINAL VERDICT after replication is the next item + §4: the marginal win did **not** hold cross-model; the surviving result is the C4 interaction.)*
- [x] **Cross-model validation — DONE 2026-08-20 15:48** (`action_enrich_xmodel_run.sh`; gemma2:9b / llama3.1:8b / mistral:7b, each 2-sharded s0→:11443 / s1→:11444, **HARD, N=11, R=8, dr=3, same 40 questions, reusing the fixed qwen clue pool** — P2 convention, clues held constant, only the model swaps; added `debate_{gemma,llama,mistral}_gpu0.yaml`). **Outcome — the `a_common` marginal-accuracy win did NOT replicate strongly:** pooled paired lift `+0.016` [−0.003,+0.037], **0/4 models significant**, top-ranked in 3/4 but mistral negative (rank 3). ⇒ **`a_common` is demoted to a secondary result** — we no longer sell it as a standalone algorithm contribution. **What survived is the C4 interaction** (§4): the `mechanism × intervention` ordering is significant and cross-model (common-heavy ⇒ common-targeted wins; copying-heavy ⇒ ordering flips, a_copy actively harmful). Framing locked as **"the effect of a decorrelation intervention depends on the mechanism generating dependence"** (reinforces C1), **not** "a_common improves accuracy." Gates `results/tables/actionenrich_gate_{gemma9b,llama8b,mistral7b}.json`; rollup `actionenrich_xmodel_summary.json`; log `action_enrich_xmodel_run.log`.
  - **Wording guard (locked):** do **not** write "mechanism-matched actions replicate 3/4 models." Mistral's copying-heavy `Δ(copy−common)=−0.446` shows a_copy is *worse* than a_common there — it proves the *response interacts with mechanism*, **not** that the right copying-targeted action exists. The matched *policy* is incomplete.

### K3 — CDCR / adaptive controller  **[OPEN branch, NOT dead — but redesigned by the C4 result; do NOT train a router now]**
The action-enrichment result does **not** kill the adaptive-algorithm idea; it **reshapes** it and
exposes the real blocker. The passive single-tier design (`M<τ ⇒ a_common` for every question) is
retired — the C4 interaction says one action is not right everywhere. A rescuable controller must be
**two-tier**: *WHEN* (susceptibility `χ`, C2) × *WHAT* (dependence mechanism, C1):
```
π(x) = STOP                if χ(x) low                     # not susceptible → don't intervene
       a_common            if χ(x) high & common-evidence-heavy
       a_copy / other       if χ(x) high & copying-heavy    # ← NO effective action known yet
```
**New bottleneck (the honest blocker):** the copying-heavy cell has **no good action** — `a_copy`
is dead (=`single`; mistral shows it is actively harmful). So there are not yet two working
"treatments" to route between, and **training a router now is premature.** *Prerequisite before any
K3 build:* find (and **pre-register before testing**) an intervention that genuinely recovers
accuracy on copying-heavy states. Only with two effective, mechanism-matched actions in hand does a
two-tier adaptive controller become buildable and testable. Until then K3 stays **parked**; the
K2 / K2.5-pre negative chain + the C4 interaction are retained as its motivation.

**Copying-targeted intervention — Counterfactual Isolation (`a_CI`): PRE-REGISTERED 2026-08-21, pre-GPU**
(`CI_PROTOCOL_FROZEN.md`). One final, frozen shot at the copying cell: after coupled debate, isolate
each agent, force the strongest case **against the group consensus `X_c`**, commit, re-vote (= the
post-contamination context of `a_copy` + the counterfactual move of `a_dissent`, aimed at `X_c`).
Frozen (Design A′ signed off): operator, prompt, and — because the natural copying-heavy label has no
power (`n_copying∈{0,1,2,7}`/40) — a **controlled copying-induction** design that makes this a
**damage–repair experiment**: induce copying on every question (`coupling=1.0`), measure the
copying-harm contrast `C_q = A_indep − A_copy` (`C_q>0` ⇔ copying hurt), then test repair on
`S_copy={C_q>0 ∧ wrong-consensus≥½}`. Primary = recovery `Δ_CI = A_CI − A_copy` + **Recovery Ratio**
`RR=(A_CI−A_copy)/(A_indep−A_copy)` (fraction of copying-destroyed accuracy restored; `RR≥100%` =
beats even not-copying = deployment-strength); secondary (weighs more) = mechanism-match slope of
`(gain_CI−gain_common)` on `C_q`. **Locked gate** on `Δ_CI` (`<0.02` KILL branch / `0.02–0.05` weak /
`≥0.05` & ≥2 models GO / `≥0.08` & positive `C_q` slope → build two-tier `π(x)`); blunt-flip
guardrails (no uniform win, no common-heavy harm). On FAIL → algorithm branch **stops**, ship C1–C4,
no operator #5.

**Status 2026-08-21 — IMPLEMENTED + no-GPU VALIDATED; pilot BLOCKED on GPU infra.** Built
`ci_enrich.py` (runner = `action_enrich.py` + the `a_CI` operator; `X_c` = debate plurality with a
soft-argmax tie fallback; own-reasoning snapshot shared by `a_copy`/`a_CI` so their re-elicits don't
cross-contaminate) + `ci_enrich_analyze.py` (`Δ_CI`/RR/`C_q`-slope/DiD/guardrails/gate, with
`--selftest`) + turnkey `ci_enrich_run.sh` (`smoke`/`qwen`/`xmodel`; health-checks each server with a
*real* generate and **aborts if the llama-server backend is dead** — guards the "backend down → cached
zeros" failure). No-GPU checks pass: analyzer selftest **10/10** known-answer; operator import +
`X_c` canonical→display mapping + frozen-prompt checks green. **⛔ Cannot run the pilot yet:** machine
inference is down — NVML driver/library mismatch (`nvidia-smi` fails) and the ollama backend crashes
on generate (`cannot get current path`); dedicated ports `:11443/:11444` down. **When GPU is
restored:** `./ci_enrich_run.sh smoke` → `./ci_enrich_run.sh qwen` (→ `results/tables/ci_gate_qwen.json`)
→ if not KILL, `./ci_enrich_run.sh xmodel` (llama + gemma) for the ≥2-model rule.

**RESOLUTION 2026-09-01 — algorithm branch CLOSED as a bound (C5 / Theorem 6).**
1. **`a_CI` pilot ran (qwen HARD, `:11443`, 1-GPU) → GATE = KILL** (`ci_enrich_qwen_s{0,1}`,
   `results/tables/ci_gate.json`): `Δ_CI = +0.000`, Recovery 0.0%; copying cell stays empty. (Only
   the qwen pilot ran; per the frozen gate, KILL ⇒ do not spend the other 3 models.)
2. **SGIR proposed** (Susceptibility-Gated Independent Resampling: `M>τ` accept, `M≤τ` discard
   contaminated trajectories and resample a *fresh independent* ensemble) as a principled successor.
   **Offline oracle (GPU-free) → KILL on both axes:** shared-evidence axis low-M recovery gain
   `+0.014` [−0.013,+0.042] (`sgir_boundary_oracle.py`, 5 configs, n=398); copying axis
   `H_repair = +0.000` (g=1) / `+0.033` (g=2), not margin-concentrated (`sgir_copying_oracle.py`).
3. **Formalized as Theorem 6** (`theory/05_theorems/thm6_repair_headroom.md`): any
   independence-restoring, no-new-info (IRIC) intervention has ceiling `A_0` and recoverable headroom
   exactly `p_+- − p_-+ ≤ ΔY`; LLM copying *amplifies* the independent plurality (preserved 91% at
   g=1) so `H_repair ≈ 0`. Identity + bound verified at 5 coupling strengths (`repair_headroom.py`) and
   generalized **cross-benchmark** to AQuA (`repair_headroom_xbench.py`): theorem holds, but `H_repair`
   *magnitude* tracks independent diversity `1−D_0` — **≈0 on TruthfulQA** (`D_0=0.83`) vs **+0.10 on
   AQuA** (`D_0=0.53`, n=20, fragile), always `≤ ΔY`. ⇒ CDCR/K2/`a_copy`/`a_CI`/SGIR are a negative
   chain **consistent with** the bound (each operator's IRIC status argued in Thm6 §1, not assumed); the
   copying cell is empty because the IRIC ceiling is `A_0`, not for lack of engineering. Do **not** write
   "explains all five" or "decorrelation never helps." **Open = NON-IRIC only** (retrieval/new evidence
   able to move `Y_0`); cross-*model* copying-axis needs GPU (only qwen g-sweep states exist) —
   deliberately not run, theorem + qwen/AQuA illustration stand.

### Multi-model / N-scaling  **[multi-model DONE 2026-08-18 · N-scaling DONE 2026-08-19]**
- **Multi-model — DONE.** Universality across qwen / llama / gemma (+ AQuA cross-dataset):
  4/5 configs pass C1/C2/C3, mistral kept as a boundary case (§4 matrix). No longer gated behind
  the controller path — this is now decoupled from CDCR and finished.
- **N-scaling — DONE (`results/tables/p2_aggregate_nscaling.json`).** Ran the finite-size test of
  **Theorem 5** on qwen×TruthfulQA `N ∈ {5, 11, 21}` (+ all P2 groups re-aggregated): **6/7 groups
  pass C1/C2/C3** (mistral7b|N11 still boundary). **But the Theorem-5 monotonicity prediction
  (`C2 slope steepens as N↑`) is NOT supported** — qwen HARD slope is non-monotone and **peaks at
  `N=11`** (`N5=−0.032 → N11=−0.097 → N21=−0.055`; pooled `−0.091/−0.125/−0.104`), all CIs
  overlapping, `easy` near-saturated. ⇒ **Reframe Theorem 5 as a non-monotone / interior-`N` peak**
  (or add `N=41`), and do **not** claim monotone `slope→0`. Detailed numbers + action in priority ①
  above. Spine still closes (generalization intact); this is a writing-time caveat, not a blocker.

**Venue calibration (honest):** current package → mid-upper **JCR Q1** (Neural
Networks / TNNLS / AI / JMLR). **NMI/PNAS ceiling** only if the hero evidence lands:
finite-size scaling + multi-model quantitative + a working controller with *real*
headroom. Theorem 4 strengthens the *idea* half, not the *evidence* half.

---

## 6. Repository layout

- `theory/` — derivations; spine = `06_mechanism_chain_MDCA.md` (the `M→D→χ→ΔY→H_repair` overview) +
  `05_theorems/thm4_mechanism_non_equivalence.md` (C1, `M→A` limits) +
  `05_theorems/thm5_susceptibility_gating.md` (C2, the `χ`-gate) +
  `05_theorems/thm6_repair_headroom.md` (C5, the IRIC repair bound).
- `src/mfg_llm/` — package: opinion state, mean field, A/B/s/σ estimation, FPK dynamics, control (HJB/LQR/RL), metrics, LLM-debate orchestration (`llm_debate/` — agent, orchestrator with coupling knob + `evidence` injection, benchmarks).
- `experiments/`
  - `exp2..exp4/` synthetic core **[DONE]**, `exp5_benchmarks/` **[DONE]**
  - `exp6_phase_transition_llm/` — `run.py` (g-sweep), `theory_overlay.py`, `flip_plot.py`, `exp6b_common_shock.py`, `exp6_ablation_plot.py`, `exp6c_shared_context.py`, `exp6c_plot.py`, `exp6c_boundary.py` (susceptibility-gating; the P2 runner, `--model-tag`), **`k1_oracle_headroom.py`** (K1 runner), **`k1_analyze.py`** (K1 gate), `k1_hm_audit.py` (H_M identity audit + honest cross-fit oracle), **`k2_router.py`** (K2 observability), **`k2p5_pre.py`** (K2.5-pre leaky screen), **`p2_aggregate.py`** (P2 cross-model C1/C2/C3 aggregator), `p2_benchmarks.py` (P2b adapter registry — now 7 benchmarks), `p2_probe.py` (P2b easy/hard stratifier), `P2_PROTOCOL_FROZEN.md`
    - **C5 / repair headroom (Thm 6):** `repair_headroom.py` (identity+bound across g), `repair_headroom_xbench.py` (cross-benchmark), `sgir_copying_oracle.py` + `sgir_boundary_oracle.py` (SGIR gates), `ci_enrich.py`/`ci_enrich_analyze.py` (a_CI, KILL), `CI_PROTOCOL_FROZEN.md`
    - **TACL hardening + scale-up:** `analysis_A_c2_controls.py` (C2 uncertainty controls), `analysis_B_matched_mechanism.py` (same-D mechanism split), `fetch_benchmarks.py` (HF→truth@0 JSONL), `tier1_breadth.py` (6×7 breadth orchestrator, resumable)
- `data/` — raw benchmarks (truthfulqa, aqua, **arc, commonsenseqa, mmlu, strategyqa, bbh**), processed `.npz`/`.json` states (incl. `exp6c_raw_*` belief states, g-sweep, `exp6_select_<b>.json` probes), estimated generators.
- `results/` — `figures/` (`exp6_ablation_copying_vs_shock_N11.png`, `exp6b_common_shock_N11.png`, `exp6_flip_lift_vs_rho_N11.png`), `tables/`, `logs/`.
- `tests/` — 38 passing (FPK ODEs, estimation, control, parsers).

## 7. Install & run

```bash
pip install -e .                       # env: /home/alex/venvs/env
pytest                                 # 38 tests

# synthetic core
python experiments/exp2_propagation_of_chaos/run.py
python experiments/exp3_phase_transition/run.py
python experiments/exp4_optimal_control/run.py

# exp6 — real-LLM mechanism experiments (needs a local Ollama serving qwen2.5:7b)
python experiments/exp6_phase_transition_llm/exp6b_common_shock.py --sanity   # Exp6A+B, offline
python experiments/exp6_phase_transition_llm/exp6_ablation_plot.py            # the headline figure

# Exp6C native shared-evidence — frozen protocol (replicate-level ΔD, R=8). Run both
# regimes in parallel on the two GPU servers (easy→:11435, hard→:11436):
python experiments/exp6_phase_transition_llm/exp6c_shared_context.py \
    --regime easy --config src/config/debate_gpu1.yaml \
    --n-questions 40 --replicates 8 --n-agents 11 --tag full
python experiments/exp6_phase_transition_llm/exp6c_shared_context.py \
    --regime hard --config src/config/debate_gpu0.yaml \
    --n-questions 40 --replicates 8 --n-agents 11 --tag full
# mini metric-validation: add  --n-questions 10 --conditions none,shared_correct_strong,\
#   indep_correct_strong,shared_misleading_strong,indep_misleading_strong  --tag minival

# K1 oracle-headroom gate (go/no-go before building CDCR). Run both regimes on the two
# GPU servers, then pool + gate (analyzer is CPU-only; the decision number is cross-fit H):
python experiments/exp6_phase_transition_llm/k1_oracle_headroom.py \
    --regime easy --config src/config/debate_gpu1.yaml \
    --n-questions 40 --n-agents 11 --replicates 6 --debate-rounds 3 --tag v1
python experiments/exp6_phase_transition_llm/k1_oracle_headroom.py \
    --regime hard --config src/config/debate_gpu0.yaml \
    --n-questions 40 --n-agents 11 --replicates 6 --debate-rounds 3 --tag v1
python experiments/exp6_phase_transition_llm/k1_analyze.py --tag v1   # → H, H_M, matrix, gate verdict
# smoke-test first: add  --n-questions 4 --replicates 2 --debate-rounds 2 --tag mini  (then analyze --tag mini)
```

### P2 — generalization (protocol FROZEN; infra built + verified)
Reproduce `M→D` (C3 ΔD), `D─χ→A` (C2 boundary slope) and the C1 mechanism split across model
families × benchmarks × N. **Full frozen protocol + run matrix:**
`experiments/exp6_phase_transition_llm/P2_PROTOCOL_FROZEN.md`.

- **`--model-tag`** (namespace-only) added to `exp6c_boundary.py` + `exp6c_shared_context.py`;
  empty ⇒ byte-identical legacy paths (verified). Fixes the belief-cache collision that would
  otherwise corrupt multi-model runs.
- **`p2_aggregate.py`** (CPU-only) — reports C1/C2/C3 **separately** per model (never collapsed),
  saves C2 `{slope, CI, margin bins, N}` for the Theorem-5 N-scaling test, and keeps failures.
  **Verified on qwen:** reproduces slope −0.125, ΔD +0.272, C1 sign-correct.
- `exp6c_boundary.py` alone is the P2 runner (superset: `none/shared/indep` → C1+C2+C3 in one pass).

```bash
# P2a (launch-ready NOW): model axis on TruthfulQA. Per model, both regimes:
python exp6c_boundary.py --regime {easy,hard} --config src/config/<cfg>.yaml \
    --n-questions 40 --replicates 8 --n-agents 11 --model-tag <stem>     # e.g. llama8b / gemma9b / mistral7b
python p2_aggregate.py --n 11     # → per-model C1/C2/C3 + N-scaling view + PASS matrix
```
**P2b** = benchmark axis × qwen (**built + verified, adapter-only**): `p2_benchmarks.py` registry
+ `--benchmark` on the runners (truthfulqa ⇒ legacy) + `p2_probe.py` stratifier. **P2c** = the
`N∈{5,21}` cell for the Theorem-5 slope test. Full plan:
`experiments/exp6_phase_transition_llm/P2_PROTOCOL_FROZEN.md`.

### C5 / repair headroom + TACL analyses (offline, GPU-free)
```bash
cd experiments/exp6_phase_transition_llm
python repair_headroom.py            # Thm 6 identity + bound across coupling g  → repair_headroom.json
python repair_headroom_xbench.py     # cross-benchmark (TruthfulQA vs AQuA), magnitude ∝ 1−D₀
python sgir_copying_oracle.py        # SGIR gate, copying axis (KILL)
python analysis_A_c2_controls.py     # C2 vs uncertainty/confidence baselines
python analysis_B_matched_mechanism.py   # same-D, different-mechanism (C1 main figure)
```

### Scale-up campaign (6 models × 7 benchmarks; §5.0) — GPU1 only
```bash
# 0) fetch + normalize the 5 new benchmarks (HF datasets-server → truth@0 JSONL)
python experiments/exp6_phase_transition_llm/fetch_benchmarks.py
# 1) dedicated GPU1 server (GPU0 is another project — do not touch)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11445 \
  OLLAMA_NUM_PARALLEL=8 ollama serve &
# 2) probes (qwen anchor) stratify easy/hard for the new benchmarks
for b in arc commonsenseqa mmlu strategyqa bbh; do \
  python p2_probe.py --benchmark $b --config src/config/probe_qwen_gpu1.yaml; done
# 3) Tier-1 breadth (resumable, continue-on-error); then Tier 2/3 + meta
python experiments/exp6_phase_transition_llm/tier1_breadth.py --dry-run   # inspect plan
python experiments/exp6_phase_transition_llm/tier1_breadth.py             # execute
```

**Paper draft:** `paper.md` — TACL skeleton (abstract + intro drafted; every section mapped to its
frozen artifact/figure; spine `M→D→χ→ΔY→H_repair`, contributions C1–C5).

## 8. Core equations

```
mean drift        dm/dt = −(A−B) m + s(q)
covariance        dΣ/dt = −AΣ − ΣAᵀ + 2σI
fixed-point stab. Re(eig(A−B)) > 0
correlated jury   P_maj(N,p,ρ) = E_Z[ Pr(Bin(N, q(Z)) > N/2) ],  q(Z)=Φ((Φ⁻¹(p)−√ρ Z)/√(1−ρ))
FLIP law          lift(ρ) = P_maj(N,p,ρ) − P_maj(N,p,0),   sign = sign(½ − p)
mechanism law     A_coll = F(p, D, M, N)  — does NOT factor through (p, D, N)   [Theorem 4]
```
