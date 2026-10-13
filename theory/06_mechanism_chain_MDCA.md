# The Mechanism Chain `M → D →χ→ A`

*Spine of the repositioned project (post literature audit, 2026-08-14). Ties the three
surviving contributions into one causal chain and states, for each edge, what is proven,
what is measured, and what is open.*

---

## 0. One picture

```
        ┌── C3: matched instrument ──┐   ┌── C2: Theorem 5 ──┐
        ▼                            ▼   ▼                   ▼
   M ───────────────────────────►  D ───────[ χ ]─────────►  A
  mechanism                     dependence   susceptibility  collective
  (WHY the                      (measured     (WHEN a change  accuracy
   dependence)                   ΔD)           in D moves the
        │                                      decision)          ▲
        └────────── C1: Theorem 4 (M sets the ρ→1 limit of A) ────┘
```

- **`M`** — the causal **mechanism** generating dependence: upstream *common/shared
  evidence* (`M₁`) vs downstream *social copying* (`M₂` synchronous, `M₃` cascade).
- **`D`** — the **measured error dependence** (replicate-level `ΔD`; identifiable, not the
  `−1/(N−1)` single-realization artifact).
- **`χ`** — the collective **decision susceptibility**: proximity of the aggregation to its
  boundary (low vote-margin `M`).
- **`A`** — the collective **accuracy** (majority/plurality correctness).

The framework answers three questions **no single prior result answers together**.

---

## 1. Three questions, three results

### Q1 — *Why?* Edge `M → D`, and `M → A` limits  **(C1 + C3)**
Different causal origins produce dependence, and the origin — not the strength — sets the
collective limit.
- **Theorem 4** (`05_theorems/thm4_mechanism_non_equivalence.md`): at `ρ→1`, the three
  mechanisms occupy three *distinct, sign-definite* accuracy limits `{p, P_maj(N,p,0),
  ≤P_maj(N,p,0)}`; for a competent crowd (`p>½`) downstream copying (`Δ=0`) is **less**
  harmful than upstream common evidence (`Δ<0`). ⇒ `A ≠ F(p, D, N)`; the label `M` is
  required (Corollary 4.1).
- **C3 instrument** (`experiments/.../exp6c_shared_context.py`): a **matched
  shared-vs-independent evidence** manipulation changes the dependence structure while
  holding evidence quality fixed, giving the identifiable, paired estimand

      ΔD = mean_q [ D_shared(q) − D_indep(q) ]  > 0        (Exp6C: 4/4 sig, ΔD 0.26–0.50).

### Q2 — *When?* Edge `D ─χ→ A`  **(C2)**
A change in dependence only moves the collective decision near the boundary.
- **Theorem 5** (`05_theorems/thm5_susceptibility_gating.md`): plurality aggregation is a
  threshold; a bounded-reach dependence perturbation flips the outcome only when the vote
  margin `M ≲ 2s`. Hence `|ΔA|` is boundary-concentrated with non-positive slope on `M`
  (pooled `−0.125` [−0.198,−0.057]), and (Cor 5.2) **`ΔD ↑↑` does not force `|ΔA| ↑↑`** —
  the two decouple through `χ`.

### Q3 — *Why isn't `ρ` enough?*  **(C1 ⊕ C2)**
Two independent reasons the correlation scalar underdetermines the outcome:
- **Corollary 4.1** — the *dynamical* endpoint depends on the mechanism `M` (same `ρ→1`,
  different `A`).
- **Corollary 5.2** — the *observed accuracy footprint* of a `D`-change depends on the
  susceptibility `χ` (same `ΔD`, different `|ΔA|` depending on boundary mass).

So a faithful predictor is `A = F(p, D, M, χ, N)` — needing the causal structure `M` **and**
the susceptibility `χ` on top of `(p, D, N)`.

### Q4 — *Can it be fixed?* Edge `ΔY → H_repair`  **(C5)**
A changed decision is not the same as a recoverable one.
- **Theorem 6** (`05_theorems/thm6_repair_headroom.md`): for any **independence-restoring,
  information-conserving** (IRIC) intervention — discard the correlated votes, re-aggregate a
  fresh *independent* ensemble, no new evidence — the recoverable accuracy is exactly the net
  directional flip,

      H_repair = A_0 − A_g = p_+-(break) − p_-+(rescue),   |H_repair| ≤ P(Y_g≠Y_0) = ΔY,

  with `A_R = A_0` the class ceiling (Prop 6.1, exact/assumption-free; Thm 6.2, the bound).
  Empirically (g-sweep, qwen): on TruthfulQA copying preserves the independent plurality on
  **91%** of questions, flips near-symmetric, so `H_repair ≈ 0` while `ΔY = 9–15%` and `D→1`.
  The bound is universal; the recoverable *magnitude* scales with independent diversity
  `1−D_0` (AQuA `D_0=0.53` → `H_repair` up to `+0.10`), always `≤ ΔY`.
- This extends the chain **below the decision**: `D → ΔY → H_repair`, each strictly weaker
  than the last. Diagnosis (`M̂,D̂`) — even a diagnosed decision change (`ΔY`) — does **not**
  imply actionability. *Decorrelation cannot manufacture information the collective never had.*

---

## 2. What each edge buys, and its status

| Edge | Claim | Evidence | Status |
|---|---|---|---|
| `M → A` (limits) | Theorem 4: mechanism sets `ρ→1` limit | Exp6/6B ablation (copying lift ≈0 vs shock −0.07) | **proven + observed** |
| `M → D` (instrument) | matched shared/indep ⇒ identifiable `ΔD` | Exp6C 4/4 sig | **DONE (qwen)** |
| `D ─χ→ A` (gate) | Theorem 5: boundary-concentrated `|ΔA|` | `exp6c_boundary` pooled slope −0.125* | **proven + observed (qwen)** |
| `ΔY → H_repair` (fix) | Theorem 6: IRIC headroom = net flip `p_+- − p_-+`, `≤ ΔY` | `repair_headroom.py` (identity ✓, bound ✓, 5 g's); `H_repair 0–4%` vs `ΔY 9–15%` | **proven + observed (qwen)** |
| generalization | all edges hold across models/benchmarks/N | — | **OPEN → P2** |

---

## 3. Positioning in one paragraph (open the paper with this)

> Prior work establishes that LLM errors are correlated (Kim et al. 2025) and that LLM
> collectives herd and conform (Bellina 2026; SNLA 2026), and recent work calibrates
> confidence against communication-induced dependence (CAGE-CAL 2026). We do not claim
> that dependence, herding, or communication-induced correlation exist. We ask **why** the
> dependence arises — its causal mechanism — and **when** a causal change in it alters the
> collective decision — its susceptibility gate — and show that statistically similar
> dependence has behaviorally different consequences depending on its origin (`M`), gated
> by the collective aggregation boundary (`χ`).

Distinctions to guard (detail in `README.md` §2c): CAGE-CAL contrasts
*communication/topology*, we contrast two *causal origins of dependence* at matched `D`;
Bellina/`2605.27288` gate *individual* conformity by *self* competence/uncertainty, we gate
the *collective* accuracy consequence by the *vote margin*; `2607.05545`'s spontaneous
floor cancels in our paired `ΔD`/`ΔA` contrast.

---

## 4. The controller, relative to the chain — **RESOLVED as a bound (C5), not parked**

CDCR would add a fifth edge — *observe `(D̂, χ̂)` → diagnose `M̂` → select an intervention
that recovers `A`*. The full gate history is now **explained by Theorem 6**, not an open
engineering problem:
- **K1** — oracle headroom thin (cross-fit `H=+0.037`, conditional).
- **K2 / K2.5-pre** — **no** router on observable (even deliberately *leaky*) features beats
  static `shared_ev`; hard oracle headroom `≈+0.003`.
- **action-enrichment** — the only positive action is `a_common` (new evidence, +0.016, weak);
  `a_copy` dead, `a_CI` (Counterfactual Isolation) gated **KILL** (2026-08-25).
- **SGIR** (susceptibility-gated independent resampling) — offline oracle **KILL** on both the
  shared-evidence axis (low-M gain `+0.014`) and the copying axis (`H_repair` = `0.000` at
  g=1, `+0.033` at g=2), 2026-09-01.

**Why they all fail — consistent with Theorem 6.** These operators are **IRIC or bounded by
it** (`a_copy`/SGIR by construction; K2 routes among IRIC actions; `a_CI` argued in Thm6 §1):
they rearrange or reset the collective with *no new information*, so their ceiling is `A_0`
and their headroom is `p_+- − p_-+ ≤ ΔY` — `≈ 0` when the independent aggregate is
near-consensus (TruthfulQA, `D_0=0.83`), rising to `≈+0.10` when it is diverse (AQuA,
`D_0=0.53`, n=20). This is a negative chain **consistent with** one bound, not five separate
failures (we do **not** claim Theorem 6 mechanically "explains all five"). The **only** action
that helped is the **non-IRIC** one (`a_common`, new evidence, weak/+0.016) — the class
Theorem 6 leaves open.

⇒ The controller edge does not "close later"; the IRIC part is **closed (bounded)**. The
genuinely open algorithmic direction is **non-IRIC**: interventions that inject new evidence
(retrieval) able to move `Y_0` itself — a data problem, not an aggregation problem.

---

## 5. Immediate plan

- **P1 (this doc + Theorem 5)** — formalization of the chain. **DONE** for the theory;
  remaining P1 polish = fold `χ` into the abstract/claims and cut the old `S=(p,D,M,N)`
  phrasing in favor of `F(p,D,M,χ,N)`.
- **P2** — reproduce `M→D` (ΔD instrument) and `D─χ→A` (boundary slope), plus the Theorem 5
  N-scaling prediction, on **≥3 model families × ≥2 benchmarks**. This is the audit's
  clearest open territory and the evidence half of the paper.
- **P3** — CDCR: **resolved as Theorem 6 / C5** (the IRIC controller class is bounded, §4),
  no longer an open build. Remaining P3 = reproduce the Thm 6 identity/bound on ≥3 model
  families (offline where states persist), and — if pursued at all — scope the *non-IRIC*
  (retrieval/new-evidence) direction, which is the only class that can exceed `A_0`.
