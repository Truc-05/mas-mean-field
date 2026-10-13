# CI_PROTOCOL_FROZEN — Counterfactual Isolation, pre-registration

> **STATUS: FROZEN 2026-08-21, BEFORE any GPU / before any A-CI number is seen.**
> Design **A′ signed off** (controlled copying-induction + copying-harm contrast `C_q`; §2). This
> document freezes the operator, the prompt, the estimands (`Δ_CI` recovery + Recovery Ratio +
> continuous `C_q` interaction), the baselines, and the GO/KILL gate for **one** copying-targeted
> intervention. No implementation number, no pilot, no prompt-shopping precedes this freeze.
> Executes **Steps 2–3** of the locked plan (design + pre-register the missing `WHAT(copying)`
> action). Step 4 (pilot) runs next; Steps 5/6 (build controller / stop) are decided by the gate.

## 0. Why this exists (one paragraph)

C1–C4 are frozen (README §2b) and stand as a complete framework→theory→causal-findings paper
**independent of any controller**. The adaptive-algorithm branch has exactly one missing piece:
the `WHAT(mechanism)` map needs an action for the **copying-heavy** cell.
`a_copy` is **dead** (cross-model: `=single`; mistral copying-heavy `Δ(copy−common)=−0.446` —
actively harmful). `a_common` (= `a_E`, independent-evidence) fills the **common-heavy** cell but
its marginal accuracy win did not replicate (pooled `+0.016`, 0/4 sig). So there are **not yet two
working, mechanism-matched treatments** to route between. This protocol is the **single, final,
pre-registered shot** at producing the copying-cell action. If it fails, the algorithm branch stops
and we ship C1–C4; we do **not** try operators 5, 6, 7.

---

## 1. The operator — Counterfactual Isolation (`a_CI`)

**Intuition.** After coupled deliberation, agents sit in a *peer-contaminated* state: a shared
(possibly correlated-error) consensus attractor. `a_copy` only told each agent to "re-decide
independently" — passive, and dead. `a_CI` is **active**: it names the current consensus answer and
forces each agent to *isolate*, build the **strongest case against** that consensus answer, privately
re-reason, **commit**, and only then re-vote — deliberately pushing the collective off the attractor.

**Pipeline (per question, per replicate):**
```
1. INDUCE   : run `debate_rounds` COUPLED rounds (same rollout as the `debate` action / a_copy)
              → a peer-contaminated final state; read its plurality label  X_c  (the consensus).
2. ISOLATE  : strip all peer context; keep each agent's OWN last (peer-influenced) reasoning.
3. CHALLENGE: instruct each agent, independently, to construct the STRONGEST case AGAINST X_c
              and to seriously weigh that a DIFFERENT option is correct.
4. RE-REASON: private, no peer signal, no evidence injected (truth-independent).
5. COMMIT   : each agent scores all labels → plurality  →  a_CI vote.
```

**Exact difference from the three dead/weak operators (frozen, so the novelty is auditable):**

| operator | applied to | move | status |
|---|---|---|---|
| `a_copy` | post-debate contaminated state | "ignore others, re-decide independently" (passive) | **dead** |
| `a_dissent` | *independent* agents (no peers) | steelman an alternative to your **own intuition** | weak/uniform |
| `a_common` | independent agents | inject **own** independent evidence | common-cell, weak marginal |
| **`a_CI`** | **post-debate contaminated state** | steelman the case **against the group's consensus `X_c`**, commit before re-seeing peers | **NEW — this test** |

`a_CI` = the *post-contamination context* of `a_copy` **+** the *counterfactual move* of `a_dissent`,
but targeted at the **consensus label `X_c`** specifically (not the agent's own prior intuition).
That combination is what neither prior operator tried, and it is the whole hypothesis.

**FROZEN PROMPT (Vietnamese, matches `action_enrich._break_copy_prompt` format so parsing/scoring
is byte-identical to every other action).** `{X_c_label}. {X_c_text}` is the induced consensus
option; `{own}` is the agent's own post-debate reasoning (via `_own_reasoning`, ≤600 chars):

```
{persona (empty for plain agents)}
Câu hỏi: {question}

Các lựa chọn:
{A. opt0 / B. opt1 / ...}

Lập luận TRƯỚC ĐÓ của chính bạn (sau khi thảo luận nhóm):
{own}

Cả nhóm hiện đang nghiêng về đáp án: {X_c_label}. {X_c_text}

Hãy TÁCH KHỎI áp lực đồng thuận của nhóm. Nhiệm vụ của bạn: xây dựng lập luận MẠNH NHẤT
CHỐNG LẠI đáp án đồng thuận "{X_c_label}", và cân nhắc NGHIÊM TÚC khả năng một phương án
KHÁC mới là đúng. Sau khi đã tự phản biện xong, hãy QUYẾT ĐỊNH ĐỘC LẬP dựa trên phán đoán
của riêng bạn (bạn ĐƯỢC PHÉP vẫn giữ "{X_c_label}" nếu sau khi phản biện bạn thấy nó thực sự
đúng). Chấm điểm mức độ tin tưởng (thang -10 đến 10) cho TỪNG nhãn ({A, B, ...}). Trả lời
DUY NHẤT bằng JSON: {"scores": {"A": <số>, "B": <số>, ...}, "reasoning": "<ngắn gọn>"}.
```
**Wording guard (frozen):** the prompt permits keeping `X_c` if it survives the challenge — this is
deliberate. `a_CI` must be a *decorrelator*, not a blunt "always flip." The mechanism-interaction
test (§4) is what distinguishes the two: a blunt flip helps/hurts uniformly; a mechanism-matched
decorrelator helps **only** on copying-heavy states.

**Implementation note (post-sign-off, mechanical — no researcher DoF).** `ci_enrich.py` =
`action_enrich.py` with one added operator: reuse `_break_copy_rollout`'s scaffold (shared coupled
debate, capture own reasoning, `_custom_elicit` on `_ci_prompt`); read `X_c` from the same `debate`
final plurality already computed. Analysis = `action_enrich_analyze.py` with `a_copy→a_CI`
substituted in `mechanism_matching`. **No new estimator is written.**

---

## 2. THE design decision that must be settled first — copying-heavy support

**Problem (quantified from the completed runs).** The observational copying-heavy label
(`|y_debate − y_indep_vote| > |y_shared_ev − y_indep_vote|`) yielded
`n_copying ∈ {qwen 2, gemma 1, llama 0, mistral 7}` out of 40. The **primary estimand lives on the
copying-heavy subset**; at `n<10` per model a naive Q=40 re-run has **essentially zero power** and the
gate would be meaningless. This must be fixed *before* the run, not explained after.

**APPROVED — Design A′: controlled copying-induction + copying-HARM contrast** (user sign-off
2026-08-21). Do **not** wait for questions to be naturally copying-heavy. Following the C3
matched-instrument logic, *drive* the collective into a copying-contaminated state on **every**
question via a strong-coupling coupled debate (the exp6 `g` knob — `coupling = 1.0`,
`debate_rounds = 3`), so the copying arm has **full support `n = Q`**. This turns the whole study into
a **damage–repair experiment**:
```
   INDEPENDENT state        COPY induction            COUNTERFACTUAL ISOLATION
   (indep_vote, A_indep)  ──copying──►  contaminated  ──a_CI──►  recovery
                           coupling=1     consensus X_c
```

**The critical refinement (why A, not the naive A).** A subset defined only as "consensus is wrong"
selects *failure cases* without proving the failure was **caused by copying** — a reviewer would say
"that's just the hard questions debate gets wrong." So `S_copy` **must carry evidence that copying
reduced performance.** Per question `q`, with `A_indep,q` = accuracy of the independent state
(`indep_vote`) and `A_copy,q` = accuracy of the coupled-debate (contaminated) state:
```
copying-harm contrast   C_q = A_indep,q − A_copy,q          # C_q > 0  ⇔  independent was better ⇒ copying HURT
S_copy (binary)         = { q :  C_q > 0  ∧  P(wrong consensus under copy | q) ≥ 0.5 }
```
`C_q > 0` is **copying-induced harm**: agents do better independently, and social coupling drags the
collective down — the *exact* damage `a_CI` claims to repair. The continuous `C_q` is preferred over
the binary subset wherever a test allows it (the interaction is run continuously against `C_q`, §3).
This is faithful to "agents ARE in a copying-contaminated state," gives real power (full `n=Q`,
weighted by `C_q`), and mirrors how Exp6B/6C *constructed* their condition rather than waiting for it.

**Rejected — Design B: enlarge Q to ~150–200** to wait for natural copying-heavy questions. At the
observed prevalence (0–7/40) this burns compute while causal identification stays weak (it selects on
outcome, not on copying-caused harm). Not used.

> **Frozen (no post-hoc tuning):** `coupling=1.0`, `debate_rounds=3`, `N=11`, `R=8`, HARD, one fixed
> question set. Coupling is **never** re-tuned after seeing results.

---

## 3. Hypotheses & estimands (FROZEN)

Notation: per question `q`, per action `a`, `y_a(q)` = mean plurality-correctness over `R` reps
(same as `action_enrich`). `A_indep(q) = y_indep_vote(q)` (independent state), `A_copy(q) = y_debate(q)`
(coupled-debate / contaminated state), `A_CI(q) = y_a_CI(q)`. Existing set
`A_prev = A0 ∪ {a_copy, a_common, a_dissent}`.

**H1 (primary) — does `a_CI` REPAIR copying-induced harm?** Recovery vs the contaminated baseline, on
the copying-harm subset:
```
Δ_CI = mean_{q∈S_copy} [ A_CI(q) − A_copy(q) ]             # paired, question-level bootstrap CI
```
Prediction: `Δ_CI > 0`. This is the damage–repair statement: on states where copying demonstrably hurt
(`C_q>0`), does isolating+counter-arguing the consensus recover accuracy over the coupled state.

**Recovery Ratio (primary companion — the interpretable headline number).** Of the accuracy copying
*destroyed*, what fraction does `a_CI` restore:
```
RR = [ A_CI − A_copy ] / [ A_indep − A_copy ]      (pooled over S_copy; A_indep − A_copy > 0 by construction)
```
e.g. `A_indep=0.60, A_copy=0.40, A_CI=0.52 ⇒ RR = 0.12/0.20 = 60%`. Reads far better than a bare
"+0.12." **Interpretation lines (frozen):** `RR ≤ 0` = no repair; `0 < RR < 100%` = **partial repair**
(scientifically real damage–repair, but `a_CI` still below the independent state — a controller would
prefer plain `indep_vote` there); **`RR ≥ 100%` (⇔ `A_CI ≥ A_indep`)** = `a_CI` beats even *not
copying* ⇒ a genuinely additive copying-cell action = the **deployment-strength** result that
justifies building the two-tier controller. Report `RR` pooled + its bootstrap CI.

**H2 (secondary, weighs MORE — the mechanism-match, run CONTINUOUSLY on `C_q`).** If `a_CI` is truly
copying-targeted, its edge over `a_common` should **grow with copying-harm** `C_q`:
```
gain_op(q) = y_op(q) − max_{a∈A0} y_a(q)                    # same gain as mechanism_matching()
continuous:  slope of ( gain_CI(q) − gain_common(q) )  on  C_q   >  0        # PRE-REGISTERED sign
discrete echo (DiD): I = mean_{S_copy}(gain_CI − gain_common) − mean_{S_common}(gain_CI − gain_common) > 0
```
where `S_common` uses the existing common-heavy label (`|y_shared_ev − y_indep_vote|` dominant). A
positive `Δ_CI` with a **non-positive slope / `I ≤ 0`** is "generic better prompting," **not**
mechanism-matched control — the two-tier `π(x)` would then be unjustified (§4).

**Blunt-flip guardrails (pre-registered falsifiers — unchanged, §user-agreed):**
- `a_CI` must **not** significantly beat `a_common` on `S_common` (else anti-consensus everywhere).
- `a_CI` must **not** hurt vs `debate` on `S_common`/non-copy controls (`C_q ≤ 0`) beyond noise
  (else it just destroys good consensus). *Three outcomes, pre-named:* wins everywhere ⇒ generic
  prompting; only reverses consensus + hurts common-heavy ⇒ anti-conformity trick; **selectively
  recovers copying-induced loss ⇒ the result we want.**

**Report all of:** per-subset (`C_q>0` vs `C_q≤0`, and `S_copy` vs `S_common`) means of
`{indep_vote, debate, a_copy, a_common, a_CI}`; `Δ_CI`; `RR` (+CI); the continuous slope on `C_q`
and the discrete `I`; and the full per-action table — so uniform-lift ≠ mechanism-match stays visible
(the mistake we refuse to repeat with `a_common`).

---

## 4. Locked GO/KILL gate (set before the run)

Applied to the recovery **`Δ_CI`** (H1) on `S_copy`, with the **mechanism-match (positive `C_q`
slope / `I`) as a necessary co-condition** for any "mechanism-matched" claim:

```
Δ_CI < +0.02                                     → KILL algorithm branch. Ship C1–C4. No operator #5.
+0.02 ≤ Δ_CI < +0.05                              → WEAK. Do NOT build the controller (report as a note).
Δ_CI ≥ +0.05  AND  Δ_CI>0 in ≥2 models            → GO (copying cell filled).
Δ_CI ≥ +0.08  AND  slope(C_q)>0 / I>0             → STRONG → build the two-tier adaptive algorithm (§6).
```
**Necessary for ANY "mechanism-matched controller" claim (all bands):** positive continuous slope of
`(gain_CI − gain_common)` on `C_q` (discrete `I > 0`) **and** both blunt-flip guardrails pass. If
`Δ_CI ≥ 0.05` but the slope/`I ≤ 0`, `a_CI` may be reported as a *generic* decorrelation op but the
two-tier `π(x)` is **not** justified — do not claim it.
**Deployment-strength (for actually building the controller, not just the science):** `RR ≥ ~100%`
(`A_CI ≥ A_indep`) on `S_copy` — otherwise the copying cell would be better served by plain
`indep_vote` and `a_CI` is a damage–repair *demonstration*, not a deployable action.

**Replication rule:** GO requires the same-signed `Δ_copy` in **≥ 2 model families**, not one. Pilot
qwen first; a single-model win is "point-in-time," never the final verdict (cf. the 08-19/08-20
`a_common` lesson — see README §5, the annotated history line).

---

## 5. Run matrix (Step 4 pilot — runs only after sign-off)

- **Regime:** HARD only (the sharpest falsifier, `H^CV_hard≈0.003`).
- **Config:** `N=11`, `R=8`, `debate_rounds=3`, `coupling=1.0`; same fixed **qwen clue pool** for
  `shared_ev`/`a_common` (P2 convention — clues held constant, only the model swaps).
- **Models / order:** (1) `qwen2.5:7b` pilot → read gate. (2) if not KILL, `llama3.1:8b` +
  `gemma2:9b` for the ≥2-model replication. `mistral:7b` optional (kept boundary; it had the most
  natural copying-heavy support, useful cross-check).
- **Sharding/servers:** identical to `action_enrich_xmodel_run.sh` (s0→:11443 / s1→:11444, 2×20 q).
- **Outputs:** `results/tables/ci_gate_{model}.json`; cross-model rollup mirrors
  `action_enrich_xmodel_summary.py` (paired `Δ_copy` per model + pooled).

**Cost guard:** `a_CI` adds ~1 induced-debate + 1 re-elicit pass per q×r on top of the existing
rollout — comparable to `a_copy`'s cost. One round, one operator, HARD-only. No re-runs to "tune."

---

## 6. If (and only if) it PASSES — the two-tier adaptive algorithm

With two working, mechanism-matched actions in hand (`a_common` = `a_E` for common-heavy; `a_CI` for
copying-heavy), and susceptibility `χ` already shown observable (C2, boundary slope), the controller
becomes buildable and testable:
```
        ⎧ STOP                    if χ(x) < τ                 # not susceptible → don't intervene
π(x) =  ⎨ a_E   (indep evidence)  if χ(x) ≥ τ,  M̂ = common
        ⎩ a_CI  (counterfact. iso) if χ(x) ≥ τ,  M̂ = copying
```
**Susceptibility says WHEN to intervene; mechanism says HOW.** Evaluate on an accuracy–compute
**Pareto** vs the pre-specified baselines: `always-debate`, `always-a_E`, `independent-vote`,
`random routing`, and `uncertainty routing` (route on margin/entropy alone, mechanism-blind). The
adaptive claim requires beating **uncertainty routing** — that is what isolates "mechanism matters
for the *intervention*," not just "susceptibility matters." `M̂` (mechanism estimate) and `χ̂` must be
built from **leakage-free observable `none`-state features** (the K2 rule), and the controller
frozen/gated before its own run.

---

## 7. Researcher-DoF guards (locked — same discipline as every prior gate)

- **Exactly ONE new operator (`a_CI`), ONE round, prompt frozen in §1 above.** No prompt-shopping,
  no "try 3 phrasings and keep the best."
- **No resurrection of `a_copy`**; it is dead and stays in the table only as the pre-registered
  contrast.
- **On FAIL (`Δ_copy < 0.02`):** the algorithm branch **stops**. Write C1–C4. Do **not** design
  operators #5/#6 — that would be exactly the prompt-shopping this discipline forbids.
- **Single-model results are point-in-time**, never final; GO needs ≥2 families.
- **`I > 0` is required** for any mechanism-matched-control language, regardless of `Δ_copy`.

---

## 8. Frozen-before-run checklist

- [x] Operator `a_CI` defined precisely; distinguished from `a_copy`/`a_dissent`/`a_common` (§1).
- [x] Prompt frozen verbatim (§1), format matches existing parse path.
- [x] **Copying-heavy support DECIDED — Design A′** (user sign-off 2026-08-21): controlled induction
      `coupling=1.0`, copying-harm contrast `C_q = A_indep − A_copy`, `S_copy = {C_q>0 ∧ wrong-consensus≥½}` (§2).
- [x] Primary estimand `Δ_CI` (recovery vs coupled debate) + **Recovery Ratio `RR`** + continuous
      interaction slope on `C_q` (discrete `I` echo) frozen with exact formulas (§3).
- [x] Blunt-flip guardrails pre-registered; three outcomes pre-named (§3).
- [x] GO/KILL gate on `Δ_CI` + slope/`I` co-condition + `RR≥100%` deployment line + ≥2-model rule (§4).
- [x] Run matrix, models, servers, cost guard fixed (§5).
- [x] Two-tier `π(x)` + Pareto baseline set (incl. uncertainty-routing bar) specified (§6).
- [x] Researcher-DoF guards locked (§7).
- [ ] Implement `ci_enrich.py` + `ci_enrich_analyze.py` (mechanical transcription; §1 note) → no-GPU synthetic validation.
- [ ] Pilot qwen HARD → gate → (if not KILL) llama + gemma for ≥2-model replication.
