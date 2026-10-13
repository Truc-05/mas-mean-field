# Offline analyses + consistency audit — 2026-09-22

No GPU. All on the frozen 4-model × 2-benchmark substrate. Two new analyses locking C1/C2
against the 2026 neighbors (CAGE-CAL, Demystifying-MAD, Not-All-Flips), plus a full audit of
`paper.md`'s load-bearing numbers.

## OFFLINE #1 — matched-D falsification of C1 (NEW)
`experiments/exp6_phase_transition_llm/analysis_C_matched_D_falsification.py`
→ `results/tables/analysis_C_matched_D_falsification.json`. shared n=80, copy n=348.

1. **Caliper matched-D (copy − shared at |ΔD|≤ε):** the mechanism gap in **ΔY survives the
   tightest caliper** — ε=.02: +0.094 [+0.063,+0.128]*, ε=.05: +0.083*, ε=.10: +0.084*.
   `p_break` +0.02–0.03* and `p_rescue` +0.02–0.03* both significant; **dA gap ≈ +0.003
   (n.s.)** at all ε.
2. **Nested LPM on ΔY:** M0(D) R²=.121 → M1(+mech) R²=.136, **F=7.40 p=.0068** → M2(+D×mech)
   **F=0.09 p=.76** (n.s.). Mechanism(copy) coef after conditioning on D = **+0.099**.
   ⇒ ΔY = f(D, M); mechanism is a level shift, no interaction.
3. **NN matching on (D, A0, margin):** E[ΔY|copy]−E[ΔY|shared] = **+0.050 [+0.017,+0.084]***.

**Honest framing point (feeds paper §7):** at matched D the mechanism difference is in
**decision-change / churn (ΔY, p_break, p_rescue), NOT net accuracy (dA≈0)** — consistent with
the "shared arm uses correct evidence" caveat. So C1's matched-D claim must be sold on ΔY, not
on a matched-D accuracy gap. This is *cleaner* (matches Thm 4: consequence ≠ magnitude), not weaker.

## OFFLINE #2 — C2 competing-predictor (already existed; re-confirmed)
`analysis_A_c2_controls.py` → n=398. margin univariate slope **−0.0345 [−0.0454,−0.0245]***.
Dissociation (margin+confidence+confidence_gap): **margin stays −0.0345 [−0.046,−0.024]***,
confidence & confidence_gap n.s. ⇒ NOT agent epistemic uncertainty (Bellina). Multivariate with
all 5: margin/vote_entropy/disagreement all lose significance (collinear). ⇒ keep wording
**"decision-boundary susceptibility *family*"**. No rerun.

## F2 main figure (NEW)
`analysis_C_figure.py` → `results/figures/exp6c_mechanism_consequence_F2.png`. ΔY vs D, two
CVD-safe curves (Okabe-Ito blue=shared / vermilion=copy, + shape/line secondary encoding). Copy
lies above shared across the whole D range ⇒ visual ΔY=f(D,M).

## Consistency audit — paper.md load-bearing numbers vs results/tables (ALL PASS)
| Claim | paper.md | table | verdict |
|---|---|---|---|
| C3 ΔD gemma / llama / qwen / qwenAQuA / mistral | +.536/.148/.272/.159/.249 | .536/.148/.272/.159/.249 | ✓ exact |
| C2 pooled \|ΔA\|~M slope | −0.125 [−0.198,−0.057] | −0.125 [−0.198,−0.059] | ✓ |
| Analysis A margin (dissociation) | −0.0345 [−0.046,−0.024]* | −0.0345 [−0.046,−0.024]* | ✓ |
| Analysis B D[.80,.90) shared / copy | dA −.042/−.125, ΔY .10/.17, H +.042/+.125 | −.042/−.125, .097/.167, +.042/+.125 | ✓ |
| Analysis B D[.95,1) copy ΔY | 0.078 | 0.078 | ✓ |
| C5 repair TQA identity/bound, D₀ | identity✓ bound✓, D₀=.83 | identity_ok/bound_ok=true, D₀=.832 | ✓ |

## Still open (needs the user's go — GPU1 campaign, prepped, not launched)
Tier-1 6×7 breadth, Tier-2 C4 (3×3), Tier-3 C5 g-sweep breadth. Infra ready per
memory `mas-mean-field-gpu`. Then freeze. Keep `H_repair ∝ (1−D₀)` as moderator hypothesis only.
