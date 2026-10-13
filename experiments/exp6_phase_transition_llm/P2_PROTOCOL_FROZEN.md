# P2 — Generalization protocol (FROZEN 2026-08-14, before any cross-model number is seen)

**Purpose.** Test whether the three claims survive across model families × benchmarks × N,
and whether Theorem 5's N-scaling prediction holds. **No CDCR here — it stays parked.**

**Freeze rules.**
1. Estimands + PASS thresholds below are locked *before* looking at any non-qwen result.
2. `--model-tag` is namespace-only: it changes cache/output filenames, **never** the
   experimental protocol (same question selection, clue cache, evidence draws, shuffles,
   estimator). Verified: empty tag ⇒ byte-identical legacy paths.
3. **Every run is reported, including failures.** A model/benchmark that breaks a claim is a
   result, not a bug to hide.
4. One runner does it all: `exp6c_boundary.py` (superset — runs `none/shared/indep`, emits
   per-question `A_q`, `D_q`, and the `none`-margin, i.e. C1+C2+C3 from one pass).

---

## Frozen estimands & PASS criteria (implemented in `p2_aggregate.py`, verified on qwen)

- **C1 — THEORY-IMPLIED mechanism contrast** = a *generalization test of Theorem 4's ρ→1
  prediction*, **not** a direct empirical two-mechanism observation. The limits are computed
  from the `none` arm (unconfounded: correct-leaning shared evidence never enters), not
  separately rolled out.
  - copying limit `A_M2 = P_maj = plurality_acc(none)`; common limit `A_M1 = p = p_single(none)`.
  - Condorcet gap `= P_maj − p`. **PASS: `sign(gap) == sign(p − ½)` and `|gap|>0`.**
  - **The MAIN empirical Mechanism Non-Equivalence stays on the controlled ablation** (copying
    vs common-shock rollout, Exp6/6B). This P2 block only tests whether the *prediction*
    generalizes; report it as such, never as an observed two-mechanism result.
  - Report the two limits separately; do **not** collapse to one correlation.
- **C2 — Boundary amplification** (Theorem 5). `|ΔA| ~ M`, `ΔA = A_shared − A_indep`,
  `M = runner-up gap`. Save `{slope, bootstrap CI, 3 equal-count margin bins, N}`.
  **PASS: slope < 0 with CI excluding 0.** Cross-run N-scaling: slope should get *more*
  negative as `N` grows (Theorem 5 finite-size exponent).
- **C3 — Matched dependence instrument** (validated estimand, unchanged). Paired
  `ΔD = mean_q[D_shared(q) − D_indep(q)]` on the shared∩indep mask, question bootstrap CI.
  **PASS: ΔD > 0 with CI excluding 0.**

**qwen2.5:7b baseline (verification, N=11):** C1 ✓ (easy gap +0.021, hard −0.043) · C2 ✓
(pooled slope −0.125 [−0.198,−0.056]) · C3 ✓ (ΔD +0.272 [+0.162,+0.393]).

---

## Frozen run matrix (proposed — adjust counts, not estimands)

**Models (5): 4 families + 1 within-family scale point.** All served locally; each needs a
dedicated `num_parallel≥4` Ollama server on its own GPU (K1 throughput fix).

| family | model | config | model-tag stem |
|---|---|---|---|
| qwen | `qwen2.5:7b` (baseline ✓) | `debate_gpu1.yaml` | `qwen7b` |
| llama | `llama3.1:8b` | `debate_llama_a.yaml` | `llama8b` |
| gemma | `gemma2:9b` | `debate_gemma.yaml` | `gemma9b` |
| mistral | `mistral:7b` | `debate_mistral.yaml` | `mistral7b` |
| qwen (scale) | `qwen2.5:14b` | (copy config, set model) | `qwen14b` |

**Benchmarks (target 6).** TruthfulQA is wired. Each additional benchmark needs a one-time
p-probe to build its `easy (p̂≥0.8)` / `hard (p̂≤0.3)` selection caches (a benchmark uniformly
hard/easy contributes only the bin it has — reported, not dropped). Proposed set:
`TruthfulQA (✓)`, `MMLU-Pro`, `GPQA` (hard-heavy), `ARC-Challenge`, `CommonsenseQA`, `AQuA`.
→ model-tag = `<model>_<bench>`, e.g. `gemma9b_arc`.

**N (for the Theorem-5 test).** Primary matrix at `N=11`. Add `N ∈ {5, 21}` for **one**
cell (e.g. `qwen7b × TruthfulQA`) — enough to test slope-steepens-with-N cheaply without an
N-sweep on every cell.

---

## Staging (do NOT run the full 5×6 Cartesian up front)

Two **orthogonal** axes first — model-generalization and benchmark-generalization — at far
lower cost than the cross-product; open selected cross-cells only if both hold.

1. **P2a — model axis** (`TruthfulQA × 5 models`), run in **parallel** with:
2. **P2b — benchmark axis** (`6 benchmarks × qwen2.5:7b`).
3. **then** selected `model × benchmark` cross-cells (only where P2a & P2b both look stable).
4. **last** N-scaling `N ∈ {5, 11, 21}` on one cell — a confirmatory test *for Theorem 5 only*.

> If C1/C2/C3 survive both orthogonal axes, the result moves from "interesting on
> Qwen/TruthfulQA" to a **general-phenomenon candidate** — the single biggest value step now.

### Adapter-only principle (P2b) — locked
- A benchmark loader **only** normalizes `question / options / gold / id` into the truth@0
  `Question` format. **No benchmark-specific prompt engineering.**
- The p-probe is used **only** to stratify easy (`p̂≥0.8`) / hard (`p̂≤0.3`); thresholds are
  benchmark-independent and frozen. A benchmark with only one bin contributes that bin.
- **Estimands and PASS criteria do not change after seeing any result.**

### P2b tooling (built + verified, adapter-only)
- `p2_benchmarks.py` — adapter registry + benchmark-aware `select_questions` (truthfulqa ⇒
  legacy path, byte-identical; others via the registry). **Verified:** AQuA normalizes to
  truth@0 (254 items, K=5); stratification picks the right easy/hard indices.
- `p2_probe.py` — single-agent p̂ probe → `data/processed/exp6_select_{benchmark}.json`
  (frozen easy≥.80 / hard≤.30 bins; no benchmark-specific prompting). **GPU.**
- `--benchmark` threaded into `exp6c_boundary.py` + `exp6c_shared_context.py`; clue cache
  benchmark-tagged (truthfulqa ⇒ legacy). All parse-verified.

### Local data status
`TruthfulQA` ✓ (frozen caches) and `AQuA` ✓ (**wired, ready — needs only its probe run**).
`MMLU-Pro / GPQA / ARC-Challenge / CommonsenseQA`: one-time download into
`data/raw/benchmarks/` + uncomment the registry entry in `p2_benchmarks.py` (a ~5-line
`load_*` normalizer) — **no other code changes**, then probe.

---

## Commands

```bash
# per model (TruthfulQA), both regimes, tagged so nothing collides:
python exp6c_boundary.py --regime easy --config src/config/<cfg>.yaml \
    --n-questions 40 --replicates 8 --n-agents 11 --model-tag <stem>
python exp6c_boundary.py --regime hard --config src/config/<cfg>.yaml \
    --n-questions 40 --replicates 8 --n-agents 11 --model-tag <stem>
# aggregate everything found (per-model C1/C2/C3 + N-scaling view + PASS matrix):
python p2_aggregate.py --n 11
```

**P2b (benchmark axis × qwen).** One-time probe per benchmark, then the same runner with
`--benchmark`; model-tag encodes the benchmark so nothing collides:
```bash
# 1) stratify easy/hard for the benchmark (GPU; AQuA is wired, others need the data file):
python p2_probe.py --benchmark aqua --config src/config/debate_gpu1.yaml --replicates 4
# 2) run the superset runner on it, both regimes, tagged <model>_<bench>:
python exp6c_boundary.py --regime {easy,hard} --config src/config/debate_gpu1.yaml \
    --n-questions 40 --replicates 8 --n-agents 11 --benchmark aqua --model-tag qwen7b_aqua
python p2_aggregate.py --n 11        # groups qwen7b_aqua alongside the legacy TruthfulQA run
```
