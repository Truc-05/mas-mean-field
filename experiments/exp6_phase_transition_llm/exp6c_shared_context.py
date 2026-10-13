"""Exp6C — LLM-native shared-information mechanism (independent validation).

Exp6B showed that a *mathematically* controlled common shock on real beliefs
reproduces the competence–correlation transition. Exp6C asks the payoff question:
does the phenomenon survive when the shock is delivered the way it would arise in
the wild — as shared *evidence/context* the LLMs read and reason over — with NO agent
ever seeing another's vote?

Mechanism. Each agent answers ONCE, independently. The only manipulation is a piece
of reference evidence C injected into its prompt (neutral framing, never "others
chose X"). Two orthogonal knobs:
  • quality  ∈ {correct, misleading}  — evidence points toward the truth / a distractor.
  • sharedness ∈ {shared, independent} — all N agents get the SAME realization C_q,
    vs each agent gets an INDEPENDENTLY sampled realization of matched quality.
The independent arm is the causal control: it holds evidence content/quality fixed and
varies ONLY whether the realization is common. So
        ShareEffect = P_maj(shared) − P_maj(independent)       (matched quality)
isolates the effect of *shared-ness* (error correlation) from the effect of the
evidence being good/bad. A `none` baseline and a weak/strong strength rung on the
shared arm complete the grid.

ESTIMATOR NOTE (2026-08-09 fix). The original "primary ρ" was
within-question pairwise correctness correlation computed from a SINGLE realization
per question. That is not identifiable: conditioning on the realized error count m of a
single N-vector forces the average pairwise covariance to −1/(N−1) (algebraically, ρ_q
= −a(1−a)/(N−1) ÷ a(1−a) = −1/(N−1) regardless of the data). Every cell therefore read
−1/(N−1) = −0.10 at N=11 — an estimator/design artifact, not a signal. The mechanism is
now measured at the REPLICATE level: for each question×condition we draw R stochastic
replicates, so for agents i,j we hold true samples (E_i1,E_j1),…,(E_iR,E_jR) of the
error processes and can estimate inter-agent error dependence. The evidence realization
C is RESAMPLED each replicate (a genuine common shock in the shared arm), which is what
induces — and lets us identify — the dependence.

PREREGISTERED HYPOTHESES (printed to the log before any answering call):
  H1  shared evidence raises inter-agent error dependence over the matched independent
        control:  ΔD = D(shared) − D(indep) > 0   at matched quality & strength.
        D = replicate-level pooled pairwise error correlation (primary).
  H2  conditional on H1, the sign of the collective effect tracks competence:
        p > ½ (easy)  ⇒  ShareEffect < 0   (shared correlation destroys the crowd)
        p < ½ (hard)  ⇒  ShareEffect > 0   (shared correlation dismantles error amplif.)

Primary D = replicate-level pooled pairwise error correlation (Var-of-sum identity,
defined whenever ≥2 agents vary). ΔD isolates shared-ness because both arms share the
question and match evidence quality/strength — only the commonality of C differs. Fleiss
ICC and pair-agreement are secondary. Regimes: hard (real single-agent p≈0.19) and a
cleaned easy set (p̂≥0.8 → realized p clearly >½).

Run (mini metric-validation, both regimes in parallel on the two GPU servers):
  python .../exp6c_shared_context.py --regime easy --config src/config/debate_gpu1.yaml \
      --n-questions 10 --replicates 8 --conditions none,shared_correct_strong,\
indep_correct_strong,shared_misleading_strong,indep_misleading_strong
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter
from mfg_llm.llm_debate.benchmarks import load_truthfulqa_mc1, plurality_prediction
from mfg_llm.metrics.stats import bootstrap_ci
from mfg_llm.opinion_state import extract_opinion_state

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
BENCH = ROOT / "data" / "raw" / "benchmarks" / "truthfulqa.csv"

# HISTORICAL ONLY (kept for provenance, no longer read). Before the English rebuild
# (2026-10-01) truthfulqa selected its questions from the g-sweep driver's own pre-probe
# caches listed here, with the predicates shown. Those caches now live under
# archive/vietnamese-era-2026-10-01/processed/. `select_questions` below delegates to the
# single unified probe path instead; the predicates were IDENTICAL to p2_benchmarks.PRED,
# so nothing about the selection rule changed when the branch was removed.
_LEGACY_REGIMES = {
    "hard": ("exp6_easy_truthfulqa_p0.0-0.3_k5_c400.json", 400, lambda p: p <= 0.30),
    "easy": ("exp6_easy_truthfulqa_p0.6-0.95_k5_c250.json", 250, lambda p: p >= 0.80),
}
REGIMES = _LEGACY_REGIMES          # name kept so older callers still import cleanly
# the 7-cell condition grid (single strength for the independent control)
CONDITIONS = ["none",
              "shared_correct_weak", "shared_correct_strong", "indep_correct_strong",
              "shared_misleading_weak", "shared_misleading_strong", "indep_misleading_strong"]


# ── question selection (single unified probe path for all benchmarks) ──────────
def select_questions(regime: str, n: int, benchmark: str = "truthfulqa") -> list:
    """Delegate to the one probe path shared by all seven benchmarks.

    This used to read `_LEGACY_REGIMES` directly, which hardwired truthfulqa to the
    g-sweep driver's pre-probe caches. Those are archived, so reading them would now
    raise. Delegating keeps every caller (exp6c_boundary via p2_benchmarks, action_enrich,
    ci_enrich) on the same selection, with thresholds `p2_benchmarks.PRED` that are
    identical to the legacy predicates above.
    """
    import p2_benchmarks as p2b          # local import: avoids an import cycle at module load
    return p2b.select_questions(regime, n, benchmark)


# ── clue generation (LLM), cached ───────────────────────────────────────────────
_NUM = re.compile(r"^\s*\d+[\.\)]\s*(.+)$")


def _parse_numbered(text: str) -> list[str]:
    out = [m.group(1).strip(" -\t") for line in text.splitlines()
           if (m := _NUM.match(line))]
    return [s for s in out if len(s) > 8]


def _gen_toward(agent: DebateAgent, question: str, options: list[str],
                target_idx: int, n: int) -> list[str]:
    tgt = options[target_idx]
    # The explicit "Write in ENGLISH only" constraint is load-bearing, not cosmetic. Without
    # it the generator drifts: the earlier Vietnamese-prompted pool was 71.9% Vietnamese,
    # 16.0% Chinese and 12.1% English over 12506 clues. Because the SHARED arm broadcasts ONE
    # realization while the INDEP arm issues N distinct ones, a free-language generator leaves
    # the independent arm additionally heterogeneous in language, so part of ΔD would measure
    # language heterogeneity rather than evidence commonality. Keep this constraint.
    prompt = (
        "You are generating data for an experiment on how information influences "
        "reasoning. Write in ENGLISH only.\n\n"
        f"Question: {question}\n\n"
        f"Write {n} SHORT (one to two sentences) and MUTUALLY DISTINCT pieces of "
        '"reference evidence or context" that sound plausible and would tend to incline '
        f"a reader toward the view: «{tgt}».\n\n"
        'Strict constraints. Do NOT write "the answer is" or any equivalent. Do NOT '
        "mention the option labels A/B/C/D. Do NOT state that this is the correct choice. "
        "Provide indirect information or argument only.\n\n"
        f"Number them 1..{n}, one per line."
    )
    items = _parse_numbered(agent._chat_with_retries(prompt))
    if not items:
        items = [f"A number of commonly cited sources connect this issue to the view: {tgt}."]
    return [items[i % len(items)] for i in range(n)]         # cycle-pad to exactly n


def build_clues(regime: str, questions: list, cfg: dict, N: int, fresh: bool,
                benchmark: str = "truthfulqa") -> list[dict]:
    bmtag = "" if benchmark == "truthfulqa" else f"_{benchmark}"   # truthfulqa ⇒ legacy cache name
    cache = PROC / f"exp6c_clues_{regime}{bmtag}.json"
    if cache.exists() and not fresh:
        data = json.loads(cache.read_text())
        # select_questions returns an ordered prefix, so cached clues align by index;
        # a longer cache (e.g. the 40-q full set) covers a shorter mini-run — reuse the
        # prefix and DON'T rewrite, so the full cache survives.
        if len(data) >= len(questions):
            print(f"clues: reusing {cache.name} (have {len(data)}, need {len(questions)})")
            return data[:len(questions)]
    gen = DebateAgent(0, model=cfg["model"], backend=cfg.get("backend", "openai"),
                      base_url=cfg.get("base_url"), temperature=1.0,
                      timeout=cfg.get("timeout", 300))
    out = []
    for qi, (q, opts) in enumerate(questions):
        K = len(opts)
        correct = _gen_toward(gen, q, opts, 0, N)                 # N variants toward truth
        mis = {d: _gen_toward(gen, q, opts, d, 4) for d in range(1, K)}  # per distractor
        out.append({"correct": correct, "mis": {str(k): v for k, v in mis.items()}})
        print(f"  clues {regime} q{qi}: correct={len(correct)} "
              f"mis={{{','.join(f'{k}:{len(v)}' for k,v in mis.items())}}}")
    cache.write_text(json.dumps(out, ensure_ascii=False))
    print(f"clues: wrote {cache.name}")
    return out


def _frame(clue: str, strength: str) -> str:
    if strength == "weak":
        return f"A reference note (reliability unverified): {clue}"
    return (f"Evidence from a source considered reliable: {clue} "
            "Please consider this information carefully.")


def _uniq(seq: list[str]) -> list[str]:
    """De-duplicate preserving order. Clue gen cycle-pads with repeats; stripping them
    means an independent draw never lands on an identical string by accident."""
    return list(dict.fromkeys(seq))


def _draw_distinct(uniq: list[str], N: int, rng: np.random.Generator) -> list[str]:
    """N draws WITHOUT replacement from the unique pool (refill by re-permuting if
    N>len). When the pool has ≥N uniques no two agents get the SAME string, so the
    independent control is not inflated by accidental duplicate evidence. NOTE: this only
    removes duplicate-artifact structure — it does NOT force D_indep→0; intrinsic shared
    failure modes and matched evidence content legitimately remain."""
    picks: list[str] = []
    while len(picks) < N:
        picks += [uniq[int(j)] for j in rng.permutation(len(uniq))]
    return picks[:N]


def evidence_for(cond: str, clues_q: dict, K: int, N: int,
                 rng: np.random.Generator) -> list[str | None]:
    """The N per-agent evidence strings for ONE replicate of one condition/question.

    C is drawn fresh from `rng` (seeded per replicate in main) so it varies across
    replicates. SHARED arm: one draw broadcast to all N agents (a common shock). INDEP
    arm: N draws without replacement from the de-duplicated pool — SAME quality/strength
    framing (_frame is identical across arms → matched control), only the commonality of
    the realization differs. The resampling makes D(shared) > D(indep) identifiable; the
    distinct indep draws keep D(indep) from being inflated by accidental duplicates."""
    if cond == "none":
        return [None] * N
    _, quality, strength = cond.split("_")
    if quality == "correct":
        uniq = _uniq(clues_q["correct"])
        if cond.startswith("shared"):
            c = uniq[int(rng.integers(len(uniq)))]                 # one common realization
            return [_frame(c, strength)] * N
        return [_frame(c, strength) for c in _draw_distinct(uniq, N, rng)]   # indep: distinct
    # misleading
    if cond.startswith("shared"):
        dkeys = list(clues_q["mis"].keys())
        d = dkeys[int(rng.integers(len(dkeys)))]                   # common target distractor
        du = _uniq(clues_q["mis"][d])
        c = du[int(rng.integers(len(du)))]
        return [_frame(c, strength)] * N
    # indep: distinct draws over the UNION of all distractor variants → spreads agents
    # across distractor-direction AND phrasing. Pool is small (~3.3 unique/distractor);
    # without-replacement is already max diversity — we do NOT synthesise extra clues,
    # matched evidence distribution matters more than pushing the control to D=0.
    all_mis = _uniq([v for pool in clues_q["mis"].values() for v in pool])
    return [_frame(c, strength) for c in _draw_distinct(all_mis, N, rng)]


# ── answering: one condition on one question, N independent agents, 1 round ─────
def answer_question(question: str, opts: list[str], evidence: list[str | None],
                    agents: list[DebateAgent], rng: np.random.Generator) -> np.ndarray:
    shuffled = list(opts)
    rng.shuffle(shuffled)
    perm = [shuffled.index(o) for o in opts]              # shuffled → canonical columns
    res: list = [None] * len(agents)
    with ThreadPoolExecutor(max_workers=min(8, len(agents))) as ex:
        futs = {ex.submit(a.elicit_belief, question, shuffled, None, evidence[i]): i
                for i, a in enumerate(agents)}
        for f in futs:
            i = futs[f]
            try:
                res[i] = f.result()
            except Exception:
                res[i] = np.zeros(len(shuffled))
    return np.stack(res, axis=0)[:, perm]                 # (N,K) canonical, truth at 0


# ── metrics ─────────────────────────────────────────────────────────────────────
def pooled_pair_corr(E: np.ndarray) -> float:
    """Primary D for ONE question: replicate-level pooled pairwise error correlation.

    E is (R, N) binary (1 = agent wrong). Using Var(ΣE) = Σ Var(E_i) + Σ_{i≠j}Cov,
        D = [Var(S) − Σ Var(E_i)] / [(Σ sd_i)² − Σ sd_i²]
          = Σ_{i≠j} Cov_ij / Σ_{i≠j} sd_i sd_j   (avg pairwise Pearson, sd-weighted).
    Defined whenever ≥2 agents vary across the R replicates; NaN if the arm saturates
    (every agent constant → error dependence is unmeasurable, which we surface).
    Unlike the single-realization formula this is NOT forced to −1/(N−1): the margin
    varies across replicates, so a common shock in C shows up as positive covariance."""
    if E.shape[0] < 2:
        return float("nan")
    var_i = E.var(axis=0)                                  # population var per agent (ddof=0)
    sd = np.sqrt(var_i)
    num = E.sum(axis=1).var() - var_i.sum()                # Σ_{i≠j} Cov_ij
    den = sd.sum() ** 2 - (sd ** 2).sum()                  # Σ_{i≠j} sd_i sd_j
    return float(num / den) if den > 1e-12 else float("nan")


def fleiss_icc(counts: list[tuple[int, int]]) -> float:
    counts = [(n, c) for n, c in counts if n >= 1]
    Q, Nt = len(counts), sum(n for n, _ in counts)
    if Q < 2 or Nt <= Q:
        return 0.0
    ybar = sum(c for _, c in counts) / Nt
    ssw = sum(c * (1 - c / n) for n, c in counts)
    ssb = sum(n * (c / n - ybar) ** 2 for n, c in counts)
    msw, msb = ssw / (Nt - Q), ssb / (Q - 1)
    m0 = (Nt - sum(n * n for n, _ in counts) / Nt) / (Q - 1)
    den = msb + (m0 - 1) * msw
    return float((msb - msw) / den) if den > 0 else 0.0


def pair_agreement(votes_per_q: list[np.ndarray]) -> float:
    vals = []
    for v in votes_per_q:
        if len(v) >= 2:
            agree = (v[:, None] == v[None, :])
            iu = np.triu_indices(len(v), 1)
            vals.append(agree[iu].mean())
    return float(np.mean(vals)) if vals else 0.0


def condition_metrics(states: list[np.ndarray], rng) -> dict:
    single_c = single_n = 0
    plur, counts, votes_q = [], [], []
    for x in states:
        info = x.max(1) > x.min(1)
        v = np.argmax(x[info], 1) if info.any() else np.array([], dtype=int)
        votes_q.append(v)
        single_c += int((v == 0).sum())
        single_n += int(info.sum())
        counts.append((int(info.sum()), int((v == 0).sum())))
        plur.append(int(plurality_prediction(x, rng) == 0))
    return {"plur": np.asarray(plur, float),
            "p_single": single_c / single_n if single_n else float("nan"),
            "plurality_acc": float(np.mean(plur)),
            "fleiss": fleiss_icc(counts),
            "pair_agree": pair_agreement(votes_q)}


def _nanmean(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(a.mean()) if a.size else float("nan")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(REGIMES), required=True)
    ap.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_gpu0.yaml"))
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--n-agents", type=int, default=11)
    ap.add_argument("--replicates", type=int, default=8,
                    help="stochastic replicates per question×condition (identifies D)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="comma-separated subset of the 7-cell grid (mini-run uses 5)")
    ap.add_argument("--tag", default="replicate", help="output-filename tag")
    ap.add_argument("--model-tag", default="", help="namespace suffix for outputs so "
                    "multi-model P2 runs do not collide (protocol unchanged)")
    ap.add_argument("--benchmark", default="truthfulqa", help="P2b benchmark adapter "
                    "(truthfulqa=legacy frozen path; others via p2_benchmarks registry)")
    ap.add_argument("--fresh-clues", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    N, R = args.n_agents, args.replicates
    mtag = f"_{args.model_tag}" if args.model_tag else ""   # "" ⇒ byte-identical to legacy paths
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    assert all(c in CONDITIONS for c in conds), f"unknown condition in {conds}"
    TABLES.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"EXP6C PREREGISTRATION — regime={args.regime}  N={N}  R={R}  model={cfg['model']}")
    print("FROZEN reading order (do not reorder post-hoc):")
    print(" H1 primary  mean_ΔD_paired = mean_q[ D_shared(q) − D_indep(q) ] > 0 (paired boot)")
    print("             D(q) = replicate-level pooled pairwise ERROR corr over R reps")
    print(" H2 primary  ShareEffect = A_shared − A_indep (plurality acc); easy p>½ ⇒ <0")
    print(" mediation   larger ΔD ⇒ larger destruction — DESCRIPTIVE only (no formal claim)")
    print(" secondary   Fleiss ICC, pair-agreement; D_pooled (per-arm NaN mask) ≠ ΔD estimand")
    print(f"conditions: {conds}")
    print("=" * 72)

    import p2_benchmarks as p2b                       # benchmark-aware (truthfulqa ⇒ legacy)
    questions = p2b.select_questions(args.regime, args.n_questions, args.benchmark)
    print(f"selected {len(questions)} {args.regime} questions [benchmark={args.benchmark}]")
    clues = build_clues(args.regime, questions, cfg, N, args.fresh_clues, args.benchmark)

    limiter = get_rate_limiter(cfg["rpm"]) if cfg.get("rpm") else None
    agents = [DebateAgent(i, model=cfg["model"], backend=cfg.get("backend", "openai"),
                          base_url=cfg.get("base_url"), temperature=cfg.get("temperature", 1.0),
                          rate_limiter=limiter, timeout=cfg.get("timeout", 300))
              for i in range(N)]

    out = TABLES / f"exp6c_{args.tag}_{args.regime}{mtag}_N{N}.csv"
    # D_pooled = mean over THIS condition's non-NaN questions (each arm its own mask); it is
    # a DESCRIPTIVE per-condition summary, NOT a term of the paired ΔD estimand (which is
    # computed on the shared shared∩indep mask in the readout). Named apart on purpose.
    fields = ["regime", "condition", "replicates", "p_single", "plurality_acc",
              "D_pooled", "D_pooled_sd", "n_D", "fleiss", "pair_agree", "n_questions"]
    rows = {}
    D_by_cond, plur_by_cond = {}, {}          # cond → per-question array (len Q)
    for cond in conds:
        ci = CONDITIONS.index(cond)           # stable seed regardless of subset
        states_all, D_q, plur_q = [], [], []
        for qi, (q, opts) in enumerate(questions):
            E_rows, plur_r = [], []
            for r in range(R):
                # display order seeded by QUESTION only → constant across conditions &
                # replicates (clean paired shared-vs-indep); evidence resampled per (qi,r,ci)
                ev = evidence_for(cond, clues[qi], len(opts), N,
                                  np.random.default_rng([qi, r, ci]))
                x = answer_question(q, opts, ev, agents, np.random.default_rng([12345, qi]))
                states_all.append(x)
                E_rows.append((x.argmax(1) != 0).astype(float))   # 1 = wrong (truth@0)
                plur_r.append(int(plurality_prediction(
                    x, np.random.default_rng([7, qi, r, ci])) == 0))
            D_q.append(pooled_pair_corr(np.array(E_rows)))         # (R,N) → scalar
            plur_q.append(float(np.mean(plur_r)))
        D_q, plur_q = np.array(D_q), np.array(plur_q)
        m = condition_metrics(states_all, np.random.default_rng([7, ci]))
        D_by_cond[cond], plur_by_cond[cond] = D_q, plur_q
        dmean = _nanmean(D_q)
        dsd = _nanmean((D_q - dmean) ** 2) ** 0.5
        n_D = int(np.count_nonzero(~np.isnan(D_q)))       # questions with a defined D
        rows[cond] = {"regime": args.regime, "condition": cond, "replicates": R,
                      "p_single": m["p_single"], "plurality_acc": m["plurality_acc"],
                      "D_pooled": dmean, "D_pooled_sd": dsd, "n_D": n_D,
                      "fleiss": m["fleiss"], "pair_agree": m["pair_agree"],
                      "n_questions": len(questions)}
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows.values())
        print(f"[{cond:26s}] p₁={m['p_single']:.2f} P_maj={m['plurality_acc']:.2f} "
              f"D_pooled={dmean:+.3f}(n={n_D}) fleiss={m['fleiss']:.2f} agree={m['pair_agree']:.2f}")

    # ── H1 (ΔD) / H2 (ShareEffect) via the shared-vs-independent control ────────
    print("\n" + "-" * 72)
    p1 = rows[conds[0]]["p_single"]
    regime_sign = "p>½ → expect ShareEffect<0" if p1 > 0.5 else "p<½ → expect ShareEffect>0"
    print(f"realized single-agent p ({conds[0]}) = {p1:.3f}  [{regime_sign}]")
    for qual in ("correct", "misleading"):
        sc, ic = f"shared_{qual}_strong", f"indep_{qual}_strong"
        if sc not in D_by_cond or ic not in D_by_cond:
            continue
        Ds, Di = D_by_cond[sc], D_by_cond[ic]
        # H1 primary: ΔD on the PAIRED subset (both arms have a defined D on that question).
        # Component means are reported on THIS SAME subset, so D_shared − D_indep == ΔD
        # exactly. The per-condition D_pooled above uses each arm's OWN mask → a different
        # estimand; we print it too, tagged [pooled], so no one reads a phantom subtraction.
        mask = ~(np.isnan(Ds) | np.isnan(Di))
        n_pair = int(mask.sum())
        if n_pair:
            dsub, isub = Ds[mask], Di[mask]
            dD = dsub - isub
            lo_d, hi_d = bootstrap_ci(dD, np.mean, n_boot=10000,
                                      rng=np.random.default_rng([42, len(qual)]))
            sig_d = "*" if lo_d > 0 else " "
            print(f"[{qual:10s}] paired(n={n_pair}): D_shared={dsub.mean():+.3f} "
                  f"D_indep={isub.mean():+.3f}  mean_ΔD_paired={dD.mean():+.3f} "
                  f"[{lo_d:+.3f},{hi_d:+.3f}]{sig_d}  H1={'OK' if lo_d > 0 else 'no'}")
            print(f"             [pooled, per-arm mask: D_shared={_nanmean(Ds):+.3f} "
                  f"(n={int(np.count_nonzero(~np.isnan(Ds)))})  "
                  f"D_indep={_nanmean(Di):+.3f} (n={int(np.count_nonzero(~np.isnan(Di)))})]")
        # H2 primary outcome: ShareEffect = A_shared − A_indep (plurality acc), paired by q
        diff = plur_by_cond[sc] - plur_by_cond[ic]
        lo, hi = bootstrap_ci(diff, np.mean, n_boot=10000,
                              rng=np.random.default_rng([99, len(qual)]))
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"             ShareEffect(A shared−indep)={float(diff.mean()):+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]{sig}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
