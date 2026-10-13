"""Exp 6: the real-LLM herding↔truth phase transition (Direction 3).

Sweeps the social-coupling knob g ∈ [0,1] (orchestrator.coupling) over a fixed
question set. g scales how much of the population each agent sees, i.e. the
social-influence matrix B in the linearised dynamics; increasing it drives
M = A - B toward the herding boundary. For each g we measure, relative to the
INDEPENDENT round-0 aggregate (= the private-evidence term s, available before any
interaction):

  indep_acc  : round-0 plurality is correct         (no coupling yet)
  debate_acc : final-round plurality is correct
  lift       : debate_acc - indep_acc               (does interaction help or hurt?)
  collapse   : 1 - Var_final / Var_0                 (consensus collapse = herding)

The predicted phenomenon: lift stays ≥0 while g is small, then flips sharply
negative past a critical g* as collapse saturates — interaction erodes truth. This
is the real-LLM analogue of exp3's synthetic b-sweep, and the g* it locates is the
control target for Direction 2 (keep g below g*, or intervene when past it).

Usage:
  python experiments/exp6_phase_transition_llm/run.py --benchmark aqua \
      --config src/config/debate_local.yaml --n-questions 20
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

from mfg_llm.llm_debate.benchmarks import (
    load_aqua, load_truthfulqa_mc1, load_mc_jsonl, run_debate_canonical, plurality_prediction,
)
from mfg_llm.metrics.stats import bootstrap_ci
from mfg_llm.opinion_state import logits_to_belief

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "data" / "raw" / "benchmarks"
# The two legacy entries are kept VERBATIM: qwen's 473 cached `exp6_{truthfulqa,aqua}_N11_g*.npz`
# were generated through exactly these calls, and any change to the loader call (e.g. routing
# truthfulqa through p2_benchmarks' explicit `k=4`) risks a different question set and a silent
# cache mismatch. The scale-up benchmarks are added through the same generic JSONL loader
# `p2_benchmarks` uses, so the copying axis can reach them; `_assert_loader_parity` below guards
# against the two registries drifting apart again.
LOADERS = {
    "aqua": lambda n: load_aqua(BENCH_DIR / "aqua_test.json", max_questions=n),
    "truthfulqa": lambda n: load_truthfulqa_mc1(BENCH_DIR / "truthfulqa.csv", max_questions=n),
    "arc": lambda n: load_mc_jsonl(BENCH_DIR / "arc_challenge.jsonl", n),
    "commonsenseqa": lambda n: load_mc_jsonl(BENCH_DIR / "commonsenseqa.jsonl", n),
    "mmlu": lambda n: load_mc_jsonl(BENCH_DIR / "mmlu.jsonl", n),
    "strategyqa": lambda n: load_mc_jsonl(BENCH_DIR / "strategyqa.jsonl", n),
    "bbh": lambda n: load_mc_jsonl(BENCH_DIR / "bbh.jsonl", n),
}


def _assert_loader_parity() -> None:
    """Fail loudly if this file's benchmark registry drifts from `p2_benchmarks.BENCHMARKS`.
    The two existed independently and silently disagreed: `run.py` offered only
    {aqua, truthfulqa} while the boundary tier ran seven, which is why the copying axis
    could not reach strategyqa at all."""
    try:
        import p2_benchmarks as p2b
    except Exception:
        return                                   # parity check is best-effort, never fatal
    missing = sorted(set(p2b.BENCHMARKS) - set(LOADERS))
    if missing:
        raise SystemExit(f"run.py LOADERS is missing benchmarks that p2_benchmarks has: {missing}")
COUPLING_GRID = [0.0, 0.5, 1.0, 2.0]


def _dispersion(state: np.ndarray) -> float:
    """Cross-agent disagreement at one round = mean pairwise total-variation
    distance between agents' belief distributions (0 = identical, 1 = disjoint).

    The opinion state is a *centered-logit* vector (scores − mean, scale ±10), not
    a distribution, so its raw variance grows just because the model gets more
    confident — comparing it across rounds is meaningless (it produced collapse
    values of −55 and NaN). We first map each agent to the belief simplex via
    softmax, where dispersion is bounded in [0, 1] and comparable across rounds."""
    p = logits_to_belief(np.asarray(state, dtype=np.float64))   # (n_agents, K)
    n = p.shape[0]
    if n < 2:
        return 0.0
    d = 0.5 * np.abs(p[:, None, :] - p[None, :, :]).sum(axis=-1)  # pairwise TV
    iu = np.triu_indices(n, k=1)
    return float(d[iu].mean())


def _collapse(states: list[np.ndarray]) -> float:
    """Consensus collapse (herding) = how much the agents' beliefs converged from
    the independent round 0 to the final round: dispersion_0 − dispersion_final,
    measured on the belief simplex. >0 = agents came together (herding), <0 = they
    pulled apart (polarisation), ≈0 = unchanged. Bounded in [−1, 1]; no division,
    so it never blows up or returns NaN the way 1 − Var_final/Var_0 did."""
    return _dispersion(states[0]) - _dispersion(states[-1])


def _single_correct(x_round: np.ndarray) -> tuple[int, int]:
    """(#correct, #informative) over individual agents at one round — truth at
    canonical index 0. This is the *marginal* single-agent accuracy, the quantity p
    in the Condorcet-with-correlation law P_maj(N,p,ρ). Measured on the REALIZED
    round-0 debate states (not the pre-probe p̂, which carries selection / regression-
    to-the-mean noise from the easy/hard filter), so it is an unbiased readout of the
    regime the subset actually lands in."""
    info = x_round.max(axis=1) > x_round.min(axis=1)
    if not info.any():
        return 0, 0
    correct = int((np.argmax(x_round[info], axis=1) == 0).sum())
    return correct, int(info.sum())


def _fleiss_icc(counts: list[tuple[int, int]]) -> float:
    """Measured vote correlation ρ = one-way random-effects (Fleiss) intraclass
    correlation of the binary correctness indicator across agents, pooled over
    questions. Each question q contributes m_q informative agents of whom c_q are
    correct. Under the equicorrelated jury model this ICC IS the ρ of P_maj(N,p,ρ):
    it compares the between-question spread of correct-counts (MSB) to the within-
    question binomial spread (MSW). ρ≈0 when agents vote independently, ρ→1 when the
    group answers as one. (Caveat: heterogeneous per-question difficulty inflates the
    g=0 baseline, so read ρ(g)−ρ(0) as the coupling-induced correlation.)"""
    counts = [(m, c) for m, c in counts if m >= 1]
    Q = len(counts)
    if Q < 2:
        return 0.0
    Ntot = sum(m for m, _ in counts)
    if Ntot <= Q:                                   # need within-group df > 0
        return 0.0
    ybar = sum(c for _, c in counts) / Ntot
    ssw = sum(c * (1.0 - c / m) for m, c in counts)          # Σ m·p̂(1-p̂)
    ssb = sum(m * (c / m - ybar) ** 2 for m, c in counts)     # Σ m·(p̂-ȳ)²
    msw = ssw / (Ntot - Q)
    msb = ssb / (Q - 1)
    m0 = (Ntot - sum(m * m for m, _ in counts) / Ntot) / (Q - 1)
    denom = msb + (m0 - 1.0) * msw
    if denom <= 0:
        return 0.0
    return float((msb - msw) / denom)


def _probe_p(question: str, options: list[str], cfg: dict, k: int,
             rng: np.random.Generator) -> float:
    """Single-agent accuracy p̂ = fraction of k INDEPENDENT one-round votes that are
    correct (truth is at canonical index 0). Reuses the debate machinery with
    coupling=0 and n_rounds=1, so the probe is genuinely independent of — and
    kept separate from — the debate used to score lift (no selection-on-evaluation
    bias). A question with no informative vote counts as p̂=0."""
    pcfg = copy.deepcopy(cfg)
    pcfg["coupling"] = 0.0
    st = run_debate_canonical(question, options, pcfg, rng, n_agents=k, n_rounds=1)
    x0 = st[0]
    informative = x0.max(axis=1) > x0.min(axis=1)
    if not informative.any():
        return 0.0
    return float((np.argmax(x0[informative], axis=1) == 0).mean())


def _select_easy(pool: list, cfg: dict, args) -> list:
    """Pre-probe the candidate pool and keep the first --n-questions with p̂ ≥ --min-p.
    Cached to a json so re-runs skip the probe."""
    base_seed = int(cfg.get("seed", 0))
    # The probe measures THIS model's competence (p̂ from _probe_p under cfg), so the
    # cache must be namespaced by model: serving one model's easy-set to another selects
    # on the wrong model's competence. `--model-tag ""` ⇒ byte-identical to legacy paths.
    mtag = f"_{args.model_tag}" if getattr(args, "model_tag", "") else ""
    cache = (ROOT / "data" / "processed" /
             f"exp6_easy_{args.benchmark}{mtag}_p{args.min_p}-{args.max_p}_k{args.probe_k}"
             f"_c{len(pool)}.json")
    if cache.exists() and not args.fresh:
        idx = json.loads(cache.read_text())["selected"]
        print(f"selection: reusing {cache.name} → {len(idx)} easy questions")
        return [pool[i] for i in idx]
    print(f"probing {len(pool)} candidates (k={args.probe_k}) for "
          f"{args.min_p} ≤ p̂ ≤ {args.max_p} ...")
    selected, phats = [], []
    for qi, (question, options) in enumerate(pool):
        rng = np.random.default_rng([base_seed, 9999, qi])   # probe RNG ≠ debate RNG
        p = _probe_p(question, options, cfg, args.probe_k, rng)
        phats.append(p)
        if args.min_p <= p <= args.max_p:
            selected.append(qi)
        if len(selected) >= args.n_questions:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"selected": selected,
                                 "phat": {i: phats[i] for i in range(len(phats))}}))
    kept_mean = np.mean([phats[i] for i in selected]) if selected else float("nan")
    print(f"selection: {len(selected)}/{qi + 1} probed passed "
          f"{args.min_p} ≤ p̂ ≤ {args.max_p} (mean p̂ of kept = {kept_mean:.2f})")
    return [pool[i] for i in selected]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=sorted(LOADERS), required=True)
    parser.add_argument("--model-tag", default="",
                        help="namespace suffix for the probe-selection cache and the per-question "
                             ".npz states, so two models cannot silently share each other's "
                             "cached debates. Empty (default) reproduces the legacy paths exactly. "
                             "Mirrors exp6c_boundary.py's --model-tag.")
    parser.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_local.yaml"))
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--grid", type=float, nargs="+", default=COUPLING_GRID)
    parser.add_argument("--fresh", action="store_true",
                        help="recompute every debate, ignoring cached .npz states")
    parser.add_argument("--n-agents", type=int, default=None,
                        help="override cfg n_agents (deeper N sharpens the p>½ flip)")
    parser.add_argument("--min-p", type=float, default=None,
                        help="keep only questions whose single-agent accuracy p̂ ≥ this "
                             "(the p>½ regime where herding erodes truth). Requires a "
                             "pre-probe; needs --candidates > --n-questions to have a pool.")
    parser.add_argument("--max-p", type=float, default=1.01,
                        help="upper bound on p̂ — exclude trivially-easy (p̂≈1) questions "
                             "that leave no room for herding to hurt; the flip is deepest "
                             "for p̂ moderately above ½ (≈0.6–0.8).")
    parser.add_argument("--probe-k", type=int, default=5,
                        help="independent single-agent samples per candidate for p̂")
    parser.add_argument("--candidates", type=int, default=None,
                        help="how many questions to probe before selecting n-questions")
    parser.add_argument("--tag", default=None,
                        help="label for the output CSV (e.g. 'hard'/'easy') so the two "
                             "realized-p regimes don't overwrite each other")
    args = parser.parse_args(argv)
    _assert_loader_parity()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.n_agents is not None:
        cfg["n_agents"] = args.n_agents
    N, R = cfg["n_agents"], cfg["n_rounds"]

    n_load = args.candidates if args.min_p is not None else args.n_questions
    pool = LOADERS[args.benchmark](n_load)
    questions = _select_easy(pool, cfg, args) if args.min_p is not None else pool
    total = len(args.grid) * len(questions) * (N * R)
    print(f"benchmark={args.benchmark}  questions={len(questions)}  "
          f"coupling grid={args.grid}  ~{total} debate calls (N={N}, R={R})\n")

    states_dir = ROOT / "data" / "processed"
    tables_dir = ROOT / "results" / "tables"
    states_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if args.tag is not None:
        tag = f"_N{N}_{args.tag}"
    elif args.min_p is not None:
        tag = f"_N{N}_easy{args.min_p}"
    else:
        tag = f"_N{N}"
    out = tables_dir / f"exp6_phase_transition_llm_{args.benchmark}{tag}.csv"
    base_seed = int(cfg.get("seed", 0))
    # p_single = realized single-agent accuracy (the theory's p); lift CI + mcnemar_p
    # are PAIRED per question (debate_correct − indep_correct on the same question,
    # same display shuffle); rho_measured = Fleiss ICC of final-round correctness.
    fields = ["coupling", "p_single", "indep_acc", "debate_acc", "lift",
              "lift_ci_low", "lift_ci_high", "mcnemar_p",
              "rho_measured", "mean_collapse", "n"]

    def write_csv(rows: list[dict]) -> None:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    rows = []
    for g in args.grid:
        gcfg = copy.deepcopy(cfg)
        gcfg["coupling"] = g
        indep, debate, collapse = [], [], []
        single_correct, single_n = 0, 0          # realized single-agent p (round 0)
        final_counts: list[tuple[int, int]] = []  # (informative, correct) at final round → ρ
        for qi, (question, options) in enumerate(questions):
            # Cache key includes an 8-char hash of the question TEXT, not just the
            # positional index qi: different --min-p/--max-p bands select different
            # questions at the same qi, so a text-blind key silently served a prior
            # run's states for the wrong question. The hash makes a stale hit impossible.
            # The hash keys the QUESTION, not the model: without a model namespace a
            # second model's run silently loads the first model's cached states (the
            # same class of stale hit the hash was added to kill, one axis over).
            # `--model-tag ""` ⇒ byte-identical to legacy paths, so qwen's existing
            # `exp6_<bench>_N11_g*.npz` and the analysis globs over them are untouched.
            qh = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
            mtag = f"_{args.model_tag}" if args.model_tag else ""
            npz_path = (states_dir /
                        f"exp6_{args.benchmark}{mtag}_N{N}_g{g:.2f}_q{qi}_{qh}.npz")
            if npz_path.exists() and not args.fresh:
                st = list(np.load(npz_path, allow_pickle=True)["states"])
            else:
                # Seed per (benchmark, g, question) so the run is reproducible AND
                # resumable at question granularity: skipping a cached question no
                # longer shifts the RNG stream of the ones still to compute. Same
                # (seed, qi) across g also means each question sees an identical
                # display shuffle at every coupling → clean paired comparison.
                rng = np.random.default_rng([base_seed, qi])
                st = run_debate_canonical(question, options, gcfg, rng, n_agents=N, n_rounds=R)
                np.savez(npz_path, states=np.stack(st, axis=0), question=question,
                         answer_space=np.array(options), coupling=g)
            indep.append(int(plurality_prediction(st[0]) == 0))
            debate.append(int(plurality_prediction(st[-1]) == 0))
            collapse.append(_collapse(st))
            sc, sn = _single_correct(st[0])
            single_correct += sc
            single_n += sn
            fc = _single_correct(st[-1])
            if fc[1] >= 1:
                final_counts.append((fc[1], fc[0]))

        indep_a = np.asarray(indep, dtype=np.float64)
        debate_a = np.asarray(debate, dtype=np.float64)
        diff = debate_a - indep_a                        # paired per-question effect
        lift = float(diff.mean())
        # Paired bootstrap CI over questions (resamples the diff vector) — the honest
        # CI for a within-question effect, unlike two independent accuracy CIs.
        boot_rng = np.random.default_rng([base_seed, 4242])
        lo, hi = bootstrap_ci(diff, np.mean, n_boot=10000, rng=boot_rng)
        # McNemar exact: discordant pairs only. b = debate broke a right indep answer,
        # c = debate fixed a wrong one; H0: b and c equally likely.
        b = int(np.sum((indep_a == 1) & (debate_a == 0)))
        c = int(np.sum((indep_a == 0) & (debate_a == 1)))
        mcnemar_p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
        p_single = single_correct / single_n if single_n else float("nan")
        rho = _fleiss_icc(final_counts)

        rows.append({"coupling": g, "p_single": p_single,
                     "indep_acc": float(indep_a.mean()), "debate_acc": float(debate_a.mean()),
                     "lift": lift, "lift_ci_low": lo, "lift_ci_high": hi,
                     "mcnemar_p": mcnemar_p, "rho_measured": rho,
                     "mean_collapse": float(np.mean(collapse)), "n": len(indep)})
        write_csv(rows)  # checkpoint after every g so an interruption keeps finished rows
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"g={g:.2f} | p₁={p_single:.2f} indep={indep_a.mean():.2f} "
              f"debate={debate_a.mean():.2f} | lift={lift:+.2f} "
              f"[{lo:+.2f},{hi:+.2f}]{sig} McN={mcnemar_p:.3f} | "
              f"ρ={rho:.2f} collapse={np.mean(collapse):+.2f}")

    if not rows:
        print("no grid points completed — nothing written")
        return
    print(f"\nwrote {out}")

    # FLIP-law readout: sign of the collective effect should track sign(½ − p_single).
    p0 = rows[0]["p_single"]
    regime = "EASY (p>½ → predict lift<0 as ρ↑)" if p0 > 0.5 else \
             "HARD (p<½ → predict lift>0 as ρ↑)" if p0 < 0.5 else "BORDERLINE (p≈½)"
    print(f"realized regime: p_single≈{p0:.2f} → {regime}")
    strong = [r for r in rows if r["rho_measured"] >= 0.15 and r["coupling"] > 0]
    if strong:
        r = max(strong, key=lambda r: r["rho_measured"])
        signed = (r["lift"] < 0) == (p0 > 0.5)
        verdict = "MATCHES" if signed and (r["lift_ci_low"] > 0 or r["lift_ci_high"] < 0) \
            else "consistent-but-n.s." if signed else "CONTRADICTS"
        print(f"strongest herding g={r['coupling']:.1f} (ρ={r['rho_measured']:.2f}): "
              f"lift={r['lift']:+.2f} [{r['lift_ci_low']:+.2f},{r['lift_ci_high']:+.2f}] "
              f"→ FLIP law {verdict}")
    else:
        print("herding never pushed ρ≥0.15 — raise coupling (add g≥2) or use a harder subset")


if __name__ == "__main__":
    main()
