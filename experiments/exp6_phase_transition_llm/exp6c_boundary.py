"""Exp6C BOUNDARY runner — same frozen protocol as exp6c_shared_context.py, but with
(a) an on-disk BELIEF CACHE (replay → no LLM re-calls when the estimator changes) and
(b) RAW per-(condition, question, replicate) persistence, and (c) a pre-intervention
`none`-based per-question MARGIN table for the susceptibility/boundary analysis.

Design (per the run spec):
  • Protocol is IDENTICAL to the frozen exp6c: same question selection, same clue cache,
    same evidence draws (rng[qi,r,ci]), same display shuffle (rng[12345,qi]). This file
    imports those functions verbatim so numbers reproduce; it only ADDS persistence.
  • Belief cache key = f"{regime}|{cond}|q{qi}|r{r}|a{agent}" (call-site coordinates, so
    per-replicate stochasticity is preserved — each realized sample is stored once). Each
    record also carries input_hash = sha256(model‖prompt) + question/opts/evidence hashes +
    the shuffled-option mapping, so a protocol change is DETECTED (hash mismatch → re-call),
    never silently served stale. NOTE: the OpenAI-compat request does not ask for logprobs,
    so the finest belief we can persist is the parsed `scores` dict + the opinion vector +
    raw response text (all stored). Add `logprobs:true` to agent._chat to capture token
    logits later — out of scope for this run.
  • round0_margin is computed ONLY from the `none` (pre-intervention) votes and reused for
    the matched shared/indep pair — never measured post-evidence (would leak the treatment).
  • We persist per-question plurality share, runner-up gap, vote entropy, #unique answers
    AND binary-correctness lead — cheap now, needed by K2/CDCR later.
  • Raw per-(cond,qi,r) canonical belief matrices x (truth@0) are dumped so ANY estimator
    (shrinkage / binomial mixed-model on D_q, etc.) can be recomputed offline with no LLM.

Run (both regimes in parallel on the two standing servers):
  easy: --regime easy --config src/config/debate_gpu1.yaml   (:11435, CUDA=1)
  hard: --regime hard --config src/config/debate_gpu0.yaml   (:11436, CUDA=0)
  conditions = none,shared_correct_strong,indep_correct_strong
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

# import the FROZEN protocol machinery verbatim (defs only; main() is __main__-guarded)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp6c_shared_context as e6c  # noqa: E402
from mfg_llm.llm_debate.agent import DebateAgent, get_rate_limiter  # noqa: E402
from mfg_llm.llm_debate.benchmarks import plurality_prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

DEFAULT_CONDS = ["none", "shared_correct_strong", "indep_correct_strong"]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class BeliefCache:
    """Append-only jsonl cache of every agent belief call. Doubles as the RAW belief store
    (text + scores + opinion vector) and the replay cache. Keyed by call-site coordinates;
    input_hash guards against serving a stale sample after a protocol change."""

    def __init__(self, path: Path):
        self.path = path
        self.mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.mem[rec["key"]] = rec
            print(f"belief-cache: loaded {len(self.mem)} records from {path.name}")
        self._fh = open(path, "a")

    def get(self, key: str, input_hash: str) -> dict | None:
        rec = self.mem.get(key)
        if rec is None:
            return None
        if rec.get("input_hash") != input_hash:      # protocol changed under this key
            return None
        return rec

    def put(self, rec: dict) -> None:
        with self._lock:
            self.mem[rec["key"]] = rec
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self):
        self._fh.close()


def cached_answer(question: str, opts: list[str], evidence: list[str | None],
                  agents: list[DebateAgent], model: str, regime: str, cond: str,
                  qi: int, r: int, cache: BeliefCache) -> tuple[np.ndarray, list[int]]:
    """Frozen answer_question, but each agent's belief is served from / written to `cache`.
    Returns (x canonical (N,K) truth@0, shuffle perm shuffled→canonical)."""
    shuffled = list(opts)
    np.random.default_rng([12345, qi]).shuffle(shuffled)     # display order: qi only (frozen)
    perm = [shuffled.index(o) for o in opts]
    N = len(agents)
    res: list = [None] * N
    misses: list[tuple[int, str, str]] = []                  # (i, prompt, input_hash)
    for i, a in enumerate(agents):
        prompt = a._build_prompt(question, shuffled, None, evidence[i])
        key = f"{regime}|{cond}|q{qi}|r{r}|a{i}"
        ih = _sha(model + "\x00" + prompt)
        hit = cache.get(key, ih)
        if hit is not None:
            res[i] = np.asarray(hit["opinion_shuffled"], float)
        else:
            misses.append((i, prompt, ih))

    def _call(i):
        # (x_shuffled, ok): ok=False signals the LLM call raised (e.g. server 404/500 after
        # retries). We still USE zeros for this pass, but must NOT persist them — a cached
        # flat zero would be served forever on replay (this is what corrupted the hard run).
        try:
            return agents[i].elicit_belief(question, shuffled, None, evidence[i]), True
        except Exception:
            return np.zeros(len(shuffled)), False

    if misses:
        with ThreadPoolExecutor(max_workers=min(8, len(misses))) as ex:
            futs = {ex.submit(_call, i): (i, prompt, ih) for i, prompt, ih in misses}
            for f in futs:
                i, prompt, ih = futs[f]
                x_sh_raw, ok = f.result()
                x_sh = np.asarray(x_sh_raw, float)
                res[i] = x_sh
                if not ok:                           # do NOT cache exception-fallback zeros
                    continue
                tr = agents[i].transcript[-1] if agents[i].transcript else {}
                cache.put({
                    "key": f"{regime}|{cond}|q{qi}|r{r}|a{i}", "input_hash": ih,
                    "model": model, "regime": regime, "cond": cond,
                    "qi": qi, "r": r, "agent": i,
                    "question_hash": _sha(question), "opts_hash": _sha("¦".join(opts)),
                    "evidence_hash": _sha(evidence[i] or ""), "shuffled": shuffled,
                    "evidence": evidence[i], "text": tr.get("response"),
                    "scores": tr.get("scores"), "opinion_shuffled": [float(v) for v in x_sh],
                })
    x = np.stack(res, axis=0)[:, perm]                       # (N,K) canonical, truth@0
    return x, perm


def margin_features(x_reps: list[np.ndarray], rng_seed: list[int]) -> dict:
    """Pre-intervention per-question margin from the `none` condition's canonical states
    (list over replicates of (N,K), truth@0). All measures are computed on the POOLED
    informative votes across replicates × agents."""
    votes = []
    plur = []
    for ri, x in enumerate(x_reps):
        info = x.max(1) > x.min(1)
        v = np.argmax(x[info], 1) if info.any() else np.array([], int)
        votes.extend(int(t) for t in v)
        plur.append(int(plurality_prediction(x, np.random.default_rng(rng_seed + [ri])) == 0))
    K = x_reps[0].shape[1]
    counts = np.bincount(votes, minlength=K).astype(float) if votes else np.zeros(K)
    tot = counts.sum()
    shares = counts / tot if tot else counts
    order = np.sort(shares)[::-1]
    top = float(order[0]) if tot else 0.0
    second = float(order[1]) if (tot and len(order) > 1) else 0.0
    nz = shares[shares > 0]
    entropy = float(-(nz * np.log2(nz)).sum()) if nz.size else 0.0
    p_correct_votes = float(shares[0]) if tot else 0.0
    wrong_lead = float(shares[1:].max()) if (tot and K > 1) else 0.0
    A_none = float(np.mean(plur)) if plur else float("nan")
    return {
        "n_votes": int(tot),
        "plurality_share": top,
        "runnerup_gap": top - second,          # PRIMARY margin M_q (collective vote margin)
        "vote_entropy": entropy,
        "n_unique": int((counts > 0).sum()),
        "p_correct_votes": p_correct_votes,
        "correctness_lead": p_correct_votes - wrong_lead,   # signed truth-vs-best-wrong lead
        "A_none": A_none,                       # plurality accuracy of the crowd, no evidence
        "dist_to_boundary": abs(A_none - 0.5),  # |P_maj − ½| susceptibility proxy χ
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(e6c.REGIMES), required=True)
    ap.add_argument("--config", default=str(ROOT / "src" / "config" / "debate_gpu0.yaml"))
    ap.add_argument("--n-questions", type=int, default=40)
    ap.add_argument("--n-agents", type=int, default=11)
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--conditions", default=",".join(DEFAULT_CONDS))
    ap.add_argument("--tag", default="boundary")
    ap.add_argument("--model-tag", default="", help="namespace suffix for cache+outputs so "
                    "multi-model P2 runs do not collide/overwrite (protocol unchanged)")
    ap.add_argument("--benchmark", default="truthfulqa", help="P2b benchmark adapter "
                    "(truthfulqa=legacy frozen path; others via p2_benchmarks registry)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    N, R = args.n_agents, args.replicates
    model = cfg["model"]
    mtag = f"_{args.model_tag}" if args.model_tag else ""   # "" ⇒ byte-identical to legacy paths
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    assert all(c in e6c.CONDITIONS for c in conds), f"unknown condition in {conds}"
    assert "none" in conds, "need `none` for the pre-intervention margin"
    TABLES.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"EXP6C BOUNDARY — regime={args.regime} N={N} R={R} model={model}")
    print(f"conditions={conds}  config={Path(args.config).name}  base_url={cfg.get('base_url')}")
    print("persist: belief-cache jsonl + raw per-(cond,q,r) states + none-margin table")
    print("=" * 72)

    import p2_benchmarks as p2b                       # benchmark-aware (truthfulqa ⇒ legacy)
    questions = p2b.select_questions(args.regime, args.n_questions, args.benchmark)
    print(f"selected {len(questions)} {args.regime} questions [benchmark={args.benchmark}]")
    clues = e6c.build_clues(args.regime, questions, cfg, N, False, args.benchmark)

    limiter = get_rate_limiter(cfg["rpm"]) if cfg.get("rpm") else None
    agents = [DebateAgent(i, model=model, backend=cfg.get("backend", "openai"),
                          base_url=cfg.get("base_url"), temperature=cfg.get("temperature", 1.0),
                          rate_limiter=limiter, timeout=cfg.get("timeout", 300))
              for i in range(N)]

    cache = BeliefCache(PROC / f"exp6c_belief_cache_{args.regime}{mtag}.jsonl")

    # per-(cond) → per-question arrays; raw[cond][qi] = list over r of (N,K) canonical
    D_by_cond, plur_by_cond, raw = {}, {}, {}
    csv_rows = {}
    csv_fields = ["regime", "condition", "replicates", "p_single", "plurality_acc",
                  "D_pooled", "D_pooled_sd", "n_D", "fleiss", "pair_agree", "n_questions"]
    csv_out = TABLES / f"exp6c_{args.tag}_{args.regime}{mtag}_N{N}.csv"

    for cond in conds:
        ci = e6c.CONDITIONS.index(cond)                      # stable seed regardless of subset
        states_all, D_q, plur_q, raw[cond] = [], [], [], []
        for qi, (q, opts) in enumerate(questions):
            E_rows, plur_r, x_reps = [], [], []
            for r in range(R):
                ev = e6c.evidence_for(cond, clues[qi], len(opts), N,
                                      np.random.default_rng([qi, r, ci]))   # frozen seed
                x, _ = cached_answer(q, opts, ev, agents, model, args.regime,
                                     cond, qi, r, cache)
                states_all.append(x)
                x_reps.append(x)
                E_rows.append((x.argmax(1) != 0).astype(float))
                plur_r.append(int(plurality_prediction(
                    x, np.random.default_rng([7, qi, r, ci])) == 0))
            D_q.append(e6c.pooled_pair_corr(np.array(E_rows)))
            plur_q.append(float(np.mean(plur_r)))
            raw[cond].append([[[float(v) for v in row] for row in x] for x in x_reps])
            print(f"  [{cond:22s} q{qi:02d}] plur={plur_q[-1]:.2f} D={D_q[-1]:+.3f}", flush=True)
        D_q, plur_q = np.array(D_q), np.array(plur_q)
        m = e6c.condition_metrics(states_all, np.random.default_rng([7, ci]))
        D_by_cond[cond], plur_by_cond[cond] = D_q, plur_q
        dmean = e6c._nanmean(D_q)
        dsd = e6c._nanmean((D_q - dmean) ** 2) ** 0.5
        csv_rows[cond] = {"regime": args.regime, "condition": cond, "replicates": R,
                          "p_single": m["p_single"], "plurality_acc": m["plurality_acc"],
                          "D_pooled": dmean, "D_pooled_sd": dsd,
                          "n_D": int(np.count_nonzero(~np.isnan(D_q))),
                          "fleiss": m["fleiss"], "pair_agree": m["pair_agree"],
                          "n_questions": len(questions)}
        with open(csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields)
            w.writeheader()
            w.writerows(csv_rows.values())
        print(f"[{cond:22s}] p1={m['p_single']:.2f} P_maj={m['plurality_acc']:.2f} "
              f"D={dmean:+.3f} fleiss={m['fleiss']:.2f} agree={m['pair_agree']:.2f}", flush=True)

    # ── per-question analysis table (margin from `none`, joined to shared/indep) ──
    margins = [margin_features([np.asarray(x) for x in raw["none"][qi]], [7, qi])
               for qi in range(len(questions))]
    per_q = []
    for qi, (q, opts) in enumerate(questions):
        rec = {"qi": qi, "question": q, "n_options": len(opts), "margin": margins[qi]}
        for cond in conds:
            rec[cond] = {"A_q": float(plur_by_cond[cond][qi]),
                         "D_q": (None if np.isnan(D_by_cond[cond][qi])
                                 else float(D_by_cond[cond][qi]))}
        per_q.append(rec)

    analysis = {"regime": args.regime, "N": N, "R": R, "model": model,
                "model_tag": args.model_tag, "conditions": conds,
                "n_questions": len(questions), "per_q": per_q}
    a_out = TABLES / f"exp6c_boundary_{args.regime}{mtag}_N{N}.json"
    a_out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    raw_out = PROC / f"exp6c_raw_{args.regime}{mtag}_N{N}.json"
    raw_out.write_text(json.dumps({"regime": args.regime, "N": N, "R": R,
                                   "conditions": conds, "raw": raw}, ensure_ascii=False))
    cache.close()
    print(f"\nwrote {a_out}\nwrote {raw_out}\nwrote {csv_out}\n"
          f"belief-cache: {len(cache.mem)} records in {cache.path.name}")


if __name__ == "__main__":
    main()
