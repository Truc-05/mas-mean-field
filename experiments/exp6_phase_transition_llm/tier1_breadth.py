"""Tier 1 — breadth screen: 6 models x 6 benchmarks, C1/C2/C3 via exp6c_boundary.

Fixed protocol (NO per-cell tuning): N=11, R=8, conditions none/shared/indep, same estimators.
One model loaded at a time on the GPU1 server (:11445). Resumable (skips existing outputs),
continue-on-error (a dead cell does not stop the sweep). Only runs a (benchmark, regime) cell
whose qwen-anchor probe bin has >= MIN_Q questions. Truthfulqa/aqua use their frozen splits.

Usage:
  python tier1_breadth.py --dry-run          # print the plan (which cells, why skipped)
  python tier1_breadth.py                     # execute (server :11445 must be up)
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = ROOT / "src" / "config"
TABLES = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"
PY = "/home/alex/trucn/venv/bin/python"

MODELS = ["qwen", "llama", "gemma", "mistral", "phi4", "deepseek"]
# bbh DROPPED 2026-10-03 after the qwen cells came in. All 80 selected bbh items (40 easy +
# 40 hard) are the single `logical_deduction` subtask -- closed constraint puzzles whose
# answer is fully determined by premises already inside the prompt. No external evidence can
# be informative there, and exp6c_shared_context._gen_toward can only emit plausible world
# knowledge, so it produced character gossip orthogonal to the ordering constraints ("Mel was
# seen lingering more at the clubhouse"). Injecting it moves the model off deduction and onto
# plausibility-weighing: A_none 0.634 -> A_shared 0.422 (easy), 0.438 -> 0.388 (hard), the only
# benchmark where correct-pointing evidence LOWERS accuracy (aqua, also closed-form, gains
# +0.084 -- a math clue can legitimately encode the answer, an ordering clue cannot).
# This cannot be repaired by regenerating clues: the task class admits no valid stimulus.
# It matters because it fails SILENTLY IN THE SUPPORTING DIRECTION -- both arms degrade from a
# broken baseline, so dA lands at +0.009/-0.013 with CIs covering 0 and reads as a clean
# Theorem-4 signature when it is really two equally damaged arms. The 2 completed qwen/bbh
# cells are kept on disk but excluded by tier1_breadth_aggregate.EXCLUDED_BENCHMARKS.
BENCHMARKS = ["truthfulqa", "aqua", "arc", "commonsenseqa", "mmlu", "strategyqa"]
REGIMES = ["easy", "hard"]
N, R, MIN_Q = 11, 8, 15


def probe_count(benchmark, regime):
    """How many questions land in this regime's bin (anchor probe).

    truthfulqa used to short-circuit to `return 40` on the assumption that a frozen legacy
    split was always populated. That assumption died with the English rebuild (2026-10-01),
    when `p2_benchmarks.select_questions` was unified onto the single `p2_probe` path: the
    short-circuit would have reported a populated bin for a benchmark whose probe cache did
    not exist, so the driver would launch the cell and only then fail inside the child.
    All seven benchmarks are now counted from their own probe cache.
    """
    cache = PROC / f"exp6_select_{benchmark}.json"
    if not cache.exists():
        return -1  # probe not run yet
    phat = json.loads(cache.read_text())["phat"].values()
    thr = (lambda p: p >= 0.80) if regime == "easy" else (lambda p: p <= 0.30)
    return sum(1 for p in phat if thr(p))


def out_path(model, benchmark, regime):
    tag = f"scaleup_{model}_{benchmark}"
    return TABLES / f"exp6c_boundary_{regime}_{tag}_N{N}.json"


def cells():
    for model in MODELS:
        for b in BENCHMARKS:
            for reg in REGIMES:
                yield model, b, reg


def run_cell(model, b, reg, nq):
    tag = f"scaleup_{model}_{b}"
    cmd = [PY, str(HERE / "exp6c_boundary.py"), "--regime", reg,
           "--config", str(CFG / f"scaleup_{model}_gpu1.yaml"),
           "--benchmark", b, "--n-questions", str(min(40, nq)),
           "--n-agents", str(N), "--replicates", str(R), "--model-tag", tag]
    env = {"PYTHONPATH": str(ROOT / "src")}
    import os
    e = {**os.environ, **env}
    t0 = time.time()
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    dt = time.time() - t0
    ok = r.returncode == 0 and out_path(model, b, reg).exists()
    return ok, dt, (r.stderr[-400:] if not ok else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=str(ROOT / "tier1_breadth.log"))
    args = ap.parse_args()
    log = open(args.log, "a")

    def say(m):
        print(m); print(m, file=log, flush=True)

    plan, skip = [], []
    for model, b, reg in cells():
        if out_path(model, b, reg).exists():
            skip.append((model, b, reg, "DONE"))
            continue
        nq = probe_count(b, reg)
        if nq == -1:
            skip.append((model, b, reg, "no-probe"))
        elif nq < MIN_Q:
            skip.append((model, b, reg, f"bin-too-small({nq})"))
        else:
            plan.append((model, b, reg, nq))
    say(f"[{time.strftime('%H:%M:%S')}] TIER1: {len(plan)} cells to run, {len(skip)} skipped "
        f"({sum(1 for s in skip if s[3]=='DONE')} done, "
        f"{sum(1 for s in skip if s[3]=='no-probe')} awaiting-probe)")
    for s in skip:
        if s[3] not in ("DONE",):
            say(f"    skip {s[0]}/{s[1]}/{s[2]}: {s[3]}")
    if args.dry_run:
        for c in plan:
            say(f"    RUN  {c[0]}/{c[1]}/{c[2]}  (n={c[3]})")
        return
    for i, (model, b, reg, nq) in enumerate(plan, 1):
        say(f"[{time.strftime('%H:%M:%S')}] ({i}/{len(plan)}) {model}/{b}/{reg} n={nq} ...")
        ok, dt, err = run_cell(model, b, reg, nq)
        say(f"    {'OK' if ok else 'FAIL'} in {dt/60:.1f}m" + (f"  ERR: {err}" if err else ""))
    say(f"[{time.strftime('%H:%M:%S')}] TIER1 sweep complete.")


if __name__ == "__main__":
    main()
