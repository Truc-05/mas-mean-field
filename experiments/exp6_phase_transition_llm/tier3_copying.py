"""Tier-3 orchestrator — the COPYING axis (g-sweep) at breadth. Resumable.

WHY TIER-3 BEFORE TIER-2 (decided 2026-10-01, inverting the README order). C1's headline —
*same dependence, different consequence* at matched `D` — needs TWO dependence mechanisms on a
common `D` scale: upstream shared-evidence (Tier-1 has it) and downstream copying (only qwen ×
truthfulqa exists). So the matched-`D` claim is currently qwen-only and the breadth heatmap
cannot speak to it (see `paper.md` §7.1). Tier-3 is what closes that; Tier-2 would only widen
C4, whose headline is already weak (`a_common` 0/4 sig, demoted). Tier-3 also needs no edits to
frozen estimator code, whereas Tier-2 requires threading `benchmark` through `action_enrich.py`
and `ci_enrich.py`, which are hardwired to truthfulqa (`action_enrich.py:264-265`).

SCOPE: 3 models {qwen, llama, gemma} × 3 benchmarks {truthfulqa, strategyqa, aqua}, coupling
grid g ∈ {0, 1, 2}, N=11, 40 questions — matching the 40 questions per g that qwen × truthfulqa
already has, so the new cells are directly comparable with the legacy one.

qwen RUNS UNNAMESPACED ON PURPOSE, matching the convention already used by hand elsewhere in
the repo (`actionenrich_s0_…` = qwen, `actionenrich_llama8b_s0_…` = llama).

NOTE (revised 2026-10-01): qwen × truthfulqa is NO LONGER FREE. It was, while the 473
Vietnamese-era `exp6_truthfulqa_N11_g*.npz` states sat in `data/processed`; those were archived
with the rest of the Vietnamese-era campaign, so all 18 cells now compute from scratch. The
probe-selection caches went with them, so each cell re-probes its band in English first.

WHY BANDED SELECTION, RESTATED. The original reason was comparability with the published qwen
g-sweep curve. That curve is Vietnamese-era and archived, so it no longer licenses the choice.
The bands are kept on their own merits: `H_repair = p_break − p_rescue` is identically zero
wherever the baseline is saturated, so a competence spread is a precondition for observing the
break/rescue decomposition at all, and banding keeps Tier-3 on the same stratified logic as the
breadth tier. The specific band values below are retained so that the Tier-3 strata remain
directly comparable with each other across models.

TWO PREREQUISITE FIXES went into `run.py` for this (both were silent-wrong-data hazards, not
conveniences):
  1. `--model-tag` now namespaces BOTH the probe-selection cache and the per-question `.npz`
     states. Without it a second model's run silently loads the FIRST model's cached debates —
     the same class of stale-cache bug the question hash was added to kill, one axis over.
  2. `LOADERS` gained the five scale-up benchmarks. It previously offered only
     {aqua, truthfulqa} while the boundary tier ran seven, so the copying axis could not reach
     strategyqa at all. `run.py:_assert_loader_parity` now fails loudly if the two registries
     drift again.

Unlike `tier1_breadth.py` this driver STREAMS each child's output to its own log file and
imposes a timeout, because `tier1_breadth.py`'s `subprocess.run(capture_output=True)` with no
timeout left no trace in the log whenever a child was killed or hung.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG = ROOT / "src" / "config"
TABLES = ROOT / "results" / "tables"
PY = "/home/alex/trucn/venv/bin/python"

MODELS = ["qwen", "llama", "gemma"]
BENCHMARKS = ["truthfulqa", "strategyqa", "aqua"]
REGIMES = ["easy", "hard"]
GRID = ["0", "1", "2"]
N, N_QUESTIONS = 11, 40
CELL_TIMEOUT_S = 6 * 3600          # a cell is ~90 min; 6 h means "wedged", not "slow"

# Band values recovered from the published Vietnamese-era cells by matching the selection
# caches to the two CSVs README:239 cites, and retained for internal comparability:
#   hard  p0.0-0.3,  k=5, c=400 -> had selected 40 -> p_single 0.186
#   easy  p0.6-0.95, k=5, c=250 -> had selected 21 -> p_single 0.630
# (README:239 claimed the easy cell was n=40; the CSV said n=21. `--n-questions` is a CAP, not
# a guarantee, and under English re-probing these counts will differ again. Do not hardcode a
# expected yield anywhere downstream.)
BAND = {
    "easy": {"min_p": "0.6", "max_p": "0.95", "probe_k": "5", "candidates": "250"},
    "hard": {"min_p": "0.0", "max_p": "0.3", "probe_k": "5", "candidates": "400"},
}

# qwen intentionally empty — see module docstring. A happy side effect: qwen × truthfulqa reuses
# the existing probe caches AND the cached debates, so it re-derives the published cell for free
# and acts as a regression check on this driver.
MODEL_TAG = {"qwen": "", "llama": "llama8b", "gemma": "gemma9b"}


def run_tag(model: str, regime: str) -> str:
    return f"tier3_{model}_{regime}"


def out_csv(model: str, benchmark: str, regime: str) -> Path:
    """run.py writes `exp6_phase_transition_llm_{benchmark}_N{N}_{tag}.csv` when --tag is set."""
    return TABLES / f"exp6_phase_transition_llm_{benchmark}_N{N}_{run_tag(model, regime)}.csv"


def cells(models=None):
    for model in (models or MODELS):
        for b in BENCHMARKS:
            for reg in REGIMES:
                yield model, b, reg


def run_cell(model: str, benchmark: str, regime: str, log_dir: Path,
            gpu_tag: str = "gpu1") -> tuple[bool, float, str]:
    band = BAND[regime]
    cmd = [PY, "-u", str(HERE / "run.py"),
           "--benchmark", benchmark,
           "--config", str(CFG / f"scaleup_{model}_{gpu_tag}.yaml"),
           "--n-questions", str(N_QUESTIONS),
           "--n-agents", str(N),
           "--grid", *GRID,
           "--min-p", band["min_p"], "--max-p", band["max_p"],
           "--probe-k", band["probe_k"], "--candidates", band["candidates"],
           "--tag", run_tag(model, regime)]
    mtag = MODEL_TAG[model]
    if mtag:
        cmd += ["--model-tag", mtag]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cell_log = log_dir / f"tier3_{model}_{benchmark}_{regime}.log"
    t0 = time.time()
    try:
        with open(cell_log, "w") as lf:
            lf.write(" ".join(cmd) + "\n\n")
            lf.flush()
            r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=CELL_TIMEOUT_S)
        rc = r.returncode
    except subprocess.TimeoutExpired:
        return False, time.time() - t0, f"TIMEOUT after {CELL_TIMEOUT_S/3600:.1f}h; see {cell_log.name}"
    dt = time.time() - t0
    if rc != 0 or not out_csv(model, benchmark, regime).exists():
        tail = ""
        try:
            tail = cell_log.read_text()[-400:].replace("\n", " ")
        except Exception:
            pass
        return False, dt, f"rc={rc}; {tail}"
    return True, dt, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log", default=str(ROOT / "tier3_copying.log"))
    ap.add_argument("--models", default="", help="comma list restricting MODELS "
                    "(default: all 3) -- for splitting the FROZEN 18-cell grid across "
                    "GPUs/processes; does not add cells")
    ap.add_argument("--gpu", default="gpu1", help="config suffix, e.g. gpu1/gpu0 -- "
                    "selects scaleup_{model}_{gpu}.yaml and which ollama port that points at")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m] or None
    if models and any(m not in MODELS for m in models):
        raise SystemExit(f"--models must be a subset of {MODELS}")
    log_dir = Path(args.log).parent / "tier3_cell_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(args.log, "a")

    def say(m: str) -> None:
        print(m)
        print(m, file=log, flush=True)

    plan, done = [], []
    for model, b, reg in cells(models):
        (done if out_csv(model, b, reg).exists() else plan).append((model, b, reg))

    # probe calls (candidates x k) + debate calls (n_q x |grid| x N x rounds), at ~96 calls/min
    est_m = sum(int(BAND[r]["candidates"]) * int(BAND[r]["probe_k"])
                + N_QUESTIONS * len(GRID) * N * 5 for _, _, r in plan) / 96
    say(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIER3 copying: {len(plan)} cells to run, "
        f"{len(done)} already done. grid={GRID} n_q<={N_QUESTIONS} N={N} regimes={REGIMES} "
        f"→ rough ETA {est_m/60:.1f}h at Tier-1's observed throughput (upper bound: cached "
        f"probes/debates and thin bands come in under it).")
    for model, b, reg in done:
        say(f"    skip {model}/{b}/{reg}: DONE ({out_csv(model, b, reg).name})")
    if args.dry_run:
        for model, b, reg in plan:
            mt = MODEL_TAG[model] or "(legacy/unnamespaced)"
            say(f"    RUN  {model}/{b}/{reg}  band=p{BAND[reg]['min_p']}-{BAND[reg]['max_p']} "
                f"c={BAND[reg]['candidates']}  model-tag={mt}")
        return

    for i, (model, b, reg) in enumerate(plan, 1):
        say(f"[{time.strftime('%H:%M:%S')}] ({i}/{len(plan)}) {model}/{b}/{reg} "
            f"[{args.gpu}] ...")
        ok, dt, err = run_cell(model, b, reg, log_dir, gpu_tag=args.gpu)
        say(f"    {'OK' if ok else 'FAIL'} in {dt/60:.1f}m" + (f"  ERR: {err}" if err else ""))
    say(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] TIER3 ({args.gpu}, models={models or MODELS}) complete.")


if __name__ == "__main__":
    main()
