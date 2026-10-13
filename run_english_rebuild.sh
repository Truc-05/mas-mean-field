#!/bin/bash
# English rebuild of the Tier-1 breadth campaign (launched 2026-10-01).
#
# WHY. The previous campaign ran Vietnamese instructions over English benchmark items, and
# the generated evidence pool drifted across languages: 71.9% Vietnamese, 16.0% Chinese,
# 12.1% English over 12506 clues. Because the SHARED arm broadcasts one evidence realization
# while the INDEP arm issues N distinct ones, that drift left the independent arm additionally
# heterogeneous in language, so part of the dependence contrast could be attributed to
# language rather than to evidence commonality. The clue cache is also shared across all six
# model families, so every model received the same 16% Chinese, which confounds exactly the
# cross-model comparison Tier-1 exists to make. All prompts are now English and the evidence
# generator is constrained to English.
#
# PHASE 1  anchor probe, 7 benchmarks x 400 candidates x N=11 x R=4 = 123,200 calls (~21h)
# PHASE 2  breadth sweep, 84 cells at ~110 min each (~6.4 days)
#
# Resumable. Phase 1 skips a benchmark whose probe cache already exists; phase 2 skips any
# cell whose output JSON exists. Safe to stop and relaunch the same unit.

set -u
ROOT=/home/alex/trucn/manew/mas-mean-field
PY=/home/alex/trucn/venv/bin/python
CFG=$ROOT/src/config/scaleup_qwen_gpu1.yaml          # qwen is the anchor for the probe
cd "$ROOT" || exit 1
export PYTHONPATH=src

# bbh stays in PHASE 1 only because its probe cache already exists, so the loop below skips
# it and nothing is spent. It is dropped from the PHASE 2 sweep in tier1_breadth.BENCHMARKS
# (2026-10-03) -- see the comment there for the logical_deduction stimulus-validity failure
# that disqualified it.
BENCHMARKS="truthfulqa aqua arc commonsenseqa mmlu strategyqa bbh"

echo "[$(date '+%F %T')] PHASE 1: anchor probe over 7 benchmarks"
for b in $BENCHMARKS; do
  cache="data/processed/exp6_select_${b}.json"
  if [ -s "$cache" ]; then
    echo "[$(date '+%F %T')]   skip $b (cache exists)"
    continue
  fi
  echo "[$(date '+%F %T')]   probe $b ..."
  if ! "$PY" -u experiments/exp6_phase_transition_llm/p2_probe.py \
        --benchmark "$b" --config "$CFG" >> probe_english.log 2>&1; then
    echo "[$(date '+%F %T')]   PROBE FAILED for $b, see probe_english.log"
  fi
done
echo "[$(date '+%F %T')] PHASE 1 done"

echo "[$(date '+%F %T')] PHASE 2: breadth sweep"
exec "$PY" -u experiments/exp6_phase_transition_llm/tier1_breadth.py \
     --log "$ROOT/tier1_breadth_english.log"
