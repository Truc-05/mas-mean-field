#!/bin/bash
# C4 (intervention response) re-run in English, truthfulqa/hard only.
#
# WHY. The four existing C4 cells (actionenrich_*_s{0,1}_hard_N11.json) were produced with
# Vietnamese prompts. Everything else in the paper is now English, so leaving them would mean
# reporting one result on a different stimulus from the rest. The prompts in action_enrich.py
# (DISSENT_PERSONA, the Break-Copy re-elicit) and ci_enrich.py (the frozen CI prompt) were
# translated on 2026-10-01; this re-runs the cells under them.
#
# SCOPE. truthfulqa/hard only, which is what C4 is reported on. Tier-2 stays dropped, so this
# does not extend C4 to other benchmarks; it only puts the existing claim on the English
# stimulus. Four models x two shards, matching the sharding of the original run.
#
# GPU DISCIPLINE. This must NOT run while english-rebuild is active: both target GPU1 and
# ollama serves one model at a time, so concurrent runs would contend and distort timings.
# The guard below refuses to start until that unit is done.
#
# COST. 4 models x 40 questions x R=6 x N=11 x (3 debate rounds + re-elicits) is roughly
# 11 hours in total at the observed throughput.

set -u
ROOT=/home/alex/trucn/manew/mas-mean-field
PY=/home/alex/trucn/venv/bin/python
cd "$ROOT" || exit 1
export PYTHONPATH=src

if systemctl --user is-active --quiet english-rebuild; then
  echo "REFUSING TO START: english-rebuild is still active and both runs target GPU1."
  echo "Wait for it to finish (watch tier1_breadth_english.log), then relaunch this script."
  exit 1
fi

# model tag -> config. Tags match the existing output naming convention so the analysis
# scripts (action_enrich_analyze.py, ci_enrich_analyze.py) keep finding the shards.
run_one () {
  local tag="$1" cfg="$2"
  for shard in 0 1; do
    local qstart=$(( shard * 20 ))
    echo "[$(date '+%F %T')] action_enrich $tag shard s$shard (q$qstart..$((qstart+19)))"
    "$PY" -u experiments/exp6_phase_transition_llm/action_enrich.py \
      --config "$cfg" --benchmark truthfulqa --regime hard \
      --n-agents 11 --replicates 6 --debate-rounds 3 \
      --q-start "$qstart" --q-count 20 --tag "${tag}s${shard}" \
      >> c4_english.log 2>&1 || echo "[$(date '+%F %T')] FAILED action_enrich $tag s$shard"
  done
}

echo "[$(date '+%F %T')] C4 English re-run starting (truthfulqa/hard)"
run_one ""          src/config/scaleup_qwen_gpu1.yaml      # qwen keeps the legacy bare tag
run_one "llama8b_"  src/config/scaleup_llama_gpu1.yaml
run_one "gemma9b_"  src/config/scaleup_gemma_gpu1.yaml
run_one "mistral7b_" src/config/scaleup_mistral_gpu1.yaml
echo "[$(date '+%F %T')] C4 English re-run done. Next: python action_enrich_analyze.py"
