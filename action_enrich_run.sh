#!/usr/bin/env bash
# Action-enrichment falsification (HARD-only) — the CDCR actionability gate.
# Self-sequences AFTER the N-scaling run frees both GPUs (waits for its COMPLETE line), then
# runs A1 in 2 question-shards in PARALLEL (s0 q0-19 -> :11443/CUDA=1, s1 q20-39 -> :11444/CUDA=0),
# N=11, R=8, debate_rounds=3, then action_enrich_analyze (merges shards -> gate). Tears down
# ONLY the two servers it starts. Continue-on-error (incremental JSON save per question).
set -u
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
AE="$PY $ROOT/experiments/exp6_phase_transition_llm/action_enrich.py"
AEA="$PY $ROOT/experiments/exp6_phase_transition_llm/action_enrich_analyze.py"
LOG="$ROOT/action_enrich_run.log"
NSLOG="$ROOT/n_scaling_run.log"
MODELS=/usr/share/ollama/.ollama/models
cd "$ROOT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

start_server(){ # gpu port
  local gpu="$1" port="$2"
  if curl -s -m1 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
    say "server :$port already up -- reuse"; return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" OLLAMA_HOST="127.0.0.1:$port" OLLAMA_NUM_PARALLEL=4 \
    OLLAMA_MODELS="$MODELS" nohup ollama serve >"$ROOT/ollama_${port}.log" 2>&1 &
  for i in $(seq 1 60); do
    curl -s -m1 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1 && { say "server :$port up (CUDA=$gpu)"; return 0; }
    sleep 1
  done
  say "server :$port FAILED to start"; return 1
}
stop_server(){ # port
  local port="$1"
  local pid=$(ss -lptnH "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && say "tore down :$port (pid $pid)"
}

say "======== ACTION-ENRICHMENT run QUEUED — waiting for N-scaling to free GPUs ========"
# wait until N-scaling has finished (its COMPLETE line is printed AFTER it tears down its servers)
while ! grep -q "P2c N-scaling run COMPLETE" "$NSLOG" 2>/dev/null; do sleep 60; done
say "N-scaling COMPLETE detected — GPUs free. Starting action-enrichment."
sleep 5

start_server 1 11443 || exit 1
start_server 0 11444 || exit 1

say ">> shard s0 (q0-19)  -> :11443  START"
$AE --config src/config/debate_qwen_gpu1.yaml --regime hard --n-agents 11 --replicates 8 \
    --debate-rounds 3 --q-start 0 --q-count 20 --tag s0 \
    >>"$ROOT/action_enrich_s0.log" 2>&1 &
PID_S0=$!
say ">> shard s1 (q20-39) -> :11444  START"
$AE --config src/config/debate_qwen_gpu0.yaml --regime hard --n-agents 11 --replicates 8 \
    --debate-rounds 3 --q-start 20 --q-count 20 --tag s1 \
    >>"$ROOT/action_enrich_s1.log" 2>&1 &
PID_S1=$!
wait $PID_S0; say "<< shard s0 DONE rc=$?"
wait $PID_S1; say "<< shard s1 DONE rc=$?"

say ">> action_enrich_analyze (merge shards -> gate verdict)"
$AEA >>"$LOG" 2>&1

stop_server 11443
stop_server 11444
say "======== ACTION-ENRICHMENT run COMPLETE — see actionenrich_gate.json ========"
