#!/usr/bin/env bash
# Counterfactual Isolation (a_CI) pilot — CI_PROTOCOL_FROZEN.md, Design A′ (damage–repair).
# HARD, N=11, R=8, debate_rounds=3, coupling=1.0. Two dedicated num_parallel=4 servers
# (GPU1 :11443, GPU0 :11444); qwen pilot sharded s0(q0-19)->:11443 / s1(q20-39)->:11444 in
# parallel, then the CI gate. Tears down ONLY the servers it starts. Continue-on-error.
#
#   ./ci_enrich_run.sh smoke     # live-smoke: N=5 R=1 1q on :11443 (validate operator end-to-end)
#   ./ci_enrich_run.sh qwen      # (default) the qwen HARD pilot → ci_gate_qwen.json → read the gate
#   ./ci_enrich_run.sh xmodel    # AFTER pilot passes: qwen + llama + gemma for the ≥2-model rule
#
# PRECONDITION: GPU healthy. This script health-checks each server with a real 1-token generate
# and ABORTS if the llama-server backend is broken (avoids the "backend dead → 275 cached zeros" bug).
set -u
MODE="${1:-qwen}"
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
CI="$PY $ROOT/experiments/exp6_phase_transition_llm/ci_enrich.py"
CIA="$PY $ROOT/experiments/exp6_phase_transition_llm/ci_enrich_analyze.py"
LOG="$ROOT/ci_enrich_run.log"
MODELS=/usr/share/ollama/.ollama/models
cd "$ROOT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

health(){ # port model  — real generate; nonzero on backend failure
  local port="$1" model="$2"
  local r; r=$(curl -s -m 60 "http://127.0.0.1:$port/api/generate" \
      -d "{\"model\":\"$model\",\"prompt\":\"Say OK\",\"stream\":false,\"options\":{\"num_predict\":4}}")
  echo "$r" | grep -q '"response"' && ! echo "$r" | grep -q '"error"'
}
start_server(){ # gpu port model
  local gpu="$1" port="$2" model="$3"
  if ! curl -s -m1 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$gpu" OLLAMA_HOST="127.0.0.1:$port" OLLAMA_NUM_PARALLEL=4 \
      OLLAMA_MODELS="$MODELS" nohup ollama serve >"$ROOT/ollama_${port}.log" 2>&1 &
    for i in $(seq 1 60); do curl -s -m1 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1 && break; sleep 1; done
  fi
  if health "$port" "$model"; then say "server :$port up + inference OK (CUDA=$gpu)"; return 0; fi
  say "server :$port BACKEND UNHEALTHY (llama-server dead / driver mismatch) — ABORT, do NOT run"; return 1
}
stop_server(){ local pid; pid=$(ss -lptnH "sport = :$1" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && say "tore down :$1 (pid $pid)"; }

run_model(){ # tag cfg1 cfg0
  local MTAG="$1" CFG1="$2" CFG0="$3"
  say ">>>> CI pilot $MTAG (s0->:11443 via $CFG1, s1->:11444 via $CFG0)  HARD N=11 R=8 dr=3 coupling=1.0"
  $CI --config src/config/$CFG1.yaml --regime hard --n-agents 11 --replicates 8 \
      --debate-rounds 3 --q-start 0 --q-count 20 --tag ${MTAG}_s0 >>"$ROOT/ci_enrich_${MTAG}_s0.log" 2>&1 & P0=$!
  $CI --config src/config/$CFG0.yaml --regime hard --n-agents 11 --replicates 8 \
      --debate-rounds 3 --q-start 20 --q-count 20 --tag ${MTAG}_s1 >>"$ROOT/ci_enrich_${MTAG}_s1.log" 2>&1 & P1=$!
  wait $P0; say "   << $MTAG s0 rc=$?"; wait $P1; say "   << $MTAG s1 rc=$?"
  $CIA --glob "ci_enrich_${MTAG}_*_hard_N*.json" --out "$ROOT/results/tables/ci_gate_${MTAG}.json" >>"$LOG" 2>&1
  say "   done $MTAG -> results/tables/ci_gate_${MTAG}.json"
}

if [ "$MODE" = smoke ]; then
  say "==== CI live-smoke (N=5 R=1 1q on :11443) ===="
  start_server 1 11443 qwen2.5:7b || exit 1
  $CI --config src/config/debate_qwen_gpu1.yaml --regime hard --n-agents 5 --replicates 1 \
      --debate-rounds 2 --q-start 0 --q-count 1 --tag smoke 2>&1 | tee -a "$LOG"
  stop_server 11443; say "==== smoke done — inspect that all 9 actions incl a_CI produced a value ===="; exit 0
fi

say "======== CI pilot run START (mode=$MODE) ========"
start_server 1 11443 qwen2.5:7b || exit 1
start_server 0 11444 qwen2.5:7b || { stop_server 11443; exit 1; }

run_model qwen debate_qwen_gpu1 debate_qwen_gpu0
if [ "$MODE" = xmodel ]; then
  # only after the qwen pilot clears the gate; reuses the P2 per-model gpu configs
  run_model llama  debate_llama_gpu1  debate_llama_gpu0
  run_model gemma  debate_gemma_gpu1  debate_gemma_gpu0
fi

stop_server 11443; stop_server 11444
say "======== CI pilot COMPLETE — read results/tables/ci_gate_*.json (Δ_CI, RR, slope, guardrails) ========"
