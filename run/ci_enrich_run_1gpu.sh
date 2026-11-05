#!/usr/bin/env bash
# Counterfactual Isolation (a_CI) — CI_PROTOCOL_FROZEN.md, Design A′ (damage–repair).
# SINGLE-GPU variant of ci_enrich_run.sh: ONE dedicated num_parallel=8 server on :11443, pinned to
# ONE GPU (CI_GPU, default 1). Per model, 2 shards (s0 q0-19 + s1 q20-39) run in PARALLEL against the
# SAME server → fills np=8 → recovers ~the per-model throughput of the old 2-GPU setup, on 1 GPU.
# Sequential over models: qwen (pilot/gate) → gemma → mistral → llama. Continue-on-error. Tears down
# ONLY the server it starts. Health-checks with a real generate (NOT nvidia-smi — NVML is broken here
# but CUDA compute works; this avoids the "backend dead → cached zeros" bug).
#
#   CI_GPU=1 ./ci_enrich_run_1gpu.sh smoke   # N=5 R=1 1q → validate a_CI end-to-end + VRAM offload
#   CI_GPU=1 ./ci_enrich_run_1gpu.sh full    # (default) all 4 models → results/tables/ci_gate_*.json
set -u
MODE="${1:-full}"
GPU="${CI_GPU:-1}"
PORT=11443
NP=8
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
CI="$PY $ROOT/experiments/exp6_phase_transition_llm/ci_enrich.py"
CIA="$PY $ROOT/experiments/exp6_phase_transition_llm/ci_enrich_analyze.py"
LOG="$ROOT/ci_enrich_run.log"
MODELS=/usr/share/ollama/.ollama/models
cd "$ROOT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

health(){ # model — real generate; nonzero on backend failure
  local r; r=$(curl -s -m 90 "http://127.0.0.1:$PORT/api/generate" \
      -d "{\"model\":\"$1\",\"prompt\":\"Say OK\",\"stream\":false,\"options\":{\"num_predict\":4}}")
  echo "$r" | grep -q '"response"' && ! echo "$r" | grep -q '"error"'
}
vram(){ curl -s -m5 "http://127.0.0.1:$PORT/api/ps" \
  | grep -oE '"name":"[^"]+"|"size":[0-9]+|"size_vram":[0-9]+' | tr '\n' ' '; }
start_server(){
  if ! curl -s -m1 "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$GPU" OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_NUM_PARALLEL=$NP \
      OLLAMA_MODELS="$MODELS" nohup ollama serve >"$ROOT/ollama_${PORT}.log" 2>&1 &
    for i in $(seq 1 60); do curl -s -m1 "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break; sleep 1; done
  fi
  if health qwen2.5:7b; then say "server :$PORT up + inference OK (CUDA=$GPU, np=$NP)"; return 0; fi
  say "server :$PORT BACKEND UNHEALTHY — ABORT (do NOT run)"; return 1
}
stop_server(){ local pid; pid=$(ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && say "tore down :$PORT (pid $pid)"; }

run_model(){ # tag config
  local MTAG="$1" CFG="$2"
  say ">>>> CI $MTAG  HARD N=11 R=8 dr=3 coupling=1.0  (s0 q0-19 + s1 q20-39 → :$PORT, 1 GPU)"
  $CI --config src/config/$CFG.yaml --regime hard --n-agents 11 --replicates 8 \
      --debate-rounds 3 --q-start 0  --q-count 20 --tag ${MTAG}_s0 >>"$ROOT/ci_enrich_${MTAG}_s0.log" 2>&1 & P0=$!
  $CI --config src/config/$CFG.yaml --regime hard --n-agents 11 --replicates 8 \
      --debate-rounds 3 --q-start 20 --q-count 20 --tag ${MTAG}_s1 >>"$ROOT/ci_enrich_${MTAG}_s1.log" 2>&1 & P1=$!
  wait $P0; say "   << $MTAG s0 rc=$?"; wait $P1; say "   << $MTAG s1 rc=$?"
  $CIA --glob "ci_enrich_${MTAG}_*_hard_N*.json" --out "$ROOT/results/tables/ci_gate_${MTAG}.json" >>"$LOG" 2>&1
  local g; g=$(grep -oE '"delta_CI": *[-0-9.]+|"gate": *"[^"]*"|"recovery_ratio": *[-0-9.]+|"slope_positive": *[a-z]+' \
      "$ROOT/results/tables/ci_gate_${MTAG}.json" 2>/dev/null | tr '\n' ' ')
  say "   done $MTAG -> ci_gate_${MTAG}.json :: $g"
}

if [ "$MODE" = smoke ]; then
  say "==== CI 1-GPU smoke (N=5 R=1 1q on :$PORT GPU$GPU np=$NP) ===="
  start_server || exit 1
  say "VRAM(before): $(vram)"
  $CI --config src/config/debate_qwen_gpu1.yaml --regime hard --n-agents 5 --replicates 1 \
      --debate-rounds 2 --q-start 0 --q-count 1 --tag smoke 2>&1 | tee -a "$LOG"
  say "VRAM(after):  $(vram)"
  say "==== smoke done — a_CI value shown above; server left UP for the full run ===="
  exit 0
fi

say "======== CI 1-GPU full run START (GPU$GPU np=$NP, models: qwen gemma mistral llama) ========"
start_server || exit 1
say "VRAM: $(vram)"
run_model qwen    debate_qwen_gpu1
run_model gemma   debate_gemma_gpu1
run_model mistral debate_mistral_gpu1
run_model llama   debate_llama_gpu1
stop_server
say "======== CI 1-GPU full run COMPLETE — read results/tables/ci_gate_*.json ========"
