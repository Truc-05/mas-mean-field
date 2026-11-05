#!/usr/bin/env bash
# Action-enrichment CROSS-MODEL validation of the standalone win `a_common`.
# qwen HARD gave Δ_fixed(a_common)=+0.038 (beats every A0 baseline) but only a WEAK router gate
# (H^CV_new=+0.035). This run tests whether the `a_common` standalone lift REPLICATES on the
# other three P2 models (gemma2:9b, llama3.1:8b, mistral:7b) — same 40 HARD questions, N=11,
# R=8, debate_rounds=3, SAME fixed qwen clue pool (the P2 convention: clues held constant, only
# the model swaps). Two dedicated num_parallel=4 servers (GPU1 :11443, GPU0 :11444); each model
# is sharded s0(q0-19)->:11443 / s1(q20-39)->:11444 in parallel, then per-model gate. Finally a
# cross-model summary. Tears down ONLY the two servers it starts. Continue-on-error.
set -u
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
AE="$PY $ROOT/experiments/exp6_phase_transition_llm/action_enrich.py"
AEA="$PY $ROOT/experiments/exp6_phase_transition_llm/action_enrich_analyze.py"
SUM="$PY $ROOT/experiments/exp6_phase_transition_llm/action_enrich_xmodel_summary.py"
LOG="$ROOT/action_enrich_xmodel_run.log"
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

# model_tag  cfg_gpu1(:11443)  cfg_gpu0(:11444)
RUNS=(
  "gemma9b    debate_gemma_gpu1    debate_gemma_gpu0"
  "llama8b    debate_llama_gpu1    debate_llama_gpu0"
  "mistral7b  debate_mistral_gpu1  debate_mistral_gpu0"
)

say "======== ACTION-ENRICHMENT CROSS-MODEL run START (gemma, llama, mistral) ========"
start_server 1 11443 || exit 1
start_server 0 11444 || exit 1

for row in "${RUNS[@]}"; do
  read -r MTAG CFG1 CFG0 <<< "$row"
  say ">>>> MODEL $MTAG  (s0->:11443 via $CFG1, s1->:11444 via $CFG0)"
  $AE --config src/config/$CFG1.yaml --regime hard --n-agents 11 --replicates 8 \
      --debate-rounds 3 --q-start 0 --q-count 20 --tag ${MTAG}_s0 \
      >>"$ROOT/action_enrich_${MTAG}_s0.log" 2>&1 &
  P0=$!
  $AE --config src/config/$CFG0.yaml --regime hard --n-agents 11 --replicates 8 \
      --debate-rounds 3 --q-start 20 --q-count 20 --tag ${MTAG}_s1 \
      >>"$ROOT/action_enrich_${MTAG}_s1.log" 2>&1 &
  P1=$!
  wait $P0; say "   << $MTAG s0 DONE rc=$?"
  wait $P1; say "   << $MTAG s1 DONE rc=$?"
  say "   >> analyze $MTAG"
  $AEA --glob "actionenrich_${MTAG}_*_hard_N*.json" \
       --out "$ROOT/results/tables/actionenrich_gate_${MTAG}.json" >>"$LOG" 2>&1
  say "   done $MTAG -> results/tables/actionenrich_gate_${MTAG}.json"
done

stop_server 11443
stop_server 11444

say ">> CROSS-MODEL SUMMARY (qwen + gemma + llama + mistral)"
$SUM >>"$LOG" 2>&1
say "======== ACTION-ENRICHMENT CROSS-MODEL run COMPLETE — see action_enrich_xmodel_run.log tail ========"
