#!/usr/bin/env bash
# P2c — Theorem-5 N-scaling test on the qwen2.5:7b x TruthfulQA cell (N=11 already DONE).
# Runs N in {5,21}, both regimes, EMPTY --model-tag so it matches the legacy N=11 baseline:
# the belief cache is keyed {regime}|{cond}|q|r|a{i} + sha(model+prompt) with NO N, and the
# per-agent evidence is a stable prefix in N (evidence_for/_draw_distinct seed on (qi,r,ci)),
# so agents 0..min(N,11)-1 are cache HITS for all 3 conds. => N=5 ~free (all hits);
# N=21 rolls out only agents 11..20 (3 conds x 40q x 8r x 10 agents x 2 regimes ~= 19.2k calls).
# easy -> dedicated :11443 (CUDA=1), hard -> dedicated :11444 (CUDA=0), regimes in PARALLEL.
# Then p2_aggregate over ALL N -> the "slope steepens as N grows" view. Tears down ONLY the
# two servers this script starts. Continue-on-error; belief cache resumes per (cond,q,r,agent).
set -u
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
BND="$PY $ROOT/experiments/exp6_phase_transition_llm/exp6c_boundary.py"
AGG="$PY $ROOT/experiments/exp6_phase_transition_llm/p2_aggregate.py"
LOG="$ROOT/n_scaling_run.log"
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

run_regime(){ # regime cfg
  local regime="$1" cfg="$2"
  for N in 5 21; do
    say ">> qwen7b TruthfulQA $regime N=$N START"
    $BND --regime "$regime" --config "src/config/$cfg" --n-questions 40 --n-agents "$N" \
         --replicates 8 --benchmark truthfulqa \
         >>"$ROOT/n_scaling_${regime}_N${N}.log" 2>&1
    say "<< qwen7b TruthfulQA $regime N=$N DONE rc=$?"
  done
}

say "======== P2c N-scaling run START ========"
start_server 1 11443 || exit 1
start_server 0 11444 || exit 1

run_regime easy debate_qwen_gpu1.yaml &   # :11443 / CUDA=1
PID_EASY=$!
run_regime hard debate_qwen_gpu0.yaml &   # :11444 / CUDA=0
PID_HARD=$!
wait $PID_EASY
wait $PID_HARD

say ">> p2_aggregate over ALL N (5,11,21) -> N-scaling view"
$AGG --out "$ROOT/results/tables/p2_aggregate_nscaling.json" >>"$LOG" 2>&1

stop_server 11443
stop_server 11444
say "======== P2c N-scaling run COMPLETE ========"
