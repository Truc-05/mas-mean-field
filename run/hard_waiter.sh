#!/usr/bin/env bash
# HARD boundary auto-relaunch waiter (non-preemptive).
# Waits for a GPU to genuinely free up (does NOT preempt the multi_sequence_3d
# swin/VLM jobs), spins up a DEDICATED qwen2.5:7b np=4 ollama on :11442
# (frozen protocol; distinct port so it never touches the other project's :11441),
# relaunches exp6c_boundary --regime hard resuming from the 231-record belief cache,
# then runs the boundary analysis and tears the server down.
set -u
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
PORT=11442
# the models live in the SYSTEM store, not ~/.ollama — a serve without this env sees an
# EMPTY model list, 404s every /v1/chat/completions, and the boundary caches flat zeros.
OLLAMA_MODELS_DIR=/usr/share/ollama/.ollama/models
MODEL=qwen2.5:7b
NEED_FREE_MIB=11000          # qwen2.5:7b (4.7G) + np=4 KV headroom on a 16G card
POLL=120
LOG="$ROOT/exp6c_boundary_hard_waiter.log"
HARD_JSON="$ROOT/results/tables/exp6c_boundary_hard_N11.json"
CFG_SRC="$ROOT/src/config/debate_gpu1_np4.yaml"
CFG="$ROOT/src/config/debate_hard_waiter_np4.yaml"

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# already done?
if [ -f "$HARD_JSON" ]; then say "HARD already complete ($HARD_JSON) — nothing to do"; exit 0; fi
# already running?
if pgrep -af "exp6c_boundary.py --regime hard" | grep -qv "$$"; then say "HARD already running — exit"; exit 0; fi

say "waiter up; need a GPU with >=${NEED_FREE_MIB}MiB free (non-preemptive)"
FREE_GPU=-1
while :; do
  while read -r idx free; do
    idx=${idx%,}; free=${free%,}            # csv leaves a trailing comma on idx → strip it
    if [ "$free" -ge "$NEED_FREE_MIB" ]; then FREE_GPU=$idx; break; fi
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  [ "$FREE_GPU" -ge 0 ] && break
  say "no free GPU yet (g0/g1 free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' '/'))"
  sleep "$POLL"
done
say "GPU$FREE_GPU is free — starting dedicated server on :$PORT"

# start dedicated frozen server (qwen2.5:7b, num_parallel=4) on the free GPU.
# OLLAMA_MODELS points at the SYSTEM store so the model is actually visible.
CUDA_VISIBLE_DEVICES=$FREE_GPU OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_NUM_PARALLEL=4 \
  OLLAMA_MODELS=$OLLAMA_MODELS_DIR OLLAMA_KEEP_ALIVE=24h \
  nohup ollama serve >"$ROOT/ollama_$PORT.log" 2>&1 &
SRV=$!
say "ollama serve pid=$SRV; waiting for readiness"
for i in $(seq 1 60); do
  curl -s "http://127.0.0.1:$PORT/api/tags" >/dev/null 2>&1 && break; sleep 2
done
# warm the model onto the GPU
curl -s "http://127.0.0.1:$PORT/api/generate" -d "{\"model\":\"$MODEL\",\"prompt\":\"ok\",\"stream\":false}" >/dev/null 2>&1
# PRE-FLIGHT: a real chat-completion must succeed. If the model is not found (404), ABORT
# instead of running the boundary and caching flat zeros (the failure that corrupted the run).
PROBE=$(curl -s -m 30 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":3}")
if ! echo "$PROBE" | grep -q '"choices"'; then
  say "PRE-FLIGHT FAILED on :$PORT — model '$MODEL' not usable (resp: ${PROBE:0:160})"
  say "ABORTING before boundary to avoid caching flat zeros. tearing down server."
  kill "$SRV" 2>/dev/null
  exit 3
fi
say "server ready; model warmed; pre-flight chat OK"

# frozen config copy, only base_url port changed
sed 's#localhost:11441/v1#localhost:'"$PORT"'/v1#' "$CFG_SRC" > "$CFG"

say "relaunching HARD (resume from belief cache)"
cd "$ROOT"
$PY experiments/exp6_phase_transition_llm/exp6c_boundary.py \
    --regime hard --config "$CFG" --n-questions 40 --n-agents 11 --replicates 8 \
    >>"$ROOT/exp6c_boundary_hard.log" 2>&1
RC=$?
say "HARD finished rc=$RC"

if [ $RC -eq 0 ] && [ -f "$HARD_JSON" ]; then
  say "running boundary analysis (easy+hard)"
  $PY experiments/exp6_phase_transition_llm/exp6c_boundary_analyze.py \
      results/tables/exp6c_boundary_easy_N11.json "$HARD_JSON" --nbins 3 \
      >>"$ROOT/exp6c_boundary_hard.log" 2>&1
  say "analysis done -> exp6c_boundary_summary.json + figures/exp6c_boundary.png"
else
  say "HARD did not produce $HARD_JSON — leaving server up? no, tearing down; inspect log"
fi

# teardown dedicated server (leave the other project's :11441 untouched)
kill "$SRV" 2>/dev/null && say "tore down :$PORT server (pid $SRV)"
say "waiter done"
