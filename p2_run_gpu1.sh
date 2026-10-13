#!/usr/bin/env bash
# P2 generalization run — GPU1 ONLY (dedicated :11443 server, CUDA=1). Never touches GPU0.
# P2a: model axis (llama8b/gemma9b/mistral7b) × TruthfulQA, easy+hard, --model-tag <stem>.
# P2b: AQuA × qwen — probe → run only regimes with >=10 stratified questions.
# Then p2_aggregate; tear down ONLY the :11443 server. Continue-on-error (belief cache resumes).
set -u
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
PORT=11443
BND="$PY $ROOT/experiments/exp6_phase_transition_llm/exp6c_boundary.py"
LOG="$ROOT/p2_run_gpu1.log"
cd "$ROOT"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

run_model(){  # stem config benchmark
  local stem="$1" cfg="$2" bench="$3"
  for regime in easy hard; do
    say ">> $stem $bench $regime START"
    $BND --regime "$regime" --config "src/config/$cfg" --n-questions 40 --n-agents 11 \
         --replicates 8 --benchmark "$bench" --model-tag "$stem" \
         >>"$ROOT/p2_${stem}_${bench}_${regime}.log" 2>&1
    say "<< $stem $bench $regime DONE rc=$?"
  done
}

say "======== P2 GPU1 run START (server :$PORT, CUDA=1) ========"

# ---- P2a: model axis × TruthfulQA (reuses qwen's clue cache = matched evidence) ----
run_model llama8b   debate_llama_gpu1.yaml   truthfulqa
run_model gemma9b   debate_gemma_gpu1.yaml   truthfulqa
run_model mistral7b debate_mistral_gpu1.yaml truthfulqa

# ---- P2b: AQuA × qwen — probe, then run viable regimes only ----
say ">> AQuA probe (qwen) START"
$PY $ROOT/experiments/exp6_phase_transition_llm/p2_probe.py \
    --benchmark aqua --config src/config/debate_qwen_gpu1.yaml \
    --replicates 3 --max-questions 200 >>"$ROOT/p2_probe_aqua.log" 2>&1
say "<< AQuA probe DONE rc=$?"
VIABLE=$($PY - <<'PYEOF'
import json
from pathlib import Path
p = Path("/home/alex/venvs/trucn/mas-mean-field/data/processed/exp6_select_aqua.json")
if not p.exists():
    print(""); raise SystemExit
v = [float(x) for x in json.loads(p.read_text())["phat"].values()]
out = [r for r, ok in (("easy", sum(x >= 0.8 for x in v) >= 10),
                       ("hard", sum(x <= 0.3 for x in v) >= 10)) if ok]
print(",".join(out))
PYEOF
)
say "AQuA viable regimes: [${VIABLE:-none}] (need >=10 q/bin)"
for regime in ${VIABLE//,/ }; do
  say ">> qwen7b_aqua aqua $regime START"
  $BND --regime "$regime" --config src/config/debate_qwen_gpu1.yaml --n-questions 40 \
       --n-agents 11 --replicates 8 --benchmark aqua --model-tag qwen7b_aqua \
       >>"$ROOT/p2_qwen7b_aqua_${regime}.log" 2>&1
  say "<< qwen7b_aqua aqua $regime DONE rc=$?"
done

# ---- aggregate the whole matrix (legacy qwen TruthfulQA + all new tagged runs) ----
say ">> p2_aggregate --n 11"
$PY $ROOT/experiments/exp6_phase_transition_llm/p2_aggregate.py --n 11 >>"$LOG" 2>&1

# ---- teardown ONLY the :11443 server (leave every other server / GPU0 untouched) ----
SRV=$(ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null && say "tore down :$PORT server (pid $SRV)"
say "======== P2 GPU1 run COMPLETE ========"
