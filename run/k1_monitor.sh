#!/usr/bin/env bash
# Waits for the two K1 runs to finish, verifies completeness, then runs k1_analyze.py.
ROOT=/home/alex/venvs/trucn/mas-mean-field
PY=/home/alex/venvs/env/bin/python
HARD_PID=$1
EASY_PID=$2
LOG=$ROOT/k1_monitor.log
: > "$LOG"

echo "[monitor] watching hard=$HARD_PID easy=$EASY_PID at $(date)" >> "$LOG"
# poll until both processes have exited
while kill -0 "$HARD_PID" 2>/dev/null || kill -0 "$EASY_PID" 2>/dev/null; do
  sleep 60
done
echo "[monitor] both runs exited at $(date)" >> "$LOG"

# verify completeness (expect 40 questions each)
for f in k1_yqa_v1_easy_N11.json k1_yqa_v1_hard_N11.json; do
  n=$("$PY" -c "import json;print(len(json.load(open('$ROOT/results/tables/$f'))['questions']))" 2>/dev/null)
  echo "[monitor] $f -> ${n:-MISSING} questions" >> "$LOG"
done

echo "[monitor] running k1_analyze.py at $(date)" >> "$LOG"
"$PY" -u "$ROOT/experiments/exp6_phase_transition_llm/k1_analyze.py" >> "$LOG" 2>&1
echo "[monitor] analyze done rc=$? at $(date)" >> "$LOG"
