#!/bin/bash
set -uo pipefail
CFG="${1:-full}"
cd "$(dirname "$0")/.."
source ./env.sh
if [ "$CFG" = "full" ]; then export OW_RESULTS="$PROJECT_ROOT/results"; else export OW_RESULTS="$PROJECT_ROOT/results_$CFG"; fi
ST="$OW_RESULTS/stages"
LOG="$OW_RESULTS/logs/pipeline.log"
mkdir -p "$ST" "$OW_RESULTS/logs"
say() { echo "$(date '+%F %T') [phase2] $*" | tee -a "$LOG"; }

if [ ! -f "$ST/dataset.done" ] || [ ! -f "$ST/control.done" ]; then
  say "phase 1 is not finished in $OW_RESULTS (need stages dataset + control). Run scripts/run_all.sh $CFG first."
  exit 1
fi

timing_add() {
  python - "$ST/timing.json" "$1" "$2" <<'PY'
import json, sys, os
p, k, v = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = json.load(open(p)) if os.path.exists(p) else {}
d[k] = d.get(k, 0.0) + v
json.dump(d, open(p, "w"), indent=1)
PY
}

stage() {
  local name="$1"; shift
  if [ -f "$ST/$name.done" ]; then say "stage $name: already done, skipping"; return 0; fi
  echo "$name $(date +%s)" > "$ST/current.txt"
  say "stage $name: START  ($*)"
  local t0=$(date +%s)
  "$@" 2>&1 | tee -a "$OW_RESULTS/logs/stage_$name.log"
  local rc=${PIPESTATUS[0]}
  local dt=$(( $(date +%s) - t0 ))
  timing_add "$name" "$dt"
  if [ $rc -ne 0 ]; then
    say "stage $name: FAILED (exit $rc). Logs: $OW_RESULTS/logs/  -- fix, then re-run the same command."
    echo "failed:$name" > "$ST/current.txt"
    exit $rc
  fi
  touch "$ST/$name.done"
  say "stage $name: done in $((dt/60)) min"
}

say "===== run_phase2.sh $CFG ====="
stage p2_model    python -m owsim.get_model
stage p2_features python -m owsim.launch features --config "$CFG"
stage p2_train    python -m owsim.launch train --config "$CFG" --phase 2
stage p2_offline  python -m owsim.eval_offline --device cuda
stage p2_control  python -m owsim.launch eval --config "$CFG" --phase 2
echo "finished $(date +%s)" > "$ST/current.txt"
say "PHASE 2 DONE"
