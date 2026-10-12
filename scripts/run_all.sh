#!/bin/bash
set -uo pipefail
CFG="${1:-full}"
cd "$(dirname "$0")/.."
source ./env.sh

if [ "$CFG" = "full" ]; then
  export OW_RESULTS="$PROJECT_ROOT/results"
else
  export OW_RESULTS="$PROJECT_ROOT/results_$CFG"
fi
mkdir -p "$OW_RESULTS/stages" "$OW_RESULTS/logs"
LOG="$OW_RESULTS/logs/pipeline.log"
ST="$OW_RESULTS/stages"
[ -f "$ST/pipeline.json" ] || echo "{\"config\": \"$CFG\", \"started\": $(date +%s)}" > "$ST/pipeline.json"

say() { echo "$(date '+%F %T') [pipeline] $*" | tee -a "$LOG"; }

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
    say "stage $name: FAILED (exit $rc) after ${dt}s. Logs: $OW_RESULTS/logs/  -- fix, then re-run the same command."
    echo "failed:$name" > "$ST/current.txt"
    exit $rc
  fi
  touch "$ST/$name.done"
  say "stage $name: done in $((dt/60)) min"
}

say "===== run_all.sh $CFG  (results in $OW_RESULTS) ====="
stage check   python -m owsim.check
stage collect python -m owsim.launch collect --config "$CFG"
stage dataset python -m owsim.dataset
stage train   python -m owsim.launch train --config "$CFG"
stage offline python -m owsim.eval_offline --device cuda
stage control python -m owsim.launch eval --config "$CFG"
echo "finished $(date +%s)" > "$ST/current.txt"
say "ALL DONE. Results: $OW_RESULTS"
