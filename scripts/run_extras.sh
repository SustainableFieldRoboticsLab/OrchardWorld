#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
source ./env.sh
export OW_RESULTS="$PROJECT_ROOT/results"
ST="$OW_RESULTS/stages"
LOG="$OW_RESULTS/logs/pipeline.log"
say() { echo "$(date '+%F %T') [extras] $*" | tee -a "$LOG"; }
if [ ! -f "$ST/p2_control.done" ]; then
  say "phases 1 and 2 must be finished first (missing $ST/p2_control.done)"; exit 1
fi
stage() {
  local name="$1"; shift
  if [ -f "$ST/$name.done" ]; then say "stage $name: already done, skipping"; return 0; fi
  echo "$name $(date +%s)" > "$ST/current.txt"
  say "stage $name: START"
  "$@" 2>&1 | tee -a "$OW_RESULTS/logs/stage_$name.log"
  local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then say "stage $name: FAILED (exit $rc) - see $OW_RESULTS/logs/"; exit $rc; fi
  touch "$ST/$name.done"
  say "stage $name: done"
}
stage x_fit     python -m owsim.extras fit
stage x_control python -m owsim.extras launch --config full
stage x_report  python -m owsim.extras report
echo "finished $(date +%s)" > "$ST/current.txt"
say "EXTRAS DONE -> $OW_RESULTS/extras/summary.md"
