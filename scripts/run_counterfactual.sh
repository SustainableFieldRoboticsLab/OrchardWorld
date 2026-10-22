#!/bin/bash
set -uo pipefail
CFG="${1:-full}"
cd "$(dirname "$0")/.."
source ./env.sh
if [ "$CFG" = "full" ]; then export OW_RESULTS="$PROJECT_ROOT/results"; else export OW_RESULTS="$PROJECT_ROOT/results_$CFG"; fi
ST="$OW_RESULTS/stages"
LOG="$OW_RESULTS/logs/pipeline.log"
mkdir -p "$ST" "$OW_RESULTS/logs"
say() { echo "$(date '+%F %T') [counterfactual] $*" | tee -a "$LOG"; }
if [ ! -f "$ST/dataset.done" ] || [ ! -f "$ST/train.done" ]; then
  say "phase 1 is not finished in $OW_RESULTS (need stages dataset + train). Run scripts/run_all.sh $CFG first."
  exit 1
fi
stage() {
  local name="$1"; shift
  if [ -f "$ST/$name.done" ]; then say "stage $name: already done, skipping"; return 0; fi
  echo "$name $(date +%s)" > "$ST/current.txt"
  say "stage $name: START"
  local t0=$(date +%s)
  "$@" 2>&1 | tee -a "$OW_RESULTS/logs/stage_$name.log"
  local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then say "stage $name: FAILED (exit $rc) - see $OW_RESULTS/logs/"; exit $rc; fi
  touch "$ST/$name.done"
  say "stage $name: done in $(( ($(date +%s) - t0) / 60 )) min"
}
stage cf_collect python -m owsim.counterfactual collect --config "$CFG"
stage cf_dataset python -m owsim.counterfactual dataset
stage cf_train   python -m owsim.counterfactual train   --config "$CFG"
stage cf_analyze python -m owsim.counterfactual analyze
stage cf_control python -m owsim.counterfactual control --config "$CFG"
stage cf_report  python -m owsim.counterfactual report
echo "finished $(date +%s)" > "$ST/current.txt"
say "COUNTERFACTUAL DONE -> $OW_RESULTS/cf/summary.md"
