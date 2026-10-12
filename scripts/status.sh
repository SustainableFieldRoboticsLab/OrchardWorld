#!/bin/bash
CFG="${1:-full}"
cd "$(dirname "$0")/.."
source ./env.sh
if [ "$CFG" = "full" ]; then export OW_RESULTS="$PROJECT_ROOT/results"; else export OW_RESULTS="$PROJECT_ROOT/results_$CFG"; fi
if [ "${2:-}" = "--once" ]; then python -m owsim.status; exit 0; fi
while true; do
  clear
  python -m owsim.status
  echo; echo "(refreshes every 30 s; Ctrl-c to stop this view - the experiment keeps running)"
  sleep 30
done
