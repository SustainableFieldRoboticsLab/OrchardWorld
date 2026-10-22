#!/bin/bash
set -uo pipefail
WHAT="${1:-}"
cd "$(dirname "$0")/.."
source ./env.sh
export OW_RESULTS="$PROJECT_ROOT/results"
RV="$OW_RESULTS/revision"
ST="$RV/stages"
LOG="$RV/logs/pipeline.log"
mkdir -p "$ST" "$RV/logs"
say() { echo "$(date '+%F %T') [revision] $*" | tee -a "$LOG"; }

stage() {
  local name="$1"; shift
  if [ -f "$ST/$name.done" ]; then say "stage $name: already done, skipping"; return 0; fi
  say "stage $name: START  ($*)"
  local t0=$(date +%s)
  "$@" 2>&1 | tee -a "$RV/logs/stage_$name.log"
  local rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then say "stage $name: FAILED (exit $rc) - see $RV/logs/stage_$name.log"; exit $rc; fi
  touch "$ST/$name.done"
  say "stage $name: done in $(( ($(date +%s) - t0) / 60 )) min"
}

run_e0()  { stage e0  python -m revision.e0_baselines; }
run_ess() { stage ess python -m revision.ess; }
run_e6()  { stage e6  python -m revision.e6_within_state; }
run_e2()  { stage e2_build python -m revision.e2_oracle build
            stage e2_train python -m revision.e2_oracle train
            stage e2_eval  python -m revision.e2_oracle eval; }
run_e3()  { stage e3_fit    python -m revision.e3_ensemble fit
            stage e3_launch python -m revision.e3_ensemble launch; }
run_e1()  { stage e1_smoke  python -m revision.e1_noise smoke
            stage e1_launch python -m revision.e1_noise launch; }
run_e5()  { stage e5_launch python -m revision.e5_robustness launch; }
run_e5f() { stage e5f_launch python -m revision.e5_robustness launch --with-foliage; }

case "$WHAT" in
  quick) run_e0; run_ess; run_e6; run_e2; run_e3 ;;
  all)   run_e0; run_ess; run_e6; run_e2; run_e3; run_e1; run_e5 ;;
  e0) run_e0 ;;  ess) run_ess ;;  e6) run_e6 ;;  e2) run_e2 ;;  e3) run_e3 ;;
  e1) run_e1 ;;  e5) run_e5 ;;  e5f) run_e5f ;;
  *) echo "usage: bash revision/run_revision.sh quick|e1|e5|all|e0|ess|e6|e2|e3|e5f"; exit 2 ;;
esac
say "'$WHAT' FINISHED -> pack with: bash revision/pack_results.sh"
