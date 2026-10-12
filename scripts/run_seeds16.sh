#!/bin/bash
cd "$(dirname "$0")/.."
source ./env.sh
export OW_RESULTS="$PROJECT_ROOT/results"
echo "$(date '+%F %T') [seeds16] START" | tee -a results/logs/pipeline.log
python -m owsim.seeds16 launch --config full 2>&1 | tee -a results/logs/stage_seeds16.log
echo "$(date '+%F %T') [seeds16] DONE  -> results/seeds16/" | tee -a results/logs/pipeline.log
