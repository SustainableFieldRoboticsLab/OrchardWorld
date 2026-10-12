#!/bin/bash
cd "$(dirname "$0")/.."
source ./env.sh
export OW_RESULTS="$PROJECT_ROOT/results"
echo "$(date '+%F %T') [theta] START" | tee -a results/logs/pipeline.log
python -m owsim.theta_sweep launch --config full --hours "${1:-1000}" 2>&1 | tee -a results/logs/stage_theta_sweep.log
echo "$(date '+%F %T') [theta] DONE  -> results/theta_sweep/" | tee -a results/logs/pipeline.log
