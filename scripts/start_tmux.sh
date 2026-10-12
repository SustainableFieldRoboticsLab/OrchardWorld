#!/bin/bash
ARG="${1:-full}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$ARG" in
  phase2)       CFG=full;  RUN="run_phase2.sh" ;;
  phase2-smoke) CFG=smoke; RUN="run_phase2.sh" ;;
  extras)       CFG=full;  RUN="run_extras.sh" ;;
  *)            CFG="$ARG"; RUN="run_all.sh" ;;
esac
S="${OW_TMUX_SESSION:-orchard}"
if [ "$CFG" != "full" ]; then RES="$ROOT/results_$CFG"; else RES="$ROOT/results"; fi

if tmux has-session -t "$S" 2>/dev/null; then
  echo "tmux session '$S' already exists.  Attach with:  tmux attach -t $S"
  echo "(to start over: tmux kill-session -t $S, then run this again - finished stages are kept)"
  exit 0
fi
tmux new-session -d -s "$S" -n run -c "$ROOT" \
  "bash -c 'bash $ROOT/scripts/$RUN $CFG; echo; echo \"[pipeline exited - this window stays open]\"; exec bash'"
tmux new-window -t "$S" -n status -c "$ROOT" "bash -c 'bash $ROOT/scripts/status.sh $CFG; exec bash'"
tmux new-window -t "$S" -n gpu -c "$ROOT" "bash -c 'watch -n 5 nvidia-smi; exec bash'"
tmux select-window -t "$S:run"
echo "Started. Results will be in $RES"
echo "  attach:          tmux attach -t $S"
echo "  switch windows:  Ctrl-b n (next) / Ctrl-b p (previous) / Ctrl-b w (list): run, status, gpu"
echo "  detach:          Ctrl-b d      (everything keeps running)"
echo
echo "The experiment runs INSIDE tmux: you can detach (Ctrl-b d) and log out safely."
echo "Do NOT type 'exit' or press Ctrl-c in the 'run' window - that stops it."
