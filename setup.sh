#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found (https://docs.astral.sh/uv/)."; exit 1
fi

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate
if ! uv pip install -r requirements.txt; then
  uv pip install -r requirements-min.txt
fi

python -m owsim.check --quick
echo "Setup done. Next:  bash scripts/start_tmux.sh smoke"
