#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found (https://docs.astral.sh/uv/)."; exit 1
fi
PY=/usr/bin/python3.12
[ -x "$PY" ] || PY=python3.12

if [ ! -x .venv/bin/python ]; then
  echo ">>> creating .venv with $PY"
  uv venv --python "$PY" .venv
fi
source .venv/bin/activate
echo ">>> installing packages"
if ! uv pip install -r requirements.txt; then
  echo ">>> full install failed; trying the minimal headless set (requirements-min.txt)"
  uv pip install -r requirements-min.txt
fi

echo ">>> quick environment check"
python -m owsim.check --quick
echo "Setup done."
