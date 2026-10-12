PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

export SHARED="${SHARED:-$PROJECT_ROOT/.cache_shared}"
case ":$PATH:" in *":$SHARED/bin:"*) ;; *) export PATH="$SHARED/bin:$PATH" ;; esac

export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu129}"
export UV_CACHE_DIR="$SHARED/uv/cache" UV_PYTHON_INSTALL_DIR="$SHARED/uv/python"
export UV_TOOL_DIR="$SHARED/uv/tools" UV_TOOL_BIN_DIR="$SHARED/bin" UV_LINK_MODE=hardlink
export PIP_CACHE_DIR="$SHARED/pip"
export HF_HOME="${HF_HOME:-$SHARED/huggingface}" HF_HUB_DISABLE_TELEMETRY=1 TORCH_HOME="$SHARED/torch"
export XDG_CACHE_HOME="$SHARED/cache" XDG_CONFIG_HOME="$SHARED/config" XDG_DATA_HOME="$SHARED/data"
export MPLCONFIGDIR="$SHARED/config/matplotlib"
export TRITON_HOME="$SHARED/triton" TRITON_CACHE_DIR="$SHARED/triton/cache"
export CUDA_CACHE_PATH="$SHARED/nv/ComputeCache" NUMBA_CACHE_DIR="$SHARED/cache/numba"
export TMPDIR="$SHARED/tmp"
export WARP_CACHE_PATH="$SHARED/cache/warp"
export NEWTON_CACHE_PATH="$SHARED/cache/newton"
mkdir -p "$SHARED/bin" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
         "$XDG_DATA_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR" \
         "$WARP_CACHE_PATH" "$NEWTON_CACHE_PATH" 2>/dev/null

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi
