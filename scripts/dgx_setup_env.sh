#!/usr/bin/env bash
# Create the dedicated Python environment for the Harry Potter RNN project.
#
# Written for the DGX (ai-n003, 8x A100-SXM4-40GB, driver CUDA 13.0).
# Idempotent: re-running only installs what is missing.
#
# Usage:
#   bash scripts/dgx_setup_env.sh              # default: $HOME/projects/hp-rnn
#   PROJECT_DIR=/path/to/repo bash scripts/dgx_setup_env.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/hp-rnn}"
VENV="$PROJECT_DIR/.venv"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"

echo "== project dir: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "== selecting base interpreter"
# The system python3 lacks ensurepip (python3.12-venv is not installed and we
# have no sudo on this cluster), so prefer a conda interpreter that ships one.
VENV_BASE_PYTHON="${VENV_BASE_PYTHON:-}"
if [ -z "$VENV_BASE_PYTHON" ]; then
  for candidate in \
    "$HOME/miniconda3/envs/ai231/bin/python" \
    "$HOME/miniconda3/bin/python" \
    "$(command -v python3)"
  do
    if [ -n "$candidate" ] && [ -x "$candidate" ] \
       && "$candidate" -c 'import ensurepip' >/dev/null 2>&1; then
      VENV_BASE_PYTHON="$candidate"
      break
    fi
  done
fi
if [ -z "$VENV_BASE_PYTHON" ]; then
  echo "ERROR: no interpreter with ensurepip found; set VENV_BASE_PYTHON" >&2
  exit 1
fi
echo "== base python: $VENV_BASE_PYTHON"
"$VENV_BASE_PYTHON" --version

if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating venv at $VENV"
  "$VENV_BASE_PYTHON" -m venv "$VENV"
else
  echo "== reusing existing venv at $VENV"
fi

echo "== upgrading pip"
"$VENV/bin/python" -m pip install --quiet --upgrade pip wheel

echo "== installing torch from $TORCH_INDEX"
"$VENV/bin/python" -m pip install --index-url "$TORCH_INDEX" torch

echo "== installing project dependencies"
"$VENV/bin/python" -m pip install EbookLib beautifulsoup4 pytest ruff

echo "== environment verification"
"$VENV/bin/python" - <<'PY'
import torch

print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
print(f"cuda version   : {torch.version.cuda}")
print(f"device count   : {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; check the driver/visible devices")

print(f"device 0       : {torch.cuda.get_device_name(0)}")

# Stage 1 gate: a real tensor operation must run on the A100.
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
print(f"matmul on A100 : {tuple(y.shape)} sum={y.sum().item():.4f}")
print("STAGE1_GPU_OK")
PY

echo "== done"
echo "activate with: source $VENV/bin/activate"
