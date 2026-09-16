#!/usr/bin/env bash
# Sync this working tree to the DGX and run a command there.
#
# The DGX is reached over SSH from this container (host VPN + WSL mirrored
# networking), and that link occasionally times out, so the sync retries.
#
# Usage:
#   bash scripts/dgx.sh "python -m pytest tests -q"
#   bash scripts/dgx.sh "python -m src.data prepare --input data/raw/book.epub --output-dir data/processed"
#   bash scripts/dgx.sh --no-sync "nvidia-smi"
#
# Environment overrides: DGX_HOST, DGX_KEY, DGX_DIR, DGX_PYTHON
set -euo pipefail

DGX_HOST="${DGX_HOST:-aldwin.fresnido@10.153.60.233}"
DGX_KEY="${DGX_KEY:-/opt/data/home/.ssh/id_ed25519}"
DGX_DIR="${DGX_DIR:-projects/hp-rnn}"
DGX_PYTHON="${DGX_PYTHON:-.venv/bin/python}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-i "$DGX_KEY" -o BatchMode=yes -o ConnectTimeout=20)

DO_SYNC=1
if [ "${1:-}" = "--no-sync" ]; then
  DO_SYNC=0
  shift
fi
REMOTE_COMMAND="${*:?usage: bash scripts/dgx.sh [--no-sync] \"<command>\"}"

if [ "$DO_SYNC" = "1" ]; then
  for attempt in 1 2 3 4 5; do
    if tar czf - -C "$PROJECT_ROOT" \
        --exclude=.git --exclude=.venv --exclude=__pycache__ \
        --exclude=.pytest_cache --exclude=.ruff_cache --exclude=logs \
        --exclude=data --exclude=checkpoints --exclude=results \
        . | ssh "${SSH_OPTS[@]}" "$DGX_HOST" \
        "mkdir -p \$HOME/$DGX_DIR && tar xzf - -C \$HOME/$DGX_DIR"; then
      break
    fi
    if [ "$attempt" = "5" ]; then
      echo "sync failed after 5 attempts" >&2
      exit 1
    fi
    echo "ssh sync attempt $attempt failed; retrying" >&2
    sleep 5
  done
fi

# Run from the project root on the DGX with the project venv activated, so
# `python`, `pytest`, and `ruff` resolve to the project environment.
ssh "${SSH_OPTS[@]}" "$DGX_HOST" \
  "cd \$HOME/$DGX_DIR && source .venv/bin/activate && $REMOTE_COMMAND"
