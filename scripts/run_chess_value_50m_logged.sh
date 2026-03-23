#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REQUESTED_DEVICE="${WAVEPDE_DEVICE:-gpu}"
export WAVEPDE_DEVICE="$REQUESTED_DEVICE"

if [[ "$REQUESTED_DEVICE" == "gpu" || "$REQUESTED_DEVICE" == "cuda" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WAVEPDE_DEVICE=$REQUESTED_DEVICE but nvidia-smi is not available. Refusing to run long CPU fallback." >&2
    exit 1
  fi
fi

DATA_DIR="${CHESS_DATA_DIR:-tmp/value_runtime/stockfish_1m_fixed/train}"
CHECKPOINT_PATH="${WAVEPDE_CHECKPOINT:-tmp/value_runtime/stockfish_1m_fixed/checkpoints/value_50m_logged_checkpoint.jls}"
LOG_PATH="${WAVEPDE_LOG_PATH:-${CHECKPOINT_PATH%.jls}.log}"

mkdir -p "$(dirname "$CHECKPOINT_PATH")"
mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "entrypoint=run_chess_value_50m_logged"
  echo "cwd=$ROOT_DIR"
  echo "data_dir=$DATA_DIR"
  echo "checkpoint=$CHECKPOINT_PATH"
  echo "log_path=$LOG_PATH"
  echo "requested_device=$REQUESTED_DEVICE"
  echo "batch_size=${WAVEPDE_BATCH_SIZE:-32}"
  echo "max_iters=${WAVEPDE_MAX_ITERS:-8000}"
  echo "chunk_rows=${WAVEPDE_CHUNK_ROWS:-20000}"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee "$LOG_PATH"

stdbuf -oL -eL julia --project=. scripts/train_chess_value_50m.jl 2>&1 | tee -a "$LOG_PATH"
