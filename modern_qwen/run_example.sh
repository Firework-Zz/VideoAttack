#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
DATASET="${DATASET:-nextqa}"
LIMIT="${LIMIT:-100}"
STEPS="${STEPS:-10}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/../outputs/qwen3vl_attack}"

python "${SCRIPT_DIR}/qwen3vl_eval_attack.py" \
  --dataset "${DATASET}" \
  --attack both \
  --model-path "${MODEL_PATH}" \
  --limit "${LIMIT}" \
  --max-frames 8 \
  --steps "${STEPS}" \
  --out-dir "${OUT_DIR}"
