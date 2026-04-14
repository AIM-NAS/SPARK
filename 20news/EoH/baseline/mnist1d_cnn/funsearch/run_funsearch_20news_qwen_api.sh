#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data1/lz/clrs/openevolve/20news/EoH/baseline/20news/funsearch"
SCRIPT_PATH="$BASE_DIR/funsearch_20news_qwen_api.py"
LOG_DIR="$BASE_DIR/logs/funsearch_news20_qwen_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/run.log"

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"

# ===== API settings =====
export FUNSEARCH_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export FUNSEARCH_API_KEY=" "
export FUNSEARCH_MODEL_NAME="qwen-plus"
export FUNSEARCH_API_TIMEOUT="${FUNSEARCH_API_TIMEOUT:-1800}"
export FUNSEARCH_TEMPERATURE="${FUNSEARCH_TEMPERATURE:-0.8}"
export FUNSEARCH_TOP_P="${FUNSEARCH_TOP_P:-0.95}"
export FUNSEARCH_MAX_NEW_TOKENS="${FUNSEARCH_MAX_NEW_TOKENS:-512}"

# ===== 20news evaluator runtime =====
export NEWS20_DEVICE="${NEWS20_DEVICE:-cpu}"
export NEWS20_SEED="${NEWS20_SEED:-42}"
export NEWS20_PROJ_DIM="${NEWS20_PROJ_DIM:-512}"
export NEWS20_NUM_CLASSES="${NEWS20_NUM_CLASSES:-20}"
export NEWS20_HIDDEN_SIZE="${NEWS20_HIDDEN_SIZE:-128}"
export NEWS20_DROPOUT="${NEWS20_DROPOUT:-0.2}"
export NEWS20_LR="${NEWS20_LR:-1e-3}"
export NEWS20_WEIGHT_DECAY="${NEWS20_WEIGHT_DECAY:-1e-4}"
export NEWS20_BATCH_SIZE="${NEWS20_BATCH_SIZE:-256}"
export NEWS20_EPOCHS="${NEWS20_EPOCHS:-10}"
export NEWS20_USE_NORMALIZER="${NEWS20_USE_NORMALIZER:-1}"
export NEWS20_STAGE2_TRAIN_LIMIT="${NEWS20_STAGE2_TRAIN_LIMIT:-0}"
export NEWS20_STAGE2_TEST_LIMIT="${NEWS20_STAGE2_TEST_LIMIT:-0}"
export NEWS20_STAGE1_TRAIN_LIMIT="${NEWS20_STAGE1_TRAIN_LIMIT:-1500}"
export NEWS20_STAGE1_TEST_LIMIT="${NEWS20_STAGE1_TEST_LIMIT:-1000}"
export NEWS20_STAGE1_EPOCHS="${NEWS20_STAGE1_EPOCHS:-2}"
export NEWS20_STAGE1_BATCH_SIZE="${NEWS20_STAGE1_BATCH_SIZE:-256}"
export NEWS20_TORCH_NUM_THREADS="${NEWS20_TORCH_NUM_THREADS:-1}"
export NEWS20_TORCH_NUM_INTEROP_THREADS="${NEWS20_TORCH_NUM_INTEROP_THREADS:-1}"

# ===== FunSearch runtime =====
export FUNSEARCH_MAX_SAMPLES="${FUNSEARCH_MAX_SAMPLES:-100}"
export FUNSEARCH_SAMPLES_PER_PROMPT="${FUNSEARCH_SAMPLES_PER_PROMPT:-4}"
export FUNSEARCH_EVAL_TIMEOUT="${FUNSEARCH_EVAL_TIMEOUT:-7200}"
export FUNSEARCH_LOG_DIR="$LOG_DIR"

cd "$BASE_DIR"

{
  echo "=================================================="
  echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "BASE_DIR: $BASE_DIR"
  echo "SCRIPT_PATH: $SCRIPT_PATH"
  echo "LOG_DIR: $LOG_DIR"
  echo "LOG_FILE: $LOG_FILE"
  echo "NEWS20_DEVICE: $NEWS20_DEVICE"
  echo "NEWS20_EPOCHS: $NEWS20_EPOCHS"
  echo "FUNSEARCH_MAX_SAMPLES: $FUNSEARCH_MAX_SAMPLES"
  echo "FUNSEARCH_SAMPLES_PER_PROMPT: $FUNSEARCH_SAMPLES_PER_PROMPT"
  echo "FUNSEARCH_EVAL_TIMEOUT: $FUNSEARCH_EVAL_TIMEOUT"
  echo "=================================================="
} | tee "$LOG_FILE"

python "$SCRIPT_PATH" 2>&1 | tee -a "$LOG_FILE"

{
  echo "=================================================="
  echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Log saved to: $LOG_FILE"
  echo "=================================================="
} | tee -a "$LOG_FILE"
