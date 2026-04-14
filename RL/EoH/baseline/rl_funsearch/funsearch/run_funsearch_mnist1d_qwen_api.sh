#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data1/lz/clrs/openevolve/RL/EoH/baseline/rl_funsearch/funsearch"
SCRIPT_PATH="$BASE_DIR/funsearch_mnist1d_qwen_api.py"
LOG_DIR="$BASE_DIR/logs/funsearch_rl_qwen_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"

# ===== API settings =====
export FUNSEARCH_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export FUNSEARCH_API_KEY=" "
export FUNSEARCH_MODEL_NAME="qwen-plus"
export FUNSEARCH_API_TIMEOUT="1800"
export FUNSEARCH_TEMPERATURE="0.8"
export FUNSEARCH_TOP_P="0.95"
export FUNSEARCH_MAX_NEW_TOKENS="512"

# ===== MNIST1D evaluator runtime =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MNIST1D_DEVICE="${MNIST1D_DEVICE:-cuda}"
export MNIST1D_SHUFFLE="${MNIST1D_SHUFFLE:-1}"
export MNIST1D_TOTAL_STEPS="${MNIST1D_TOTAL_STEPS:-8000}"
export MNIST1D_PRINT_EVERY="${MNIST1D_PRINT_EVERY:-500}"
export MNIST1D_EVAL_EVERY="${MNIST1D_EVAL_EVERY:-250}"
export MNIST1D_CHECKPOINT_EVERY="${MNIST1D_CHECKPOINT_EVERY:-1000}"
export MNIST1D_SEED="${MNIST1D_SEED:-42}"
export MNIST1D_TORCH_NUM_THREADS="${MNIST1D_TORCH_NUM_THREADS:-1}"
export MNIST1D_TORCH_NUM_INTEROP_THREADS="${MNIST1D_TORCH_NUM_INTEROP_THREADS:-1}"

# ===== FunSearch runtime =====
export FUNSEARCH_MAX_SAMPLES="${FUNSEARCH_MAX_SAMPLES:-100}"
export FUNSEARCH_SAMPLES_PER_PROMPT="${FUNSEARCH_SAMPLES_PER_PROMPT:-4}"
export FUNSEARCH_LOG_DIR="$LOG_DIR"

cd "$BASE_DIR"
python "$SCRIPT_PATH" 2>&1 | tee "$LOG_DIR/run.log"

