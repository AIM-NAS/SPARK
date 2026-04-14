#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data1/lz/clrs/openevolve/mnist1d/EoH/EoH/baseline/funsearch"
SCRIPT_PATH="$BASE_DIR/funsearch_mnist1d_local_llm.py"
LOG_DIR="$BASE_DIR/logs/funsearch_mnist1d_local_llm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"

# ===== MNIST1D evaluator runtime =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MNIST1D_DEVICE="${MNIST1D_DEVICE:-cuda}"
export MNIST1D_SHUFFLE="${MNIST1D_SHUFFLE:-1}"
export MNIST1D_TOTAL_STEPS="${MNIST1D_TOTAL_STEPS:-2000}"
export MNIST1D_PRINT_EVERY="${MNIST1D_PRINT_EVERY:-500}"
export MNIST1D_EVAL_EVERY="${MNIST1D_EVAL_EVERY:-250}"
export MNIST1D_CHECKPOINT_EVERY="${MNIST1D_CHECKPOINT_EVERY:-1000}"
export MNIST1D_SEED="${MNIST1D_SEED:-42}"
export MNIST1D_TORCH_NUM_THREADS="${MNIST1D_TORCH_NUM_THREADS:-1}"
export MNIST1D_TORCH_NUM_INTEROP_THREADS="${MNIST1D_TORCH_NUM_INTEROP_THREADS:-1}"

# ===== FunSearch runtime =====
export FUNSEARCH_MAX_SAMPLES="${FUNSEARCH_MAX_SAMPLES:-20}"
export FUNSEARCH_SAMPLES_PER_PROMPT="${FUNSEARCH_SAMPLES_PER_PROMPT:-4}"
export FUNSEARCH_LOG_DIR="$LOG_DIR"
# 如果你的本地 LLM server 不是这个地址，再单独 export 覆盖即可
export FUNSEARCH_LLM_URL="${FUNSEARCH_LLM_URL:-http://127.0.0.1:11011/completions}"

cd "$BASE_DIR"
python "$SCRIPT_PATH" 2>&1 | tee "$LOG_DIR/run.log"
