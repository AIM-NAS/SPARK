#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$BASE_DIR/funsearch_rl_qwen_api.py"
INITIAL_PROGRAM_PATH="$BASE_DIR/initial_program.py"
EVALUATOR_PATH="$BASE_DIR/evaluator.py"
LOG_DIR="$BASE_DIR/logs/funsearch_rl_qwen_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"
export FUNSEARCH_API_KEY=" "
# ===== Task files =====
export FUNSEARCH_INITIAL_PROGRAM="$INITIAL_PROGRAM_PATH"
export FUNSEARCH_EVALUATOR_PATH="$EVALUATOR_PATH"

# ===== API settings =====
export FUNSEARCH_API_BASE="${FUNSEARCH_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export FUNSEARCH_API_KEY="${FUNSEARCH_API_KEY:-}"
export FUNSEARCH_MODEL_NAME="${FUNSEARCH_MODEL_NAME:-qwen-plus}"
export FUNSEARCH_API_TIMEOUT="${FUNSEARCH_API_TIMEOUT:-1800}"
export FUNSEARCH_TEMPERATURE="${FUNSEARCH_TEMPERATURE:-0.8}"
export FUNSEARCH_TOP_P="${FUNSEARCH_TOP_P:-0.95}"
export FUNSEARCH_MAX_NEW_TOKENS="${FUNSEARCH_MAX_NEW_TOKENS:-768}"
export FUNSEARCH_MAX_RETRIES_PER_SAMPLE="${FUNSEARCH_MAX_RETRIES_PER_SAMPLE:-8}"

if [[ -z "${FUNSEARCH_API_KEY}" ]]; then
  echo "[ERROR] FUNSEARCH_API_KEY is empty. Please export it before running." >&2
  exit 1
fi

# ===== RL evaluator runtime =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RL_DEVICE="${RL_DEVICE:-auto}"
export RL_ENV_NAME="${RL_ENV_NAME:-CartPole-v1}"
export RL_INPUT_DIM="${RL_INPUT_DIM:-4}"
export RL_OUTPUT_DIM="${RL_OUTPUT_DIM:-2}"
export RL_MAX_PARAM_COUNT="${RL_MAX_PARAM_COUNT:-10000}"
export RL_MAX_STEPS_PER_EPISODE="${RL_MAX_STEPS_PER_EPISODE:-500}"
export RL_TRAIN_EPISODES="${RL_TRAIN_EPISODES:-50}"
export RL_TEST_EPISODES="${RL_TEST_EPISODES:-10}"
export RL_GAMMA="${RL_GAMMA:-0.99}"
export RL_LR="${RL_LR:-1e-2}"
export RL_EVAL_SEEDS="${RL_EVAL_SEEDS:-42,123}"
export RL_PARAM_PENALTY="${RL_PARAM_PENALTY:-0.0001}"
export RL_TORCH_NUM_THREADS="${RL_TORCH_NUM_THREADS:-1}"
export RL_TORCH_NUM_INTEROP_THREADS="${RL_TORCH_NUM_INTEROP_THREADS:-1}"

# ===== FunSearch runtime =====
export FUNSEARCH_MAX_SAMPLES="${FUNSEARCH_MAX_SAMPLES:-100}"
export FUNSEARCH_SAMPLES_PER_PROMPT="${FUNSEARCH_SAMPLES_PER_PROMPT:-4}"
export FUNSEARCH_LOG_DIR="$LOG_DIR"

cd "$BASE_DIR"
python "$SCRIPT_PATH" 2>&1 | tee "$LOG_DIR/run.log"

