#!/usr/bin/env bash

# =========================================================
# 4-GPU queue runner for CLRS remaining tasks
# - 固定 4 个 worker，对应 GPU 0/1/2/3
# - 每张卡同一时刻只跑 1 个任务
# - 某个任务失败不影响其他任务
# - 每个 (algorithm, train_length) 单独日志、单独 checkpoint
# - 日志名: 任务名_训练长度.log
# =========================================================

set -u
set -o pipefail

############################
# 可修改区域
############################

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_PY="${RUN_PY:-./run.py}"

# 4 张 GPU
GPUS=(0 1 2 3)

# 失败重试次数（例如 1 表示失败后再试 1 次，总共最多跑 2 次）
RETRIES="${RETRIES:-1}"

# 单个任务超时（秒），0 表示不限制
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"

# 剩余任务
TASKS=(
  binary_search
  bridges
  bubble_sort
  graham_scan
  heapsort
  insertion_sort
  jarvis_march
  kmp_matcher
  mst_kruskal
  naive_string_matcher
  optimal_bst
  quicksort
  strongly_connected_components
)

# train_lengths 全测
#TRAIN_LENGTHS=(4 7 11 13 16)
TRAIN_LENGTHS=(16)
# 输出目录
ROOT_DIR="$(pwd)"
LOG_DIR="${ROOT_DIR}/logs_gpu_queue"
CKPT_DIR="${ROOT_DIR}/checkpoints_gpu_queue"
STATUS_DIR="${ROOT_DIR}/status_gpu_queue"
QUEUE_FILE="${ROOT_DIR}/jobs_gpu_queue.txt"
SUMMARY_FILE="${ROOT_DIR}/summary_gpu_queue.txt"
LOCK_DIR="${ROOT_DIR}/locks_gpu_queue"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$STATUS_DIR" "$LOCK_DIR"

# 你平时 run.py 还要补的其他参数，直接填在这里
EXTRA_ARGS=(
  # 例如：
  # "--processor_type=triplet_gmpnn"
  # "--batch_size=32"
  # "--train_steps=10000"
  # "--eval_every=50"
  # "--use_ln=true"
  # "--dataset_path=/data1/lz/clrs/openevolve/dataset/CLRS30_v1.0.0.tar/CLRS30_v1.0.0/clrs_dataset"
)

############################
# 基础检查
############################

if [[ ! -f "$RUN_PY" ]]; then
  echo "[FATAL] run.py 不存在: $RUN_PY"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[FATAL] Python 不存在: $PYTHON_BIN"
  exit 1
fi

: > "$SUMMARY_FILE"
: > "$QUEUE_FILE"

############################
# 构建任务队列
############################

for algo in "${TASKS[@]}"; do
  for train_len in "${TRAIN_LENGTHS[@]}"; do
    echo "${algo},${train_len}" >> "$QUEUE_FILE"
  done
done

TOTAL_JOBS=$(wc -l < "$QUEUE_FILE")
echo "[INFO] total jobs = ${TOTAL_JOBS}" | tee -a "$SUMMARY_FILE"

############################
# 原子取任务
############################
# 用 mkdir 充当轻量锁，避免多个 worker 同时取到同一行

pop_job() {
  local lock="${LOCK_DIR}/queue.lock"

  while ! mkdir "$lock" 2>/dev/null; do
    sleep 0.2
  done

  # 临界区开始
  if [[ ! -s "$QUEUE_FILE" ]]; then
    rmdir "$lock"
    return 1
  fi

  local job
  job="$(head -n 1 "$QUEUE_FILE")"

  # 删除第一行
  tail -n +2 "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"

  rmdir "$lock"

  echo "$job"
  return 0
}

############################
# 单个任务执行
############################

run_one_job() {
  local gpu="$1"
  local algo="$2"
  local train_len="$3"

  local job_name="${algo}_${train_len}"
  local log_file="${LOG_DIR}/${job_name}.log"
  local ckpt_path="${CKPT_DIR}/${job_name}"
  local status_file="${STATUS_DIR}/${job_name}.status"

  mkdir -p "$ckpt_path"
  rm -f "$status_file"

  {
    echo "=================================================="
    echo "[START] $(date '+%F %T')"
    echo "JOB_NAME=${job_name}"
    echo "ALGORITHM=${algo}"
    echo "TRAIN_LENGTH=${train_len}"
    echo "GPU=${gpu}"
    echo "CHECKPOINT_PATH=${ckpt_path}"
    echo "LOG_FILE=${log_file}"
    echo "=================================================="
    echo
  } > "$log_file"

  local attempt=0
  local rc=1

  while (( attempt <= RETRIES )); do
    attempt=$((attempt + 1))

    {
      echo "[ATTEMPT ${attempt}] $(date '+%F %T')"
      echo "[CMD] CUDA_VISIBLE_DEVICES=${gpu} ${PYTHON_BIN} -u ${RUN_PY} --algorithms=${algo} --train_lengths=${train_len} --checkpoint_path=${ckpt_path} ${EXTRA_ARGS[*]}"
      echo
    } >> "$log_file"

    if [[ "$TIMEOUT_SEC" -gt 0 ]] && command -v timeout >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="${gpu}" \
      timeout --signal=TERM --kill-after=60 "${TIMEOUT_SEC}" \
        "${PYTHON_BIN}" -u "${RUN_PY}" \
        --algorithms="${algo}" \
        --train_lengths="${train_len}" \
        --checkpoint_path="${ckpt_path}" \
        "${EXTRA_ARGS[@]}" >> "$log_file" 2>&1
      rc=$?
    else
      CUDA_VISIBLE_DEVICES="${gpu}" \
        "${PYTHON_BIN}" -u "${RUN_PY}" \
        --algorithms="${algo}" \
        --train_lengths="${train_len}" \
        --checkpoint_path="${ckpt_path}" \
        "${EXTRA_ARGS[@]}" >> "$log_file" 2>&1
      rc=$?
    fi

    if [[ "$rc" -eq 0 ]]; then
      {
        echo
        echo "[SUCCESS] $(date '+%F %T')"
        echo "EXIT_CODE=${rc}"
      } >> "$log_file"
      echo "SUCCESS ${job_name} GPU=${gpu}" > "$status_file"
      echo "[OK] ${job_name} on GPU ${gpu}"
      return 0
    else
      {
        echo
        echo "[FAIL] $(date '+%F %T')"
        echo "EXIT_CODE=${rc}"
      } >> "$log_file"
      echo "[WARN] ${job_name} failed on GPU ${gpu}, attempt=${attempt}, exit_code=${rc}"
      sleep 2
    fi
  done

  echo "FAIL ${job_name} GPU=${gpu} EXIT_CODE=${rc}" > "$status_file"
  echo "[FAIL] ${job_name} on GPU ${gpu}"
  return 0
}

############################
# Worker：每个 GPU 一个
############################

worker_loop() {
  local gpu="$1"

  while true; do
    local item
    item="$(pop_job)" || break

    local algo="${item%%,*}"
    local train_len="${item##*,}"

    run_one_job "$gpu" "$algo" "$train_len"
  done

  echo "[WORKER DONE] GPU ${gpu}"
}

############################
# 启动 4 个 worker
############################

echo "[INFO] queue file: $QUEUE_FILE"
echo "[INFO] logs dir  : $LOG_DIR"
echo "[INFO] ckpt dir  : $CKPT_DIR"
echo "[INFO] status dir: $STATUS_DIR"
echo "[INFO] gpus      : ${GPUS[*]}"
echo

for gpu in "${GPUS[@]}"; do
  worker_loop "$gpu" &
done

wait

############################
# 汇总
############################

ok_count=0
fail_count=0

{
  echo
  echo "================ SUMMARY ================"
  echo "TIME: $(date '+%F %T')"
  echo "TOTAL_JOBS=${TOTAL_JOBS}"
  echo
} >> "$SUMMARY_FILE"

for algo in "${TASKS[@]}"; do
  for train_len in "${TRAIN_LENGTHS[@]}"; do
    job_name="${algo}_${train_len}"
    status_file="${STATUS_DIR}/${job_name}.status"

    if [[ -f "$status_file" ]]; then
      cat "$status_file" >> "$SUMMARY_FILE"
      if grep -q "^SUCCESS " "$status_file"; then
        ok_count=$((ok_count + 1))
      else
        fail_count=$((fail_count + 1))
      fi
    else
      echo "FAIL ${job_name} GPU=UNKNOWN EXIT_CODE=UNKNOWN" >> "$SUMMARY_FILE"
      fail_count=$((fail_count + 1))
    fi
  done
done

{
  echo
  echo "OK_COUNT=${ok_count}"
  echo "FAIL_COUNT=${fail_count}"
  echo "SUMMARY_FILE=${SUMMARY_FILE}"
} | tee -a "$SUMMARY_FILE"

echo
echo "[DONE] All jobs finished."
echo "[DONE] Summary: $SUMMARY_FILE"