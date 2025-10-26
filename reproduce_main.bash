#!/bin/bash


rm -rf general_dir
rm -rf work_dir

# Available GPUs
GPUS=(0 1 2 5 6 7)
NUM_GPUS=${#GPUS[@]}

# Number of tasks
NUM_TASKS=6

# Directory for logs
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for TASK_ID in $(seq 0 $((NUM_TASKS - 1))); do
  GPU_ID=${GPUS[$((TASK_ID % NUM_GPUS))]}

  echo "Launching task $TASK_ID on GPU $GPU_ID ..."

  (
    # Log file names
    TRAIN_LOG="${LOG_DIR}/task_${TASK_ID}_train.log"
    MAIN_LOG="${LOG_DIR}/task_${TASK_ID}_attack.log"

    # Run the two Python scripts sequentially with logging
    echo "[Task $TASK_ID] Starting train_model_clean.py on GPU $GPU_ID" | tee "$TRAIN_LOG"
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_model_clean.py --task_id=$TASK_ID >> "$TRAIN_LOG" 2>&1

    echo "[Task $TASK_ID] Starting main.py on GPU $GPU_ID" | tee "$MAIN_LOG"
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --task_id=$TASK_ID >> "$MAIN_LOG" 2>&1

    echo "[Task $TASK_ID] Completed." | tee -a "$MAIN_LOG"
  ) &
done

# Wait for all background jobs to finish
wait
echo "✅ All tasks completed. Logs saved in $LOG_DIR/"
