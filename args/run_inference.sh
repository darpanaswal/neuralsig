#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################

EXPERIMENT="inference"
MODEL="gemma"
ENV="bioterrorism"

# ENVIRONMENTS = [
#     "cybersecurity", "medical", "hate_harassment",
#     "general_crime", "bioterrorism",
# ]

# Number of GPUs to run in Data-Parallel mode
N_GPUS=4

MAX_TOKENS=3000
TEMPERATURE=0.0
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=4096
DTYPE="float16"

PARTITION="gpu_p2"
WALLTIME="12:00:00"

########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${ENV}.txt"

SCRIPT_PATH="$(readlink -f "$0")"

########################################
# Submit to SLURM if not already inside a job
########################################

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"

    sbatch \
        --job-name="${EXPERIMENT}_${ENV}" \
        --output="${LOG_FILE}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$((N_GPUS * 4)) \
        --gres=gpu:${N_GPUS} \
        --time="${WALLTIME}" \
        "${SCRIPT_PATH}"

    exit 0
fi

########################################
# Inside allocated job
########################################
set +u

# ==============================================================================
# SAFE SYMLINK TRICK: Protect $HOME quota from massive Pip/Conda installs
# ==============================================================================
# 1. Route ~/.config (used by vLLM for NCCL binaries)
if [ ! -L "$HOME/.config" ]; then
    mkdir -p "$WORK/.config"
    # Copy contents if the directory exists and isn't empty, then remove the original
    if [ -d "$HOME/.config" ]; then
        cp -a "$HOME/.config/." "$WORK/.config/" 2>/dev/null || true
        rm -rf "$HOME/.config"
    fi
    ln -s "$WORK/.config" "$HOME/.config"
fi

# 2. Route ~/.conda (used by Anaconda for massive environment binaries)
if [ ! -L "$HOME/.conda" ]; then
    mkdir -p "$WORK/.conda"
    # Copy contents if the directory exists and isn't empty, then remove the original
    if [ -d "$HOME/.conda" ]; then
        cp -a "$HOME/.conda/." "$WORK/.conda/" 2>/dev/null || true
        rm -rf "$HOME/.conda"
    fi
    ln -s "$WORK/.conda" "$HOME/.conda"
fi
# ==============================================================================

cd "$WORK/neuralsig"
module purge
module load anaconda-py3/2024.06
source $WORK/env_cache_guard.sh
conda activate neusig

set -u

########################################
# Logging
########################################

: > "${LOG_FILE}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "Batched vLLM inference (Data Parallelism)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

echo "SLURM Job ID    : ${SLURM_JOB_ID}" | tee -a "${LOG_FILE}"
echo "Node            : ${SLURMD_NODENAME}" | tee -a "${LOG_FILE}"
echo "Partition       : ${SLURM_JOB_PARTITION}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

echo "Environment     : ${ENV}" | tee -a "${LOG_FILE}"
echo "GPUs for DP     : ${N_GPUS}" | tee -a "${LOG_FILE}"
echo "Max tokens      : ${MAX_TOKENS}" | tee -a "${LOG_FILE}"
echo "Temperature     : ${TEMPERATURE}" | tee -a "${LOG_FILE}"
echo "Log file        : ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# Safely extract the available GPUs provided by SLURM
IFS=',' read -ra AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"

export NCCL_NET_PLUGIN=none
export VLLM_WORKER_MULTIPROC_METHOD=spawn

########################################
# Run Inference (Data Parallel shards)
########################################

echo "Launching ${N_GPUS} parallel vLLM workers..." | tee -a "${LOG_FILE}"

for i in $(seq 0 $((N_GPUS - 1))); do
    # Ensure we don't index past what SLURM gave us
    GPU_IDX=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}
    
    SHARD_LOG="${LOG_DIR}/${ENV}_shard${i}.txt"
    echo " -> Worker $i assigned to GPU ${GPU_IDX}. Logging to ${SHARD_LOG}" | tee -a "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_IDX}" python -u run_inference.py \
        --env "${ENV}" \
        --model "${MODEL}" \
        --num_shards "${N_GPUS}" \
        --shard_id "$i" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --gpu_mem_util "${GPU_MEM_UTIL}" \
        --max_model_len "${MAX_MODEL_LEN}" \
        --dtype "${DTYPE}" \
        > "${SHARD_LOG}" 2>&1 &
done

# Wait for all background workers to finish
wait

echo "All workers completed. Merging JSON shards..." | tee -a "${LOG_FILE}"

########################################
# Merge script to reconstruct JSON lists
########################################
python -c "
import json, glob
from pathlib import Path

env = '${ENV}'
model = '${MODEL}'
out_dir = Path('outputs/responses') / model / env

# Discover jailbreak-type subfolders written under transformed/ (dan,
# opposite_mode, payload_split, bon_augment, ...) rather than hardcoding them.
jb_types = sorted(d.name for d in (out_dir / 'transformed').iterdir() if d.is_dir()) \
    if (out_dir / 'transformed').exists() else []
splits = ['base/harmful', 'base/safe'] + [
    f'transformed/{t}/{s}' for t in jb_types for s in ('harmful', 'safe')
]

for split in splits:
    pattern = str(out_dir / f'{split}_*.json')
    files = glob.glob(pattern)
    if not files: 
        continue

    data = []
    for f in files:
        with open(f, 'r') as fp:
            data.extend(json.load(fp))

    # Sort deterministically by orig_index
    data.sort(key=lambda x: x.get('orig_index', 0))

    out_path = out_dir / f'{split}.json'
    with open(out_path, 'w') as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)

    print(f'Merged {len(files)} shards for {split} -> {out_path} ({len(data)} entries)')

    # Clean up intermediate shards
    for f in files:
        Path(f).unlink()
" | tee -a "${LOG_FILE}"

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "Done." | tee -a "${LOG_FILE}"