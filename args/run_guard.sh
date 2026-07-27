#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################

EXPERIMENT="guard"
GUARD_DIR="models/meta-llama/Llama-Guard-4-12B/"
MODEL="gemma"
ENV="bioterrorism"

# ENVIRONMENTS = [
#     "cybersecurity", "medical", "hate_harassment",
#     "general_crime", "bioterrorism",
# ]

# Data-parallel: one LG4 worker per GPU. V100 32GB fits 12B fp16 (~24GB).
N_GPUS=2

BATCH_SIZE=2
MAX_NEW_TOKENS=20
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
echo "Llama-Guard-4 refusal scoring (Data Parallel)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

echo "SLURM Job ID    : ${SLURM_JOB_ID}"   | tee -a "${LOG_FILE}"
echo "Node            : ${SLURMD_NODENAME}" | tee -a "${LOG_FILE}"
echo "Environment     : ${ENV}"            | tee -a "${LOG_FILE}"
echo "Guard model     : ${GUARD_DIR}"      | tee -a "${LOG_FILE}"
echo "GPUs for DP     : ${N_GPUS}"         | tee -a "${LOG_FILE}"
echo "Batch size      : ${BATCH_SIZE}"     | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# Safely extract the GPUs SLURM gave us.
IFS=',' read -ra AVAILABLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"

########################################
# Run scoring (one worker per GPU)
########################################

echo "Launching ${N_GPUS} parallel LG4 workers..." | tee -a "${LOG_FILE}"

for i in $(seq 0 $((N_GPUS - 1))); do
    GPU_IDX=${AVAILABLE_GPUS[$((i % ${#AVAILABLE_GPUS[@]}))]}

    SHARD_LOG="${LOG_DIR}/${ENV}_shard${i}.txt"
    echo " -> Worker $i assigned to GPU ${GPU_IDX}. Logging to ${SHARD_LOG}" | tee -a "${LOG_FILE}"

    CUDA_VISIBLE_DEVICES="${GPU_IDX}" python -u run_guard.py \
        --env "${ENV}" \
        --model "${MODEL}" \
        --guard_model "${GUARD_DIR}" \
        --num_shards "${N_GPUS}" \
        --shard_id "$i" \
        --batch_size "${BATCH_SIZE}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --dtype "${DTYPE}" \
        > "${SHARD_LOG}" 2>&1 &
done

wait

echo "All workers done. Merging scored shards..." | tee -a "${LOG_FILE}"

########################################
# Merge: fold <split>_scored_<shard>.json back into <split>.json in place.
# Merge key is orig_index + transformed_prompt (transformed sets have variants).
########################################
python -c "
import json, glob
from pathlib import Path

env = '${ENV}'
model = '${MODEL}'
root = Path('outputs/responses') / model / env

# Discover jailbreak-type subfolders under transformed/ (dan, opposite_mode,
# payload_split, bon_augment, ...) rather than hardcoding them.
jb_types = sorted(d.name for d in (root / 'transformed').iterdir() if d.is_dir()) \
    if (root / 'transformed').exists() else []
sets = ['base/harmful', 'base/safe'] + [
    f'transformed/{t}/{s}' for t in jb_types for s in ('harmful', 'safe')
]

def key(e):
    return (e.get('orig_index'), e.get('transformed_prompt'))

for s in sets:
    tgt = root / f'{s}.json'
    if not tgt.exists():
        continue
    stem = tgt.stem
    shards = glob.glob(str(tgt.parent / f'{stem}_scored_*.json'))
    if not shards:
        print(f'no scored shards for {s}; skipping')
        continue

    scored = {}
    for f in shards:
        for e in json.load(open(f)):
            scored[key(e)] = e

    base = json.load(open(tgt))
    hits = 0
    for i, e in enumerate(base):
        k = key(e)
        if k in scored:
            base[i] = scored[k]
            hits += 1

    json.dump(base, open(tgt, 'w'), indent=2, ensure_ascii=False)
    print(f'{s}: merged {hits}/{len(base)} scored entries -> {tgt}')

    for f in shards:
        Path(f).unlink()
" | tee -a "${LOG_FILE}"

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "Done." | tee -a "${LOG_FILE}"