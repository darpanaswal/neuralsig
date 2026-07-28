#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################

EXPERIMENT="judge"
MODEL="gemma"
ENV="cybersecurity"

# ENVIRONMENTS = [
#     "cybersecurity", "medical", "hate_harassment",
#     "general_crime", "bioterrorism",
# ]

# Leave empty to score every set discovered for ENV (base + all
# transformed/<type>). Otherwise list specific sets. 
SETS=("base") # E.g. ("base" "dan" "artprompt" "opposite_mode" "payload_split" "prefix_injection").

# Leave empty to score both splits. Otherwise "harmful" or "safe".
SPLIT=""

BACKEND="openai"
JUDGE_MODEL=""          # default: gpt-4o-mini (openai) / LLAMA3 path (local)
CONCURRENCY=8            # openai backend: parallel API requests
BATCH_SIZE=8

PARTITION="cpu_p1"       # CPU-only partition: no GPU held, so this job
                         # doesn't compete for GPU allocations with others.
N_CPUS=4
WALLTIME="12:00:00"

########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${MODEL}_${ENV}.txt"

SCRIPT_PATH="$(readlink -f "$0")"

########################################
# Submit to SLURM if not already inside a job
########################################

if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"

    sbatch \
        --job-name="${EXPERIMENT}_${MODEL}_${ENV}" \
        --output="${LOG_FILE}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${N_CPUS}" \
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
if [ ! -L "$HOME/.config" ]; then
    mkdir -p "$WORK/.config"
    if [ -d "$HOME/.config" ]; then
        cp -a "$HOME/.config/." "$WORK/.config/" 2>/dev/null || true
        rm -rf "$HOME/.config"
    fi
    ln -s "$WORK/.config" "$HOME/.config"
fi

if [ ! -L "$HOME/.conda" ]; then
    mkdir -p "$WORK/.conda"
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

# CPU-only: hide any GPUs so the local backend (if used) and any CUDA-aware
# libraries can't grab one, freeing GPU nodes for other jobs. Submit this
# script again with a different MODEL/ENV/SETS to run more jobs in parallel.
export CUDA_VISIBLE_DEVICES=""

EXTRA_ARGS=()
if [ -n "${JUDGE_MODEL}" ]; then
    EXTRA_ARGS+=(--judge_model "${JUDGE_MODEL}")
fi
if [ "${BACKEND}" = "local" ]; then
    EXTRA_ARGS+=(--device cpu)
fi
if [ "${#SETS[@]}" -gt 0 ]; then
    EXTRA_ARGS+=(--sets "${SETS[@]}")
fi
if [ -n "${SPLIT}" ]; then
    EXTRA_ARGS+=(--split "${SPLIT}")
fi

########################################
# Logging
########################################

: > "${LOG_FILE}"

echo "========================================" | tee -a "${LOG_FILE}"
echo "LLM-Judge scoring (CPU-only)" | tee -a "${LOG_FILE}"
echo "========================================" | tee -a "${LOG_FILE}"

echo "SLURM Job ID    : ${SLURM_JOB_ID}"   | tee -a "${LOG_FILE}"
echo "Node            : ${SLURMD_NODENAME}" | tee -a "${LOG_FILE}"
echo "Model           : ${MODEL}"          | tee -a "${LOG_FILE}"
echo "Environment     : ${ENV}"            | tee -a "${LOG_FILE}"
echo "Sets            : ${SETS[*]:-<all>}" | tee -a "${LOG_FILE}"
echo "Split           : ${SPLIT:-<both>}"  | tee -a "${LOG_FILE}"
echo "Backend         : ${BACKEND}"        | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

python -u run_judge.py --rescore \
    --env "${ENV}" \
    --model "${MODEL}" \
    --backend "${BACKEND}" \
    --concurrency "${CONCURRENCY}" \
    --batch_size "${BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}" \
    >> "${LOG_FILE}" 2>&1

echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "Done." | tee -a "${LOG_FILE}"
