#!/usr/bin/env bash
# =============================================================================
#  Sourced by every .job file. EDIT THE "SITE SETTINGS" BLOCK FOR YOUR CLUSTER.
# =============================================================================
set -euo pipefail

# ----------------------------- SITE SETTINGS ---------------------------------
# Modules (uncomment / adapt; run `module avail` on your cluster)
# module purge
# module load cuda/12.1
# module load anaconda3        # or: module load python/3.10

# Python environment (created once with: bash scripts/setup_env.sh)
ENV_NAME="${ENV_NAME:-pidetect}"
set +u   # conda/venv activation scripts reference unset variables
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}" || true
elif [ -f "${PROJECT_DIR:-$PWD}/.venv/bin/activate" ]; then
  source "${PROJECT_DIR:-$PWD}/.venv/bin/activate"
fi
set -u

# Large shared cache for HF models/datasets (scratch/work filesystem, NOT $HOME)
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/hf_cache}"
# -----------------------------------------------------------------------------

# Project root = directory containing this slurm/ folder
export PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "${PROJECT_DIR}"
export CONFIG="${CONFIG:-configs/default.yaml}"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Compute nodes are usually offline: everything must be pre-fetched by 00_download.job
if [ "${ALLOW_DOWNLOAD:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

mkdir -p logs
echo "=== $(date) | job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-NA} | host=$(hostname)"
echo "=== project=${PROJECT_DIR} config=${CONFIG} HF_HOME=${HF_HOME}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch,sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
