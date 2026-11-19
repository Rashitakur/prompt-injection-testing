#!/usr/bin/env bash
# Create the Python environment once (login node, with internet).
#   bash scripts/setup_env.sh            -> conda env "pidetect"
#   USE_VENV=1 bash scripts/setup_env.sh -> ./.venv
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${USE_VENV:-0}" = "1" ] || ! command -v conda >/dev/null 2>&1; then
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  conda env create -f environment.yml || conda env update -f environment.yml
fi
echo "Environment ready. Test with: sbatch slurm/smoke.job   (or: bash scripts/run_smoke.sh)"
