#!/usr/bin/env bash
# =============================================================================
#  Submit the full pipeline with SLURM dependencies.
#
#  Usage:   bash slurm/submit_all.sh
#  Options (env vars):
#     ACCOUNT=myproj        -> --account
#     GPU_PARTITION=gpu     -> partition for GPU jobs
#     CPU_PARTITION=cpu     -> partition for CPU jobs
#     QOS=normal            -> --qos
#     SKIP_DOWNLOAD=1       -> assets already cached (ran 00_download on login node)
#     CONFIG=configs/default.yaml
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
GPU_PARTITION="${GPU_PARTITION:-gpu}"
CPU_PARTITION="${CPU_PARTITION:-cpu}"
COMMON=(--parsable --export=ALL)
[ -n "${ACCOUNT:-}" ] && COMMON+=(--account="${ACCOUNT}")
[ -n "${QOS:-}" ] && COMMON+=(--qos="${QOS}")
G=(--partition="${GPU_PARTITION}")
C=(--partition="${CPU_PARTITION}")

sub() { sbatch "${COMMON[@]}" "$@"; }

if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  DATA=$(sub "${C[@]}" slurm/01_prepare_data.job)
else
  DL=$(sub "${C[@]}" slurm/00_download.job)
  DATA=$(sub "${C[@]}" --dependency=afterok:${DL} slurm/01_prepare_data.job)
fi
echo "data job: ${DATA}"

D="--dependency=afterok:${DATA}"
J1=$(sub "${C[@]}" $D slurm/02_tfidf_lr.job)
J2=$(sub "${G[@]}" $D slurm/03_embed_lr.job)
J3=$(sub "${G[@]}" $D slurm/04_distilbert.job)
J4=$(sub "${G[@]}" $D slurm/05_perplexity.job)
J5=$(sub "${G[@]}" $D slurm/06_llm_judge.job)
J6=$(sub "${G[@]}" $D slurm/07_reference.job)
TRAIN="${J1}:${J2}:${J3}:${J4}:${J5}:${J6}"
echo "detector jobs: ${TRAIN}"

# afterany: latency/aggregation still run if one optional detector failed
L1=$(sub "${G[@]}" --dependency=afterany:${TRAIN} slurm/08_latency_gpu.job)
L2=$(sub "${C[@]}" --dependency=afterany:${TRAIN} slurm/09_latency_cpu.job)
AG=$(sub "${C[@]}" --dependency=afterany:${L1}:${L2} slurm/10_aggregate.job)
echo "latency: ${L1} ${L2} | aggregate: ${AG}"
echo "monitor with: squeue -u \$USER   |   logs in ./logs/"
