#!/bin/bash -l
#$ -N vsem_analyze
#$ -j y
#$ -l h_rt=4:00:00
#$ -l mem_per_core=8G
#$ -P bayesij
#$ -pe omp 4
#
# Batch job for VSEM post-hoc analysis (W2 computation is CPU-intensive).
# Coverage plots and diagnostics are fast; W2 is the bottleneck.
#
# Usage:
#   cd experiments/vsem
#   qsub submit_analyze.sh

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"
export JAX_ENABLE_X64=1

echo "JOB_ID=$JOB_ID"
hostname
python --version

cd "${SGE_O_WORKDIR}"

exec python -u analyze.py \
    --experiment-name vsem
