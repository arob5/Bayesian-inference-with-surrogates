#!/bin/bash -l
#$ -N rkpcn_analyze
#$ -j y
#$ -l h_rt=2:00:00
#$ -l mem_per_core=8G
#$ -P robustvb
#$ -pe omp 4
#
# Analyze an existing benchmark run (print summaries, compute W2 where
# applicable, produce all plots).
#
# Usage:
#   cd experiments/benchmarks
#   qsub -v RUN_DIR=/path/to/benchmark_run_20260422 submit_analyze.sh
#
# Required:
#   RUN_DIR - absolute path to the benchmark run directory
#             (the directory containing benchmark_manifest.json)
#
# Optional:
#   THIN - thinning factor for scatter plots and KDE (default: 5)
#
# Uses omp 4 because Sinkhorn/KDE can escape OMP_NUM_THREADS=1 via
# their internal thread pools.

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"
export JAX_ENABLE_X64=1

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4 inter_op_parallelism_threads=2"
export TMPDIR=/scratch/$USER/$JOB_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR must be set (qsub -v RUN_DIR=/path/to/benchmark_run submit_analyze.sh)"
    exit 1
fi

THIN="${THIN:-5}"

echo "JOB_ID=$JOB_ID"
hostname
python --version
echo "RUN_DIR=$RUN_DIR"
echo "THIN=$THIN"

cd "${SGE_O_WORKDIR}"

exec python -u -m benchmark analyze --output-dir "$RUN_DIR" --thin "$THIN"
