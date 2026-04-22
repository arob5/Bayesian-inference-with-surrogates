#!/bin/bash -l
#$ -N rkpcn_bench
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=8G
#$ -P robustvb
#$ -pe omp 1
#
# Run a multi-replicate, multi-variant RKPCN benchmark.
#
# Usage:
#   cd experiments/benchmarks
#   qsub submit_benchmark.sh
#   qsub -v CONFIG=my_sweep.yaml,OUTPUT_SUFFIX=abl1 submit_benchmark.sh
#
# Configurable via environment variables:
#   CONFIG         - YAML config (default: multichain_sweep.yaml)
#   OUTPUT_DIR     - full output path; if unset, built from CONFIG name +
#                    timestamp under out/benchmarks/
#   OUTPUT_SUFFIX  - appended to the auto-generated OUTPUT_DIR name
#   SEED           - PRNG seed (default: 42)
#   SAVE_TRACE     - "1" to save full single-chain traces (default: 0)

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"
export JAX_ENABLE_X64=1

# Thread-limiting for JAX
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
export TMPDIR=/scratch/$USER/$JOB_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

CONFIG="${CONFIG:-multichain_sweep.yaml}"
SEED="${SEED:-42}"
SAVE_TRACE="${SAVE_TRACE:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-}"

if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)
    BASENAME=$(basename "$CONFIG" .yaml)
    SLUG="${BASENAME}_${TIMESTAMP}"
    if [ -n "$OUTPUT_SUFFIX" ]; then
        SLUG="${SLUG}_${OUTPUT_SUFFIX}"
    fi
    OUTPUT_DIR="${REPO_DIR}/out/benchmarks/${SLUG}"
fi

echo "JOB_ID=$JOB_ID"
hostname
python --version
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SEED=$SEED"
echo "SAVE_TRACE=$SAVE_TRACE"

cd "${SGE_O_WORKDIR}"

TRACE_FLAG=""
if [ "$SAVE_TRACE" = "1" ]; then
    TRACE_FLAG="--save-trace"
fi

exec python -u -m benchmark run \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    $TRACE_FLAG
