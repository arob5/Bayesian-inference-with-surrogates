#!/bin/bash -l
#$ -N rkpcn_smoke
#$ -j y
#$ -l h_rt=0:30:00
#$ -l mem_per_core=8G
#$ -P robustvb
#$ -pe omp 1
#
# Smoke test: small variant sweep on one replicate. Validates the
# benchmark pipeline end-to-end before launching a full sweep.
#
# Usage:
#   cd experiments/benchmarks
#   qsub submit_smoke_test.sh                   # VSEM rep
#   qsub -v CONFIG=smoke_test_pde.yaml submit_smoke_test.sh   # PDE rep

REPO_DIR="$(cd "${SGE_O_WORKDIR}/../.." && pwd)"
source "${REPO_DIR}/.venv/bin/activate"
export JAX_ENABLE_X64=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
export TMPDIR=/scratch/$USER/$JOB_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

CONFIG="${CONFIG:-smoke_test.yaml}"
SEED="${SEED:-42}"

if [ -z "$OUTPUT_DIR" ]; then
    BASENAME=$(basename "$CONFIG" .yaml)
    OUTPUT_DIR="${REPO_DIR}/out/benchmarks/${BASENAME}"
fi

echo "JOB_ID=$JOB_ID"
hostname
python --version
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"

cd "${SGE_O_WORKDIR}"

exec python -u -m benchmark run \
    --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --save-trace
